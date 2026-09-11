use super::super::moe::launch_moe_kernel;
use super::super::tensor::{launch_tensor_kernel, validate_resident};
use super::*;

mod loader;
use loader::paged_mla_functions;

const PAGED_MLA_SOURCE: &str = include_str!("paged_mla/source.hip");

pub(crate) struct CtMlaWeightRef<'a> {
    pub packed: &'a DeviceBuffer,
    pub scales: &'a DeviceBuffer,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub scale_dtype: u32,
    pub bits: u32,
}

#[derive(Clone, Copy)]
struct PagedMlaFunctions {
    cache_append: usize,
    cache_append_q8: usize,
    cache_append_dsa_prologue_q8: usize,
    cache_append_q8_hadamard: usize,
    cache_transform_q8_hadamard: usize,
    dsa_mean_pool: usize,
    dsa_interval_pool: usize,
    dsa_interval_bounds: usize,
    dsa_kpool_compress: usize,
    cache_copy_q8_pair: usize,
    cache_append_mla_q8_bf16: usize,
    cache_append_mla_q8_bf16_indirect: usize,
    mla_hot_scatter_q8: usize,
    mla_hot_gather_q8: usize,
    dsa_clear: usize,
    dsa_gather_selection_scores: usize,
    mla_gather_selected_q8: usize,
    mla_gpu_hot_gather_q8: usize,
    mla_gpu_hot_pin: usize,
    mla_gpu_hot_invalidate: usize,
    dsa_merge_sequence_shards: usize,
    dsa_score: usize,
    dsa_quantize_query_i8: usize,
    dsa_score_i8: usize,
    dsa_score_native_wmma_i8: usize,
    dsa_score_wmma: usize,
    dsa_score_native_wmma: usize,
    dsa_score_native_wmma_rows2: usize,
    dsa_score_native_wmma_decode: usize,
    dsa_score_native_wmma_kpool: usize,
    dsa_score_prefix_native_wmma: usize,
    dsa_score_selected_native_wmma: usize,
    dsa_map_candidate_selection: usize,
    dsa_compact_prefixes: usize,
    dsa_select_prefix: usize,
    dsa_select: usize,
    dsa_select_compact: usize,
    dsa_select_threshold: usize,
    dsa_select_radix_stage: usize,
    dsa_select_tile_counts: usize,
    dsa_select_tile_scan: usize,
    dsa_select_tile_scatter: usize,
    dsa_expand_kpool_selection: usize,
    absorb_query: usize,
    absorb_query_wmma: usize,
    dense_attention: usize,
    sparse_attention: usize,
    dense_wmma: bool,
    decode_attention: usize,
    decode_partial: usize,
    decode_partial_wmma_q8: usize,
    prefill_wmma_q8_heads32: usize,
    decode_partial_wmma_q8_colpar: usize,
    decode_partial_wmma_q8_colpar512: usize,
    decode_partial_wmma_q8_colpar512_shared: usize,
    decode_partial_wmma_q8_colpar512_abl: usize,
    split_merge: usize,
    split_merge_pl: usize,
    selection_split: usize,
    shard_scale: usize,
    shard_merge_heads: usize,
    project_value: usize,
    project_value_wmma: usize,
    project_value_perm: usize,
    wavefront_size: u32,
}

#[cfg(test)]
static TEST_SPARSE_PREFILL_HEADS32: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[cfg(test)]
static TEST_SPARSE_PREFILL_WMMA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[cfg(test)]
static TEST_MLA_DECODE_WMMA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn mla_decode_wmma_enabled() -> bool {
    #[cfg(test)]
    {
        if !TEST_MLA_DECODE_WMMA.load(std::sync::atomic::Ordering::Relaxed) {
            return false;
        }
        true
    }
    #[cfg(not(test))]
    true
}
fn sparse_prefill_wmma_enabled() -> bool {
    #[cfg(test)]
    {
        return TEST_SPARSE_PREFILL_WMMA.load(std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(test))]
    true
}

/// 在节点注册前完成 paged MLA/DSA HIPRTC 编译，避免首个请求承担一次性开销。
pub fn warmup_paged_mla(device_id: i32) -> Result<(), String> {
    paged_mla_functions(device_id).map(|_| ())
}

pub(crate) struct PagedMlaSelectionShards {
    pub owner: std::sync::Arc<DeviceBuffer>,
    pub owner_counts: std::sync::Arc<DeviceBuffer>,
    pub peer: std::sync::Arc<DeviceBuffer>,
    pub peer_counts: std::sync::Arc<DeviceBuffer>,
}

/// 全局 top-k 只做一次，然后按固定 KV block parity 拆成两张紧凑候选表。
/// counts 使 attention 只扫本卡实际候选数，不把另一半填充槽当作计算量。
pub(crate) fn try_split_paged_mla_selection_parity(device_id: i32, selection: &DeviceBuffer, rows: usize, width: usize, block_size: usize) -> Result<PagedMlaSelectionShards, String> {
    if rows == 0 || width == 0 || block_size == 0 {
        return Err(format!("paged MLA selection split shape rows={rows} width={width} block={block_size} 非法"));
    }
    let table_bytes = rows.checked_mul(width).and_then(|n| n.checked_mul(std::mem::size_of::<u32>())).ok_or("paged MLA selection split table 大小溢出")?;
    let count_bytes = rows.checked_mul(std::mem::size_of::<u32>()).ok_or("paged MLA selection split counts 大小溢出")?;
    validate_resident(selection, device_id, table_bytes, "paged MLA global selection")?;
    let owner = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, table_bytes)?);
    let peer = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, table_bytes)?);
    let owner_counts = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, count_bytes)?);
    let peer_counts = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, count_bytes)?);
    let functions = paged_mla_functions(device_id)?;
    let mut d_selection = selection.pointer;
    let mut d_owner = owner.pointer;
    let mut d_peer = peer.pointer;
    let mut d_owner_counts = owner_counts.pointer;
    let mut d_peer_counts = peer_counts.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "paged MLA selection split rows 超过 u32")?;
    let mut width = u32::try_from(width).map_err(|_| "paged MLA selection split width 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "paged MLA selection split block 超过 u32")?;
    let mut arguments = [
        (&mut d_selection as *mut *mut c_void).cast(),
        (&mut d_owner as *mut *mut c_void).cast(),
        (&mut d_peer as *mut *mut c_void).cast(),
        (&mut d_owner_counts as *mut *mut c_void).cast(),
        (&mut d_peer_counts as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut width as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
    ];
    launch_moe_kernel(functions.selection_split, rows, 1, 256, 0, &mut arguments, "HIP MLA split global selection parity")?;
    Ok(PagedMlaSelectionShards { owner, owner_counts, peer, peer_counts })
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_f32_bf16(device_id: i32, input: &DeviceBuffer, cache: &DeviceBuffer, block_table: &DeviceBuffer, position: usize, rows: usize, columns: usize, block_size: usize) -> Result<(), String> {
    let elements = rows.checked_mul(columns).ok_or("paged cache append 元素数溢出")?;
    let end = position.checked_add(rows).ok_or("paged cache append position 溢出")?;
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("paged cache input 大小溢出")?, "paged cache input")?;
    validate_resident(cache, device_id, end.checked_mul(columns).and_then(|n| n.checked_mul(2)).ok_or("paged cache 大小溢出")?, "paged cache")?;
    validate_resident(block_table, device_id, end.div_ceil(block_size).checked_mul(4).ok_or("paged block table 大小溢出")?, "paged block table")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_cache = cache.pointer;
    let mut d_table = block_table.pointer;
    let mut position = u32::try_from(position).map_err(|_| "paged cache position 超过 u32")?;
    let mut rows = u32::try_from(rows).map_err(|_| "paged cache rows 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "paged cache columns 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "paged cache block_size 超过 u32")?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_cache as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut position as *mut u32).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.cache_append, u32::try_from(elements.div_ceil(256)).map_err(|_| "paged cache grid 超过 u32")?, 256, &mut arguments, "HIP paged cache append")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_mla_gpu_hot_gather_q8(
    device_id: i32,
    host: [&RegisteredHostBuffer; 3],
    cache: [&DeviceBuffer; 3],
    metadata: &DeviceBuffer,
    map_rows: usize,
    epoch: u32,
    selection: &DeviceBuffer,
    output: [&DeviceBuffer; 3],
    count: usize,
    columns: [usize; 3],
    cache_rows: usize,
    recent_rows: usize,
    host_rows: usize,
    context_rows: usize,
    trace_counts: Option<&DeviceBuffer>,
) -> Result<(), String> {
    if count == 0 || cache_rows == 0 || recent_rows == 0 || columns.contains(&0) || epoch == 0 || host_rows > context_rows || context_rows - host_rows > recent_rows || context_rows > map_rows {
        return Err(format!("GPU hot gather 状态非法: count={count} columns={columns:?} cache={cache_rows} recent={recent_rows} host={host_rows} context={context_rows} map={map_rows}"));
    }
    let row_bytes = [columns[0], columns[1].checked_mul(2).ok_or("GPU hot scale 行溢出")?, columns[2].checked_mul(2).ok_or("GPU hot rope 行溢出")?];
    let stored_rows = cache_rows.checked_add(recent_rows).ok_or("GPU hot cache 行数溢出")?;
    for i in 0..3 {
        let required = host_rows.checked_mul(row_bytes[i]).ok_or("GPU hot host 大小溢出")?;
        if host[i].bytes() < required {
            return Err(format!("GPU hot host[{i}] bytes={}，期望 {required}", host[i].bytes()));
        }
        validate_resident(cache[i], device_id, stored_rows.checked_mul(row_bytes[i]).ok_or("GPU hot cache 大小溢出")?, "GPU hot cache")?;
        validate_resident(output[i], device_id, count.checked_mul(row_bytes[i]).ok_or("GPU hot output 大小溢出")?, "GPU hot gather output")?;
    }
    let meta_words = cache_rows.checked_mul(2).and_then(|n| n.checked_add(map_rows)).and_then(|n| n.checked_add(1)).ok_or("GPU hot metadata 溢出")?;
    validate_resident(metadata, device_id, meta_words.checked_mul(4).ok_or("GPU hot metadata bytes 溢出")?, "GPU hot metadata")?;
    validate_resident(selection, device_id, count.checked_mul(4).ok_or("GPU hot selection 溢出")?, "GPU hot selection")?;
    if let Some(counts) = trace_counts {
        validate_resident(counts, device_id, 12, "GPU hot trace counts")?;
    }
    set_device(device_id)?;
    let functions = paged_mla_functions(device_id)?;
    let mut dimensions = [count, columns[0], columns[1], columns[2], cache_rows, recent_rows, host_rows, context_rows, map_rows, epoch as usize]
        .map(|v| u32::try_from(v).map_err(|_| "GPU hot dimension 超过 u32".to_owned()))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    if count < cache_rows {
        let mut pointers = [metadata.device_pointer(), selection.device_pointer()];
        let mut pin_dimensions = [dimensions[0], dimensions[8], dimensions[4], dimensions[6], epoch];
        let mut args = [ptr::null_mut(); 7];
        for (target, value) in args.iter_mut().zip(pointers.iter_mut()) {
            *target = (value as *mut usize).cast();
        }
        for (target, value) in args[2..].iter_mut().zip(pin_dimensions.iter_mut()) {
            *target = (value as *mut u32).cast();
        }
        launch_tensor_kernel(functions.mla_gpu_hot_pin, dimensions[0].div_ceil(256), 256, &mut args, "HIP GPU hot pin")?;
    }
    let mut pointers = [
        host[0].device_pointer(),
        host[1].device_pointer(),
        host[2].device_pointer(),
        cache[0].device_pointer(),
        cache[1].device_pointer(),
        cache[2].device_pointer(),
        metadata.device_pointer(),
        selection.device_pointer(),
        output[0].device_pointer(),
        output[1].device_pointer(),
        output[2].device_pointer(),
        trace_counts.map_or(0, DeviceBuffer::device_pointer),
    ];
    let mut args = [ptr::null_mut(); 22];
    for (target, value) in args.iter_mut().zip(pointers.iter_mut()) {
        *target = (value as *mut usize).cast();
    }
    for (target, value) in args[12..].iter_mut().zip(dimensions.iter_mut()) {
        *target = (value as *mut u32).cast();
    }
    // 行内仅搬运字节；较小 block 减少空闲 wave 与跨 wave barrier。
    launch_tensor_kernel(functions.mla_gpu_hot_gather_q8, dimensions[0], 64, &mut args, "HIP GPU hot gather Q8")
}

pub(crate) fn try_mla_gpu_hot_invalidate(device_id: i32, metadata: &DeviceBuffer, map_rows: usize, cache_rows: usize, keep: usize, end: usize) -> Result<(), String> {
    if end <= keep {
        return Ok(());
    }
    set_device(device_id)?;
    let mut pointer = metadata.device_pointer();
    let mut dimensions = [map_rows, cache_rows, keep, end].map(|v| u32::try_from(v).map_err(|_| "GPU hot invalidate dimension 超过 u32".to_owned())).into_iter().collect::<Result<Vec<_>, _>>()?;
    let mut args = [(&mut pointer as *mut usize).cast(), (&mut dimensions[0] as *mut u32).cast(), (&mut dimensions[1] as *mut u32).cast(), (&mut dimensions[2] as *mut u32).cast(), (&mut dimensions[3] as *mut u32).cast()];
    launch_tensor_kernel(paged_mla_functions(device_id)?.mla_gpu_hot_invalidate, ((end - keep).div_ceil(256)) as u32, 256, &mut args, "HIP GPU hot invalidate")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_mla_gather_selected_q8(
    device_id: i32,
    latent: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope: &DeviceBuffer,
    block_table: &DeviceBuffer,
    selection: &DeviceBuffer,
    gathered_latent: &DeviceBuffer,
    gathered_scales: &DeviceBuffer,
    gathered_rope: &DeviceBuffer,
    count: usize,
    latent_dim: usize,
    latent_group_size: usize,
    rope_dim: usize,
    block_size: usize,
) -> Result<(), String> {
    if count == 0 || latent_dim == 0 || latent_group_size == 0 || !latent_dim.is_multiple_of(latent_group_size) || block_size == 0 {
        return Err("MLA gather selected shape 非法".to_owned());
    }
    let groups = latent_dim / latent_group_size;
    validate_resident(latent, device_id, count.checked_mul(latent_dim).ok_or("MLA gather latent 溢出")?, "MLA gather latent")?;
    validate_resident(latent_scales, device_id, count.checked_mul(groups).and_then(|n| n.checked_mul(2)).ok_or("MLA gather scales 溢出")?, "MLA gather scales")?;
    validate_resident(rope, device_id, count.checked_mul(rope_dim).and_then(|n| n.checked_mul(2)).ok_or("MLA gather rope 溢出")?, "MLA gather rope")?;
    validate_resident(selection, device_id, count.checked_mul(4).ok_or("MLA gather selection 溢出")?, "MLA gather selection")?;
    validate_resident(gathered_latent, device_id, count.checked_mul(latent_dim).ok_or("MLA gather latent output 溢出")?, "MLA gather latent output")?;
    validate_resident(gathered_scales, device_id, count.checked_mul(groups).and_then(|n| n.checked_mul(2)).ok_or("MLA gather scales output 溢出")?, "MLA gather scales output")?;
    validate_resident(gathered_rope, device_id, count.checked_mul(rope_dim).and_then(|n| n.checked_mul(2)).ok_or("MLA gather rope output 溢出")?, "MLA gather rope output")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_latent = latent.pointer;
    let mut d_scales = latent_scales.pointer;
    let mut d_rope = rope.pointer;
    let mut d_table = block_table.pointer;
    let mut d_selection = selection.pointer;
    let mut d_g_latent = gathered_latent.pointer;
    let mut d_g_scales = gathered_scales.pointer;
    let mut d_g_rope = gathered_rope.pointer;
    let mut count_u32 = u32::try_from(count).map_err(|_| "MLA gather count 超过 u32")?;
    let mut latent_dim_u32 = u32::try_from(latent_dim).map_err(|_| "MLA gather latent_dim 超过 u32")?;
    let mut group_u32 = u32::try_from(latent_group_size).map_err(|_| "MLA gather group 超过 u32")?;
    let mut rope_dim_u32 = u32::try_from(rope_dim).map_err(|_| "MLA gather rope_dim 超过 u32")?;
    let mut block_u32 = u32::try_from(block_size).map_err(|_| "MLA gather block 超过 u32")?;
    let mut arguments = [
        (&mut d_latent as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_rope as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut d_selection as *mut *mut c_void).cast(),
        (&mut d_g_latent as *mut *mut c_void).cast(),
        (&mut d_g_scales as *mut *mut c_void).cast(),
        (&mut d_g_rope as *mut *mut c_void).cast(),
        (&mut count_u32 as *mut u32).cast(),
        (&mut latent_dim_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut rope_dim_u32 as *mut u32).cast(),
        (&mut block_u32 as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.mla_gather_selected_q8, count_u32, 256, &mut arguments, "HIP MLA gather selected Q8")?;
    Ok(())
}

pub fn try_paged_cache_append_f32_q8(
    device_id: i32,
    input: &DeviceBuffer,
    cache: &DeviceBuffer,
    scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    position: usize,
    rows: usize,
    columns: usize,
    group_size: usize,
    block_size: usize,
) -> Result<(), String> {
    if group_size == 0 || group_size > 256 || !group_size.is_power_of_two() || !columns.is_multiple_of(group_size) {
        return Err(format!("paged Q8 cache group_size={group_size} columns={columns} 非法"));
    }
    let elements = rows.checked_mul(columns).ok_or("paged Q8 cache append 元素数溢出")?;
    let groups_per_row = columns / group_size;
    let groups = rows.checked_mul(groups_per_row).ok_or("paged Q8 cache group 数溢出")?;
    let end = position.checked_add(rows).ok_or("paged Q8 cache append position 溢出")?;
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("paged Q8 cache input 大小溢出")?, "paged Q8 cache input")?;
    validate_resident(cache, device_id, end.checked_mul(columns).ok_or("paged Q8 cache 大小溢出")?, "paged Q8 cache")?;
    validate_resident(scales, device_id, end.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("paged Q8 cache scale 大小溢出")?, "paged Q8 cache scales")?;
    validate_resident(block_table, device_id, end.div_ceil(block_size).checked_mul(4).ok_or("paged Q8 block table 大小溢出")?, "paged Q8 block table")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_cache = cache.pointer;
    let mut d_scales = scales.pointer;
    let mut d_table = block_table.pointer;
    let mut position = u32::try_from(position).map_err(|_| "paged Q8 cache position 超过 u32")?;
    let mut rows = u32::try_from(rows).map_err(|_| "paged Q8 cache rows 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "paged Q8 cache columns 超过 u32")?;
    let mut group_size = u32::try_from(group_size).map_err(|_| "paged Q8 cache group_size 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "paged Q8 cache block_size 超过 u32")?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_cache as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut position as *mut u32).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut group_size as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.cache_append_q8, u32::try_from(groups).map_err(|_| "paged Q8 cache grid 超过 u32")?, group_size, &mut arguments, "HIP paged cache append Q8G")
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged_dsa_append_layernorm_rope_q8(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: &DeviceBuffer,
    cache: &DeviceBuffer,
    scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    position: usize,
    rows: usize,
    columns: usize,
    rotary_dim: usize,
    layout: crate::attention::rope::RotaryLayout,
    cos: &[f32],
    sin: &[f32],
    block_size: usize,
    eps: f32,
) -> Result<(), String> {
    try_paged_dsa_append_layernorm_rope_q8_at(device_id, input, weight, bias, cache, scales, block_table, position, position, rows, columns, rotary_dim, layout, cos, sin, block_size, eps)
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged_dsa_append_layernorm_rope_q8_at(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: &DeviceBuffer,
    cache: &DeviceBuffer,
    scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    cache_position: usize,
    rope_position: usize,
    rows: usize,
    columns: usize,
    rotary_dim: usize,
    layout: crate::attention::rope::RotaryLayout,
    cos: &[f32],
    sin: &[f32],
    block_size: usize,
    eps: f32,
) -> Result<(), String> {
    try_paged_dsa_append_layernorm_rope_q8_remote(device_id, device_id, input, weight, bias, cache, scales, block_table, cache_position, rope_position, rows, columns, rotary_dim, layout, cos, sin, block_size, eps)
}

/// 与 `_at` 相同，但 cache/scales/table 允许驻留在 cache_device_id（cooperative
/// 单行 append 直写 peer 卡）。kernel 仍在 device_id 的流上提交。
#[allow(clippy::too_many_arguments)]
pub fn try_paged_dsa_append_layernorm_rope_q8_remote(
    device_id: i32,
    cache_device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: &DeviceBuffer,
    cache: &DeviceBuffer,
    scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    cache_position: usize,
    rope_position: usize,
    rows: usize,
    columns: usize,
    rotary_dim: usize,
    layout: crate::attention::rope::RotaryLayout,
    cos: &[f32],
    sin: &[f32],
    block_size: usize,
    eps: f32,
) -> Result<(), String> {
    if rows == 0 || columns == 0 || columns > 256 || !columns.is_power_of_two() || rotary_dim == 0 || rotary_dim > columns || !rotary_dim.is_multiple_of(2) {
        return Err(format!("paged DSA fused prologue rows={rows} columns={columns} rotary_dim={rotary_dim} 非法"));
    }
    let elements = rows.checked_mul(columns).ok_or("paged DSA fused prologue 元素数溢出")?;
    let end = cache_position.checked_add(rows).ok_or("paged DSA fused prologue cache position 溢出")?;
    let rope_end = rope_position.checked_add(rows).ok_or("paged DSA fused prologue rope position 溢出")?;
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("paged DSA fused input 大小溢出")?, "paged DSA fused input")?;
    validate_resident(weight, device_id, columns.checked_mul(4).ok_or("paged DSA fused weight 大小溢出")?, "paged DSA fused weight")?;
    validate_resident(bias, device_id, columns.checked_mul(4).ok_or("paged DSA fused bias 大小溢出")?, "paged DSA fused bias")?;
    validate_resident(cache, cache_device_id, end.checked_mul(columns).ok_or("paged DSA fused cache 大小溢出")?, "paged DSA fused cache")?;
    validate_resident(scales, cache_device_id, end.checked_mul(2).ok_or("paged DSA fused scale 大小溢出")?, "paged DSA fused scales")?;
    validate_resident(block_table, cache_device_id, end.div_ceil(block_size).checked_mul(4).ok_or("paged DSA fused block table 大小溢出")?, "paged DSA fused block table")?;
    let half = rotary_dim / 2;
    let (cosine, sine) = super::super::tensor::resident_rope_tables(device_id, cos, sin, half, rope_position..rope_end)?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = weight.pointer;
    let mut d_bias = bias.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_cache = cache.pointer;
    let mut d_scales = scales.pointer;
    let mut d_table = block_table.pointer;
    let mut cache_position = u32::try_from(cache_position).map_err(|_| "paged DSA fused cache position 超过 u32")?;
    let mut rope_position = u32::try_from(rope_position).map_err(|_| "paged DSA fused rope position 超过 u32")?;
    let mut rows = u32::try_from(rows).map_err(|_| "paged DSA fused rows 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "paged DSA fused columns 超过 u32")?;
    let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "paged DSA fused rotary_dim 超过 u32")?;
    let mut split_half = u32::from(layout == crate::attention::rope::RotaryLayout::SplitHalf);
    let mut block_size = u32::try_from(block_size).map_err(|_| "paged DSA fused block_size 超过 u32")?;
    let mut eps = eps;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_weight as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_cache as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut cache_position as *mut u32).cast(),
        (&mut rope_position as *mut u32).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
        (&mut split_half as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
        (&mut eps as *mut f32).cast(),
    ];
    launch_tensor_kernel(functions.cache_append_dsa_prologue_q8, rows, 256, &mut arguments, "HIP paged DSA LayerNorm+RoPE+Q8 append")
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_f32_q8_hadamard(
    device_id: i32,
    input: &DeviceBuffer,
    cache: &DeviceBuffer,
    scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    position: usize,
    rows: usize,
    columns: usize,
    group_size: usize,
    block_size: usize,
) -> Result<(), String> {
    if rows == 0 || block_size == 0 || columns == 0 || columns > 256 || !columns.is_power_of_two() || group_size != columns {
        return Err(format!("paged Hadamard Q8 cache rows={rows} columns={columns} group_size={group_size} block_size={block_size} 非法"));
    }
    let elements = rows.checked_mul(columns).ok_or("paged Hadamard Q8 cache append 元素数溢出")?;
    let end = position.checked_add(rows).ok_or("paged Hadamard Q8 cache append position 溢出")?;
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("paged Hadamard Q8 cache input 大小溢出")?, "paged Hadamard Q8 cache input")?;
    validate_resident(cache, device_id, end.checked_mul(columns).ok_or("paged Hadamard Q8 cache 大小溢出")?, "paged Hadamard Q8 cache")?;
    validate_resident(scales, device_id, end.checked_mul(2).ok_or("paged Hadamard Q8 cache scale 大小溢出")?, "paged Hadamard Q8 cache scales")?;
    validate_resident(block_table, device_id, end.div_ceil(block_size).checked_mul(4).ok_or("paged Hadamard Q8 block table 大小溢出")?, "paged Hadamard Q8 block table")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_cache = cache.pointer;
    let mut d_scales = scales.pointer;
    let mut d_table = block_table.pointer;
    let mut position = u32::try_from(position).map_err(|_| "paged Hadamard Q8 cache position 超过 u32")?;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "paged Hadamard Q8 cache rows 超过 u32")?;
    let mut columns_u32 = u32::try_from(columns).map_err(|_| "paged Hadamard Q8 cache columns 超过 u32")?;
    let mut group_u32 = u32::try_from(group_size).map_err(|_| "paged Hadamard Q8 cache group_size 超过 u32")?;
    let mut block_u32 = u32::try_from(block_size).map_err(|_| "paged Hadamard Q8 cache block_size 超过 u32")?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_cache as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut position as *mut u32).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut block_u32 as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.cache_append_q8_hadamard, u32::try_from(rows).map_err(|_| "paged Hadamard Q8 cache grid 超过 u32")?, columns_u32, &mut arguments, "HIP paged cache append Hadamard Q8")
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_transform_q8_hadamard(
    device_id: i32,
    source: &DeviceBuffer,
    source_scales: &DeviceBuffer,
    target: &DeviceBuffer,
    target_scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    position: usize,
    rows: usize,
    columns: usize,
    block_size: usize,
) -> Result<(), String> {
    if rows == 0 || block_size == 0 || columns == 0 || columns > 256 || !columns.is_power_of_two() {
        return Err(format!("paged raw-Q8→Hadamard-Q8 rows={rows} columns={columns} block_size={block_size} 非法"));
    }
    let end = position.checked_add(rows).ok_or("paged raw-Q8→Hadamard-Q8 position 溢出")?;
    validate_resident(source, device_id, end.checked_mul(columns).ok_or("paged raw-Q8 source 大小溢出")?, "paged raw-Q8 source")?;
    validate_resident(source_scales, device_id, end.checked_mul(2).ok_or("paged raw-Q8 source scale 大小溢出")?, "paged raw-Q8 source scales")?;
    validate_resident(target, device_id, end.checked_mul(columns).ok_or("paged Hadamard target 大小溢出")?, "paged Hadamard target")?;
    validate_resident(target_scales, device_id, end.checked_mul(2).ok_or("paged Hadamard target scale 大小溢出")?, "paged Hadamard target scales")?;
    validate_resident(block_table, device_id, end.div_ceil(block_size).checked_mul(4).ok_or("paged Hadamard block table 大小溢出")?, "paged Hadamard block table")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_source = source.pointer;
    let mut d_source_scales = source_scales.pointer;
    let mut d_target = target.pointer;
    let mut d_target_scales = target_scales.pointer;
    let mut d_table = block_table.pointer;
    let mut position = u32::try_from(position).map_err(|_| "paged raw-Q8→Hadamard-Q8 position 超过 u32")?;
    let mut rows = u32::try_from(rows).map_err(|_| "paged raw-Q8→Hadamard-Q8 rows 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "paged raw-Q8→Hadamard-Q8 columns 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "paged raw-Q8→Hadamard-Q8 block_size 超过 u32")?;
    let mut arguments = [
        (&mut d_source as *mut *mut c_void).cast(),
        (&mut d_source_scales as *mut *mut c_void).cast(),
        (&mut d_target as *mut *mut c_void).cast(),
        (&mut d_target_scales as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut position as *mut u32).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.cache_transform_q8_hadamard, rows, columns, &mut arguments, "HIP paged raw-Q8 to Hadamard-Q8")
}

/// 从生产 raw-Q8/G cache 生成 HISA mean-pooled Q8/G block summary。
/// 该算子只负责布局转换；调用方决定 summary 的生命周期与是否参与 selection。
#[allow(clippy::too_many_arguments)]
pub fn try_dsa_mean_pool_q8(
    device_id: i32,
    keys: &DeviceBuffer,
    key_scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    pooled_keys: &DeviceBuffer,
    pooled_scales: &DeviceBuffer,
    pool_block_table: &DeviceBuffer,
    first_pool: usize,
    pool_rows: usize,
    context_rows: usize,
    columns: usize,
    group_size: usize,
    pool_size: usize,
    block_size: usize,
) -> Result<(), String> {
    if pool_rows == 0 || context_rows == 0 || columns == 0 || group_size == 0 || group_size > 256 || !group_size.is_power_of_two() || !columns.is_multiple_of(group_size) || pool_size == 0 || block_size == 0 {
        return Err("DSA HISA mean pool shape 非法".to_owned());
    }
    let total_pool_rows = context_rows.div_ceil(pool_size);
    if first_pool.checked_add(pool_rows).is_none_or(|end| end > total_pool_rows) {
        return Err(format!("DSA HISA mean pool range={first_pool}+{pool_rows} 超过 {total_pool_rows}"));
    }
    let groups_per_row = columns / group_size;
    validate_resident(keys, device_id, context_rows.checked_mul(columns).ok_or("DSA HISA raw keys 大小溢出")?, "DSA HISA raw keys")?;
    validate_resident(key_scales, device_id, context_rows.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("DSA HISA raw scales 大小溢出")?, "DSA HISA raw scales")?;
    validate_resident(block_table, device_id, context_rows.div_ceil(block_size).checked_mul(4).ok_or("DSA HISA raw table 大小溢出")?, "DSA HISA raw table")?;
    validate_resident(pooled_keys, device_id, total_pool_rows.checked_mul(columns).ok_or("DSA HISA pooled keys 大小溢出")?, "DSA HISA pooled keys")?;
    validate_resident(pooled_scales, device_id, total_pool_rows.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("DSA HISA pooled scales 大小溢出")?, "DSA HISA pooled scales")?;
    validate_resident(pool_block_table, device_id, total_pool_rows.div_ceil(block_size).checked_mul(4).ok_or("DSA HISA pool table 大小溢出")?, "DSA HISA pool table")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_keys = keys.pointer;
    let mut d_scales = key_scales.pointer;
    let mut d_table = block_table.pointer;
    let mut d_pooled_keys = pooled_keys.pointer;
    let mut d_pooled_scales = pooled_scales.pointer;
    let mut d_pool_table = pool_block_table.pointer;
    let mut first_pool = u32::try_from(first_pool).map_err(|_| "DSA HISA first_pool 超过 u32")?;
    let mut pool_rows = u32::try_from(pool_rows).map_err(|_| "DSA HISA pool_rows 超过 u32")?;
    let mut context_rows = u32::try_from(context_rows).map_err(|_| "DSA HISA context 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "DSA HISA columns 超过 u32")?;
    let mut group_size = u32::try_from(group_size).map_err(|_| "DSA HISA group 超过 u32")?;
    let mut pool_size = u32::try_from(pool_size).map_err(|_| "DSA HISA pool_size 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "DSA HISA block_size 超过 u32")?;
    let mut arguments = [
        (&mut d_keys as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut d_pooled_keys as *mut *mut c_void).cast(),
        (&mut d_pooled_scales as *mut *mut c_void).cast(),
        (&mut d_pool_table as *mut *mut c_void).cast(),
        (&mut first_pool as *mut u32).cast(),
        (&mut pool_rows as *mut u32).cast(),
        (&mut context_rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut group_size as *mut u32).cast(),
        (&mut pool_size as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
    ];
    let groups = pool_rows.checked_mul(columns / group_size).ok_or("DSA HISA mean pool grid 溢出")?;
    launch_tensor_kernel(functions.dsa_mean_pool, groups, group_size, &mut arguments, "HIP DSA HISA mean pool Q8G")
}

#[allow(clippy::too_many_arguments)]
pub fn try_dsa_interval_pool_q8(
    device_id: i32,
    keys: &DeviceBuffer,
    key_scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    lower: &DeviceBuffer,
    upper: &DeviceBuffer,
    first_pool: usize,
    pool_rows: usize,
    context_rows: usize,
    columns: usize,
    group_size: usize,
    pool_size: usize,
    block_size: usize,
) -> Result<(), String> {
    if pool_rows == 0 || context_rows == 0 || columns == 0 || group_size == 0 || group_size > 256 || !group_size.is_power_of_two() || !columns.is_multiple_of(group_size) || pool_size == 0 || block_size == 0 {
        return Err("DSA interval pool shape 非法".to_owned());
    }
    let total_pool_rows = context_rows.div_ceil(pool_size);
    if first_pool.checked_add(pool_rows).is_none_or(|end| end > total_pool_rows) {
        return Err(format!("DSA interval pool range={first_pool}+{pool_rows} 超过 {total_pool_rows}"));
    }
    let groups_per_row = columns / group_size;
    validate_resident(keys, device_id, context_rows.checked_mul(columns).ok_or("DSA interval raw keys 大小溢出")?, "DSA interval raw keys")?;
    validate_resident(key_scales, device_id, context_rows.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("DSA interval raw scales 大小溢出")?, "DSA interval raw scales")?;
    validate_resident(block_table, device_id, context_rows.div_ceil(block_size).checked_mul(4).ok_or("DSA interval raw table 大小溢出")?, "DSA interval raw table")?;
    let summary_bytes = total_pool_rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("DSA interval summary 大小溢出")?;
    validate_resident(lower, device_id, summary_bytes, "DSA interval lower")?;
    validate_resident(upper, device_id, summary_bytes, "DSA interval upper")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_keys = keys.pointer;
    let mut d_scales = key_scales.pointer;
    let mut d_table = block_table.pointer;
    let mut d_lower = lower.pointer;
    let mut d_upper = upper.pointer;
    let mut first_pool = u32::try_from(first_pool).map_err(|_| "DSA interval first_pool 超过 u32")?;
    let mut pool_rows = u32::try_from(pool_rows).map_err(|_| "DSA interval pool_rows 超过 u32")?;
    let mut context_rows = u32::try_from(context_rows).map_err(|_| "DSA interval context 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "DSA interval columns 超过 u32")?;
    let mut group_size = u32::try_from(group_size).map_err(|_| "DSA interval group 超过 u32")?;
    let mut pool_size = u32::try_from(pool_size).map_err(|_| "DSA interval pool_size 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "DSA interval block_size 超过 u32")?;
    let mut arguments = [
        (&mut d_keys as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut d_lower as *mut *mut c_void).cast(),
        (&mut d_upper as *mut *mut c_void).cast(),
        (&mut first_pool as *mut u32).cast(),
        (&mut pool_rows as *mut u32).cast(),
        (&mut context_rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut group_size as *mut u32).cast(),
        (&mut pool_size as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
    ];
    let groups = pool_rows.checked_mul(u32::try_from(groups_per_row).map_err(|_| "DSA interval pool group 数超过 u32")?).ok_or("DSA interval pool grid 溢出")?;
    launch_tensor_kernel(functions.dsa_interval_pool, groups, group_size, &mut arguments, "HIP DSA interval pool Q8G")
}

pub fn try_dsa_interval_score_bounds(
    device_id: i32,
    lower: &DeviceBuffer,
    upper: &DeviceBuffer,
    query: &DeviceBuffer,
    head_weights: &DeviceBuffer,
    bounds: &DeviceBuffer,
    pool_rows: usize,
    head_count: usize,
    head_dim: usize,
) -> Result<(), String> {
    if pool_rows == 0 || head_count == 0 || head_dim == 0 || head_dim > 256 || !head_dim.is_power_of_two() {
        return Err("DSA interval bound shape 非法".to_owned());
    }
    let summary_bytes = pool_rows.checked_mul(head_dim).and_then(|n| n.checked_mul(4)).ok_or("DSA interval bound summary 大小溢出")?;
    validate_resident(lower, device_id, summary_bytes, "DSA interval bound lower")?;
    validate_resident(upper, device_id, summary_bytes, "DSA interval bound upper")?;
    validate_resident(query, device_id, head_count.checked_mul(head_dim).and_then(|n| n.checked_mul(4)).ok_or("DSA interval bound query 大小溢出")?, "DSA interval bound query")?;
    validate_resident(head_weights, device_id, head_count.checked_mul(4).ok_or("DSA interval bound weights 大小溢出")?, "DSA interval bound weights")?;
    validate_resident(bounds, device_id, pool_rows.checked_mul(4).ok_or("DSA interval bounds 大小溢出")?, "DSA interval bounds")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_lower = lower.pointer;
    let mut d_upper = upper.pointer;
    let mut d_query = query.pointer;
    let mut d_weights = head_weights.pointer;
    let mut d_bounds = bounds.pointer;
    let mut pool_rows = u32::try_from(pool_rows).map_err(|_| "DSA interval pool_rows 超过 u32")?;
    let mut head_count = u32::try_from(head_count).map_err(|_| "DSA interval head_count 超过 u32")?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "DSA interval head_dim 超过 u32")?;
    let mut arguments = [
        (&mut d_lower as *mut *mut c_void).cast(),
        (&mut d_upper as *mut *mut c_void).cast(),
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_weights as *mut *mut c_void).cast(),
        (&mut d_bounds as *mut *mut c_void).cast(),
        (&mut pool_rows as *mut u32).cast(),
        (&mut head_count as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.dsa_interval_bounds, pool_rows, head_dim, &mut arguments, "HIP DSA interval score bounds")
}

#[allow(clippy::too_many_arguments)]
pub fn try_dsa_kpool_compress_q8(
    device_id: i32,
    keys: &DeviceBuffer,
    key_scales: &DeviceBuffer,
    block_table: &DeviceBuffer,
    gates: &DeviceBuffer,
    ape: &DeviceBuffer,
    pooled_keys: &DeviceBuffer,
    pooled_scales: &DeviceBuffer,
    pool_block_table: &DeviceBuffer,
    first_pool: usize,
    pool_rows: usize,
    context_rows: usize,
    capacity: usize,
    columns: usize,
    group_size: usize,
    kpool: usize,
    block_size: usize,
) -> Result<(), String> {
    if pool_rows == 0 {
        return Ok(());
    }
    if kpool == 0 || kpool > 8 || group_size == 0 || group_size > 256 || !group_size.is_power_of_two() || !columns.is_multiple_of(group_size) || first_pool.checked_add(pool_rows).is_none_or(|end| end > context_rows / kpool) {
        return Err(format!("DSA kpool 压缩 shape 非法: first={first_pool} pools={pool_rows} context={context_rows} columns={columns} group={group_size} kpool={kpool}"));
    }
    let groups_per_row = columns / group_size;
    let pool_capacity = capacity / kpool;
    validate_resident(keys, device_id, context_rows.checked_mul(columns).ok_or("DSA kpool key 大小溢出")?, "DSA kpool keys")?;
    validate_resident(key_scales, device_id, context_rows.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("DSA kpool key scale 大小溢出")?, "DSA kpool key scales")?;
    validate_resident(gates, device_id, context_rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool gate 大小溢出")?, "DSA kpool gates")?;
    validate_resident(ape, device_id, kpool.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool APE 大小溢出")?, "DSA kpool APE")?;
    validate_resident(pooled_keys, device_id, pool_capacity.checked_mul(columns).ok_or("DSA pooled key 大小溢出")?, "DSA pooled keys")?;
    validate_resident(pooled_scales, device_id, pool_capacity.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("DSA pooled scale 大小溢出")?, "DSA pooled scales")?;
    validate_resident(block_table, device_id, context_rows.div_ceil(block_size).checked_mul(4).ok_or("DSA block table 大小溢出")?, "DSA block table")?;
    validate_resident(pool_block_table, device_id, pool_capacity.div_ceil(block_size).checked_mul(4).ok_or("DSA pool block table 大小溢出")?, "DSA pool block table")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_keys = keys.pointer;
    let mut d_key_scales = key_scales.pointer;
    let mut d_table = block_table.pointer;
    let mut d_gates = gates.pointer;
    let mut d_ape = ape.pointer;
    let mut d_pooled_keys = pooled_keys.pointer;
    let mut d_pooled_scales = pooled_scales.pointer;
    let mut d_pool_table = pool_block_table.pointer;
    let mut first_pool = u32::try_from(first_pool).map_err(|_| "DSA first_pool 超过 u32")?;
    let mut pool_rows = u32::try_from(pool_rows).map_err(|_| "DSA pool_rows 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "DSA columns 超过 u32")?;
    let mut group_size = u32::try_from(group_size).map_err(|_| "DSA group_size 超过 u32")?;
    let mut kpool = u32::try_from(kpool).map_err(|_| "DSA kpool 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "DSA block_size 超过 u32")?;
    let mut arguments = [
        (&mut d_keys as *mut *mut c_void).cast(),
        (&mut d_key_scales as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut d_gates as *mut *mut c_void).cast(),
        (&mut d_ape as *mut *mut c_void).cast(),
        (&mut d_pooled_keys as *mut *mut c_void).cast(),
        (&mut d_pooled_scales as *mut *mut c_void).cast(),
        (&mut d_pool_table as *mut *mut c_void).cast(),
        (&mut first_pool as *mut u32).cast(),
        (&mut pool_rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut group_size as *mut u32).cast(),
        (&mut kpool as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
    ];
    let groups = pool_rows.checked_mul(u32::try_from(groups_per_row).map_err(|_| "DSA groups_per_row 超过 u32")?).ok_or("DSA kpool grid 溢出")?;
    launch_tensor_kernel(functions.dsa_kpool_compress, groups, group_size, &mut arguments, "HIP DSA kpool compress Q8G")
}

#[allow(clippy::too_many_arguments)]
pub fn try_q8_cache_copy_pair(
    device_id: i32,
    source_key: &DeviceBuffer,
    source_key_scales: &DeviceBuffer,
    source_value: &DeviceBuffer,
    source_value_scales: &DeviceBuffer,
    target_key: &DeviceBuffer,
    target_key_scales: &DeviceBuffer,
    target_value: &DeviceBuffer,
    target_value_scales: &DeviceBuffer,
    source_row: usize,
    target_row: usize,
    rows: usize,
    columns: usize,
    scale_columns: usize,
    target_capacity: usize,
) -> Result<(), String> {
    if rows == 0 || columns == 0 || scale_columns == 0 || target_capacity == 0 || target_row >= target_capacity {
        return Err(format!("Q8 cache pair copy shape 非法: source_row={source_row} target_row={target_row} rows={rows} columns={columns} scales={scale_columns} capacity={target_capacity}"));
    }
    let source_rows = source_row.checked_add(rows).ok_or("Q8 cache pair source rows 溢出")?;
    let code_source_bytes = source_rows.checked_mul(columns).ok_or("Q8 cache pair source code 大小溢出")?;
    let scale_source_bytes = source_rows.checked_mul(scale_columns).and_then(|n| n.checked_mul(2)).ok_or("Q8 cache pair source scale 大小溢出")?;
    let code_target_bytes = target_capacity.checked_mul(columns).ok_or("Q8 cache pair target code 大小溢出")?;
    let scale_target_bytes = target_capacity.checked_mul(scale_columns).and_then(|n| n.checked_mul(2)).ok_or("Q8 cache pair target scale 大小溢出")?;
    for (buffer, bytes, name) in [
        (source_key, code_source_bytes, "source key"),
        (source_value, code_source_bytes, "source value"),
        (source_key_scales, scale_source_bytes, "source key scales"),
        (source_value_scales, scale_source_bytes, "source value scales"),
        (target_key, code_target_bytes, "target key"),
        (target_value, code_target_bytes, "target value"),
        (target_key_scales, scale_target_bytes, "target key scales"),
        (target_value_scales, scale_target_bytes, "target value scales"),
    ] {
        validate_resident(buffer, device_id, bytes, name)?;
    }
    set_device(device_id)?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_source_key = source_key.pointer;
    let mut d_source_key_scales = source_key_scales.pointer;
    let mut d_source_value = source_value.pointer;
    let mut d_source_value_scales = source_value_scales.pointer;
    let mut d_target_key = target_key.pointer;
    let mut d_target_key_scales = target_key_scales.pointer;
    let mut d_target_value = target_value.pointer;
    let mut d_target_value_scales = target_value_scales.pointer;
    let mut source_row = u32::try_from(source_row).map_err(|_| "Q8 cache pair source_row 超过 u32")?;
    let mut target_row = u32::try_from(target_row).map_err(|_| "Q8 cache pair target_row 超过 u32")?;
    let mut rows = u32::try_from(rows).map_err(|_| "Q8 cache pair rows 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "Q8 cache pair columns 超过 u32")?;
    let mut scale_columns = u32::try_from(scale_columns).map_err(|_| "Q8 cache pair scale_columns 超过 u32")?;
    let mut target_capacity = u32::try_from(target_capacity).map_err(|_| "Q8 cache pair target_capacity 超过 u32")?;
    let mut arguments = [
        (&mut d_source_key as *mut *mut c_void).cast(),
        (&mut d_source_key_scales as *mut *mut c_void).cast(),
        (&mut d_source_value as *mut *mut c_void).cast(),
        (&mut d_source_value_scales as *mut *mut c_void).cast(),
        (&mut d_target_key as *mut *mut c_void).cast(),
        (&mut d_target_key_scales as *mut *mut c_void).cast(),
        (&mut d_target_value as *mut *mut c_void).cast(),
        (&mut d_target_value_scales as *mut *mut c_void).cast(),
        (&mut source_row as *mut u32).cast(),
        (&mut target_row as *mut u32).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut scale_columns as *mut u32).cast(),
        (&mut target_capacity as *mut u32).cast(),
    ];
    let elements = rows.checked_mul(columns.max(scale_columns)).ok_or("Q8 cache pair grid 溢出")?;
    launch_tensor_kernel(functions.cache_copy_q8_pair, elements.div_ceil(256), 256, &mut arguments, "HIP Q8 cache pair copy")
}

/// 间接 position 变体（graph 用）：positions = [cache_position, rope_position]
/// 的常驻 u32 buffer；replay 前 host 覆写该 buffer，graph 参数保持不变。
#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_mla_rope_indirect_f32_q8_bf16(
    device_id: i32,
    latent_input: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope_input: &DeviceBuffer,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    positions: &DeviceBuffer,
    rows: usize,
    latent_columns: usize,
    rope_columns: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    cos: &[f32],
    sin: &[f32],
    group_size: usize,
    block_size: usize,
) -> Result<(), String> {
    if rows == 0 || positions.bytes() < 8 {
        return Err("paged MLA indirect append positions buffer 非法".to_owned());
    }
    if group_size == 0 || group_size > 256 || !group_size.is_power_of_two() || !latent_columns.is_multiple_of(group_size) {
        return Err(format!("paged MLA Q8 cache group_size={group_size} latent_columns={latent_columns} 非法"));
    }
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > rope_columns {
        return Err(format!("paged MLA indirect RoPE dim={rotary_dim} rope_columns={rope_columns} 非法"));
    }
    let latent_elements = rows.checked_mul(latent_columns).ok_or("paged MLA indirect latent 元素数溢出")?;
    let rope_elements = rows.checked_mul(rope_columns).ok_or("paged MLA indirect rope 元素数溢出")?;
    let groups_per_row = latent_columns / group_size;
    let latent_groups = rows.checked_mul(groups_per_row).ok_or("paged MLA indirect group 数溢出")?;
    let blocks = latent_groups.checked_add(rope_elements.div_ceil(group_size)).ok_or("paged MLA indirect grid 溢出")?;
    validate_resident(latent_input, device_id, latent_elements.checked_mul(4).ok_or("paged MLA indirect latent input 溢出")?, "paged MLA indirect latent input")?;
    validate_resident(rope_input, device_id, rope_elements.checked_mul(4).ok_or("paged MLA indirect rope input 溢出")?, "paged MLA indirect rope input")?;
    validate_resident(positions, device_id, 8, "paged MLA indirect positions")?;
    let half = rotary_dim / 2;
    let table_rows = cos.len() / half;
    let (cosine, sine) = super::super::tensor::resident_rope_tables(device_id, cos, sin, half, 0..table_rows)?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_latent_input = latent_input.pointer;
    let mut d_latent_cache = latent_cache.pointer;
    let mut d_latent_scales = latent_scales.pointer;
    let mut d_rope_input = rope_input.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_rope_cache = rope_cache.pointer;
    let mut d_table = block_table.pointer;
    let mut d_positions = positions.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "paged MLA indirect rows 超过 u32")?;
    let mut latent_columns = u32::try_from(latent_columns).map_err(|_| "paged MLA indirect latent columns 超过 u32")?;
    let mut rope_columns = u32::try_from(rope_columns).map_err(|_| "paged MLA indirect rope columns 超过 u32")?;
    let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "paged MLA indirect rope dim 超过 u32")?;
    let mut split_half = u32::from(layout == RotaryLayout::SplitHalf);
    let mut group_size = u32::try_from(group_size).map_err(|_| "paged MLA indirect group_size 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "paged MLA indirect block_size 超过 u32")?;
    let mut arguments = [
        (&mut d_latent_input as *mut *mut c_void).cast(),
        (&mut d_latent_cache as *mut *mut c_void).cast(),
        (&mut d_latent_scales as *mut *mut c_void).cast(),
        (&mut d_rope_input as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_rope_cache as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut d_positions as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut latent_columns as *mut u32).cast(),
        (&mut rope_columns as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
        (&mut split_half as *mut u32).cast(),
        (&mut group_size as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.cache_append_mla_q8_bf16_indirect, u32::try_from(blocks).map_err(|_| "paged MLA indirect grid 超过 u32")?, group_size, &mut arguments, "HIP paged MLA cache append Q8G+BF16 indirect")
}

#[allow(clippy::too_many_arguments)]
fn try_paged_cache_append_mla_f32_q8_bf16_inner(
    device_id: i32,
    latent_input: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope_input: &DeviceBuffer,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    position: usize,
    rows: usize,
    latent_columns: usize,
    rope_columns: usize,
    group_size: usize,
    block_size: usize,
    rope_rotation: Option<(usize, usize, RotaryLayout, &[f32], &[f32])>,
    cache_device: Option<i32>,
    log: Option<(&DeviceBuffer, &DeviceBuffer, &DeviceBuffer, usize)>,
) -> Result<(), String> {
    if group_size == 0 || group_size > 256 || !group_size.is_power_of_two() || !latent_columns.is_multiple_of(group_size) {
        return Err(format!("paged MLA Q8 cache group_size={group_size} latent_columns={latent_columns} 非法"));
    }
    let latent_elements = rows.checked_mul(latent_columns).ok_or("paged MLA Q8 cache latent 元素数溢出")?;
    let rope_elements = rows.checked_mul(rope_columns).ok_or("paged MLA Q8 cache rope 元素数溢出")?;
    let groups_per_row = latent_columns / group_size;
    let latent_groups = rows.checked_mul(groups_per_row).ok_or("paged MLA Q8 cache group 数溢出")?;
    let rope_blocks = rope_elements.div_ceil(group_size);
    let blocks = latent_groups.checked_add(rope_blocks).ok_or("paged MLA Q8 cache grid 溢出")?;
    let end = position.checked_add(rows).ok_or("paged MLA Q8 cache position 溢出")?;
    // cooperative 单行 append 直写对端卡时，cache/scales/table 驻留在 peer 卡：
    // 校验按它们的实际驻留卡做，launch 仍在本卡流上（输入在本卡，写出跨卡）。
    let cache_device = cache_device.unwrap_or(device_id);
    validate_resident(latent_input, device_id, latent_elements.checked_mul(4).ok_or("paged MLA Q8 latent input 大小溢出")?, "paged MLA Q8 latent input")?;
    validate_resident(latent_cache, cache_device, end.checked_mul(latent_columns).ok_or("paged MLA Q8 latent cache 大小溢出")?, "paged MLA Q8 latent cache")?;
    validate_resident(latent_scales, cache_device, end.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("paged MLA Q8 latent scale 大小溢出")?, "paged MLA Q8 latent scales")?;
    validate_resident(rope_input, device_id, rope_elements.checked_mul(4).ok_or("paged MLA rope input 大小溢出")?, "paged MLA rope input")?;
    validate_resident(rope_cache, cache_device, end.checked_mul(rope_columns).and_then(|n| n.checked_mul(2)).ok_or("paged MLA rope cache 大小溢出")?, "paged MLA rope cache")?;
    validate_resident(block_table, cache_device, end.div_ceil(block_size).checked_mul(4).ok_or("paged MLA block table 大小溢出")?, "paged MLA block table")?;
    let resident_rotation = match rope_rotation {
        Some((rope_position, rotary_dim, layout, cos, sin)) => {
            if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > rope_columns {
                return Err(format!("paged MLA fused RoPE dim={rotary_dim} rope_columns={rope_columns} 非法"));
            }
            let half = rotary_dim / 2;
            let rope_end = rope_position.checked_add(rows).ok_or("paged MLA RoPE position 溢出")?;
            let tables = super::super::tensor::resident_rope_tables(device_id, cos, sin, half, rope_position..rope_end)?;
            Some((tables, rope_position, rotary_dim, layout))
        }
        None => None,
    };
    let functions = paged_mla_functions(device_id)?;
    let mut d_latent_input = latent_input.pointer;
    let mut d_latent_cache = latent_cache.pointer;
    let mut d_latent_scales = latent_scales.pointer;
    let mut d_rope_input = rope_input.pointer;
    let mut d_cosine = resident_rotation.as_ref().map_or(std::ptr::null_mut(), |((cosine, _), _, _, _)| cosine.pointer);
    let mut d_sine = resident_rotation.as_ref().map_or(std::ptr::null_mut(), |((_, sine), _, _, _)| sine.pointer);
    let mut d_rope_cache = rope_cache.pointer;
    let mut d_table = block_table.pointer;
    let mut position = u32::try_from(position).map_err(|_| "paged MLA cache position 超过 u32")?;
    let mut rope_position = u32::try_from(resident_rotation.as_ref().map_or(position as usize, |(_, rope_position, _, _)| *rope_position)).map_err(|_| "paged MLA RoPE position 超过 u32")?;
    let mut rows = u32::try_from(rows).map_err(|_| "paged MLA cache rows 超过 u32")?;
    let mut latent_columns = u32::try_from(latent_columns).map_err(|_| "paged MLA latent columns 超过 u32")?;
    let mut rope_columns = u32::try_from(rope_columns).map_err(|_| "paged MLA rope columns 超过 u32")?;
    let mut rotary_dim = u32::try_from(resident_rotation.as_ref().map_or(0, |(_, _, rotary_dim, _)| *rotary_dim)).map_err(|_| "paged MLA RoPE dim 超过 u32")?;
    let mut split_half = u32::from(resident_rotation.as_ref().is_some_and(|(_, _, _, layout)| *layout == RotaryLayout::SplitHalf));
    let mut group_size = u32::try_from(group_size).map_err(|_| "paged MLA Q8 group_size 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "paged MLA block_size 超过 u32")?;
    let log_span = log.map(|(_, _, _, row)| row + rows as usize);
    let (mut d_log_latent, mut d_log_scales, mut d_log_rope, mut log_row) = match log {
        Some((log_latent, log_scales, log_rope, row)) => {
            let span = log_span.expect("log span 已计算");
            validate_resident(log_latent, device_id, span.checked_mul(latent_columns as usize).ok_or("paged MLA log latent 大小溢出")?, "paged MLA log latent")?;
            validate_resident(log_scales, device_id, span.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("paged MLA log scales 大小溢出")?, "paged MLA log scales")?;
            validate_resident(log_rope, device_id, span.checked_mul(rope_columns as usize).and_then(|n| n.checked_mul(2)).ok_or("paged MLA log rope 大小溢出")?, "paged MLA log rope")?;
            (log_latent.pointer, log_scales.pointer, log_rope.pointer, u32::try_from(row).map_err(|_| "paged MLA log row 超过 u32")?)
        }
        None => (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), 0u32),
    };
    let mut arguments = [
        (&mut d_latent_input as *mut *mut c_void).cast(),
        (&mut d_latent_cache as *mut *mut c_void).cast(),
        (&mut d_latent_scales as *mut *mut c_void).cast(),
        (&mut d_rope_input as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_rope_cache as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut position as *mut u32).cast(),
        (&mut rope_position as *mut u32).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut latent_columns as *mut u32).cast(),
        (&mut rope_columns as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
        (&mut split_half as *mut u32).cast(),
        (&mut group_size as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
        (&mut d_log_latent as *mut *mut c_void).cast(),
        (&mut d_log_scales as *mut *mut c_void).cast(),
        (&mut d_log_rope as *mut *mut c_void).cast(),
        (&mut log_row as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.cache_append_mla_q8_bf16, u32::try_from(blocks).map_err(|_| "paged MLA Q8 cache grid 超过 u32")?, group_size, &mut arguments, "HIP paged MLA cache append Q8G+BF16")
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_mla_f32_q8_bf16(
    device_id: i32,
    latent_input: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope_input: &DeviceBuffer,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    position: usize,
    rows: usize,
    latent_columns: usize,
    rope_columns: usize,
    group_size: usize,
    block_size: usize,
) -> Result<(), String> {
    try_paged_cache_append_mla_f32_q8_bf16_inner(device_id, latent_input, latent_cache, latent_scales, rope_input, rope_cache, block_table, position, rows, latent_columns, rope_columns, group_size, block_size, None, None, None)
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_mla_rope_f32_q8_bf16(
    device_id: i32,
    latent_input: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope_input: &DeviceBuffer,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    position: usize,
    rows: usize,
    latent_columns: usize,
    rope_columns: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    group_size: usize,
    block_size: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), String> {
    try_paged_cache_append_mla_f32_q8_bf16_inner(
        device_id,
        latent_input,
        latent_cache,
        latent_scales,
        rope_input,
        rope_cache,
        block_table,
        position,
        rows,
        latent_columns,
        rope_columns,
        group_size,
        block_size,
        Some((position, rotary_dim, layout, cos, sin)),
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_mla_rope_at_f32_q8_bf16(
    device_id: i32,
    latent_input: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope_input: &DeviceBuffer,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    cache_position: usize,
    rope_position: usize,
    rows: usize,
    latent_columns: usize,
    rope_columns: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    group_size: usize,
    block_size: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), String> {
    try_paged_cache_append_mla_f32_q8_bf16_inner(
        device_id,
        latent_input,
        latent_cache,
        latent_scales,
        rope_input,
        rope_cache,
        block_table,
        cache_position,
        rows,
        latent_columns,
        rope_columns,
        group_size,
        block_size,
        Some((rope_position, rotary_dim, layout, cos, sin)),
        None,
        None,
    )
}

/// 热窗单行 append 的日志双写变体:量化结果同时写入窗口槽位与 64 行环形
/// mirror 日志,供批量 D2H 喂养 CPU mirror(append 路径零额外流操作)。
#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_mla_f32_q8_bf16_with_log(
    device_id: i32,
    latent_input: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope_input: &DeviceBuffer,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    position: usize,
    rows: usize,
    latent_columns: usize,
    rope_columns: usize,
    group_size: usize,
    block_size: usize,
    log_latent: &DeviceBuffer,
    log_scales: &DeviceBuffer,
    log_rope: &DeviceBuffer,
    log_row: usize,
) -> Result<(), String> {
    try_paged_cache_append_mla_f32_q8_bf16_inner(
        device_id,
        latent_input,
        latent_cache,
        latent_scales,
        rope_input,
        rope_cache,
        block_table,
        position,
        rows,
        latent_columns,
        rope_columns,
        group_size,
        block_size,
        None,
        None,
        Some((log_latent, log_scales, log_rope, log_row)),
    )
}

/// 热窗单行 append(带 RoPE)的日志双写变体,语义同上。
#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_mla_rope_at_f32_q8_bf16_with_log(
    device_id: i32,
    latent_input: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope_input: &DeviceBuffer,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    cache_position: usize,
    rope_position: usize,
    rows: usize,
    latent_columns: usize,
    rope_columns: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    group_size: usize,
    block_size: usize,
    cos: &[f32],
    sin: &[f32],
    log_latent: &DeviceBuffer,
    log_scales: &DeviceBuffer,
    log_rope: &DeviceBuffer,
    log_row: usize,
) -> Result<(), String> {
    try_paged_cache_append_mla_f32_q8_bf16_inner(
        device_id,
        latent_input,
        latent_cache,
        latent_scales,
        rope_input,
        rope_cache,
        block_table,
        cache_position,
        rows,
        latent_columns,
        rope_columns,
        group_size,
        block_size,
        Some((rope_position, rotary_dim, layout, cos, sin)),
        None,
        Some((log_latent, log_scales, log_rope, log_row)),
    )
}

/// cooperative 单行 append 的直写变体：目标 cache/scales/table 允许驻留在 peer 卡
/// （cache_device_id != device_id）。kernel 仍在 device_id 的流上提交——输入在本卡、
/// 输出跨卡直写，省掉 staging + parity 拷贝 + P2P 整段。调用方负责把写入完成事件
/// 经既有 event/P2P 链串到对端消费点。
#[allow(clippy::too_many_arguments)]
pub fn try_paged_cache_append_mla_rope_remote_f32_q8_bf16(
    device_id: i32,
    cache_device_id: i32,
    latent_input: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: &DeviceBuffer,
    rope_input: &DeviceBuffer,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    cache_position: usize,
    rope_position: usize,
    rows: usize,
    latent_columns: usize,
    rope_columns: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    group_size: usize,
    block_size: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), String> {
    try_paged_cache_append_mla_f32_q8_bf16_inner(
        device_id,
        latent_input,
        latent_cache,
        latent_scales,
        rope_input,
        rope_cache,
        block_table,
        cache_position,
        rows,
        latent_columns,
        rope_columns,
        group_size,
        block_size,
        Some((rope_position, rotary_dim, layout, cos, sin)),
        Some(cache_device_id),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn try_mla_hot_scatter_q8(
    device_id: i32,
    source_latent: &DeviceBuffer,
    source_scales: &DeviceBuffer,
    source_rope: &DeviceBuffer,
    slots: &DeviceBuffer,
    target_latent: &DeviceBuffer,
    target_scales: &DeviceBuffer,
    target_rope: &DeviceBuffer,
    rows: usize,
    latent_columns: usize,
    scale_columns: usize,
    rope_columns: usize,
    target_rows: usize,
) -> Result<(), String> {
    if rows == 0 || latent_columns == 0 || scale_columns == 0 || rope_columns == 0 || target_rows == 0 {
        return Err("MLA hot scatter shape 非法".to_owned());
    }
    validate_resident(source_latent, device_id, rows.checked_mul(latent_columns).ok_or("MLA hot latent 大小溢出")?, "MLA hot source latent")?;
    validate_resident(source_scales, device_id, rows.checked_mul(scale_columns).and_then(|n| n.checked_mul(2)).ok_or("MLA hot scales 大小溢出")?, "MLA hot source scales")?;
    validate_resident(source_rope, device_id, rows.checked_mul(rope_columns).and_then(|n| n.checked_mul(2)).ok_or("MLA hot rope 大小溢出")?, "MLA hot source rope")?;
    validate_resident(slots, device_id, rows.checked_mul(4).ok_or("MLA hot slots 大小溢出")?, "MLA hot slots")?;
    validate_resident(target_latent, device_id, target_rows.checked_mul(latent_columns).ok_or("MLA hot target latent 大小溢出")?, "MLA hot target latent")?;
    validate_resident(target_scales, device_id, target_rows.checked_mul(scale_columns).and_then(|n| n.checked_mul(2)).ok_or("MLA hot target scales 大小溢出")?, "MLA hot target scales")?;
    validate_resident(target_rope, device_id, target_rows.checked_mul(rope_columns).and_then(|n| n.checked_mul(2)).ok_or("MLA hot target rope 大小溢出")?, "MLA hot target rope")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_source_latent = source_latent.pointer;
    let mut d_source_scales = source_scales.pointer;
    let mut d_source_rope = source_rope.pointer;
    let mut d_slots = slots.pointer;
    let mut d_target_latent = target_latent.pointer;
    let mut d_target_scales = target_scales.pointer;
    let mut d_target_rope = target_rope.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "MLA hot rows 超过 u32")?;
    let mut latent_columns = u32::try_from(latent_columns).map_err(|_| "MLA hot latent columns 超过 u32")?;
    let mut scale_columns = u32::try_from(scale_columns).map_err(|_| "MLA hot scale columns 超过 u32")?;
    let mut rope_columns = u32::try_from(rope_columns).map_err(|_| "MLA hot rope columns 超过 u32")?;
    let mut target_rows = u32::try_from(target_rows).map_err(|_| "MLA hot target rows 超过 u32")?;
    let mut arguments = [
        (&mut d_source_latent as *mut *mut c_void).cast(),
        (&mut d_source_scales as *mut *mut c_void).cast(),
        (&mut d_source_rope as *mut *mut c_void).cast(),
        (&mut d_slots as *mut *mut c_void).cast(),
        (&mut d_target_latent as *mut *mut c_void).cast(),
        (&mut d_target_scales as *mut *mut c_void).cast(),
        (&mut d_target_rope as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut latent_columns as *mut u32).cast(),
        (&mut scale_columns as *mut u32).cast(),
        (&mut rope_columns as *mut u32).cast(),
        (&mut target_rows as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.mla_hot_scatter_q8, rows, 256, &mut arguments, "HIP MLA hot scatter Q8+BF16")
}

#[allow(clippy::too_many_arguments)]
pub fn try_mla_hot_gather_q8(
    device_id: i32,
    source_latent: &DeviceBuffer,
    source_scales: Option<&DeviceBuffer>,
    source_rope: &DeviceBuffer,
    tokens: &DeviceBuffer,
    slots: &DeviceBuffer,
    target_latent: &DeviceBuffer,
    target_scales: &DeviceBuffer,
    target_rope: &DeviceBuffer,
    rows: usize,
    latent_columns: usize,
    scale_columns: usize,
    rope_columns: usize,
    target_rows: usize,
) -> Result<(), String> {
    if rows == 0 || latent_columns == 0 || scale_columns == 0 || rope_columns == 0 || target_rows == 0 {
        return Err("MLA hot gather shape 非法".to_owned());
    }
    let source_scales = source_scales.ok_or("MLA hot gather 需要 Q8 source scales")?;
    // source 是全量旧 buffer（H2D+扩容 后已远超 rows*cols），按 rows 取下界足以覆盖 kernel 访问范围。
    validate_resident(source_latent, device_id, rows.checked_mul(latent_columns).ok_or("MLA hot latent 大小溢出")?, "MLA hot source latent")?;
    validate_resident(source_scales, device_id, rows.checked_mul(scale_columns).and_then(|n| n.checked_mul(2)).ok_or("MLA hot scales 大小溢出")?, "MLA hot source scales")?;
    validate_resident(source_rope, device_id, rows.checked_mul(rope_columns).and_then(|n| n.checked_mul(2)).ok_or("MLA hot rope 大小溢出")?, "MLA hot source rope")?;
    validate_resident(tokens, device_id, rows.checked_mul(4).ok_or("MLA hot tokens 大小溢出")?, "MLA hot gather tokens")?;
    validate_resident(slots, device_id, rows.checked_mul(4).ok_or("MLA hot slots 大小溢出")?, "MLA hot slots")?;
    validate_resident(target_latent, device_id, target_rows.checked_mul(latent_columns).ok_or("MLA hot target latent 大小溢出")?, "MLA hot target latent")?;
    validate_resident(target_scales, device_id, target_rows.checked_mul(scale_columns).and_then(|n| n.checked_mul(2)).ok_or("MLA hot target scales 大小溢出")?, "MLA hot target scales")?;
    validate_resident(target_rope, device_id, target_rows.checked_mul(rope_columns).and_then(|n| n.checked_mul(2)).ok_or("MLA hot target rope 大小溢出")?, "MLA hot target rope")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_source_latent = source_latent.pointer;
    let mut d_source_scales = source_scales.pointer;
    let mut d_source_rope = source_rope.pointer;
    let mut d_tokens = tokens.pointer;
    let mut d_slots = slots.pointer;
    let mut d_target_latent = target_latent.pointer;
    let mut d_target_scales = target_scales.pointer;
    let mut d_target_rope = target_rope.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "MLA hot rows 超过 u32")?;
    let mut latent_columns = u32::try_from(latent_columns).map_err(|_| "MLA hot latent columns 超过 u32")?;
    let mut scale_columns = u32::try_from(scale_columns).map_err(|_| "MLA hot scale columns 超过 u32")?;
    let mut rope_columns = u32::try_from(rope_columns).map_err(|_| "MLA hot rope columns 超过 u32")?;
    let mut target_rows = u32::try_from(target_rows).map_err(|_| "MLA hot target rows 超过 u32")?;
    let mut arguments = [
        (&mut d_source_latent as *mut *mut c_void).cast(),
        (&mut d_source_scales as *mut *mut c_void).cast(),
        (&mut d_source_rope as *mut *mut c_void).cast(),
        (&mut d_tokens as *mut *mut c_void).cast(),
        (&mut d_slots as *mut *mut c_void).cast(),
        (&mut d_target_latent as *mut *mut c_void).cast(),
        (&mut d_target_scales as *mut *mut c_void).cast(),
        (&mut d_target_rope as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut latent_columns as *mut u32).cast(),
        (&mut scale_columns as *mut u32).cast(),
        (&mut rope_columns as *mut u32).cast(),
        (&mut target_rows as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.mla_hot_gather_q8, rows, 256, &mut arguments, "HIP MLA hot gather Q8+BF16")
}

#[derive(Default)]
struct PagedDsaWorkspace {
    quantized_query: Option<std::rc::Rc<DeviceBuffer>>,
    quantized_query_bytes: usize,
    query_scales: Option<std::rc::Rc<DeviceBuffer>>,
    query_scale_bytes: usize,
    scores: Option<std::rc::Rc<DeviceBuffer>>,
    scores_bytes: usize,
    score_prefixes: Option<std::rc::Rc<DeviceBuffer>>,
    score_prefixes_bytes: usize,
    candidates: Option<std::rc::Rc<DeviceBuffer>>,
    candidates_bytes: usize,
    candidate_metadata: Option<std::rc::Rc<DeviceBuffer>>,
    candidate_metadata_bytes: usize,
    coarse_histograms: Option<std::rc::Rc<DeviceBuffer>>,
    coarse_histograms_bytes: usize,
    select_tiles: Option<std::rc::Rc<DeviceBuffer>>,
    select_tiles_bytes: usize,
}

fn reserve_paged_dsa_buffer(buffer: &mut Option<std::rc::Rc<DeviceBuffer>>, capacity: &mut usize, device_id: i32, bytes: usize) -> Result<std::rc::Rc<DeviceBuffer>, String> {
    if *capacity < bytes {
        // score 工作区随上下文增长；按小块预留，减少跨 score tile 时的
        // hipMalloc/hipFree。大 prefill 最多额外保留不足 64 KiB。
        let reserved = if bytes < 65536 { bytes.next_power_of_two() } else { bytes.checked_add(65535).ok_or("paged DSA scratch 容量溢出")? / 65536 * 65536 };
        *buffer = Some(std::rc::Rc::new(DeviceBuffer::allocate(device_id, reserved)?));
        *capacity = reserved;
    }
    buffer.as_ref().cloned().ok_or_else(|| "paged DSA scratch 分配失败".to_owned())
}

thread_local! {
    static PAGED_DSA_WORKSPACES: std::cell::RefCell<std::collections::HashMap<(i32, usize), PagedDsaWorkspace>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

#[derive(Default)]
struct PagedMlaWorkspace {
    absorbed: Option<std::rc::Rc<DeviceBuffer>>,
    absorbed_bytes: usize,
    weighted: Option<std::rc::Rc<DeviceBuffer>>,
    weighted_bytes: usize,
    // 重排结果必须活到 attention 消费之后；按提交线程/流复用，避免每层
    // 反复取还池块，也避免非 stage 调用在 scan 发射前回收裸指针的 owner。
    gathered_latent: Option<std::rc::Rc<DeviceBuffer>>,
    gathered_latent_bytes: usize,
    gathered_scales: Option<std::rc::Rc<DeviceBuffer>>,
    gathered_scales_bytes: usize,
    gathered_rope: Option<std::rc::Rc<DeviceBuffer>>,
    gathered_rope_bytes: usize,
}

thread_local! {
    static PAGED_MLA_WORKSPACES: std::cell::RefCell<std::collections::HashMap<(i32, usize), PagedMlaWorkspace>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

#[derive(Default)]
struct PagedMlaSplitWorkspace {
    partial: Option<std::rc::Rc<DeviceBuffer>>,
    partial_bytes: usize,
    stats: Option<std::rc::Rc<DeviceBuffer>>,
    stats_bytes: usize,
}

thread_local! {
    static PAGED_MLA_SPLIT_WORKSPACES: std::cell::RefCell<std::collections::HashMap<(i32, usize), PagedMlaSplitWorkspace>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

pub(crate) fn release_paged_mla_workspaces(device_id: i32) {
    PAGED_DSA_WORKSPACES.with(|workspaces| {
        workspaces.borrow_mut().retain(|&(device, _), _| device != device_id);
    });
    PAGED_MLA_WORKSPACES.with(|workspaces| {
        workspaces.borrow_mut().retain(|&(device, _), _| device != device_id);
    });
    PAGED_MLA_SPLIT_WORKSPACES.with(|workspaces| {
        workspaces.borrow_mut().retain(|&(device, _), _| device != device_id);
    });
}

/// shadow/oracle 在下一次 DSA score 覆盖 workspace 前同步读取最近一行有序分数。
/// 只由显式 profile 使用，生产路径不会发生 D2H。
pub fn try_download_last_dsa_score_keys(device_id: i32, elements: usize) -> Result<Vec<u32>, String> {
    if elements == 0 {
        return Err("DSA shadow score elements=0 非法".to_owned());
    }
    let key = crate::kernel::rocm::hip::compute_workspace_key(device_id);
    let scores = PAGED_DSA_WORKSPACES.with(|workspaces| workspaces.borrow().get(&key).and_then(|workspace| workspace.scores.clone()).ok_or_else(|| "DSA shadow 缺少 score workspace".to_owned()))?;
    let bytes = elements.checked_mul(std::mem::size_of::<u32>()).ok_or("DSA shadow score 字节数溢出")?;
    if bytes > scores.bytes() {
        return Err(format!("DSA shadow score bytes={bytes} 超过 workspace={}", scores.bytes()));
    }
    let mut values = vec![0_u32; elements];
    scores.copy_to_host(unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), bytes) })?;
    Ok(values)
}

/// 候选域 exact rerank：用生产 raw-Q8 K + BF16 Q 重算候选，并在
/// candidate-local score 上做稳定 Top-K。候选必须按原 token 顺序传入。
#[allow(clippy::too_many_arguments)]
pub fn try_dsa_rerank_paged_q8_candidates(
    device_id: i32,
    keys: &DeviceBuffer,
    key_scales: &DeviceBuffer,
    key_group_size: usize,
    block_table: &DeviceBuffer,
    query: &DeviceBuffer,
    head_weights: &DeviceBuffer,
    candidates: &DeviceBuffer,
    query_rows: usize,
    context_rows: usize,
    query_start: usize,
    head_count: usize,
    head_dim: usize,
    candidate_count: usize,
    top_k: usize,
    block_size: usize,
) -> Result<DeviceBuffer, String> {
    if query_rows != 1
        || context_rows <= candidate_count
        || query_start + query_rows != context_rows
        || candidate_count <= top_k
        || top_k == 0
        || head_count == 0
        || head_dim == 0
        || key_group_size == 0
        || !head_dim.is_multiple_of(key_group_size)
        || block_size == 0
    {
        return Err(format!("DSA candidate rerank shape rows={query_rows} context={context_rows} start={query_start} candidates={candidate_count} top_k={top_k} 非法"));
    }
    let functions = paged_mla_functions(device_id)?;
    if !functions.dense_wmma {
        return Err("DSA candidate rerank 需要 native BF16 WMMA".to_owned());
    }
    let key_groups = head_dim / key_group_size;
    validate_resident(keys, device_id, context_rows.checked_mul(head_dim).ok_or("DSA rerank keys 大小溢出")?, "DSA rerank keys")?;
    validate_resident(key_scales, device_id, context_rows.checked_mul(key_groups).and_then(|n| n.checked_mul(2)).ok_or("DSA rerank scales 大小溢出")?, "DSA rerank scales")?;
    validate_resident(query, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).and_then(|n| n.checked_mul(4)).ok_or("DSA rerank query 大小溢出")?, "DSA rerank query")?;
    validate_resident(head_weights, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(4)).ok_or("DSA rerank weights 大小溢出")?, "DSA rerank weights")?;
    validate_resident(candidates, device_id, query_rows.checked_mul(candidate_count).and_then(|n| n.checked_mul(4)).ok_or("DSA rerank candidates 大小溢出")?, "DSA rerank candidates")?;
    validate_resident(block_table, device_id, context_rows.div_ceil(block_size).checked_mul(4).ok_or("DSA rerank block table 大小溢出")?, "DSA rerank block table")?;

    let score_bytes = query_rows.checked_mul(candidate_count).and_then(|n| n.checked_mul(4)).ok_or("DSA rerank score 大小溢出")?;
    let selection_bytes = query_rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("DSA rerank selection 大小溢出")?;
    let scores = DeviceBuffer::allocate(device_id, score_bytes)?;
    let slots = DeviceBuffer::allocate(device_id, selection_bytes)?;
    let selection = DeviceBuffer::allocate_reusable(device_id, selection_bytes)?;

    let profile = super::options().profile_dsa;
    if profile {
        super::synchronize_device(device_id, "hipDeviceSynchronize DSA candidate rerank boundary")?;
    }
    let score_started = profile.then(std::time::Instant::now);
    let mut d_keys = keys.pointer;
    let mut d_key_scales = key_scales.pointer;
    let mut d_table = block_table.pointer;
    let mut d_query = query.pointer;
    let mut d_weights = head_weights.pointer;
    let mut d_candidates = candidates.pointer;
    let mut d_candidate_counts = ptr::null_mut();
    let mut candidate_capacity = u32::try_from(candidate_count).map_err(|_| "DSA rerank candidate_count 超过 u32")?;
    let mut d_overflow_rows = ptr::null_mut();
    let mut d_overflow_count = ptr::null_mut();
    let mut d_overflow_cursor = ptr::null_mut();
    let mut d_scores = scores.pointer;
    let mut rows = u32::try_from(query_rows).map_err(|_| "DSA rerank rows 超过 u32")?;
    let mut context = u32::try_from(context_rows).map_err(|_| "DSA rerank context 超过 u32")?;
    let mut start = u32::try_from(query_start).map_err(|_| "DSA rerank start 超过 u32")?;
    let mut heads = u32::try_from(head_count).map_err(|_| "DSA rerank heads 超过 u32")?;
    let mut dim = u32::try_from(head_dim).map_err(|_| "DSA rerank head_dim 超过 u32")?;
    let mut group = u32::try_from(key_group_size).map_err(|_| "DSA rerank group 超过 u32")?;
    let mut block = u32::try_from(block_size).map_err(|_| "DSA rerank block 超过 u32")?;
    let mut score_stride = candidate_capacity;
    let mut candidate_local = 1u32;
    let mut score_args = [
        (&mut d_keys as *mut *mut c_void).cast(),
        (&mut d_key_scales as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_weights as *mut *mut c_void).cast(),
        (&mut d_candidates as *mut *mut c_void).cast(),
        (&mut d_candidate_counts as *mut *mut c_void).cast(),
        (&mut candidate_capacity as *mut u32).cast(),
        (&mut d_overflow_rows as *mut *mut c_void).cast(),
        (&mut d_overflow_count as *mut *mut c_void).cast(),
        (&mut d_overflow_cursor as *mut *mut c_void).cast(),
        (&mut d_scores as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut context as *mut u32).cast(),
        (&mut start as *mut u32).cast(),
        (&mut heads as *mut u32).cast(),
        (&mut dim as *mut u32).cast(),
        (&mut group as *mut u32).cast(),
        (&mut block as *mut u32).cast(),
        (&mut score_stride as *mut u32).cast(),
        (&mut candidate_local as *mut u32).cast(),
    ];
    launch_moe_kernel(functions.dsa_score_selected_native_wmma, candidate_capacity.div_ceil(128), rows, 256, 0, &mut score_args, "HIP DSA raw candidate rerank score")?;
    if profile {
        super::synchronize_device(device_id, "hipDeviceSynchronize DSA candidate rerank score")?;
    }
    let score_ms = score_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e3);

    let select_started = profile.then(std::time::Instant::now);
    let mut d_slots = slots.pointer;
    let mut candidate_start = candidate_capacity - 1;
    let mut top_k = u32::try_from(top_k).map_err(|_| "DSA rerank top_k 超过 u32")?;
    let mut select_args = [
        (&mut d_scores as *mut *mut c_void).cast(),
        (&mut d_slots as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut score_stride as *mut u32).cast(),
        (&mut candidate_start as *mut u32).cast(),
        (&mut top_k as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.dsa_select, rows, 256, &mut select_args, "HIP DSA candidate-local exact radix")?;
    let mut d_selection = selection.pointer;
    let elements = rows.checked_mul(top_k).ok_or("DSA rerank output elements 溢出")?;
    let mut map_args = [
        (&mut d_candidates as *mut *mut c_void).cast(),
        (&mut d_slots as *mut *mut c_void).cast(),
        (&mut d_selection as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut candidate_capacity as *mut u32).cast(),
        (&mut top_k as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.dsa_map_candidate_selection, elements.div_ceil(256), 256, &mut map_args, "HIP DSA map candidate exact selection")?;
    if profile {
        super::synchronize_device(device_id, "hipDeviceSynchronize DSA candidate rerank select")?;
        let select_ms = select_started.expect("DSA rerank profile timer 已创建").elapsed().as_secs_f64() * 1e3;
        eprintln!("[dsa-rerank-profile] device={device_id} rows={query_rows} context={context_rows} candidates={candidate_count} top_k={} score_ms={score_ms:.3} select_ms={select_ms:.3} total_ms={:.3}", top_k, score_ms + select_ms,);
    }
    Ok(selection)
}

const PARALLEL_SELECT_TILE_ROWS: usize = 1024;
const PARALLEL_SELECT_MEDIUM_TILE_ROWS: usize = 2048;
const PARALLEL_SELECT_LONG_TILE_ROWS: usize = 4096;
const PARALLEL_SELECT_MEDIUM_CONTEXT: usize = 60 * PARALLEL_SELECT_MEDIUM_TILE_ROWS;
const PARALLEL_SELECT_LONG_CONTEXT: usize = 60 * PARALLEL_SELECT_LONG_TILE_ROWS;

fn parallel_select_tile_rows(context_rows: usize, prefer_medium: bool) -> usize {
    if context_rows >= PARALLEL_SELECT_LONG_CONTEXT {
        PARALLEL_SELECT_LONG_TILE_ROWS
    } else if prefer_medium && context_rows >= PARALLEL_SELECT_MEDIUM_CONTEXT {
        PARALLEL_SELECT_MEDIUM_TILE_ROWS
    } else {
        PARALLEL_SELECT_TILE_ROWS
    }
}

#[allow(clippy::too_many_arguments)]
pub fn try_dsa_select_paged_q8(
    device_id: i32,
    keys: &DeviceBuffer,
    key_scales: &DeviceBuffer,
    key_group_size: usize,
    hadamard_i8: bool,
    block_table: &DeviceBuffer,
    query: &DeviceBuffer,
    head_weights: &DeviceBuffer,
    query_rows: usize,
    context_rows: usize,
    query_start: usize,
    head_count: usize,
    head_dim: usize,
    top_k: usize,
    prefer_medium_select_tile: bool,
    block_size: usize,
) -> Result<DeviceBuffer, String> {
    try_dsa_select_paged_q8_impl(
        device_id,
        keys,
        key_scales,
        key_group_size,
        hadamard_i8,
        block_table,
        query,
        head_weights,
        query_rows,
        context_rows,
        query_start,
        head_count,
        head_dim,
        top_k,
        prefer_medium_select_tile,
        block_size,
        2,
        false,
    )
    .map(|(selection, _)| selection)
}

pub struct DsaSequenceShardSelection {
    pub selection: DeviceBuffer,
    pub scores: DeviceBuffer,
    pub width: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn try_dsa_select_paged_q8_sequence_shard(
    device_id: i32,
    keys: &DeviceBuffer,
    key_scales: &DeviceBuffer,
    key_group_size: usize,
    block_table: &DeviceBuffer,
    query: &DeviceBuffer,
    head_weights: &DeviceBuffer,
    query_rows: usize,
    context_rows: usize,
    query_start: usize,
    head_count: usize,
    head_dim: usize,
    width: usize,
    block_size: usize,
    shard_parity: usize,
) -> Result<DsaSequenceShardSelection, String> {
    let (selection, scores) =
        try_dsa_select_paged_q8_impl(device_id, keys, key_scales, key_group_size, false, block_table, query, head_weights, query_rows, context_rows, query_start, head_count, head_dim, width, false, block_size, shard_parity, true)?;
    Ok(DsaSequenceShardSelection { selection, scores: scores.expect("sequence shard 必须收集 selection score"), width })
}

#[allow(clippy::too_many_arguments)]
fn try_dsa_select_paged_q8_impl(
    device_id: i32,
    keys: &DeviceBuffer,
    key_scales: &DeviceBuffer,
    key_group_size: usize,
    hadamard_i8: bool,
    block_table: &DeviceBuffer,
    query: &DeviceBuffer,
    head_weights: &DeviceBuffer,
    query_rows: usize,
    context_rows: usize,
    query_start: usize,
    head_count: usize,
    head_dim: usize,
    top_k: usize,
    prefer_medium_select_tile: bool,
    block_size: usize,
    shard_parity: usize,
    collect_selection_scores: bool,
) -> Result<(DeviceBuffer, Option<DeviceBuffer>), String> {
    if query_rows == 0
        || context_rows <= top_k
        || query_start.checked_add(query_rows) != Some(context_rows)
        || head_count == 0
        || head_dim == 0
        || key_group_size == 0
        || key_group_size > 256
        || !key_group_size.is_power_of_two()
        || !head_dim.is_multiple_of(key_group_size)
        || (hadamard_i8 && (head_dim != key_group_size || head_dim > 256 || !head_dim.is_power_of_two()))
        || top_k == 0
        || block_size == 0
        || shard_parity > 2
    {
        return Err("paged DSA selection shape 非法".to_owned());
    }

    let functions = paged_mla_functions(device_id)?;
    let profile_dsa = options().profile_dsa;
    let use_wmma = !hadamard_i8 && head_count.is_multiple_of(16) && head_dim.is_multiple_of(16);
    let use_native_wmma = use_wmma && functions.dense_wmma && options().native_dsa_wmma;
    let single_row_native_wmma = use_native_wmma && query_rows == 1;
    let rows2_native_wmma = use_native_wmma && query_rows == 2;
    let use_native_i8 = hadamard_i8 && head_count.is_multiple_of(16) && functions.dense_wmma && options().native_dsa_wmma;
    let compact_select = top_k <= 4096;
    let compact_shard_scores = shard_parity <= 1 && query_rows == 1;
    const PREFIX_CANDIDATE_CAPACITY: usize = 4096;
    // 128K 的 2K-row append 会因 prefix candidate overflow 重算大量 score；
    // 只在更长上下文使用压缩路径，目标档位继续走精确 compact top-k。
    const PREFIX_CONTEXT_THRESHOLD: usize = 256 * 1024;
    let prefix_select = use_native_wmma && compact_select && query_rows >= 8 && context_rows >= PREFIX_CONTEXT_THRESHOLD;
    if shard_parity <= 1 && (!use_native_wmma || hadamard_i8 || prefix_select || top_k > 2048) {
        return Err(format!("paged DSA sequence shard 仅支持 raw-Q8 native WMMA、width<=2048，当前 native={use_native_wmma} hadamard={hadamard_i8} prefix={prefix_select} width={top_k}"));
    }
    const PARALLEL_SELECT_CONTEXT_THRESHOLD: usize = 32 * 1024;
    // 中档只在 stage 已观察到至少三路 decode 时启用；C1/C2 保持 1K tile。
    // 当前 stage 不能跨 verify 行等待形成 DSA-only cohort；长上下文 decode
    // 改为单行内部按 history tile 并行，selection 集合与稳定顺序都不变。
    // sequence-shard decode 的半片只有 1/2 行数，单 block compact select 已够快；
    // 关掉 4-kernel tile 链后每侧少 3 次 launch 与 3 个流上间隙。
    let parallel_select = compact_select && !prefix_select && query_rows <= 4 && context_rows >= PARALLEL_SELECT_CONTEXT_THRESHOLD && shard_parity > 1;
    let score_histogram_shared_bytes: u32 = if compact_select {
        if single_row_native_wmma {
            256 * 4
        } else if rows2_native_wmma {
            2 * 256 * 4
        } else if use_native_wmma || use_native_i8 {
            4 * 256 * 4
        } else {
            256 * 4
        }
    } else {
        0
    };
    let score_tile_rows = if single_row_native_wmma {
        128
    } else if use_native_wmma || use_native_i8 {
        256
    } else {
        128
    };
    let key_groups = head_dim / key_group_size;
    let resident_rows = if shard_parity <= 1 {
        let blocks = context_rows / block_size;
        let tail = context_rows % block_size;
        (blocks / 2) * block_size + usize::from(blocks % 2 > shard_parity) * block_size + usize::from(blocks % 2 == shard_parity) * tail
    } else {
        context_rows
    };
    let max_score_rows = if compact_shard_scores { resident_rows } else { context_rows };
    let max_tile_count = max_score_rows.div_ceil(score_tile_rows);
    validate_resident(keys, device_id, resident_rows.checked_mul(head_dim).ok_or("DSA Q8 keys 大小溢出")?, "DSA Q8 keys")?;
    validate_resident(key_scales, device_id, resident_rows.checked_mul(key_groups).and_then(|n| n.checked_mul(2)).ok_or("DSA Q8 scales 大小溢出")?, "DSA Q8 scales")?;
    validate_resident(query, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).and_then(|n| n.checked_mul(4)).ok_or("DSA query 大小溢出")?, "DSA query")?;
    validate_resident(head_weights, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(4)).ok_or("DSA weights 大小溢出")?, "DSA head weights")?;
    validate_resident(block_table, device_id, resident_rows.max(1).div_ceil(block_size).checked_mul(4).ok_or("DSA block table 大小溢出")?, "DSA block table")?;

    let selection_elements = query_rows.checked_mul(top_k).ok_or("DSA selection 大小溢出")?;
    let quantized_query_bytes = if hadamard_i8 { query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).ok_or("DSA i8 query 大小溢出")? } else { 1 };
    let query_scale_bytes = if hadamard_i8 { query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(4)).ok_or("DSA i8 query scale 大小溢出")? } else { 4 };
    let max_score_stride = max_tile_count.checked_mul(score_tile_rows).ok_or("DSA score stride 溢出")?;
    let score_bytes_per_row = max_score_stride.checked_mul(4).ok_or("DSA score 行字节数溢出")?;

    // 长上下文按 query 行流式处理，避免展开整个 query×context score 矩阵。
    // 128K append 下 96 MiB 在末个 2K chunk 只能容纳约 186 行，导致
    // score/select 被拆成 12 次。固定的 128 MiB 上限可压到 9 次，且 workspace
    // 仍按 device 复用并在请求结束释放，不随 KV cache 或 session 数量增长。
    const SCRATCH_LIMIT_BYTES: usize = 128 * 1024 * 1024;
    const MAX_BATCH_ROWS: usize = 256;
    let batch_rows = query_rows.min(MAX_BATCH_ROWS).min((SCRATCH_LIMIT_BYTES / score_bytes_per_row).max(1));
    let score_bytes = batch_rows.checked_mul(score_bytes_per_row).ok_or("DSA score 字节数溢出")?;
    let score_prefix_bytes = if prefix_select { batch_rows.checked_mul(max_score_stride).ok_or("DSA score prefix 字节数溢出")? } else { 1 };
    let candidate_bytes = if prefix_select { batch_rows.checked_mul(PREFIX_CANDIDATE_CAPACITY).and_then(|n| n.checked_mul(4)).ok_or("DSA prefix candidate 字节数溢出")? } else { 4 };
    // counts、threshold bytes、overflow rows，加 overflow count/cursor 两个控制字。
    // radix_state 平面（prefix/mask/rank 每 row 3 个 u32）挂在 metadata 尾部，
    // 供 threshold→stage→stage 的逐字节接力。
    let candidate_metadata_bytes = batch_rows.checked_mul(6).and_then(|n| n.checked_add(2)).and_then(|n| n.checked_mul(4)).ok_or("DSA prefix metadata 字节数溢出")?;
    let coarse_histogram_bytes = if compact_select { batch_rows.checked_mul(256).and_then(|n| n.checked_mul(4)).ok_or("DSA coarse histogram 字节数溢出")? } else { 4 };
    let max_select_tile_count = max_score_stride.div_ceil(PARALLEL_SELECT_TILE_ROWS);
    // 每个 tile 保存第二字节 radix histogram，以及 greater/equal 的 count
    // 与 exclusive offset。并行 select 只用于最多 4 个 query row，scratch
    // 即使在 1M context 也不超过约 4 MiB/device。
    let select_tiles_bytes = if parallel_select {
        let tile_entries = batch_rows.checked_mul(max_select_tile_count).ok_or("DSA parallel select tile 数溢出")?;
        let count_offset_bytes = tile_entries.checked_mul(4).and_then(|n| n.checked_mul(4)).ok_or("DSA parallel select count scratch 字节数溢出")?;
        let histogram_bytes = tile_entries.checked_mul(256).and_then(|n| n.checked_mul(4)).ok_or("DSA parallel select histogram scratch 字节数溢出")?;
        count_offset_bytes.checked_add(histogram_bytes).ok_or("DSA parallel select scratch 字节数溢出")?
    } else {
        4
    };
    let selection_bytes = selection_elements.checked_mul(4).ok_or("DSA selection 字节数溢出")?;

    let (quantized_query, query_scales, scores, score_prefixes, candidates, candidate_metadata, coarse_histograms, select_tiles) = PAGED_DSA_WORKSPACES.with(|workspaces| -> Result<_, String> {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        let quantized_query = reserve_paged_dsa_buffer(&mut workspace.quantized_query, &mut workspace.quantized_query_bytes, device_id, quantized_query_bytes)?;
        let query_scales = reserve_paged_dsa_buffer(&mut workspace.query_scales, &mut workspace.query_scale_bytes, device_id, query_scale_bytes)?;
        let scores = reserve_paged_dsa_buffer(&mut workspace.scores, &mut workspace.scores_bytes, device_id, score_bytes)?;
        let score_prefixes = reserve_paged_dsa_buffer(&mut workspace.score_prefixes, &mut workspace.score_prefixes_bytes, device_id, score_prefix_bytes)?;
        let candidates = reserve_paged_dsa_buffer(&mut workspace.candidates, &mut workspace.candidates_bytes, device_id, candidate_bytes)?;
        let candidate_metadata = reserve_paged_dsa_buffer(&mut workspace.candidate_metadata, &mut workspace.candidate_metadata_bytes, device_id, candidate_metadata_bytes)?;
        let coarse_histograms = reserve_paged_dsa_buffer(&mut workspace.coarse_histograms, &mut workspace.coarse_histograms_bytes, device_id, coarse_histogram_bytes)?;
        let select_tiles = reserve_paged_dsa_buffer(&mut workspace.select_tiles, &mut workspace.select_tiles_bytes, device_id, select_tiles_bytes)?;
        Ok((quantized_query, query_scales, scores, score_prefixes, candidates, candidate_metadata, coarse_histograms, select_tiles))
    })?;
    let selection = DeviceBuffer::allocate_reusable(device_id, selection_bytes)?;
    let selection_scores = collect_selection_scores.then(|| DeviceBuffer::allocate_reusable(device_id, selection_bytes)).transpose()?;
    let mut dim_u32 = u32::try_from(head_dim).map_err(|_| "DSA head_dim 超过 u32")?;
    let mut profile_quant_ms = 0.0f64;
    let mut profile_score_ms = 0.0f64;
    let mut profile_select_ms = 0.0f64;
    let mut profile_select_stage_ms = [0.0f64; 4];
    let mut profile_batches = 0usize;
    let mut profile_candidate_total = 0usize;
    let mut profile_candidate_max = 0usize;
    let mut profile_overflow_rows = 0usize;
    if hadamard_i8 {
        if profile_dsa {
            super::synchronize_device(device_id, "hipDeviceSynchronize DSA i8 quantize profile boundary")?;
        }
        let quant_started = profile_dsa.then(std::time::Instant::now);
        let mut d_query = query.pointer;
        let mut d_quantized = quantized_query.pointer;
        let mut d_query_scales = query_scales.pointer;
        let query_vectors = query_rows.checked_mul(head_count).ok_or("DSA i8 query vector 数溢出")?;
        let mut query_vectors_u32 = u32::try_from(query_vectors).map_err(|_| "DSA i8 query vector 数超过 u32")?;
        let mut query_dim_u32 = dim_u32;
        let mut quant_args =
            [(&mut d_query as *mut *mut c_void).cast(), (&mut d_quantized as *mut *mut c_void).cast(), (&mut d_query_scales as *mut *mut c_void).cast(), (&mut query_vectors_u32 as *mut u32).cast(), (&mut query_dim_u32 as *mut u32).cast()];
        launch_tensor_kernel(functions.dsa_quantize_query_i8, query_vectors_u32, query_dim_u32, &mut quant_args, "HIP DSA query Hadamard Q8")?;
        if profile_dsa || options().debug_dsa_sync {
            super::synchronize_device(device_id, "hipDeviceSynchronize DSA query Hadamard Q8")?;
        }
        if let Some(started) = quant_started {
            profile_quant_ms = started.elapsed().as_secs_f64() * 1e3;
        }
    }
    for query_offset in (0..query_rows).step_by(batch_rows) {
        let current_rows = (query_rows - query_offset).min(batch_rows);
        let query_offset_bytes = query_offset.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).and_then(|n| n.checked_mul(4)).ok_or("DSA query offset 溢出")?;
        let weight_offset_bytes = query_offset.checked_mul(head_count).and_then(|n| n.checked_mul(4)).ok_or("DSA weight offset 溢出")?;
        let selection_offset = query_offset.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("DSA selection offset 溢出")?;
        let batch_start = query_start.checked_add(query_offset).ok_or("DSA batch query_start 溢出")?;
        // Causal prefill 的当前 batch 只看得到自身末尾；不要为未来 K 启动 WMMA 后再丢弃。
        let batch_context_rows = batch_start.checked_add(current_rows).ok_or("DSA batch context 溢出")?;
        let score_context_rows = if shard_parity <= 1 {
            let blocks = batch_context_rows / block_size;
            let tail = batch_context_rows % block_size;
            (blocks / 2) * block_size + usize::from(blocks % 2 > shard_parity) * block_size + usize::from(blocks % 2 == shard_parity) * tail
        } else {
            batch_context_rows
        };
        let score_stride_rows = if compact_shard_scores { score_context_rows } else { batch_context_rows };
        let batch_score_stride = score_stride_rows.div_ceil(score_tile_rows).checked_mul(score_tile_rows).ok_or("DSA batch score stride 溢出")?;
        let batch_tile_count = score_context_rows.div_ceil(score_tile_rows);

        let mut d_keys = keys.pointer;
        let mut d_key_scales = key_scales.pointer;
        let mut d_table = block_table.pointer;
        let i8_query_offset_bytes = query_offset.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).ok_or("DSA i8 query offset 溢出")?;
        let query_scale_offset_bytes = query_offset.checked_mul(head_count).and_then(|n| n.checked_mul(4)).ok_or("DSA i8 query scale offset 溢出")?;
        let mut d_query_batch = unsafe { query.pointer.cast::<u8>().add(query_offset_bytes).cast::<c_void>() };
        let mut d_query_i8_batch = unsafe { quantized_query.pointer.cast::<u8>().add(i8_query_offset_bytes).cast::<c_void>() };
        let mut d_query_scale_batch = unsafe { query_scales.pointer.cast::<u8>().add(query_scale_offset_bytes).cast::<c_void>() };
        let mut d_weight_batch = unsafe { head_weights.pointer.cast::<u8>().add(weight_offset_bytes).cast::<c_void>() };
        let mut d_scores = scores.pointer;
        let mut d_score_prefixes = if prefix_select { score_prefixes.pointer } else { ptr::null_mut() };
        let mut d_native_score_output = if prefix_select { d_score_prefixes } else { d_scores };
        let mut d_candidates = candidates.pointer;
        let metadata_base = candidate_metadata.pointer.cast::<u8>();
        let metadata_row_bytes = batch_rows.checked_mul(4).ok_or("DSA prefix metadata offset 溢出")?;
        let mut d_candidate_counts = candidate_metadata.pointer;
        let mut d_threshold_bytes = unsafe { metadata_base.add(metadata_row_bytes).cast::<c_void>() };
        let mut d_overflow_rows = unsafe { metadata_base.add(metadata_row_bytes * 2).cast::<c_void>() };
        let mut d_overflow_count = unsafe { metadata_base.add(metadata_row_bytes * 3).cast::<c_void>() };
        let mut d_overflow_cursor = unsafe { metadata_base.add(metadata_row_bytes * 3 + 4).cast::<c_void>() };
        let mut d_radix_state = unsafe { metadata_base.add(metadata_row_bytes * 3 + 8).cast::<c_void>() };
        let mut d_coarse_histograms = if compact_select { coarse_histograms.pointer } else { ptr::null_mut() };
        let select_tile_plane_bytes = batch_rows.checked_mul(max_select_tile_count).and_then(|n| n.checked_mul(2)).and_then(|n| n.checked_mul(4)).ok_or("DSA parallel select tile plane offset 溢出")?;
        let mut d_select_tile_counts = select_tiles.pointer;
        let mut d_select_tile_offsets = if parallel_select { unsafe { select_tiles.pointer.cast::<u8>().add(select_tile_plane_bytes).cast::<c_void>() } } else { select_tiles.pointer };
        let mut d_select_tile_histograms = if parallel_select { unsafe { select_tiles.pointer.cast::<u8>().add(select_tile_plane_bytes * 2).cast::<c_void>() } } else { select_tiles.pointer };
        let mut current_rows_u32 = u32::try_from(current_rows).map_err(|_| "DSA batch rows 超过 u32")?;
        let mut context_u32 = u32::try_from(batch_context_rows).map_err(|_| "DSA context 超过 u32")?;
        let mut start_u32 = u32::try_from(batch_start).map_err(|_| "DSA query_start 超过 u32")?;
        let mut selection_start_u32 = u32::try_from(if compact_shard_scores { score_context_rows - 1 } else { batch_start }).map_err(|_| "DSA selection start 超过 u32")?;
        let mut heads_u32 = u32::try_from(head_count).map_err(|_| "DSA heads 超过 u32")?;
        let mut key_group_u32 = u32::try_from(key_group_size).map_err(|_| "DSA key_group_size 超过 u32")?;
        let mut block_u32 = u32::try_from(block_size).map_err(|_| "DSA block_size 超过 u32")?;
        let mut score_tile_rows_u32 = u32::try_from(score_tile_rows).map_err(|_| "DSA score tile 超过 u32")?;
        let mut score_stride_u32 = u32::try_from(batch_score_stride).map_err(|_| "DSA score stride 超过 u32")?;
        let mut shard_parity_u32 = u32::try_from(shard_parity).map_err(|_| "DSA shard parity 超过 u32")?;
        let mut compact_shard_scores_u32 = u32::from(compact_shard_scores);
        let mut candidate_capacity_u32 = u32::try_from(PREFIX_CANDIDATE_CAPACITY).map_err(|_| "DSA prefix candidate capacity 超过 u32")?;
        let mut score_args = [
            (&mut d_keys as *mut *mut c_void).cast(),
            (&mut d_key_scales as *mut *mut c_void).cast(),
            (&mut d_table as *mut *mut c_void).cast(),
            (&mut d_query_batch as *mut *mut c_void).cast(),
            (&mut d_weight_batch as *mut *mut c_void).cast(),
            (&mut d_scores as *mut *mut c_void).cast(),
            (&mut d_coarse_histograms as *mut *mut c_void).cast(),
            (&mut current_rows_u32 as *mut u32).cast(),
            (&mut context_u32 as *mut u32).cast(),
            (&mut start_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut dim_u32 as *mut u32).cast(),
            (&mut key_group_u32 as *mut u32).cast(),
            (&mut block_u32 as *mut u32).cast(),
            (&mut score_tile_rows_u32 as *mut u32).cast(),
        ];
        let mut native_score_args = [
            (&mut d_keys as *mut *mut c_void).cast(),
            (&mut d_key_scales as *mut *mut c_void).cast(),
            (&mut d_table as *mut *mut c_void).cast(),
            (&mut d_query_batch as *mut *mut c_void).cast(),
            (&mut d_weight_batch as *mut *mut c_void).cast(),
            (&mut d_native_score_output as *mut *mut c_void).cast(),
            (&mut d_coarse_histograms as *mut *mut c_void).cast(),
            (&mut current_rows_u32 as *mut u32).cast(),
            (&mut context_u32 as *mut u32).cast(),
            (&mut start_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut dim_u32 as *mut u32).cast(),
            (&mut key_group_u32 as *mut u32).cast(),
            (&mut block_u32 as *mut u32).cast(),
            (&mut score_tile_rows_u32 as *mut u32).cast(),
            (&mut score_stride_u32 as *mut u32).cast(),
            (&mut shard_parity_u32 as *mut u32).cast(),
        ];
        let mut i8_score_args = [
            (&mut d_keys as *mut *mut c_void).cast(),
            (&mut d_key_scales as *mut *mut c_void).cast(),
            (&mut d_table as *mut *mut c_void).cast(),
            (&mut d_query_i8_batch as *mut *mut c_void).cast(),
            (&mut d_query_scale_batch as *mut *mut c_void).cast(),
            (&mut d_weight_batch as *mut *mut c_void).cast(),
            (&mut d_scores as *mut *mut c_void).cast(),
            (&mut d_coarse_histograms as *mut *mut c_void).cast(),
            (&mut current_rows_u32 as *mut u32).cast(),
            (&mut context_u32 as *mut u32).cast(),
            (&mut start_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut dim_u32 as *mut u32).cast(),
            (&mut block_u32 as *mut u32).cast(),
            (&mut score_tile_rows_u32 as *mut u32).cast(),
        ];
        if profile_dsa && query_offset == 0 && !hadamard_i8 {
            // profile-only 计时先排空前序同流 MLA；后续 batch 已由 select 后同步建立边界。
            super::synchronize_device(device_id, "hipDeviceSynchronize DSA profile boundary")?;
        }
        let score_started = profile_dsa.then(std::time::Instant::now);
        if compact_select {
            let mut coarse_histogram_elements = current_rows_u32.checked_mul(256).ok_or("DSA coarse histogram elements 溢出")?;
            let mut clear_args = [(&mut d_coarse_histograms as *mut *mut c_void).cast(), (&mut coarse_histogram_elements as *mut u32).cast()];
            // clear、score、select 与后续 MLA 都由 launch helper 提交到同一 legacy
            // default stream；跨 device 复制另由 ordered P2P 的 event/wait 串接。
            launch_tensor_kernel(functions.dsa_clear, coarse_histogram_elements.div_ceil(256), 256, &mut clear_args, "HIP DSA clear coarse histogram")?;
            if prefix_select {
                let mut control_elements = 2u32;
                let mut clear_control_args = [(&mut d_overflow_count as *mut *mut c_void).cast(), (&mut control_elements as *mut u32).cast()];
                launch_tensor_kernel(functions.dsa_clear, 1, 256, &mut clear_control_args, "HIP DSA clear prefix controls")?;
            }
        }
        if shard_parity <= 1 {
            let mut score_elements = current_rows_u32.checked_mul(u32::try_from(batch_score_stride).map_err(|_| "DSA shard score stride 超过 u32")?).ok_or("DSA shard score clear 大小溢出")?;
            let mut clear_score_args = [(&mut d_scores as *mut *mut c_void).cast(), (&mut score_elements as *mut u32).cast()];
            launch_tensor_kernel(functions.dsa_clear, score_elements.div_ceil(256), 256, &mut clear_score_args, "HIP DSA clear sequence-shard scores")?;
        }
        launch_moe_kernel(
            if use_native_i8 {
                functions.dsa_score_native_wmma_i8
            } else if hadamard_i8 {
                functions.dsa_score_i8
            } else if use_native_wmma {
                if prefix_select {
                    functions.dsa_score_prefix_native_wmma
                } else if single_row_native_wmma {
                    functions.dsa_score_native_wmma_decode
                } else if rows2_native_wmma {
                    functions.dsa_score_native_wmma_rows2
                } else {
                    functions.dsa_score_native_wmma
                }
            } else if use_wmma {
                functions.dsa_score_wmma
            } else {
                functions.dsa_score
            },
            u32::try_from(batch_tile_count).map_err(|_| "DSA tile grid 超过 u32")?,
            if single_row_native_wmma {
                current_rows_u32
            } else if rows2_native_wmma {
                current_rows_u32.div_ceil(2)
            } else if use_native_wmma || use_native_i8 {
                current_rows_u32.div_ceil(4)
            } else {
                current_rows_u32
            },
            if single_row_native_wmma {
                256
            } else if use_native_wmma || use_native_i8 {
                512
            } else {
                256
            },
            score_histogram_shared_bytes,
            if hadamard_i8 {
                &mut i8_score_args[..]
            } else if use_native_wmma {
                &mut native_score_args[..]
            } else {
                &mut score_args[..]
            },
            if use_native_i8 {
                "HIP DSA score Hadamard i8 native WMMA"
            } else if hadamard_i8 {
                "HIP DSA score Hadamard i8 scalar"
            } else if use_native_wmma {
                if prefix_select {
                    "HIP DSA score prefix native WMMA"
                } else if single_row_native_wmma {
                    "HIP DSA score decode native WMMA"
                } else if rows2_native_wmma {
                    "HIP DSA score rows2 native WMMA"
                } else {
                    "HIP DSA score native WMMA"
                }
            } else if use_wmma {
                "HIP DSA score rocWMMA"
            } else {
                "HIP DSA score scalar"
            },
        )?;
        if profile_dsa || options().debug_dsa_sync {
            super::synchronize_device(device_id, if use_wmma || use_native_i8 { "hipDeviceSynchronize DSA score WMMA" } else { "hipDeviceSynchronize DSA score scalar" })?;
        }
        if let Some(started) = score_started {
            profile_score_ms += started.elapsed().as_secs_f64() * 1e3;
        }

        let mut d_selection = unsafe { selection.pointer.cast::<u8>().add(selection_offset).cast::<c_void>() };
        let mut topk_u32 = u32::try_from(top_k).map_err(|_| "DSA top_k 超过 u32")?;
        let mut visibility_divisor_u32 = 1u32;
        let mut compact_args = [
            (&mut d_scores as *mut *mut c_void).cast(),
            (&mut d_coarse_histograms as *mut *mut c_void).cast(),
            (&mut d_selection as *mut *mut c_void).cast(),
            (&mut current_rows_u32 as *mut u32).cast(),
            (&mut score_stride_u32 as *mut u32).cast(),
            (&mut selection_start_u32 as *mut u32).cast(),
            (&mut topk_u32 as *mut u32).cast(),
            (&mut visibility_divisor_u32 as *mut u32).cast(),
        ];
        let select_started = profile_dsa.then(std::time::Instant::now);
        if prefix_select {
            let mut compact_prefix_args = [
                (&mut d_score_prefixes as *mut *mut c_void).cast(),
                (&mut d_coarse_histograms as *mut *mut c_void).cast(),
                (&mut d_candidates as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut d_threshold_bytes as *mut *mut c_void).cast(),
                (&mut d_overflow_rows as *mut *mut c_void).cast(),
                (&mut d_overflow_count as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut candidate_capacity_u32 as *mut u32).cast(),
            ];
            launch_tensor_kernel(functions.dsa_compact_prefixes, current_rows_u32, 256, &mut compact_prefix_args, "HIP DSA compact score prefixes")?;

            let mut candidate_local = 0u32;
            let mut selected_score_args = [
                (&mut d_keys as *mut *mut c_void).cast(),
                (&mut d_key_scales as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_query_batch as *mut *mut c_void).cast(),
                (&mut d_weight_batch as *mut *mut c_void).cast(),
                (&mut d_candidates as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut candidate_capacity_u32 as *mut u32).cast(),
                (&mut d_overflow_rows as *mut *mut c_void).cast(),
                (&mut d_overflow_count as *mut *mut c_void).cast(),
                (&mut d_overflow_cursor as *mut *mut c_void).cast(),
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut context_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut heads_u32 as *mut u32).cast(),
                (&mut dim_u32 as *mut u32).cast(),
                (&mut key_group_u32 as *mut u32).cast(),
                (&mut block_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut candidate_local as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_score_selected_native_wmma, candidate_capacity_u32.div_ceil(128), current_rows_u32, 256, 0, &mut selected_score_args, "HIP DSA score compact candidates")?;

            let mut d_no_candidates = ptr::null_mut();
            let mut overflow_score_args = [
                (&mut d_keys as *mut *mut c_void).cast(),
                (&mut d_key_scales as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_query_batch as *mut *mut c_void).cast(),
                (&mut d_weight_batch as *mut *mut c_void).cast(),
                (&mut d_no_candidates as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut candidate_capacity_u32 as *mut u32).cast(),
                (&mut d_overflow_rows as *mut *mut c_void).cast(),
                (&mut d_overflow_count as *mut *mut c_void).cast(),
                (&mut d_overflow_cursor as *mut *mut c_void).cast(),
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut context_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut heads_u32 as *mut u32).cast(),
                (&mut dim_u32 as *mut u32).cast(),
                (&mut key_group_u32 as *mut u32).cast(),
                (&mut block_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut candidate_local as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_score_selected_native_wmma, 256, 1, 256, 0, &mut overflow_score_args, "HIP DSA score prefix overflow")?;

            let mut prefix_select_args = [
                (&mut d_score_prefixes as *mut *mut c_void).cast(),
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_coarse_histograms as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut d_threshold_bytes as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut candidate_capacity_u32 as *mut u32).cast(),
            ];
            launch_tensor_kernel(functions.dsa_select_prefix, current_rows_u32, 256, &mut prefix_select_args, "HIP DSA prefix radix select top-k")?;
        } else if parallel_select {
            // 240K 起 4K tile；128K 只有高并发才用 2K，避免 C1/C2 欠占用。
            let select_tile_rows = parallel_select_tile_rows(score_stride_rows, prefer_medium_select_tile);
            let mut select_tile_count_u32 = u32::try_from(batch_score_stride.div_ceil(select_tile_rows)).map_err(|_| "DSA parallel select tile count 超过 u32")?;
            let mut select_tile_rows_u32 = u32::try_from(select_tile_rows).map_err(|_| "DSA parallel select tile rows 超过 u32")?;
            let mut clear_rows = current_rows_u32;
            let mut clear_threshold_args = [(&mut d_overflow_rows as *mut *mut c_void).cast(), (&mut clear_rows as *mut u32).cast()];
            // profile_dsa 下逐 kernel 同步计时（threshold/counts/scan/scatter），
            // 定位 select 链固定开销 vs 净 kernel 时间。
            macro_rules! dsa_stage_sync {
                ($stage:expr) => {
                    if profile_dsa {
                        super::synchronize_device(device_id, concat!("HIP DSA select stage ", $stage))?;
                    }
                };
            }
            let mut stage_started = profile_dsa.then(std::time::Instant::now);
            let stage_ms = &mut profile_select_stage_ms;
            launch_tensor_kernel(functions.dsa_clear, 1, 256, &mut clear_threshold_args, "HIP DSA clear radix tile counters")?;
            dsa_stage_sync!("clear");
            let mut threshold_args = [
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_coarse_histograms as *mut *mut c_void).cast(),
                (&mut d_select_tile_histograms as *mut *mut c_void).cast(),
                (&mut d_overflow_rows as *mut *mut c_void).cast(),
                (&mut d_radix_state as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut selection_start_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut select_tile_count_u32 as *mut u32).cast(),
                (&mut select_tile_rows_u32 as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_select_threshold, select_tile_count_u32, current_rows_u32, 256, 0, &mut threshold_args, "HIP DSA parallel compact radix threshold")?;
            // byte3/byte4 接力：每级全 tile 并行重扫 + last CTA 只做桶决策，
            // 替代原先"最后 CTA 内 shift 循环 O(visible) 串行尾巴"。
            for (stage_shift, stage_label) in [(8_u32, "HIP DSA radix stage byte3"), (0_u32, "HIP DSA radix stage byte4")] {
                let mut stage_shift_value = stage_shift;
                let mut stage_args = [
                    (&mut d_scores as *mut *mut c_void).cast(),
                    (&mut d_select_tile_histograms as *mut *mut c_void).cast(),
                    (&mut d_radix_state as *mut *mut c_void).cast(),
                    (&mut d_overflow_rows as *mut *mut c_void).cast(),
                    (&mut d_candidate_counts as *mut *mut c_void).cast(),
                    (&mut d_threshold_bytes as *mut *mut c_void).cast(),
                    (&mut current_rows_u32 as *mut u32).cast(),
                    (&mut score_stride_u32 as *mut u32).cast(),
                    (&mut selection_start_u32 as *mut u32).cast(),
                    (&mut select_tile_count_u32 as *mut u32).cast(),
                    (&mut select_tile_rows_u32 as *mut u32).cast(),
                    (&mut stage_shift_value as *mut u32).cast(),
                ];
                launch_moe_kernel(functions.dsa_select_radix_stage, select_tile_count_u32, current_rows_u32, 256, 0, &mut stage_args, stage_label)?;
            }
            dsa_stage_sync!("threshold+stages");
            if let Some(started) = stage_started.take() {
                stage_ms[0] = started.elapsed().as_secs_f64() * 1e3;
            }

            let mut count_args = [
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut d_select_tile_counts as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut selection_start_u32 as *mut u32).cast(),
                (&mut select_tile_count_u32 as *mut u32).cast(),
                (&mut select_tile_rows_u32 as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_select_tile_counts, select_tile_count_u32, current_rows_u32, 256, 0, &mut count_args, "HIP DSA count selected tiles")?;
            dsa_stage_sync!("counts");
            if let Some(started) = stage_started.take() {
                stage_ms[1] = started.elapsed().as_secs_f64() * 1e3;
            }

            let mut scan_args = [(&mut d_select_tile_counts as *mut *mut c_void).cast(), (&mut d_select_tile_offsets as *mut *mut c_void).cast(), (&mut current_rows_u32 as *mut u32).cast(), (&mut select_tile_count_u32 as *mut u32).cast()];
            launch_tensor_kernel(functions.dsa_select_tile_scan, current_rows_u32, 256, &mut scan_args, "HIP DSA scan selected tiles")?;
            dsa_stage_sync!("scan");
            if let Some(started) = stage_started.take() {
                stage_ms[2] = started.elapsed().as_secs_f64() * 1e3;
            }

            let mut scatter_args = [
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut d_threshold_bytes as *mut *mut c_void).cast(),
                (&mut d_select_tile_offsets as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut selection_start_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut select_tile_count_u32 as *mut u32).cast(),
                (&mut select_tile_rows_u32 as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_select_tile_scatter, select_tile_count_u32, current_rows_u32, 256, 0, &mut scatter_args, "HIP DSA scatter selected tiles")?;
            dsa_stage_sync!("scatter");
            if let Some(started) = stage_started.take() {
                stage_ms[3] = started.elapsed().as_secs_f64() * 1e3;
            }
        } else if compact_select {
            launch_tensor_kernel(functions.dsa_select_compact, current_rows_u32, 256, &mut compact_args, "HIP DSA compact radix select top-k")?;
            if options().debug_dsa_sync {
                super::synchronize_device(device_id, "hipDeviceSynchronize DSA compact select debug")?;
            }
        } else {
            super::super::try_stable_radix_topk_u32_into(device_id, &scores, None, &selection, current_rows, batch_score_stride, batch_start, top_k, query_offset)?;
        }
        if let Some(selection_scores) = selection_scores.as_ref() {
            let mut d_selected_scores = unsafe { selection_scores.pointer.cast::<u8>().add(selection_offset).cast::<c_void>() };
            let gather_elements = current_rows_u32.checked_mul(topk_u32).ok_or("DSA shard gather 元素数溢出")?;
            let mut gather_args = [
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut d_selected_scores as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut block_u32 as *mut u32).cast(),
                (&mut shard_parity_u32 as *mut u32).cast(),
                (&mut compact_shard_scores_u32 as *mut u32).cast(),
            ];
            launch_tensor_kernel(functions.dsa_gather_selection_scores, gather_elements.div_ceil(256), 256, &mut gather_args, "HIP DSA gather sequence-shard scores")?;
        }
        if profile_dsa || options().debug_dsa_sync {
            super::synchronize_device(device_id, "hipDeviceSynchronize DSA radix select top-k")?;
        }
        if profile_dsa && prefix_select {
            let mut metadata = vec![0u32; candidate_metadata_bytes / 4];
            candidate_metadata.copy_to_host(unsafe { std::slice::from_raw_parts_mut(metadata.as_mut_ptr().cast(), candidate_metadata_bytes) })?;
            for &count in &metadata[..current_rows] {
                let count = count as usize;
                profile_candidate_total += count;
                profile_candidate_max = profile_candidate_max.max(count);
            }
            profile_overflow_rows += metadata[batch_rows * 3] as usize;
        }
        if let Some(started) = select_started {
            profile_select_ms += started.elapsed().as_secs_f64() * 1e3;
        }
        profile_batches += 1;
    }
    if profile_dsa {
        eprintln!(
            "[dsa-profile] device={device_id} rows={query_rows} context={context_rows} batches={profile_batches} hadamard_i8={hadamard_i8} quant_ms={profile_quant_ms:.3} score_ms={profile_score_ms:.3} select_ms={profile_select_ms:.3} select_stage_ms={:.3}/{:.3}/{:.3}/{:.3} prefix={} parallel_select={} candidate_avg={:.1} candidate_max={} overflow_rows={}",
            profile_select_stage_ms[0],
            profile_select_stage_ms[1],
            profile_select_stage_ms[2],
            profile_select_stage_ms[3],
            prefix_select,
            parallel_select,
            profile_candidate_total as f64 / query_rows as f64,
            profile_candidate_max,
            profile_overflow_rows
        );
    }

    if options().debug_selection {
        let mut host = vec![0_u32; selection_elements];
        selection.copy_to_host(unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast(), selection_bytes) })?;
        for row in 0..query_rows {
            let visible = (query_start + row + 1).min(context_rows);
            let target = top_k.min(visible);
            for rank in 0..target {
                let token = host[row * top_k + rank] as usize;
                if token >= visible {
                    return Err(format!("DSA selection 越界: row={row} rank={rank} token={token} visible={visible} query_start={query_start}"));
                }
            }
        }
    }
    Ok((selection, selection_scores))
}

pub fn try_dsa_merge_sequence_shard_topk(device_id: i32, owner: &DsaSequenceShardSelection, peer: &DsaSequenceShardSelection, rows: usize, top_k: usize) -> Result<DeviceBuffer, String> {
    let candidates = owner.width.checked_add(peer.width).ok_or("DSA shard merge candidate 数溢出")?;
    if rows == 0 || top_k == 0 || top_k > candidates || candidates > 4096 {
        return Err(format!("DSA shard merge rows={rows} owner={} peer={} top_k={top_k} 非法", owner.width, peer.width));
    }
    for (buffer, width, label) in [(&owner.selection, owner.width, "owner tokens"), (&owner.scores, owner.width, "owner scores"), (&peer.selection, peer.width, "peer tokens"), (&peer.scores, peer.width, "peer scores")] {
        validate_resident(buffer, device_id, rows.checked_mul(width).and_then(|n| n.checked_mul(4)).ok_or("DSA shard merge 输入大小溢出")?, label)?;
    }
    let output = DeviceBuffer::allocate_reusable(device_id, rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("DSA shard merge 输出大小溢出")?)?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_owner_tokens = owner.selection.pointer;
    let mut d_owner_scores = owner.scores.pointer;
    let mut owner_width_u32 = u32::try_from(owner.width).map_err(|_| "DSA owner width 超过 u32")?;
    let mut d_peer_tokens = peer.selection.pointer;
    let mut d_peer_scores = peer.scores.pointer;
    let mut peer_width_u32 = u32::try_from(peer.width).map_err(|_| "DSA peer width 超过 u32")?;
    let mut d_output = output.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "DSA shard merge rows 超过 u32")?;
    let mut top_k_u32 = u32::try_from(top_k).map_err(|_| "DSA shard merge top_k 超过 u32")?;
    let mut arguments = [
        (&mut d_owner_tokens as *mut *mut c_void).cast(),
        (&mut d_owner_scores as *mut *mut c_void).cast(),
        (&mut owner_width_u32 as *mut u32).cast(),
        (&mut d_peer_tokens as *mut *mut c_void).cast(),
        (&mut d_peer_scores as *mut *mut c_void).cast(),
        (&mut peer_width_u32 as *mut u32).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut top_k_u32 as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.dsa_merge_sequence_shards, rows_u32, 256, &mut arguments, "HIP DSA merge sequence-shard top-k")?;
    Ok(output)
}

/// glm5_next kpool 全 GPU 选择：对已池化 Q8 key 做 WMMA 评分，按原 token
/// 的因果位置选池，再展开为 attention 直接消费的 token 索引。
#[allow(clippy::too_many_arguments)]
pub fn try_dsa_select_paged_q8_kpool(
    device_id: i32,
    pooled_keys: &DeviceBuffer,
    pooled_scales: &DeviceBuffer,
    key_group_size: usize,
    pool_block_table: &DeviceBuffer,
    query: &DeviceBuffer,
    head_weights: &DeviceBuffer,
    query_rows: usize,
    context_rows: usize,
    query_start: usize,
    head_count: usize,
    head_dim: usize,
    top_k: usize,
    kpool: usize,
    block_size: usize,
) -> Result<DeviceBuffer, String> {
    if query_rows == 0 || query_start.checked_add(query_rows) != Some(context_rows) || context_rows <= top_k || kpool == 0 || top_k == 0 || !top_k.is_multiple_of(kpool) || key_group_size == 0 || !head_dim.is_multiple_of(key_group_size) {
        return Err("paged DSA kpool selection shape 非法".to_owned());
    }
    let pool_rows = context_rows / kpool;
    let pool_top_k = top_k / kpool;
    if pool_rows <= pool_top_k {
        return Err("paged DSA kpool selection 池数不足".to_owned());
    }
    let functions = paged_mla_functions(device_id)?;
    if !functions.dense_wmma {
        return Err("paged DSA kpool 需要 gfx11+ native WMMA".to_owned());
    }
    let groups_per_row = head_dim / key_group_size;
    let selection_width = top_k.checked_add(kpool - 1).ok_or("DSA kpool selection width 溢出")?;
    validate_resident(pooled_keys, device_id, pool_rows.checked_mul(head_dim).ok_or("DSA pooled keys 大小溢出")?, "DSA pooled keys")?;
    validate_resident(pooled_scales, device_id, pool_rows.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("DSA pooled scales 大小溢出")?, "DSA pooled scales")?;
    validate_resident(pool_block_table, device_id, pool_rows.div_ceil(block_size).checked_mul(4).ok_or("DSA pool table 大小溢出")?, "DSA pool table")?;
    validate_resident(query, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool query 大小溢出")?, "DSA kpool query")?;
    validate_resident(head_weights, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool head weights 大小溢出")?, "DSA kpool head weights")?;

    const SCORE_TILE_ROWS: usize = 256;
    const MAX_BATCH_ROWS: usize = 256;
    const SCRATCH_LIMIT_BYTES: usize = 128 * 1024 * 1024;
    let score_stride = pool_rows.div_ceil(SCORE_TILE_ROWS).checked_mul(SCORE_TILE_ROWS).ok_or("DSA kpool score stride 溢出")?;
    let score_bytes_per_row = score_stride.checked_mul(4).ok_or("DSA kpool score row bytes 溢出")?;
    let batch_rows = query_rows.min(MAX_BATCH_ROWS).min((SCRATCH_LIMIT_BYTES / score_bytes_per_row).max(1));
    let score_bytes = batch_rows.checked_mul(score_bytes_per_row).ok_or("DSA kpool score bytes 溢出")?;
    let histogram_bytes = batch_rows.checked_mul(256).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool histogram bytes 溢出")?;
    let (scores, histograms) = PAGED_DSA_WORKSPACES.with(|workspaces| -> Result<_, String> {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        let scores = reserve_paged_dsa_buffer(&mut workspace.scores, &mut workspace.scores_bytes, device_id, score_bytes)?;
        let histograms = reserve_paged_dsa_buffer(&mut workspace.coarse_histograms, &mut workspace.coarse_histograms_bytes, device_id, histogram_bytes)?;
        Ok((scores, histograms))
    })?;
    let pool_selection_bytes = query_rows.checked_mul(pool_top_k).and_then(|n| n.checked_mul(4)).ok_or("DSA pool selection bytes 溢出")?;
    let pool_selection = DeviceBuffer::allocate_reusable(device_id, pool_selection_bytes)?;
    for query_offset in (0..query_rows).step_by(batch_rows) {
        let current_rows = (query_rows - query_offset).min(batch_rows);
        let batch_start = query_start.checked_add(query_offset).ok_or("DSA kpool batch start 溢出")?;
        let batch_pool_rows = batch_start.checked_add(current_rows).ok_or("DSA kpool batch end 溢出")? / kpool;
        let batch_tiles = batch_pool_rows.div_ceil(SCORE_TILE_ROWS);
        let batch_score_stride = batch_tiles.checked_mul(SCORE_TILE_ROWS).ok_or("DSA kpool batch stride 溢出")?;
        let query_offset_bytes = query_offset.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool query offset 溢出")?;
        let weight_offset_bytes = query_offset.checked_mul(head_count).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool weight offset 溢出")?;
        let selection_offset_bytes = query_offset.checked_mul(pool_top_k).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool selection offset 溢出")?;
        let mut d_keys = pooled_keys.pointer;
        let mut d_scales = pooled_scales.pointer;
        let mut d_table = pool_block_table.pointer;
        let mut d_query = unsafe { query.pointer.cast::<u8>().add(query_offset_bytes).cast::<c_void>() };
        let mut d_weights = unsafe { head_weights.pointer.cast::<u8>().add(weight_offset_bytes).cast::<c_void>() };
        let mut d_scores = scores.pointer;
        let mut d_histograms = histograms.pointer;
        let mut d_selection = unsafe { pool_selection.pointer.cast::<u8>().add(selection_offset_bytes).cast::<c_void>() };
        let mut current_rows_u32 = u32::try_from(current_rows).map_err(|_| "DSA kpool rows 超过 u32")?;
        let mut pool_rows_u32 = u32::try_from(batch_pool_rows).map_err(|_| "DSA kpool context 超过 u32")?;
        let mut start_u32 = u32::try_from(batch_start).map_err(|_| "DSA kpool start 超过 u32")?;
        let mut heads_u32 = u32::try_from(head_count).map_err(|_| "DSA kpool heads 超过 u32")?;
        let mut dim_u32 = u32::try_from(head_dim).map_err(|_| "DSA kpool dim 超过 u32")?;
        let mut group_u32 = u32::try_from(key_group_size).map_err(|_| "DSA kpool group 超过 u32")?;
        let mut block_u32 = u32::try_from(block_size).map_err(|_| "DSA kpool block 超过 u32")?;
        let mut tile_u32 = SCORE_TILE_ROWS as u32;
        let mut kpool_u32 = u32::try_from(kpool).map_err(|_| "DSA kpool 超过 u32")?;
        let mut histogram_elements = current_rows_u32.checked_mul(256).ok_or("DSA kpool histogram elements 溢出")?;
        let mut clear_args = [(&mut d_histograms as *mut *mut c_void).cast(), (&mut histogram_elements as *mut u32).cast()];
        launch_tensor_kernel(functions.dsa_clear, histogram_elements.div_ceil(256), 256, &mut clear_args, "HIP DSA kpool clear histogram")?;
        let mut score_args = [
            (&mut d_keys as *mut *mut c_void).cast(),
            (&mut d_scales as *mut *mut c_void).cast(),
            (&mut d_table as *mut *mut c_void).cast(),
            (&mut d_query as *mut *mut c_void).cast(),
            (&mut d_weights as *mut *mut c_void).cast(),
            (&mut d_scores as *mut *mut c_void).cast(),
            (&mut d_histograms as *mut *mut c_void).cast(),
            (&mut current_rows_u32 as *mut u32).cast(),
            (&mut pool_rows_u32 as *mut u32).cast(),
            (&mut start_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut dim_u32 as *mut u32).cast(),
            (&mut group_u32 as *mut u32).cast(),
            (&mut block_u32 as *mut u32).cast(),
            (&mut tile_u32 as *mut u32).cast(),
            (&mut kpool_u32 as *mut u32).cast(),
        ];
        launch_moe_kernel(functions.dsa_score_native_wmma_kpool, u32::try_from(batch_tiles).map_err(|_| "DSA kpool tile grid 超过 u32")?, current_rows_u32.div_ceil(4), 512, 4 * 256 * 4, &mut score_args, "HIP DSA kpool score native WMMA")?;
        let mut stride_u32 = u32::try_from(batch_score_stride).map_err(|_| "DSA kpool stride 超过 u32")?;
        let mut pool_top_k_u32 = u32::try_from(pool_top_k).map_err(|_| "DSA pool top-k 超过 u32")?;
        let mut select_args = [
            (&mut d_scores as *mut *mut c_void).cast(),
            (&mut d_histograms as *mut *mut c_void).cast(),
            (&mut d_selection as *mut *mut c_void).cast(),
            (&mut current_rows_u32 as *mut u32).cast(),
            (&mut stride_u32 as *mut u32).cast(),
            (&mut start_u32 as *mut u32).cast(),
            (&mut pool_top_k_u32 as *mut u32).cast(),
            (&mut kpool_u32 as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.dsa_select_compact, current_rows_u32, 256, &mut select_args, "HIP DSA kpool compact select")?;
    }
    let output_bytes = query_rows.checked_mul(selection_width).and_then(|n| n.checked_mul(4)).ok_or("DSA kpool output bytes 溢出")?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    let mut d_pool_selection = pool_selection.pointer;
    let mut d_output = output.pointer;
    let mut query_rows_u32 = u32::try_from(query_rows).map_err(|_| "DSA kpool output rows 超过 u32")?;
    let mut query_start_u32 = u32::try_from(query_start).map_err(|_| "DSA kpool output start 超过 u32")?;
    let mut pool_top_k_u32 = u32::try_from(pool_top_k).map_err(|_| "DSA pool top-k 超过 u32")?;
    let mut kpool_u32 = u32::try_from(kpool).map_err(|_| "DSA kpool 超过 u32")?;
    let mut width_u32 = u32::try_from(selection_width).map_err(|_| "DSA kpool width 超过 u32")?;
    let mut expand_args = [
        (&mut d_pool_selection as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut query_rows_u32 as *mut u32).cast(),
        (&mut query_start_u32 as *mut u32).cast(),
        (&mut pool_top_k_u32 as *mut u32).cast(),
        (&mut kpool_u32 as *mut u32).cast(),
        (&mut width_u32 as *mut u32).cast(),
    ];
    let elements = query_rows_u32.checked_mul(width_u32).ok_or("DSA kpool expand grid 溢出")?;
    launch_tensor_kernel(functions.dsa_expand_kpool_selection, elements.div_ceil(256), 256, &mut expand_args, "HIP DSA expand kpool selection")?;
    Ok(output)
}

pub(crate) struct CtMlaPrefillSegmentRef<'a> {
    pub(crate) latent_cache: &'a DeviceBuffer,
    pub(crate) latent_scales: Option<&'a DeviceBuffer>,
    pub(crate) latent_group_size: usize,
    pub(crate) rope_cache: &'a DeviceBuffer,
    pub(crate) block_table: &'a DeviceBuffer,
    pub(crate) selection: &'a DeviceBuffer,
    pub(crate) query_rows: usize,
    pub(crate) context_rows: usize,
    pub(crate) query_start: usize,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_paged_mla_attention_ct_segmented(
    device_id: i32,
    query: &DeviceBuffer,
    weight: CtMlaWeightRef<'_>,
    segments: &[CtMlaPrefillSegmentRef<'_>],
    q_projection: usize,
    head_count: usize,
    rope_dim: usize,
    top_k: usize,
    block_size: usize,
    output: &DeviceBuffer,
) -> Result<(), String> {
    let query_rows = segments.iter().try_fold(0usize, |rows, segment| rows.checked_add(segment.query_rows).ok_or("segmented paged MLA query rows 溢出"))?;
    if query_rows == 0 || top_k == 0 || !q_projection.is_multiple_of(head_count) || !weight.rows.is_multiple_of(head_count) || weight.cols == 0 || weight.group_size == 0 || !weight.cols.is_multiple_of(weight.group_size) {
        return Err("segmented paged MLA shape 非法".to_owned());
    }
    let q_head_dim = q_projection / head_count;
    let kv_head_dim = weight.rows / head_count;
    let latent_dim = weight.cols;
    if rope_dim == 0 || rope_dim > q_head_dim {
        return Err("segmented paged MLA rope shape 非法".to_owned());
    }
    let query_bytes = query_rows.checked_mul(q_projection).and_then(|n| n.checked_mul(4)).ok_or("segmented paged MLA query 大小溢出")?;
    let output_bytes = query_rows.checked_mul(q_projection).and_then(|n| n.checked_mul(2)).ok_or("segmented paged MLA output 大小溢出")?;
    validate_resident(query, device_id, query_bytes, "segmented paged MLA query")?;
    validate_resident(output, device_id, output_bytes, "segmented paged MLA output")?;
    let intermediate_elements = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(latent_dim)).ok_or("segmented paged MLA intermediate 大小溢出")?;
    let intermediate_bytes = intermediate_elements.checked_mul(2).ok_or("segmented paged MLA intermediate 字节数溢出")?;
    let (absorbed, weighted) = PAGED_MLA_WORKSPACES.with(|workspaces| -> Result<_, String> {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        Ok((reserve_paged_dsa_buffer(&mut workspace.absorbed, &mut workspace.absorbed_bytes, device_id, intermediate_bytes)?, reserve_paged_dsa_buffer(&mut workspace.weighted, &mut workspace.weighted_bytes, device_id, intermediate_bytes)?))
    })?;
    let functions = paged_mla_functions(device_id)?;

    let mut d_query = query.pointer;
    let mut d_packed = weight.packed.pointer;
    let mut d_scales = weight.scales.pointer;
    let mut d_absorbed = absorbed.pointer;
    let mut query_rows_u32 = u32::try_from(query_rows).map_err(|_| "segmented paged MLA rows 超过 u32")?;
    let mut heads_u32 = u32::try_from(head_count).map_err(|_| "segmented paged MLA heads 超过 u32")?;
    let mut q_head_u32 = u32::try_from(q_head_dim).map_err(|_| "segmented paged MLA q head 超过 u32")?;
    let mut kv_head_u32 = u32::try_from(kv_head_dim).map_err(|_| "segmented paged MLA kv head 超过 u32")?;
    let mut latent_u32 = u32::try_from(latent_dim).map_err(|_| "segmented paged MLA latent 超过 u32")?;
    let mut rope_u32 = u32::try_from(rope_dim).map_err(|_| "segmented paged MLA rope 超过 u32")?;
    let mut group_u32 = u32::try_from(weight.group_size).map_err(|_| "segmented paged MLA group 超过 u32")?;
    let mut scale_u32 = weight.scale_dtype;
    let mut bits_u32 = weight.bits;
    let mut absorb_args = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_absorbed as *mut *mut c_void).cast(),
        (&mut query_rows_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut q_head_u32 as *mut u32).cast(),
        (&mut kv_head_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
        (&mut rope_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut scale_u32 as *mut u32).cast(),
        (&mut bits_u32 as *mut u32).cast(),
    ];
    let tile = 128usize;
    let absorb_tiles = head_count.checked_mul(latent_dim.div_ceil(tile)).ok_or("segmented paged MLA absorb grid 溢出")?;
    launch_moe_kernel(
        functions.absorb_query_wmma,
        u32::try_from(absorb_tiles).map_err(|_| "segmented paged MLA absorb grid 超过 u32")?,
        u32::try_from(query_rows.div_ceil(tile)).map_err(|_| "segmented paged MLA absorb rows 超过 u32")?,
        functions.wavefront_size * 8,
        0,
        &mut absorb_args,
        "HIP segmented MLA absorb query WMMA",
    )?;

    let shared_per_head = latent_dim.checked_mul(std::mem::size_of::<u16>()).and_then(|latent| rope_dim.checked_mul(std::mem::size_of::<f32>()).and_then(|rope| latent.checked_add(rope))).ok_or("segmented paged MLA shared memory 溢出")?;
    let shared_bytes = 8usize.checked_mul(shared_per_head).ok_or("segmented paged MLA shared memory 溢出")?;
    let shared_u32 = u32::try_from(shared_bytes).map_err(|_| "segmented paged MLA shared memory 超过 u32")?;
    let row_stride_bytes = head_count.checked_mul(latent_dim).and_then(|n| n.checked_mul(2)).ok_or("segmented paged MLA row stride 溢出")?;
    let query_stride_bytes = q_projection.checked_mul(4).ok_or("segmented paged MLA query stride 溢出")?;
    let mut row_offset = 0usize;
    for segment in segments {
        if segment.query_rows <= 1 || segment.query_start.checked_add(segment.query_rows) != Some(segment.context_rows) {
            return Err("segmented paged MLA segment shape 非法".to_owned());
        }
        if segment.latent_group_size != 0 && (!latent_dim.is_multiple_of(segment.latent_group_size) || segment.latent_scales.is_none()) {
            return Err("segmented paged MLA Q8 cache 非法".to_owned());
        }
        let latent_element_bytes = if segment.latent_group_size == 0 { 2 } else { 1 };
        validate_resident(segment.latent_cache, device_id, segment.context_rows.checked_mul(latent_dim).and_then(|n| n.checked_mul(latent_element_bytes)).ok_or("segmented paged MLA latent 大小溢出")?, "segmented paged MLA latent")?;
        if let Some(scales) = segment.latent_scales {
            validate_resident(scales, device_id, segment.context_rows.checked_mul(latent_dim / segment.latent_group_size).and_then(|n| n.checked_mul(2)).ok_or("segmented paged MLA scales 大小溢出")?, "segmented paged MLA scales")?;
        }
        validate_resident(segment.rope_cache, device_id, segment.context_rows.checked_mul(rope_dim).and_then(|n| n.checked_mul(2)).ok_or("segmented paged MLA rope 大小溢出")?, "segmented paged MLA rope")?;
        validate_resident(segment.block_table, device_id, segment.context_rows.div_ceil(block_size).checked_mul(4).ok_or("segmented paged MLA block table 大小溢出")?, "segmented paged MLA block table")?;
        validate_resident(segment.selection, device_id, segment.query_rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("segmented paged MLA selection 大小溢出")?, "segmented paged MLA selection")?;

        let query_offset = row_offset.checked_mul(query_stride_bytes).ok_or("segmented paged MLA query offset 溢出")?;
        let intermediate_offset = row_offset.checked_mul(row_stride_bytes).ok_or("segmented paged MLA intermediate offset 溢出")?;
        let mut d_segment_query = unsafe { query.pointer.cast::<u8>().add(query_offset).cast() };
        let mut d_segment_absorbed = unsafe { absorbed.pointer.cast::<u8>().add(intermediate_offset).cast() };
        let mut d_latent = segment.latent_cache.pointer;
        let mut d_latent_scales = segment.latent_scales.map_or(ptr::null_mut(), |buffer| buffer.pointer);
        let mut d_rope = segment.rope_cache.pointer;
        let mut d_table = segment.block_table.pointer;
        let mut d_selection = segment.selection.pointer;
        let mut d_segment_weighted = unsafe { weighted.pointer.cast::<u8>().add(intermediate_offset).cast() };
        let mut rows_u32 = u32::try_from(segment.query_rows).map_err(|_| "segmented paged MLA segment rows 超过 u32")?;
        let mut context_u32 = u32::try_from(segment.context_rows).map_err(|_| "segmented paged MLA context 超过 u32")?;
        let mut start_u32 = u32::try_from(segment.query_start).map_err(|_| "segmented paged MLA start 超过 u32")?;
        let mut latent_group_u32 = u32::try_from(segment.latent_group_size).map_err(|_| "segmented paged MLA latent group 超过 u32")?;
        let mut topk_u32 = u32::try_from(top_k).map_err(|_| "segmented paged MLA top-k 超过 u32")?;
        let mut block_u32 = u32::try_from(block_size).map_err(|_| "segmented paged MLA block size 超过 u32")?;
        let mut selected_u32 = 1u32;
        let mut attention_args = [
            (&mut d_segment_query as *mut *mut c_void).cast(),
            (&mut d_segment_absorbed as *mut *mut c_void).cast(),
            (&mut d_latent as *mut *mut c_void).cast(),
            (&mut d_latent_scales as *mut *mut c_void).cast(),
            (&mut d_rope as *mut *mut c_void).cast(),
            (&mut d_table as *mut *mut c_void).cast(),
            (&mut d_selection as *mut *mut c_void).cast(),
            (&mut d_segment_weighted as *mut *mut c_void).cast(),
            (&mut rows_u32 as *mut u32).cast(),
            (&mut context_u32 as *mut u32).cast(),
            (&mut start_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut q_head_u32 as *mut u32).cast(),
            (&mut latent_u32 as *mut u32).cast(),
            (&mut latent_group_u32 as *mut u32).cast(),
            (&mut rope_u32 as *mut u32).cast(),
            (&mut topk_u32 as *mut u32).cast(),
            (&mut block_u32 as *mut u32).cast(),
            (&mut selected_u32 as *mut u32).cast(),
        ];
        launch_moe_kernel(functions.sparse_attention, heads_u32.div_ceil(8), rows_u32, 256, shared_u32, &mut attention_args, "HIP segmented paged MLA sparse prefill")?;
        row_offset += segment.query_rows;
    }

    let mut d_weighted = weighted.pointer;
    let mut d_output = output.pointer;
    let mut weight_head_start_u32 = 0u32;
    let mut project_args = [
        (&mut d_weighted as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut query_rows_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut weight_head_start_u32 as *mut u32).cast(),
        (&mut q_head_u32 as *mut u32).cast(),
        (&mut kv_head_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
        (&mut rope_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut scale_u32 as *mut u32).cast(),
        (&mut bits_u32 as *mut u32).cast(),
    ];
    let value_dim = kv_head_dim.checked_sub(q_head_dim - rope_dim).ok_or("segmented paged MLA value dim 下溢")?;
    let project_tiles = head_count.checked_mul(value_dim.div_ceil(tile)).ok_or("segmented paged MLA project grid 溢出")?;
    launch_moe_kernel(
        functions.project_value_wmma,
        u32::try_from(project_tiles).map_err(|_| "segmented paged MLA project grid 超过 u32")?,
        u32::try_from(query_rows.div_ceil(tile)).map_err(|_| "segmented paged MLA project rows 超过 u32")?,
        functions.wavefront_size * 8,
        0,
        &mut project_args,
        "HIP segmented MLA project value WMMA",
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_paged_mla_attention_ct_into(
    device_id: i32,
    query: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: Option<&DeviceBuffer>,
    latent_group_size: usize,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    selection: Option<&DeviceBuffer>,
    weight: CtMlaWeightRef<'_>,
    query_rows: usize,
    context_rows: usize,
    query_start: usize,
    q_projection: usize,
    head_count: usize,
    rope_dim: usize,
    top_k: usize,
    block_size: usize,
    output: &DeviceBuffer,
    split_decode_override: Option<bool>,
) -> Result<(), String> {
    try_paged_mla_attention_ct_inner(
        device_id,
        query,
        latent_cache,
        latent_scales,
        latent_group_size,
        rope_cache,
        block_table,
        selection,
        None,
        weight,
        query_rows,
        context_rows,
        query_start,
        q_projection,
        head_count,
        rope_dim,
        top_k,
        block_size,
        output,
        split_decode_override,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_paged_mla_attention_ct_inner(
    device_id: i32,
    query: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: Option<&DeviceBuffer>,
    latent_group_size: usize,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    selection: Option<&DeviceBuffer>,
    selection_counts: Option<&DeviceBuffer>,
    weight: CtMlaWeightRef<'_>,
    query_rows: usize,
    context_rows: usize,
    query_start: usize,
    q_projection: usize,
    head_count: usize,
    rope_dim: usize,
    top_k: usize,
    block_size: usize,
    output: &DeviceBuffer,
    split_decode_override: Option<bool>,
    shard: Option<(usize, &DeviceBuffer, &DeviceBuffer)>,
) -> Result<(), String> {
    if query_rows == 0
        || query_start.checked_add(query_rows).is_none_or(|end| end > context_rows)
        || !q_projection.is_multiple_of(head_count)
        || !weight.rows.is_multiple_of(head_count)
        || weight.cols == 0
        || !weight.cols.is_multiple_of(weight.group_size)
    {
        return Err("paged MLA shape 非法".to_owned());
    }
    let q_head_dim = q_projection / head_count;
    let kv_head_dim = weight.rows / head_count;
    let latent_dim = weight.cols;
    let shard_rows = |rows: usize, parity: usize| {
        let blocks = rows / block_size;
        let tail = rows % block_size;
        (blocks / 2) * block_size + usize::from(blocks % 2 > parity) * block_size + usize::from(blocks % 2 == parity) * tail
    };
    let resident_rows = shard.map_or(context_rows, |(parity, _, _)| shard_rows(context_rows, parity));
    if latent_group_size != 0 && (!latent_dim.is_multiple_of(latent_group_size) || latent_scales.is_none()) {
        return Err(format!("paged MLA latent Q8G{latent_group_size} cache 非法"));
    }
    validate_resident(query, device_id, query_rows.checked_mul(q_projection).and_then(|n| n.checked_mul(4)).ok_or("paged MLA query 大小溢出")?, "paged MLA query")?;
    let latent_element_bytes = if latent_group_size == 0 { 2 } else { 1 };
    validate_resident(latent_cache, device_id, resident_rows.checked_mul(latent_dim).and_then(|n| n.checked_mul(latent_element_bytes)).ok_or("paged MLA latent 大小溢出")?, "paged MLA latent")?;
    if let Some(scales) = latent_scales {
        validate_resident(scales, device_id, resident_rows.checked_mul(latent_dim / latent_group_size).and_then(|n| n.checked_mul(2)).ok_or("paged MLA latent scale 大小溢出")?, "paged MLA latent scales")?;
    }
    validate_resident(rope_cache, device_id, resident_rows.checked_mul(rope_dim).and_then(|n| n.checked_mul(2)).ok_or("paged MLA rope 大小溢出")?, "paged MLA rope")?;
    validate_resident(block_table, device_id, resident_rows.max(1).div_ceil(block_size).checked_mul(4).ok_or("paged MLA table 大小溢出")?, "paged MLA table")?;
    if let Some(selection) = selection {
        validate_resident(selection, device_id, query_rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("paged MLA selection 大小溢出")?, "paged MLA selection")?;
    }
    if let Some(counts) = selection_counts {
        if selection.is_none() || shard.is_none() {
            return Err("paged MLA selection counts 只能用于 pair shard selection".to_owned());
        }
        validate_resident(counts, device_id, query_rows.checked_mul(4).ok_or("paged MLA selection counts 大小溢出")?, "paged MLA selection counts")?;
    }
    if options().debug_finite {
        try_validate_finite_resident_range_f32(device_id, query, (query_rows - 1) * q_projection, q_projection).map_err(|error| format!("paged MLA query 包含非有限值或异常幅值: {error}"))?;
        if let Some(scales) = latent_scales {
            try_validate_finite_resident_range_bf16(device_id, scales, 0, resident_rows * (latent_dim / latent_group_size)).map_err(|error| format!("paged MLA latent scale 包含非有限值或异常幅值: {error}"))?;
        } else {
            try_validate_finite_resident_range_bf16(device_id, latent_cache, 0, resident_rows * latent_dim).map_err(|error| format!("paged MLA latent cache 包含非有限值或异常幅值: {error}"))?;
        }
        try_validate_finite_resident_range_bf16(device_id, rope_cache, 0, resident_rows * rope_dim).map_err(|error| format!("paged MLA rope cache 包含非有限值或异常幅值: {error}"))?;
    }
    let absorbed_elements = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(latent_dim)).ok_or("paged MLA absorbed 大小溢出")?;
    let output_elements = query_rows.checked_mul(q_projection).ok_or("paged MLA output 大小溢出")?;
    let intermediate_bytes = absorbed_elements.checked_mul(2).ok_or("paged MLA intermediate 字节数溢出")?;
    let (absorbed, workspace_weighted) = PAGED_MLA_WORKSPACES.with(|workspaces| -> Result<_, String> {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        Ok((reserve_paged_dsa_buffer(&mut workspace.absorbed, &mut workspace.absorbed_bytes, device_id, intermediate_bytes)?, reserve_paged_dsa_buffer(&mut workspace.weighted, &mut workspace.weighted_bytes, device_id, intermediate_bytes)?))
    })?;
    let weighted = shard.map_or(workspace_weighted.as_ref(), |(_, weighted, _)| weighted);
    let output_element_bytes = if query_rows > 1 { 2 } else { 4 };
    if let Some((_, shard_weighted, shard_stats)) = shard {
        validate_resident(shard_weighted, device_id, intermediate_bytes, "paged MLA shard weighted")?;
        validate_resident(shard_stats, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(2 * std::mem::size_of::<f32>())).ok_or("paged MLA shard stats 大小溢出")?, "paged MLA shard stats")?;
    } else {
        validate_resident(output, device_id, output_elements.checked_mul(output_element_bytes).ok_or("paged MLA output 字节数溢出")?, "paged MLA output")?;
    }
    let functions = paged_mla_functions(device_id)?;
    let profile_mla = options().kernel_profile;
    if options().log_mla_pointers {
        eprintln!(
            "[mla-pointers] device={device_id} start={query_start} rows={query_rows} query={:p}+{} latent={:p}+{} scales={:p}+{} rope={:p}+{} table={:p}+{} selection={:p}+{} absorbed={:p}+{} weighted={:p}+{} output={:p}+{}",
            query.pointer,
            query.bytes,
            latent_cache.pointer,
            latent_cache.bytes,
            latent_scales.map_or(ptr::null_mut(), |buffer| buffer.pointer),
            latent_scales.map_or(0, |buffer| buffer.bytes),
            rope_cache.pointer,
            rope_cache.bytes,
            block_table.pointer,
            block_table.bytes,
            selection.map_or(ptr::null_mut(), |buffer| buffer.pointer),
            selection.map_or(0, |buffer| buffer.bytes),
            absorbed.pointer,
            absorbed.bytes,
            weighted.pointer,
            weighted.bytes,
            output.pointer,
            output.bytes
        );
    }

    let mut d_query = query.pointer;
    let mut d_packed = weight.packed.pointer;
    let mut d_scales = weight.scales.pointer;
    let mut d_absorbed = absorbed.pointer;
    let mut query_rows_u32 = u32::try_from(query_rows).map_err(|_| "paged MLA query_rows 超过 u32")?;
    let mut heads_u32 = u32::try_from(head_count).map_err(|_| "paged MLA heads 超过 u32")?;
    let mut q_head_u32 = u32::try_from(q_head_dim).map_err(|_| "paged MLA q_head_dim 超过 u32")?;
    let mut kv_head_u32 = u32::try_from(kv_head_dim).map_err(|_| "paged MLA kv_head_dim 超过 u32")?;
    let mut latent_u32 = u32::try_from(latent_dim).map_err(|_| "paged MLA latent_dim 超过 u32")?;
    let mut rope_u32 = u32::try_from(rope_dim).map_err(|_| "paged MLA rope_dim 超过 u32")?;
    let mut group_u32 = u32::try_from(weight.group_size).map_err(|_| "paged MLA group_size 超过 u32")?;
    let mut scale_u32 = weight.scale_dtype;
    let mut bits_u32 = weight.bits;
    let mut absorb_args = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_absorbed as *mut *mut c_void).cast(),
        (&mut query_rows_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut q_head_u32 as *mut u32).cast(),
        (&mut kv_head_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
        (&mut rope_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut scale_u32 as *mut u32).cast(),
        (&mut bits_u32 as *mut u32).cast(),
    ];
    let absorb_wmma = query_rows > 2 && options().mla_absorb_wmma;
    // decode W8G32 向量化臂按 64 latent 列/block 发射（kernel 内同条件门控）。
    let absorb_w8v = query_rows <= 2 && weight.bits == 8 && weight.group_size == 32 && latent_dim.is_multiple_of(64) && q_head_dim > rope_dim && (q_head_dim - rope_dim).is_multiple_of(16) && q_head_dim - rope_dim <= 1024;
    let absorb_tile = if absorb_wmma {
        128
    } else if absorb_w8v {
        64
    } else {
        16
    };
    let absorb_tiles = head_count.checked_mul(latent_dim.div_ceil(absorb_tile)).ok_or("paged MLA absorb grid x 溢出")?;
    let absorb_started = profile_mla.then(std::time::Instant::now);
    launch_moe_kernel(
        if absorb_wmma { functions.absorb_query_wmma } else { functions.absorb_query },
        u32::try_from(absorb_tiles).map_err(|_| "paged MLA absorb grid x 超过 u32")?,
        u32::try_from(query_rows.div_ceil(absorb_tile)).map_err(|_| "paged MLA absorb grid y 超过 u32")?,
        if absorb_wmma { functions.wavefront_size * 8 } else { 256 },
        0,
        &mut absorb_args,
        if absorb_wmma { "HIP MLA absorb query WMMA" } else { "HIP MLA absorb query decode" },
    )?;
    if let Some(started) = absorb_started {
        super::synchronize_device(device_id, "hipDeviceSynchronize MLA absorb profile")?;
        eprintln!("[mla-profile] device={device_id} rows={query_rows} context={context_rows} absorb_ms={:.3}", started.elapsed().as_secs_f64() * 1e3,);
    }
    if query_rows != 0 && options().debug_finite {
        let row_elements = head_count * latent_dim;
        try_validate_finite_resident_range_bf16(device_id, &absorbed, (query_rows - 1) * row_elements, row_elements).map_err(|error| format!("paged MLA absorb query 包含非有限值: {error}"))?;
    }

    let mut d_latent = latent_cache.pointer;
    let mut d_latent_scales = latent_scales.map_or(ptr::null_mut(), |buffer| buffer.pointer);
    let mut latent_group_u32 = u32::try_from(latent_group_size).map_err(|_| "paged MLA latent group_size 超过 u32")?;
    let mut d_rope = rope_cache.pointer;
    let mut d_table = block_table.pointer;
    let mut d_selection = selection.map_or(ptr::null_mut(), |buffer| buffer.pointer);
    let mut d_selection_counts = selection_counts.map_or(ptr::null_mut(), |buffer| buffer.pointer);
    let mut d_weighted = weighted.pointer;
    let mut d_split_partial = ptr::null_mut();
    let mut d_split_stats = ptr::null_mut();
    let mut d_direct_weighted = ptr::null_mut();
    let mut context_u32 = u32::try_from(context_rows).map_err(|_| "paged MLA context 超过 u32")?;
    let mut start_u32 = u32::try_from(query_start).map_err(|_| "paged MLA query_start 超过 u32")?;
    let mut topk_u32 = u32::try_from(top_k).map_err(|_| "paged MLA top_k 超过 u32")?;
    let mut block_u32 = u32::try_from(block_size).map_err(|_| "paged MLA block_size 超过 u32")?;
    let mut shard_u32 = shard.map_or(2, |(parity, _, _)| parity as u32);
    let mut d_merged_stats = shard.map_or(ptr::null_mut(), |(_, _, stats)| stats.pointer);
    let force_dense_prefill = query_rows > 1 && options().force_dense_prefill;
    let mut selected_u32 = u32::from(selection.is_some() && !force_dense_prefill);
    let mut split_count_u32 = 1u32;
    let mut split_size_u32 = context_u32;
    let mut attention_args = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_absorbed as *mut *mut c_void).cast(),
        (&mut d_latent as *mut *mut c_void).cast(),
        (&mut d_latent_scales as *mut *mut c_void).cast(),
        (&mut d_rope as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut d_selection as *mut *mut c_void).cast(),
        (&mut d_selection_counts as *mut *mut c_void).cast(),
        (&mut d_weighted as *mut *mut c_void).cast(),
        (&mut d_split_partial as *mut *mut c_void).cast(),
        (&mut d_split_stats as *mut *mut c_void).cast(),
        (&mut query_rows_u32 as *mut u32).cast(),
        (&mut context_u32 as *mut u32).cast(),
        (&mut start_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut q_head_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
        (&mut latent_group_u32 as *mut u32).cast(),
        (&mut rope_u32 as *mut u32).cast(),
        (&mut topk_u32 as *mut u32).cast(),
        (&mut block_u32 as *mut u32).cast(),
        (&mut selected_u32 as *mut u32).cast(),
        (&mut split_count_u32 as *mut u32).cast(),
        (&mut split_size_u32 as *mut u32).cast(),
        (&mut shard_u32 as *mut u32).cast(),
    ];
    let mut sparse_args = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_absorbed as *mut *mut c_void).cast(),
        (&mut d_latent as *mut *mut c_void).cast(),
        (&mut d_latent_scales as *mut *mut c_void).cast(),
        (&mut d_rope as *mut *mut c_void).cast(),
        (&mut d_table as *mut *mut c_void).cast(),
        (&mut d_selection as *mut *mut c_void).cast(),
        (&mut d_weighted as *mut *mut c_void).cast(),
        (&mut query_rows_u32 as *mut u32).cast(),
        (&mut context_u32 as *mut u32).cast(),
        (&mut start_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut q_head_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
        (&mut latent_group_u32 as *mut u32).cast(),
        (&mut rope_u32 as *mut u32).cast(),
        (&mut topk_u32 as *mut u32).cast(),
        (&mut block_u32 as *mut u32).cast(),
        (&mut selected_u32 as *mut u32).cast(),
    ];
    let split_decode = shard.is_some() || split_decode_override.or(options().mla_decode_split).unwrap_or(context_rows >= options().mla_decode_split_threshold);
    let attention_started = profile_mla.then(std::time::Instant::now);
    const WMMA_HEADS_PER_BLOCK: usize = 16;
    const WMMA_TOKEN_TILE: usize = 32;
    const WMMA_OUTPUT_STRIDE: usize = 256;
    const WMMA_MAX_SHARED_BYTES: usize = 64 * 1024;
    let selected_wmma_shape = mla_decode_wmma_enabled()
        && options().mla_decode_wmma
        && functions.decode_partial_wmma_q8 != 0
        && latent_group_size != 0
        && latent_group_size.is_multiple_of(16)
        && head_count.is_multiple_of(WMMA_HEADS_PER_BLOCK)
        && rope_dim.is_multiple_of(16)
        && latent_dim.is_multiple_of(WMMA_OUTPUT_STRIDE)
        && latent_dim / WMMA_OUTPUT_STRIDE == 2;
    let selected_wmma_shared = if selected_wmma_shape {
        Some(
            (WMMA_TOKEN_TILE * 16)
                .checked_add(WMMA_HEADS_PER_BLOCK.checked_mul(16).ok_or("paged MLA WMMA probability tile 溢出")?)
                .and_then(|elements| elements.checked_add(latent_dim.checked_mul(16)?))
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                .and_then(|bytes| bytes.checked_add(WMMA_HEADS_PER_BLOCK.checked_mul(std::mem::size_of::<f32>())?))
                .and_then(|bytes| bytes.checked_add(WMMA_TOKEN_TILE * std::mem::size_of::<u32>()))
                .and_then(|bytes| bytes.checked_add(WMMA_TOKEN_TILE.checked_mul(latent_dim)?))
                .and_then(|bytes| bytes.checked_add(WMMA_TOKEN_TILE.checked_mul(latent_dim.checked_div(latent_group_size)?)?.checked_mul(std::mem::size_of::<u16>())?))
                .ok_or("paged MLA WMMA shared memory 字节数溢出")?,
        )
    } else {
        None
    };
    // gfx11 每个 workgroup 最多使用 64 KiB LDS；超限形态保留原标量路径。
    let selected_wmma_q8 = selected_wmma_shared.is_some_and(|bytes| bytes <= WMMA_MAX_SHARED_BYTES);
    let batch_split_decode = query_rows > 1 && query_rows <= 8 && split_decode && shard.is_none() && selection.is_some() && selected_wmma_q8;
    // 区分 dense / sparse / decode 路径的 profile 标签，仅用于 [mla-profile] 归因。
    let mut attention_kind = "decode_split";
    if query_rows == 1 && split_decode || batch_split_decode {
        // selected decode 的 latent 预重排：一次散读把 top-k 行收集到连续
        // workspace，scan 各 head-group 不再对同一批行做冗余散读。仅非分片路径
        // （分片下 gathered 行序与 parity compact 行号不一致，不适用）。
        if query_rows == 1 && selected_wmma_q8 && shard.is_none() && selection.is_some() && latent_scales.is_some() && latent_group_size != 0 && top_k >= 1024 {
            let visible_rows = query_start.checked_add(1).ok_or("paged MLA decode visible rows 溢出")?.min(context_rows);
            let gather_rows = visible_rows.min(top_k);
            let selection = selection.expect("selected 已检查");
            let scales_source = latent_scales.expect("Q8 scales 已检查");
            let groups = latent_dim / latent_group_size;
            let (gathered_latent, gathered_scales, gathered_rope) = PAGED_MLA_WORKSPACES.with(|workspaces| -> Result<_, String> {
                let mut workspaces = workspaces.borrow_mut();
                let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
                Ok((
                    reserve_paged_dsa_buffer(&mut workspace.gathered_latent, &mut workspace.gathered_latent_bytes, device_id, gather_rows * latent_dim)?,
                    reserve_paged_dsa_buffer(&mut workspace.gathered_scales, &mut workspace.gathered_scales_bytes, device_id, gather_rows * groups * 2)?,
                    reserve_paged_dsa_buffer(&mut workspace.gathered_rope, &mut workspace.gathered_rope_bytes, device_id, gather_rows * rope_dim * 2)?,
                ))
            })?;
            try_mla_gather_selected_q8(device_id, latent_cache, scales_source, rope_cache, block_table, selection, &gathered_latent, &gathered_scales, &gathered_rope, gather_rows, latent_dim, latent_group_size, rope_dim, block_size)?;
            d_latent = gathered_latent.pointer;
            d_latent_scales = gathered_scales.pointer;
            d_rope = gathered_rope.pointer;
            // gather 已完成逻辑页表映射，结果是按 selection 顺序排列的紧凑行。
            // scan 直接读行号，不能再次套用源页表或原上下文的可见行数。
            d_table = ptr::null_mut();
            d_selection = ptr::null_mut();
            selected_u32 = 0;
            context_u32 = gather_rows as u32;
            start_u32 = context_u32 - 1;
        }
        let requested_tile_size = options().mla_decode_tile_size;
        let visible_rows = query_start.checked_add(query_rows).ok_or("paged MLA decode visible rows 溢出")?.min(context_rows);
        // DSA selection 按逻辑候选区间拆分，tile 内再映射到真实分页位置。
        let decode_rows = if selection.is_some() { visible_rows.min(top_k) } else { visible_rows };
        // WMMA kernel 已按 shard_parity 把全局 token 映射到本卡紧凑物理行；
        // sequence shard 必须与未分片 decode 使用同级算子，才能兑现半程扫描收益。
        let decode_wmma_q8 = selected_wmma_q8;
        // merge kernel 的 shared scale 数组容量决定 tile 绝对上限。
        const DECODE_MAX_TILES: usize = 512;
        let tile_alignment = if decode_wmma_q8 { WMMA_TOKEN_TILE } else { 128 };
        // 单个 tile launch 的 grid.x：WMMA 每 block 16 头、fdot2 每 block 4 头。
        // cooperative 半头拆分下 head_count 已是半值，目标 tile 数自动翻倍，
        // 保证每卡总 block 数仍接近 mla_decode_target_blocks。
        let blocks_per_tile = if decode_wmma_q8 { head_count.div_ceil(WMMA_HEADS_PER_BLOCK) } else { head_count.div_ceil(4) }.max(1);
        let max_decode_tiles = (options().mla_decode_target_blocks / blocks_per_tile).clamp(1, DECODE_MAX_TILES);
        // split 数受 merge kernel 上限约束；长上下文自动放大 tile，避免退化为运行时错误。
        let minimum_tile_size = decode_rows.div_ceil(max_decode_tiles).div_ceil(tile_alignment).checked_mul(tile_alignment).ok_or("paged MLA decode minimum tile size 溢出")?;
        let decode_tile_size = requested_tile_size.max(minimum_tile_size).div_ceil(tile_alignment).checked_mul(tile_alignment).ok_or("paged MLA decode tile size 溢出")?;
        let tile_count = decode_rows.div_ceil(decode_tile_size);
        if tile_count == 0 || tile_count > DECODE_MAX_TILES {
            return Err(format!("paged MLA decode tile_count={tile_count} 非法"));
        }
        let colpar512 = decode_wmma_q8 && functions.decode_partial_wmma_q8_colpar512 != 0 && latent_dim == 512 && latent_group_size == 64 && rope_dim == 64;
        // 相同 tile 的交替 A/B：单行及 3..8 行受益，2 行会退化；
        // 不改变 split 大小与求和顺序，未验证的形状保留原路径。
        let shared_kv_tile = colpar512 && head_count == 32 && decode_rows >= 1024 && decode_tile_size == 64 && (query_rows == 1 || (3..=8).contains(&query_rows));
        // 共享 BF16 tile 的单行路径直接从原 KV 扫描，省去一次 top-k gather
        // 和三个临时输出；旧路径继续先把 top-k 行收集到连续
        // workspace，scan 各 head-group 不再对同一批行做冗余散读。仅非分片路径
        // （分片下 gathered 行序与 parity compact 行号不一致，不适用）。
        if !shared_kv_tile && query_rows == 1 && selected_wmma_q8 && shard.is_none() && selection.is_some() && latent_scales.is_some() && latent_group_size != 0 && top_k >= 1024 {
            let visible_rows = query_start.checked_add(1).ok_or("paged MLA decode visible rows 溢出")?.min(context_rows);
            let gather_rows = visible_rows.min(top_k);
            let selection = selection.expect("selected 已检查");
            let scales_source = latent_scales.expect("Q8 scales 已检查");
            let groups = latent_dim / latent_group_size;
            let (gathered_latent, gathered_scales, gathered_rope) = PAGED_MLA_WORKSPACES.with(|workspaces| -> Result<_, String> {
                let mut workspaces = workspaces.borrow_mut();
                let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
                Ok((
                    reserve_paged_dsa_buffer(&mut workspace.gathered_latent, &mut workspace.gathered_latent_bytes, device_id, gather_rows * latent_dim)?,
                    reserve_paged_dsa_buffer(&mut workspace.gathered_scales, &mut workspace.gathered_scales_bytes, device_id, gather_rows * groups * 2)?,
                    reserve_paged_dsa_buffer(&mut workspace.gathered_rope, &mut workspace.gathered_rope_bytes, device_id, gather_rows * rope_dim * 2)?,
                ))
            })?;
            try_mla_gather_selected_q8(device_id, latent_cache, scales_source, rope_cache, block_table, selection, &gathered_latent, &gathered_scales, &gathered_rope, gather_rows, latent_dim, latent_group_size, rope_dim, block_size)?;
            d_latent = gathered_latent.pointer;
            d_latent_scales = gathered_scales.pointer;
            d_rope = gathered_rope.pointer;
            // gather 已解析页表和 selection，scan 直接读取紧凑行，避免二次映射。
            d_table = ptr::null_mut();
            d_selection = ptr::null_mut();
            selected_u32 = 0;
            context_u32 = gather_rows as u32;
            start_u32 = context_u32 - 1;
        }
        let partial_elements = query_rows.checked_mul(tile_count).and_then(|elements| elements.checked_mul(head_count)).and_then(|elements| elements.checked_mul(latent_dim)).ok_or("paged MLA decode partial 元素数溢出")?;
        let stats_elements = query_rows.checked_mul(tile_count).and_then(|elements| elements.checked_mul(head_count)).and_then(|elements| elements.checked_mul(2)).ok_or("paged MLA decode stats 元素数溢出")?;
        let partial_bytes = partial_elements.checked_mul(std::mem::size_of::<f32>()).ok_or("paged MLA decode partial 字节数溢出")?;
        let stats_bytes = stats_elements.checked_mul(std::mem::size_of::<f32>()).ok_or("paged MLA decode stats 字节数溢出")?;
        let (partial, stats) = PAGED_MLA_SPLIT_WORKSPACES.with(|workspaces| -> Result<_, String> {
            let mut workspaces = workspaces.borrow_mut();
            let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
            let PagedMlaSplitWorkspace { partial, partial_bytes: partial_capacity, stats, stats_bytes: stats_capacity } = workspace;
            Ok((reserve_paged_dsa_buffer(partial, partial_capacity, device_id, partial_bytes)?, reserve_paged_dsa_buffer(stats, stats_capacity, device_id, stats_bytes)?))
        })?;
        let mut d_partial = partial.pointer;
        let mut d_stats = stats.pointer;
        let mut tile_size_u32 = u32::try_from(decode_tile_size).map_err(|_| "paged MLA decode tile_size 超过 u32")?;
        let mut tile_count_u32 = u32::try_from(tile_count).map_err(|_| "paged MLA decode tile_count 超过 u32")?;
        let mut split_tile_count_u32 = tile_count_u32;
        let mut stage_chunk_u32 = 0_u32;
        let mut partial_args = [
            (&mut d_query as *mut *mut c_void).cast(),
            (&mut d_absorbed as *mut *mut c_void).cast(),
            (&mut d_latent as *mut *mut c_void).cast(),
            (&mut d_latent_scales as *mut *mut c_void).cast(),
            (&mut d_rope as *mut *mut c_void).cast(),
            (&mut d_table as *mut *mut c_void).cast(),
            (&mut d_selection as *mut *mut c_void).cast(),
            (&mut d_selection_counts as *mut *mut c_void).cast(),
            (&mut d_partial as *mut *mut c_void).cast(),
            (&mut d_stats as *mut *mut c_void).cast(),
            (&mut d_direct_weighted as *mut *mut c_void).cast(),
            (&mut context_u32 as *mut u32).cast(),
            (&mut start_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut q_head_u32 as *mut u32).cast(),
            (&mut latent_u32 as *mut u32).cast(),
            (&mut latent_group_u32 as *mut u32).cast(),
            (&mut rope_u32 as *mut u32).cast(),
            (&mut topk_u32 as *mut u32).cast(),
            (&mut block_u32 as *mut u32).cast(),
            (&mut selected_u32 as *mut u32).cast(),
            (&mut tile_size_u32 as *mut u32).cast(),
            (&mut stage_chunk_u32 as *mut u32).cast(),
            (&mut shard_u32 as *mut u32).cast(),
        ];
        let decode_shared = if decode_wmma_q8 {
            selected_wmma_shared.expect("WMMA Q8 路径已经检查 shared memory")
        } else {
            let per_head = latent_dim
                .checked_mul(std::mem::size_of::<u16>())
                .and_then(|latent_bytes| rope_dim.checked_mul(std::mem::size_of::<f32>()).and_then(|rope_bytes| latent_bytes.checked_add(rope_bytes)))
                .ok_or("paged MLA decode query cache 字节数溢出")?;
            let query_cache = 4usize.checked_mul(per_head).ok_or("paged MLA decode query cache 字节数溢出")?;
            // Step2 软件流水：fdot2 路径把 latent tile 装进 LDS，QK/PV 共用一份。
            // Q8 存 code+scale（chunk 64），f16 存原始 bf16（chunk 32），都加 rope tile。
            let stage_chunk = if latent_group_size != 0 { 64usize } else { 32 };
            let stage_bytes = (if latent_group_size != 0 {
                stage_chunk.checked_mul(latent_dim).and_then(|codes| stage_chunk.checked_mul(latent_dim / latent_group_size).and_then(|groups| groups.checked_mul(2)).and_then(|scales| codes.checked_add(scales)))
            } else {
                stage_chunk.checked_mul(latent_dim).and_then(|latent| latent.checked_mul(2))
            })
            .and_then(|latent_tile| stage_chunk.checked_mul(rope_dim).and_then(|rope| rope.checked_mul(2)).and_then(|rope_tile| latent_tile.checked_add(rope_tile)))
            .ok_or("paged MLA decode staging 字节数溢出")?;
            // gfx11 每 workgroup 最多 64 KiB LDS；放不下时回退 legacy 直读路径。
            if query_cache.checked_add(stage_bytes).is_some_and(|shared| shared <= 64 * 1024) {
                stage_chunk_u32 = u32::try_from(stage_chunk).map_err(|_| "paged MLA decode stage chunk 超过 u32")?;
                query_cache.checked_add(stage_bytes).expect("staging 字节数已检查")
            } else {
                query_cache
            }
        };
        let decode_shared_u32 = u32::try_from(decode_shared).map_err(|_| "paged MLA decode query cache 超过 u32")?;
        if decode_wmma_q8 {
            // GLM-5.3 生产形状走列并行 QK 特化（8 wave 分列段 + LDS 归约，全常量展开）；
            // 生产实际 q_head_dim=256（nope 192+rope 64），不门控 q_head。
            // 数值为容差族（score 归约顺序变化），bench 位级对照见 decode_scan_bench。
            let launch_shared = if shared_kv_tile {
                // 16 KiB scratch 供 rope/QK 归约分时复用，完整 BF16 KV tile 供 QK/PV 共用。
                let bytes = (8 * 2 * 8 * 32 * 2 + WMMA_HEADS_PER_BLOCK * 16 + WMMA_TOKEN_TILE * (latent_dim + 8)) * 2 + (WMMA_HEADS_PER_BLOCK + WMMA_TOKEN_TILE) * 4;
                u32::try_from(bytes).map_err(|_| "paged MLA decoded KV shared 超过 u32")?
            } else if colpar512 {
                let bytes =
                    (8 * WMMA_TOKEN_TILE * 16 + WMMA_HEADS_PER_BLOCK * 16 + 16 * latent_dim) * 2 + WMMA_HEADS_PER_BLOCK * 4 + WMMA_TOKEN_TILE * 4 + WMMA_TOKEN_TILE * latent_dim + WMMA_TOKEN_TILE * (latent_dim / latent_group_size) * 2;
                u32::try_from(bytes).map_err(|_| "paged MLA colpar shared 超过 u32")?
            } else {
                decode_shared_u32
            };
            let launch_function = if shared_kv_tile {
                functions.decode_partial_wmma_q8_colpar512_shared
            } else if colpar512 {
                functions.decode_partial_wmma_q8_colpar512
            } else {
                functions.decode_partial_wmma_q8
            };
            // 一次性打印 kernel 选择，供生产 engagement 核查。
            static COLPAR_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !COLPAR_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[mla-decode-scan] kernel={} tile_size={decode_tile_size} tiles={tile_count} heads={head_count} latent={latent_dim} group={latent_group_size} rope={rope_dim} q_head={q_head_dim}",
                    if shared_kv_tile {
                        "colpar512_shared"
                    } else if colpar512 {
                        "colpar512"
                    } else {
                        "baseline"
                    }
                );
            }
            // WMMA kernel 没有标量软件流水的 stage_chunk 参数；必须单独维护
            // 参数表，否则 shard_parity 会错读前一项的 0，peer shard 被当成 owner。
            let mut wmma_args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_absorbed as *mut *mut c_void).cast(),
                (&mut d_latent as *mut *mut c_void).cast(),
                (&mut d_latent_scales as *mut *mut c_void).cast(),
                (&mut d_rope as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut d_selection_counts as *mut *mut c_void).cast(),
                (&mut d_partial as *mut *mut c_void).cast(),
                (&mut d_stats as *mut *mut c_void).cast(),
                (&mut d_direct_weighted as *mut *mut c_void).cast(),
                (&mut context_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut heads_u32 as *mut u32).cast(),
                (&mut q_head_u32 as *mut u32).cast(),
                (&mut latent_u32 as *mut u32).cast(),
                (&mut latent_group_u32 as *mut u32).cast(),
                (&mut rope_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut block_u32 as *mut u32).cast(),
                (&mut selected_u32 as *mut u32).cast(),
                (&mut tile_size_u32 as *mut u32).cast(),
                (&mut split_tile_count_u32 as *mut u32).cast(),
                (&mut shard_u32 as *mut u32).cast(),
            ];
            let grid_y = tile_count_u32.checked_mul(query_rows_u32).ok_or("paged MLA decode grid y 溢出")?;
            launch_moe_kernel(launch_function, heads_u32.div_ceil(WMMA_HEADS_PER_BLOCK as u32), grid_y, 256, launch_shared, &mut wmma_args, "HIP paged MLA decode partial Q8 WMMA")?;
        } else {
            launch_moe_kernel(functions.decode_partial, heads_u32.div_ceil(4), tile_count_u32, 256, decode_shared_u32, &mut partial_args, "HIP paged MLA decode partial")?;
        }
        let mut merge_args = [
            (&mut d_partial as *mut *mut c_void).cast(),
            (&mut d_stats as *mut *mut c_void).cast(),
            (&mut d_weighted as *mut *mut c_void).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut latent_u32 as *mut u32).cast(),
            (&mut tile_count_u32 as *mut u32).cast(),
            (&mut d_merged_stats as *mut *mut c_void).cast(),
        ];
        launch_moe_kernel(functions.split_merge_pl, heads_u32 * latent_u32.div_ceil(128), query_rows_u32, 128, 0, &mut merge_args, "HIP paged MLA decode merge")?;
    } else {
        let dense_prefill =
            functions.dense_wmma && (selection.is_none() || force_dense_prefill) && query_rows > 1 && latent_dim.is_multiple_of(16) && rope_dim.is_multiple_of(16) && (latent_group_size == 0 || latent_group_size.is_multiple_of(16));
        let sparse_prefill_wmma = query_rows > 1 && !dense_prefill && selection.is_some() && selected_wmma_q8 && sparse_prefill_wmma_enabled();
        let sparse_prefill_heads4 = shard.is_none() && query_rows > 1 && !dense_prefill && options().sparse_prefill_heads4;
        attention_kind = if query_rows == 1 {
            "decode"
        } else if dense_prefill {
            "dense"
        } else if sparse_prefill_wmma {
            "sparse_wmma"
        } else {
            "sparse"
        };
        let (attention_function, attention_heads_per_block, attention_shared) = if query_rows == 1 {
            let elements = latent_dim.checked_add(rope_dim).ok_or("paged MLA decode query cache 元素数溢出")?;
            let bytes = 4usize.checked_mul(elements).and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>())).ok_or("paged MLA decode query cache 字节数溢出")?;
            (functions.decode_attention, 4u32, bytes)
        } else {
            let per_head =
                latent_dim.checked_mul(std::mem::size_of::<u16>()).and_then(|latent| rope_dim.checked_mul(std::mem::size_of::<f32>()).and_then(|rope| latent.checked_add(rope))).ok_or("paged MLA prefill query cache 字节数溢出")?;
            let heads_per_block = if sparse_prefill_heads4 { 4usize } else { 8 };
            let bytes = heads_per_block.checked_mul(per_head).ok_or("paged MLA prefill query cache 字节数溢出")?;
            let function = if dense_prefill {
                functions.dense_attention
            } else if sparse_prefill_heads4 {
                functions.decode_attention
            } else {
                functions.sparse_attention
            };
            (function, heads_per_block as u32, bytes)
        };
        let attention_shared_u32 = u32::try_from(attention_shared).map_err(|_| "paged MLA query cache 超过 u32")?;
        let attention_rows = if dense_prefill && selection.is_none() { query_rows.div_ceil(2) } else { query_rows };
        let query_blocks = query_rows.div_ceil(2).checked_mul(head_count.div_ceil(8)).ok_or("paged MLA prefill query block 数溢出")?;
        let target_blocks = options().mla_prefill_target_blocks;
        let max_splits = (context_rows / 2048).clamp(1, 64);
        let requested_splits = if dense_prefill && selection.is_none() && query_blocks < target_blocks { target_blocks.div_ceil(query_blocks).min(max_splits) } else { 1 };
        let split_size = context_rows.div_ceil(requested_splits).div_ceil(128).checked_mul(128).ok_or("paged MLA prefill split size 溢出")?;
        let split_count = context_rows.div_ceil(split_size);
        // sparse WMMA 直接把每个 query/head 的最终 weighted latent 写到输出，
        // 不经过 split partial。parity shard 也不能因此白白保留一份约 128 MiB
        // 的 F32 workspace，否则长上下文的首次 chunk 会在热路径触发 hipMalloc。
        let split_buffers = if (split_count > 1 || shard.is_some()) && !sparse_prefill_wmma {
            let partial_elements = query_rows.checked_mul(split_count).and_then(|elements| elements.checked_mul(head_count)).and_then(|elements| elements.checked_mul(latent_dim)).ok_or("paged MLA prefill partial 元素数溢出")?;
            let stats_elements = query_rows.checked_mul(split_count).and_then(|elements| elements.checked_mul(head_count)).and_then(|elements| elements.checked_mul(2)).ok_or("paged MLA prefill stats 元素数溢出")?;
            let partial_bytes = partial_elements.checked_mul(std::mem::size_of::<f32>()).ok_or("paged MLA prefill partial 字节数溢出")?;
            let stats_bytes = stats_elements.checked_mul(std::mem::size_of::<f32>()).ok_or("paged MLA prefill stats 字节数溢出")?;
            Some(PAGED_MLA_SPLIT_WORKSPACES.with(|workspaces| -> Result<_, String> {
                let mut workspaces = workspaces.borrow_mut();
                let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
                let PagedMlaSplitWorkspace { partial, partial_bytes: partial_capacity, stats, stats_bytes: stats_capacity } = workspace;
                Ok((reserve_paged_dsa_buffer(partial, partial_capacity, device_id, partial_bytes)?, reserve_paged_dsa_buffer(stats, stats_capacity, device_id, stats_bytes)?))
            })?)
        } else {
            None
        };
        if let Some((partial, stats)) = split_buffers.as_ref() {
            d_split_partial = partial.pointer;
            d_split_stats = stats.pointer;
            split_count_u32 = u32::try_from(split_count).map_err(|_| "paged MLA prefill split_count 超过 u32")?;
            split_size_u32 = u32::try_from(split_size).map_err(|_| "paged MLA prefill split_size 超过 u32")?;
            // `attention_args` 持有该值的 raw pointer；显式读取让 lint 看见真实数据流。
            std::hint::black_box(split_size_u32);
        }
        let attention_grid_y = attention_rows.checked_mul(split_count).ok_or("paged MLA attention grid y 溢出")?;
        if sparse_prefill_wmma {
            d_direct_weighted = d_weighted;
            let mut direct_tile_size_u32 = topk_u32;
            let mut direct_split_tile_count_u32 = 1u32;
            let mut direct_args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_absorbed as *mut *mut c_void).cast(),
                (&mut d_latent as *mut *mut c_void).cast(),
                (&mut d_latent_scales as *mut *mut c_void).cast(),
                (&mut d_rope as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut d_selection_counts as *mut *mut c_void).cast(),
                (&mut d_split_partial as *mut *mut c_void).cast(),
                (&mut d_merged_stats as *mut *mut c_void).cast(),
                (&mut d_direct_weighted as *mut *mut c_void).cast(),
                (&mut context_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut heads_u32 as *mut u32).cast(),
                (&mut q_head_u32 as *mut u32).cast(),
                (&mut latent_u32 as *mut u32).cast(),
                (&mut latent_group_u32 as *mut u32).cast(),
                (&mut rope_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut block_u32 as *mut u32).cast(),
                (&mut selected_u32 as *mut u32).cast(),
                (&mut direct_tile_size_u32 as *mut u32).cast(),
                (&mut direct_split_tile_count_u32 as *mut u32).cast(),
                (&mut shard_u32 as *mut u32).cast(),
            ];
            let shared_bytes = selected_wmma_shared.expect("sparse WMMA 已检查 shared memory");
            let prefill_shared_bytes = latent_dim
                .checked_add(8)
                .and_then(|stride| stride.checked_mul(WMMA_TOKEN_TILE))
                .and_then(|elements| elements.checked_add(WMMA_TOKEN_TILE * 16 + 32 * 16))
                .and_then(|elements| elements.checked_mul(2))
                .and_then(|bytes| bytes.checked_add((32 + WMMA_TOKEN_TILE) * 4))
                .ok_or("paged MLA prefill BF16 KV tile 字节数溢出")?;
            let heads32 = query_rows >= 16 && head_count.is_multiple_of(32) && prefill_shared_bytes <= WMMA_MAX_SHARED_BYTES;
            #[cfg(test)]
            let heads32 = heads32 && TEST_SPARSE_PREFILL_HEADS32.load(std::sync::atomic::Ordering::Relaxed);
            let heads_per_block = if heads32 { 32 } else { WMMA_HEADS_PER_BLOCK };
            // Q8G64 的完整 BF16 tile 与旧 code/scale + PV scratch 占用相同 LDS，
            // QK/PV 共享一次反量化，decode 保持原来的暂存路径。
            let shared_bytes = if heads32 { prefill_shared_bytes } else { shared_bytes };
            launch_moe_kernel(
                if heads32 { functions.prefill_wmma_q8_heads32 } else { functions.decode_partial_wmma_q8 },
                heads_u32.div_ceil(heads_per_block as u32),
                query_rows_u32,
                256,
                u32::try_from(shared_bytes).map_err(|_| "paged MLA sparse WMMA shared memory 超过 u32")?,
                &mut direct_args,
                "HIP paged MLA sparse prefill Q8 WMMA",
            )?;
        } else if query_rows == 1 || sparse_prefill_heads4 {
            // Decode kernel 没有 split workspace 参数，不能复用 prefill 的参数表。
            let mut decode_args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_absorbed as *mut *mut c_void).cast(),
                (&mut d_latent as *mut *mut c_void).cast(),
                (&mut d_latent_scales as *mut *mut c_void).cast(),
                (&mut d_rope as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut d_weighted as *mut *mut c_void).cast(),
                (&mut query_rows_u32 as *mut u32).cast(),
                (&mut context_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut heads_u32 as *mut u32).cast(),
                (&mut q_head_u32 as *mut u32).cast(),
                (&mut latent_u32 as *mut u32).cast(),
                (&mut latent_group_u32 as *mut u32).cast(),
                (&mut rope_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut block_u32 as *mut u32).cast(),
                (&mut selected_u32 as *mut u32).cast(),
            ];
            launch_moe_kernel(attention_function, heads_u32.div_ceil(attention_heads_per_block), if query_rows == 1 { 1 } else { query_rows_u32 }, 256, attention_shared_u32, &mut decode_args, "HIP paged MLA decode")?;
        } else if dense_prefill {
            launch_moe_kernel(
                attention_function,
                heads_u32.div_ceil(attention_heads_per_block),
                u32::try_from(attention_grid_y).map_err(|_| "paged MLA attention grid y 超过 u32")?,
                256,
                attention_shared_u32,
                &mut attention_args,
                "HIP paged MLA dense prefill",
            )?;
        } else {
            launch_moe_kernel(
                attention_function,
                heads_u32.div_ceil(attention_heads_per_block),
                u32::try_from(attention_grid_y).map_err(|_| "paged MLA attention grid y 超过 u32")?,
                256,
                attention_shared_u32,
                &mut sparse_args,
                "HIP paged MLA sparse prefill",
            )?;
        }
        if (split_count > 1 || shard.is_some()) && !sparse_prefill_wmma {
            let mut merge_args = [
                (&mut d_split_partial as *mut *mut c_void).cast(),
                (&mut d_split_stats as *mut *mut c_void).cast(),
                (&mut d_weighted as *mut *mut c_void).cast(),
                (&mut heads_u32 as *mut u32).cast(),
                (&mut latent_u32 as *mut u32).cast(),
                (&mut split_count_u32 as *mut u32).cast(),
                (&mut d_merged_stats as *mut *mut c_void).cast(),
            ];
            launch_moe_kernel(functions.split_merge_pl, heads_u32 * latent_u32.div_ceil(128), query_rows_u32, 128, 0, &mut merge_args, "HIP paged MLA prefill split merge")?;
        }
    }
    if let Some(started) = attention_started {
        super::synchronize_device(device_id, "hipDeviceSynchronize MLA attention profile")?;
        eprintln!("[mla-profile] device={device_id} rows={query_rows} context={context_rows} kind={attention_kind} attention_ms={:.3}", started.elapsed().as_secs_f64() * 1e3,);
    }
    if query_rows != 0 && options().debug_finite {
        let row_elements = head_count * latent_dim;
        try_validate_finite_resident_range_bf16(device_id, weighted, (query_rows - 1) * row_elements, row_elements).map_err(|error| format!("paged MLA weighted latent 包含非有限值: {error}"))?;
    }
    if shard.is_some() {
        return Ok(());
    }

    let mut d_output = output.pointer;
    let mut weight_head_start_u32 = 0u32;
    let mut project_args = [
        (&mut d_weighted as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut query_rows_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut weight_head_start_u32 as *mut u32).cast(),
        (&mut q_head_u32 as *mut u32).cast(),
        (&mut kv_head_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
        (&mut rope_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut scale_u32 as *mut u32).cast(),
        (&mut bits_u32 as *mut u32).cast(),
    ];
    let value_dim = kv_head_dim.checked_sub(q_head_dim - rope_dim).ok_or("paged MLA value_dim 下溢")?;
    let project_wmma = query_rows > 2;
    // PV perm 生产切换已回退:服务进程中该 kernel 发射返回成功但整体不执行
    // (入口盖章不落、输出保持池内旧值,输入/参数/发射路径六轮取证全部正常,
    // 同源码 cargo test 进程 oracle 逐位一致)——根因未明,取证链见
    // docs/rocm-glm53-kernel-probe-20260903.md。kernel 保留在模块内待后续排查。
    let project_tile = if project_wmma { 128 } else { 16 };
    let project_tiles = head_count.checked_mul(value_dim.div_ceil(project_tile)).ok_or("paged MLA project grid x 溢出")?;
    let project_started = profile_mla.then(std::time::Instant::now);
    launch_moe_kernel(
        if project_wmma { functions.project_value_wmma } else { functions.project_value },
        u32::try_from(project_tiles).map_err(|_| "paged MLA project grid x 超过 u32")?,
        u32::try_from(query_rows.div_ceil(project_tile)).map_err(|_| "paged MLA project grid y 超过 u32")?,
        if project_wmma { functions.wavefront_size * 8 } else { 256 },
        0,
        &mut project_args,
        if project_wmma { "HIP MLA project value WMMA" } else { "HIP MLA project value decode" },
    )?;
    if let Some(started) = project_started {
        super::synchronize_device(device_id, "hipDeviceSynchronize MLA project profile")?;
        eprintln!("[mla-profile] device={device_id} rows={query_rows} context={context_rows} project_ms={:.3}", started.elapsed().as_secs_f64() * 1e3,);
    }
    if query_rows != 0 && options().debug_finite {
        if query_rows > 1 {
            try_validate_finite_resident_range_bf16(device_id, output, (query_rows - 1) * q_projection, q_projection).map_err(|error| format!("paged MLA projected value 包含非有限值: {error}"))?;
        } else {
            try_validate_finite_resident_range_f32(device_id, output, 0, q_projection).map_err(|error| format!("paged MLA projected value 包含非有限值: {error}"))?;
        }
    }
    Ok(())
}

pub(crate) struct PagedMlaShardAttention {
    pub weighted: std::sync::Arc<DeviceBuffer>,
    pub stats: std::sync::Arc<DeviceBuffer>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_paged_mla_attention_ct_shard(
    device_id: i32,
    query: &DeviceBuffer,
    latent_cache: &DeviceBuffer,
    latent_scales: Option<&DeviceBuffer>,
    latent_group_size: usize,
    rope_cache: &DeviceBuffer,
    block_table: &DeviceBuffer,
    selection: Option<&DeviceBuffer>,
    selection_counts: Option<&DeviceBuffer>,
    weight: CtMlaWeightRef<'_>,
    query_rows: usize,
    context_rows: usize,
    query_start: usize,
    q_projection: usize,
    head_count: usize,
    rope_dim: usize,
    top_k: usize,
    block_size: usize,
    parity: usize,
) -> Result<PagedMlaShardAttention, String> {
    if parity > 1 || block_size == 0 {
        return Err(format!("paged MLA shard parity={parity} block_size={block_size} 非法"));
    }
    let latent_dim = weight.cols;
    let weighted_bytes = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(latent_dim)).and_then(|n| n.checked_mul(2)).ok_or("paged MLA shard weighted 大小溢出")?;
    let stats_bytes = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(2 * std::mem::size_of::<f32>())).ok_or("paged MLA shard stats 大小溢出")?;
    let weighted = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, weighted_bytes)?);
    let stats = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, stats_bytes)?);
    let full_blocks = context_rows / block_size;
    let tail = context_rows % block_size;
    let resident_rows = (full_blocks / 2) * block_size + usize::from(full_blocks % 2 > parity) * block_size + usize::from(full_blocks % 2 == parity) * tail;
    if resident_rows == 0 {
        if !weighted_bytes.is_multiple_of(std::mem::size_of::<u32>()) {
            return Err(format!("paged MLA 空 shard weighted bytes={weighted_bytes} 未按 u32 对齐"));
        }
        let functions = paged_mla_functions(device_id)?;
        let mut d_weighted = weighted.pointer;
        let mut weighted_words = u32::try_from(weighted_bytes / std::mem::size_of::<u32>()).map_err(|_| "paged MLA 空 shard weighted words 超过 u32")?;
        let mut clear_weighted_args = [(&mut d_weighted as *mut *mut c_void).cast(), (&mut weighted_words as *mut u32).cast()];
        launch_tensor_kernel(functions.dsa_clear, weighted_words.div_ceil(256), 256, &mut clear_weighted_args, "HIP paged MLA empty shard weighted clear")?;
        let mut d_stats = stats.pointer;
        let mut stats_words = u32::try_from(stats_bytes / std::mem::size_of::<u32>()).map_err(|_| "paged MLA 空 shard stats words 超过 u32")?;
        let mut clear_stats_args = [(&mut d_stats as *mut *mut c_void).cast(), (&mut stats_words as *mut u32).cast()];
        launch_tensor_kernel(functions.dsa_clear, stats_words.div_ceil(256), 256, &mut clear_stats_args, "HIP paged MLA empty shard stats clear")?;
        return Ok(PagedMlaShardAttention { weighted, stats });
    }
    try_paged_mla_attention_ct_inner(
        device_id,
        query,
        latent_cache,
        latent_scales,
        latent_group_size,
        rope_cache,
        block_table,
        selection,
        selection_counts,
        weight,
        query_rows,
        context_rows,
        query_start,
        q_projection,
        head_count,
        rope_dim,
        top_k,
        block_size,
        &weighted,
        Some(true),
        Some((parity, &weighted, &stats)),
    )?;
    Ok(PagedMlaShardAttention { weighted, stats })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_paged_mla_shard_scale_project_ct(
    device_id: i32,
    weighted: &DeviceBuffer,
    local_stats: &DeviceBuffer,
    remote_stats: &DeviceBuffer,
    weight: CtMlaWeightRef<'_>,
    query_rows: usize,
    q_projection: usize,
    head_count: usize,
    rope_dim: usize,
    output: &DeviceBuffer,
) -> Result<(), String> {
    if query_rows == 0 || !q_projection.is_multiple_of(head_count) || !weight.rows.is_multiple_of(head_count) || weight.cols == 0 || !weight.cols.is_multiple_of(weight.group_size) {
        return Err("paged MLA shard project shape 非法".to_owned());
    }
    let q_head_dim = q_projection / head_count;
    let kv_head_dim = weight.rows / head_count;
    let latent_dim = weight.cols;
    let weighted_elements = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(latent_dim)).ok_or("paged MLA shard project weighted 大小溢出")?;
    let stats_bytes = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(2 * std::mem::size_of::<f32>())).ok_or("paged MLA shard project stats 大小溢出")?;
    validate_resident(weighted, device_id, weighted_elements.checked_mul(2).ok_or("paged MLA shard project weighted 字节溢出")?, "paged MLA shard weighted")?;
    validate_resident(local_stats, device_id, stats_bytes, "paged MLA shard local stats")?;
    validate_resident(remote_stats, device_id, stats_bytes, "paged MLA shard remote stats")?;
    validate_resident(output, device_id, query_rows.checked_mul(q_projection).and_then(|n| n.checked_mul(if query_rows > 1 { 2 } else { 4 })).ok_or("paged MLA shard output 大小溢出")?, "paged MLA shard output")?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_weighted = weighted.pointer;
    let mut d_local_stats = local_stats.pointer;
    let mut d_remote_stats = remote_stats.pointer;
    let mut rows_u32 = u32::try_from(query_rows).map_err(|_| "paged MLA shard rows 超过 u32")?;
    let mut heads_u32 = u32::try_from(head_count).map_err(|_| "paged MLA shard heads 超过 u32")?;
    let mut weight_head_start_u32 = 0u32;
    let mut latent_u32 = u32::try_from(latent_dim).map_err(|_| "paged MLA shard latent 超过 u32")?;
    let mut scale_args = [
        (&mut d_weighted as *mut *mut c_void).cast(),
        (&mut d_local_stats as *mut *mut c_void).cast(),
        (&mut d_remote_stats as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.shard_scale, u32::try_from(weighted_elements.div_ceil(256)).map_err(|_| "paged MLA shard scale grid 超过 u32")?, 256, &mut scale_args, "HIP paged MLA shard scale")?;

    let mut d_packed = weight.packed.pointer;
    let mut d_scales = weight.scales.pointer;
    let mut d_output = output.pointer;
    let mut q_head_u32 = u32::try_from(q_head_dim).map_err(|_| "paged MLA shard q head 超过 u32")?;
    let mut kv_head_u32 = u32::try_from(kv_head_dim).map_err(|_| "paged MLA shard kv head 超过 u32")?;
    let mut rope_u32 = u32::try_from(rope_dim).map_err(|_| "paged MLA shard rope 超过 u32")?;
    let mut group_u32 = u32::try_from(weight.group_size).map_err(|_| "paged MLA shard group 超过 u32")?;
    let mut scale_u32 = weight.scale_dtype;
    let mut bits_u32 = weight.bits;
    let mut project_args = [
        (&mut d_weighted as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut weight_head_start_u32 as *mut u32).cast(),
        (&mut q_head_u32 as *mut u32).cast(),
        (&mut kv_head_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
        (&mut rope_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut scale_u32 as *mut u32).cast(),
        (&mut bits_u32 as *mut u32).cast(),
    ];
    let value_dim = kv_head_dim.checked_sub(q_head_dim - rope_dim).ok_or("paged MLA shard value_dim 下溢")?;
    let project_wmma = query_rows > 1;
    let project_tile = if project_wmma { 128 } else { 16 };
    let project_tiles = head_count.checked_mul(value_dim.div_ceil(project_tile)).ok_or("paged MLA shard project grid 溢出")?;
    launch_moe_kernel(
        if project_wmma { functions.project_value_wmma } else { functions.project_value },
        u32::try_from(project_tiles).map_err(|_| "paged MLA shard project grid 超过 u32")?,
        u32::try_from(query_rows.div_ceil(project_tile)).map_err(|_| "paged MLA shard project rows 超过 u32")?,
        if project_wmma { functions.wavefront_size * 8 } else { 256 },
        0,
        &mut project_args,
        "HIP MLA shard project value",
    )
}

/// 精确合并两个 sequence shard，但只物化本卡负责的 query heads。随后 value
/// projection 也只计算这些 heads，形成 attention 的 head reduce-scatter。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_paged_mla_shard_merge_project_heads_ct(
    device_id: i32,
    local_weighted: &DeviceBuffer,
    remote_weighted: &DeviceBuffer,
    local_stats: &DeviceBuffer,
    remote_stats: &DeviceBuffer,
    weight: CtMlaWeightRef<'_>,
    query_rows: usize,
    q_projection: usize,
    total_heads: usize,
    head_start: usize,
    head_count: usize,
    rope_dim: usize,
    output: &DeviceBuffer,
) -> Result<(), String> {
    if query_rows == 0
        || head_count == 0
        || head_start.checked_add(head_count).map_or(true, |end| end > total_heads)
        || !q_projection.is_multiple_of(total_heads)
        || !weight.rows.is_multiple_of(total_heads)
        || weight.cols == 0
        || weight.group_size == 0
        || !weight.cols.is_multiple_of(weight.group_size)
    {
        return Err("paged MLA shard head merge shape 非法".to_owned());
    }
    let q_head_dim = q_projection / total_heads;
    let latent_dim = weight.cols;
    let full_weighted_elements = query_rows.checked_mul(total_heads).and_then(|n| n.checked_mul(latent_dim)).ok_or("paged MLA shard head merge weighted 大小溢出")?;
    let stats_bytes = query_rows.checked_mul(total_heads).and_then(|n| n.checked_mul(2 * std::mem::size_of::<f32>())).ok_or("paged MLA shard head merge stats 大小溢出")?;
    let compact_elements = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(latent_dim)).ok_or("paged MLA shard head merge compact 大小溢出")?;
    validate_resident(local_weighted, device_id, full_weighted_elements.checked_mul(2).ok_or("paged MLA shard head merge weighted 字节溢出")?, "paged MLA shard local weighted")?;
    validate_resident(local_stats, device_id, stats_bytes, "paged MLA shard local stats")?;
    if remote_weighted.bytes < full_weighted_elements.checked_mul(2).ok_or("paged MLA remote weighted 字节溢出")? || remote_stats.bytes < stats_bytes {
        return Err("paged MLA shard remote weighted/stats 大小不足".to_owned());
    }
    if remote_weighted.device_id != device_id {
        enable_peer_access(device_id, remote_weighted.device_id)?;
    }
    if remote_stats.device_id != device_id {
        enable_peer_access(device_id, remote_stats.device_id)?;
    }
    let compact = DeviceBuffer::allocate_reusable(device_id, compact_elements.checked_mul(2).ok_or("paged MLA shard compact 字节溢出")?)?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_local_weighted = local_weighted.pointer;
    let mut d_remote_weighted = remote_weighted.pointer;
    let mut d_local_stats = local_stats.pointer;
    let mut d_remote_stats = remote_stats.pointer;
    let mut d_compact = compact.pointer;
    let mut rows_u32 = u32::try_from(query_rows).map_err(|_| "paged MLA shard head merge rows 超过 u32")?;
    let mut total_heads_u32 = u32::try_from(total_heads).map_err(|_| "paged MLA shard total heads 超过 u32")?;
    let mut head_start_u32 = u32::try_from(head_start).map_err(|_| "paged MLA shard head start 超过 u32")?;
    let mut head_count_u32 = u32::try_from(head_count).map_err(|_| "paged MLA shard head count 超过 u32")?;
    let mut latent_u32 = u32::try_from(latent_dim).map_err(|_| "paged MLA shard latent 超过 u32")?;
    let mut merge_args = [
        (&mut d_local_weighted as *mut *mut c_void).cast(),
        (&mut d_remote_weighted as *mut *mut c_void).cast(),
        (&mut d_local_stats as *mut *mut c_void).cast(),
        (&mut d_remote_stats as *mut *mut c_void).cast(),
        (&mut d_compact as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut total_heads_u32 as *mut u32).cast(),
        (&mut head_start_u32 as *mut u32).cast(),
        (&mut head_count_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
    ];
    let merge_heads = query_rows.checked_mul(head_count).ok_or("paged MLA shard head merge grid 溢出")?;
    launch_tensor_kernel(functions.shard_merge_heads, u32::try_from(merge_heads).map_err(|_| "paged MLA shard head merge grid 超过 u32")?, 256, &mut merge_args, "HIP paged MLA shard merge heads")?;

    let kv_head_dim = weight.rows / total_heads;
    let output_elements = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(q_head_dim)).ok_or("paged MLA shard head project output 大小溢出")?;
    validate_resident(output, device_id, output_elements.checked_mul(if query_rows > 1 { 2 } else { 4 }).ok_or("paged MLA shard head project output 字节溢出")?, "paged MLA shard head project output")?;
    let mut d_packed = weight.packed.pointer;
    let mut d_scales = weight.scales.pointer;
    let mut d_output = output.pointer;
    let mut q_head_u32 = u32::try_from(q_head_dim).map_err(|_| "paged MLA shard q head 超过 u32")?;
    let mut kv_head_u32 = u32::try_from(kv_head_dim).map_err(|_| "paged MLA shard kv head 超过 u32")?;
    let mut rope_u32 = u32::try_from(rope_dim).map_err(|_| "paged MLA shard rope 超过 u32")?;
    let mut group_u32 = u32::try_from(weight.group_size).map_err(|_| "paged MLA shard group 超过 u32")?;
    let mut scale_u32 = weight.scale_dtype;
    let mut bits_u32 = weight.bits;
    let mut project_args = [
        (&mut d_compact as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut head_count_u32 as *mut u32).cast(),
        (&mut head_start_u32 as *mut u32).cast(),
        (&mut q_head_u32 as *mut u32).cast(),
        (&mut kv_head_u32 as *mut u32).cast(),
        (&mut latent_u32 as *mut u32).cast(),
        (&mut rope_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut scale_u32 as *mut u32).cast(),
        (&mut bits_u32 as *mut u32).cast(),
    ];
    let value_dim = kv_head_dim.checked_sub(q_head_dim - rope_dim).ok_or("paged MLA shard head value_dim 下溢")?;
    let project_wmma = query_rows > 1;
    let project_tile = if project_wmma { 128 } else { 16 };
    let project_tiles = head_count.checked_mul(value_dim.div_ceil(project_tile)).ok_or("paged MLA shard head project grid 溢出")?;
    launch_moe_kernel(
        if project_wmma { functions.project_value_wmma } else { functions.project_value },
        u32::try_from(project_tiles).map_err(|_| "paged MLA shard head project grid 超过 u32")?,
        u32::try_from(query_rows.div_ceil(project_tile)).map_err(|_| "paged MLA shard head project rows 超过 u32")?,
        if project_wmma { functions.wavefront_size * 8 } else { 256 },
        0,
        &mut project_args,
        "HIP MLA shard merge/project heads",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_select_medium_tile_requires_decode_parallelism_hint() {
        assert_eq!(parallel_select_tile_rows(128 * 1024, false), 1024);
        assert_eq!(parallel_select_tile_rows(128 * 1024, true), 2048);
        assert_eq!(parallel_select_tile_rows(119 * 1024, true), 1024);
        assert_eq!(parallel_select_tile_rows(256 * 1024, false), 4096);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_project_value_rows2_bf16_preserves_output_guard() {
        let bf16 = |value: f32| (value.to_bits() >> 16) as u16;
        let as_bytes = |values: &[u16]| unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 2) };
        let (mut rows, mut heads, mut head_start, mut q_head, mut kv_head, mut latent, mut rope, mut group, mut scale_type, mut bits) = (2_u32, 2_u32, 0_u32, 8_u32, 8_u32, 16_u32, 4_u32, 16_u32, 0_u32, 16_u32);
        let inputs = (0..rows * heads * latent).map(|i| bf16((i as i32 % 7 - 3) as f32 / 8.0)).collect::<Vec<_>>();
        let weights = (0..heads * kv_head * latent).map(|i| bf16((i as i32 % 11 - 5) as f32 / 16.0)).collect::<Vec<_>>();
        let input = DeviceBuffer::upload(0, as_bytes(&inputs)).unwrap();
        let weight = DeviceBuffer::upload(0, as_bytes(&weights)).unwrap();
        let scales = DeviceBuffer::upload(0, as_bytes(&[bf16(1.0)])).unwrap();
        let elements = (rows * heads * q_head) as usize;
        // guard 与输出等大，错误的 F32 写入也不会破坏其他 allocation。
        let output = DeviceBuffer::upload(0, &vec![0xa5; elements * 4]).unwrap();
        let (mut d_input, mut d_weight, mut d_scales, mut d_output) = (input.pointer, weight.pointer, scales.pointer, output.pointer);
        let mut args = [
            (&mut d_input as *mut *mut c_void).cast(), (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_scales as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(), (&mut heads as *mut u32).cast(), (&mut head_start as *mut u32).cast(),
            (&mut q_head as *mut u32).cast(), (&mut kv_head as *mut u32).cast(), (&mut latent as *mut u32).cast(),
            (&mut rope as *mut u32).cast(), (&mut group as *mut u32).cast(), (&mut scale_type as *mut u32).cast(), (&mut bits as *mut u32).cast(),
        ];
        let functions = paged_mla_functions(0).unwrap();
        launch_moe_kernel(functions.project_value, heads, 1, 256, 0, &mut args, "MLA 双行 BF16 输出门禁").unwrap();
        let mut actual = vec![0; elements * 4];
        output.copy_to_host(&mut actual).unwrap();
        assert!(actual[elements * 2..].iter().all(|&byte| byte == 0xa5), "双行 projection 越过 BF16 输出边界");
        let mut expected = vec![0_u16; elements];
        for row in 0..rows as usize {
            for head in 0..heads as usize {
                for value in 0..4 {
                    let sum = (0..latent as usize).map(|column| {
                        let x = f32::from_bits(u32::from(inputs[(row * heads as usize + head) * latent as usize + column]) << 16);
                        let w = f32::from_bits(u32::from(weights[(head * kv_head as usize + 4 + value) * latent as usize + column]) << 16);
                        x * w
                    }).sum::<f32>();
                    expected[(row * heads as usize + head) * q_head as usize + value] = bf16(sum);
                }
            }
        }
        assert_eq!(&actual[..elements * 2], as_bytes(&expected), "双行 projection 与 CPU oracle 不一致");
    }

    fn ordered_score(score: f32) -> u32 {
        let bits = score.to_bits();
        bits ^ if bits & 0x8000_0000 != 0 { 0xffff_ffff } else { 0x8000_0000 }
    }

    fn as_bytes<T>(values: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn as_bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values)) }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_hot_scatter_preserves_exact_rows() {
        const DEVICE_ID: i32 = 0;
        const ROWS: usize = 3;
        const TARGET_ROWS: usize = 5;
        const LATENT_COLUMNS: usize = 7;
        const SCALE_COLUMNS: usize = 2;
        const ROPE_COLUMNS: usize = 3;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let latent = (0..ROWS * LATENT_COLUMNS).map(|value| value as u8 + 1).collect::<Vec<_>>();
        let scales = (0..ROWS * SCALE_COLUMNS).map(|value| value as u16 + 101).collect::<Vec<_>>();
        let rope = (0..ROWS * ROPE_COLUMNS).map(|value| value as u16 + 201).collect::<Vec<_>>();
        let slots = [4_u32, 1, 3];
        let source_latent = DeviceBuffer::upload(DEVICE_ID, &latent).unwrap();
        let source_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scales)).unwrap();
        let source_rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
        let slots_device = DeviceBuffer::upload(DEVICE_ID, as_bytes(&slots)).unwrap();
        let target_latent = DeviceBuffer::upload(DEVICE_ID, &vec![0_u8; TARGET_ROWS * LATENT_COLUMNS]).unwrap();
        let target_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&vec![0_u16; TARGET_ROWS * SCALE_COLUMNS])).unwrap();
        let target_rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&vec![0_u16; TARGET_ROWS * ROPE_COLUMNS])).unwrap();
        try_mla_hot_scatter_q8(DEVICE_ID, &source_latent, &source_scales, &source_rope, &slots_device, &target_latent, &target_scales, &target_rope, ROWS, LATENT_COLUMNS, SCALE_COLUMNS, ROPE_COLUMNS, TARGET_ROWS).unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP MLA hot scatter oracle").unwrap();

        let mut actual_latent = vec![0_u8; TARGET_ROWS * LATENT_COLUMNS];
        let mut actual_scales = vec![0_u16; TARGET_ROWS * SCALE_COLUMNS];
        let mut actual_rope = vec![0_u16; TARGET_ROWS * ROPE_COLUMNS];
        target_latent.copy_to_host(&mut actual_latent).unwrap();
        target_scales.copy_to_host(as_bytes_mut(&mut actual_scales)).unwrap();
        target_rope.copy_to_host(as_bytes_mut(&mut actual_rope)).unwrap();
        for (source, &slot) in slots.iter().enumerate() {
            let slot = slot as usize;
            assert_eq!(&actual_latent[slot * LATENT_COLUMNS..(slot + 1) * LATENT_COLUMNS], &latent[source * LATENT_COLUMNS..(source + 1) * LATENT_COLUMNS]);
            assert_eq!(&actual_scales[slot * SCALE_COLUMNS..(slot + 1) * SCALE_COLUMNS], &scales[source * SCALE_COLUMNS..(source + 1) * SCALE_COLUMNS]);
            assert_eq!(&actual_rope[slot * ROPE_COLUMNS..(slot + 1) * ROPE_COLUMNS], &rope[source * ROPE_COLUMNS..(source + 1) * ROPE_COLUMNS]);
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_hot_gather_preserves_token_to_slot_routing() {
        const DEVICE_ID: i32 = 0;
        const ROWS: usize = 3;
        const TARGET_ROWS: usize = 6;
        const LATENT_COLUMNS: usize = 7;
        const SCALE_COLUMNS: usize = 2;
        const ROPE_COLUMNS: usize = 3;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let source_rows = 8;
        let latent = (0..source_rows * LATENT_COLUMNS).map(|value| value as u8 + 11).collect::<Vec<_>>();
        let scales = (0..source_rows * SCALE_COLUMNS).map(|value| value as u16 + 51).collect::<Vec<_>>();
        let rope = (0..source_rows * ROPE_COLUMNS).map(|value| value as u16 + 71).collect::<Vec<_>>();
        let tokens = [2_u32, 5, 0];
        let slots = [4_u32, 1, 5];
        let source_latent = DeviceBuffer::upload(DEVICE_ID, &latent).unwrap();
        let source_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scales)).unwrap();
        let source_rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
        let tokens_device = DeviceBuffer::upload(DEVICE_ID, as_bytes(&tokens)).unwrap();
        let slots_device = DeviceBuffer::upload(DEVICE_ID, as_bytes(&slots)).unwrap();
        let target_latent = DeviceBuffer::upload(DEVICE_ID, &vec![0_u8; TARGET_ROWS * LATENT_COLUMNS]).unwrap();
        let target_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&vec![0_u16; TARGET_ROWS * SCALE_COLUMNS])).unwrap();
        let target_rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&vec![0_u16; TARGET_ROWS * ROPE_COLUMNS])).unwrap();
        try_mla_hot_gather_q8(DEVICE_ID, &source_latent, Some(&source_scales), &source_rope, &tokens_device, &slots_device, &target_latent, &target_scales, &target_rope, ROWS, LATENT_COLUMNS, SCALE_COLUMNS, ROPE_COLUMNS, TARGET_ROWS)
            .unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP MLA hot gather oracle").unwrap();

        let mut actual_latent = vec![0_u8; TARGET_ROWS * LATENT_COLUMNS];
        let mut actual_scales = vec![0_u16; TARGET_ROWS * SCALE_COLUMNS];
        let mut actual_rope = vec![0_u16; TARGET_ROWS * ROPE_COLUMNS];
        target_latent.copy_to_host(&mut actual_latent).unwrap();
        target_scales.copy_to_host(as_bytes_mut(&mut actual_scales)).unwrap();
        target_rope.copy_to_host(as_bytes_mut(&mut actual_rope)).unwrap();
        for (index, (&token, &slot)) in tokens.iter().zip(slots.iter()).enumerate() {
            let token = token as usize;
            let slot = slot as usize;
            assert_eq!(&actual_latent[slot * LATENT_COLUMNS..(slot + 1) * LATENT_COLUMNS], &latent[token * LATENT_COLUMNS..(token + 1) * LATENT_COLUMNS], "latent row {index}");
            assert_eq!(&actual_scales[slot * SCALE_COLUMNS..(slot + 1) * SCALE_COLUMNS], &scales[token * SCALE_COLUMNS..(token + 1) * SCALE_COLUMNS], "scales row {index}");
            assert_eq!(&actual_rope[slot * ROPE_COLUMNS..(slot + 1) * ROPE_COLUMNS], &rope[token * ROPE_COLUMNS..(token + 1) * ROPE_COLUMNS], "rope row {index}");
        }
    }

    fn compare_compact_topk(context_rows: usize) -> Result<(), String> {
        const DEVICE_ID: i32 = 0;
        const QUERY_ROWS: usize = 4;
        const TOP_K: usize = 2048;
        let query_start = context_rows.checked_sub(QUERY_ROWS).ok_or("DSA oracle context 太短")?;
        let mut scores = vec![0_u32; QUERY_ROWS * context_rows];
        let mut histograms = vec![0_u32; QUERY_ROWS * 256];
        for row in 0..QUERY_ROWS {
            let visible = query_start + row + 1;
            for token in 0..context_rows {
                // 同一批覆盖散列、重复值、全相等和严格单调四类排序形状。
                let score = match row {
                    0 => {
                        let hash = (token as u32).wrapping_mul(0x9e37_79b9).rotate_left(13);
                        (hash % 1_000_003) as f32 * (1.0 / 4096.0) - 128.0
                    }
                    1 => (token % 8192) as f32 * 0.125 - 512.0,
                    2 => 1.0,
                    _ => token as f32 - context_rows as f32 * 0.5,
                };
                let key = ordered_score(score);
                scores[row * context_rows + token] = key;
                if token < visible {
                    histograms[row * 256 + (key >> 24) as usize] += 1;
                }
            }
        }

        let functions = paged_mla_functions(DEVICE_ID)?;
        let scores = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scores))?;
        let histograms = DeviceBuffer::upload(DEVICE_ID, as_bytes(&histograms))?;
        let bytes = QUERY_ROWS * TOP_K * std::mem::size_of::<u32>();
        let baseline = DeviceBuffer::allocate(DEVICE_ID, bytes)?;
        let compact = DeviceBuffer::allocate(DEVICE_ID, bytes)?;
        let mut d_scores = scores.pointer;
        let mut d_histograms = histograms.pointer;
        let mut d_baseline = baseline.pointer;
        let mut d_compact = compact.pointer;
        let mut rows = QUERY_ROWS as u32;
        let mut stride = context_rows as u32;
        let mut start = query_start as u32;
        let mut top_k = TOP_K as u32;
        let mut baseline_args =
            [(&mut d_scores as *mut *mut c_void).cast(), (&mut d_baseline as *mut *mut c_void).cast(), (&mut rows as *mut u32).cast(), (&mut stride as *mut u32).cast(), (&mut start as *mut u32).cast(), (&mut top_k as *mut u32).cast()];
        launch_tensor_kernel(functions.dsa_select, rows, 256, &mut baseline_args, "HIP DSA baseline oracle")?;
        let mut visibility_divisor = 1u32;
        let mut compact_args = [
            (&mut d_scores as *mut *mut c_void).cast(),
            (&mut d_histograms as *mut *mut c_void).cast(),
            (&mut d_compact as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut stride as *mut u32).cast(),
            (&mut start as *mut u32).cast(),
            (&mut top_k as *mut u32).cast(),
            (&mut visibility_divisor as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.dsa_select_compact, rows, 256, &mut compact_args, "HIP DSA compact oracle")?;
        super::super::synchronize_device(DEVICE_ID, "HIP DSA compact oracle")?;

        let mut baseline_host = vec![0_u32; QUERY_ROWS * TOP_K];
        let mut compact_host = vec![0_u32; QUERY_ROWS * TOP_K];
        baseline.copy_to_host(unsafe { std::slice::from_raw_parts_mut(baseline_host.as_mut_ptr().cast(), bytes) })?;
        compact.copy_to_host(unsafe { std::slice::from_raw_parts_mut(compact_host.as_mut_ptr().cast(), bytes) })?;
        for row in 0..QUERY_ROWS {
            let target = TOP_K.min(query_start + row + 1);
            let range = row * TOP_K..row * TOP_K + target;
            if baseline_host[range.clone()] != compact_host[range.clone()] {
                let mismatch = baseline_host[range.clone()].iter().zip(&compact_host[range.clone()]).position(|(baseline, compact)| baseline != compact).unwrap();
                return Err(format!("DSA compact oracle 不一致: context={context_rows} row={row} rank={mismatch} baseline={} compact={}", baseline_host[range.start + mismatch], compact_host[range.start + mismatch]));
            }
        }
        Ok(())
    }

    pub(crate) fn compare_score_pipeline(context_rows: usize, query_rows: usize, equal_scores: bool) -> Result<(), String> {
        const DEVICE_ID: i32 = 0;
        const HEAD_COUNT: usize = 64;
        const HEAD_DIM: usize = 128;
        const KEY_GROUP_SIZE: usize = 128;
        const BLOCK_SIZE: usize = 1024;
        const TOP_K: usize = 2048;
        let query_start = context_rows.checked_sub(query_rows).ok_or("DSA score oracle context 太短")?;
        let selection_width = TOP_K.min(context_rows.saturating_sub(1));
        let keys = (0..context_rows * HEAD_DIM).map(|index| ((index.wrapping_mul(17).wrapping_add(index / HEAD_DIM * 13)) % 255) as u8).collect::<Vec<_>>();
        let scales = vec![0x3f80_u16; context_rows * (HEAD_DIM / KEY_GROUP_SIZE)];
        let block_table = (0..context_rows.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
        let query = (0..query_rows * HEAD_COUNT * HEAD_DIM).map(|index| if equal_scores { 0.0 } else { ((index.wrapping_mul(29) % 257) as f32 - 128.0) * (1.0 / 127.0) }).collect::<Vec<_>>();
        let head_weights = (0..query_rows * HEAD_COUNT).map(|index| ((index % HEAD_COUNT) + 1) as f32 * (1.0 / HEAD_COUNT as f32)).collect::<Vec<_>>();
        let keys = DeviceBuffer::upload(DEVICE_ID, &keys)?;
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scales))?;
        let block_table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&block_table))?;
        let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query))?;
        let head_weights = DeviceBuffer::upload(DEVICE_ID, as_bytes(&head_weights))?;
        let compact = try_dsa_select_paged_q8(DEVICE_ID, &keys, &scales, KEY_GROUP_SIZE, false, &block_table, &query, &head_weights, query_rows, context_rows, query_start, HEAD_COUNT, HEAD_DIM, selection_width, false, BLOCK_SIZE)?;

        let functions = paged_mla_functions(DEVICE_ID)?;
        let use_native_wmma = functions.dense_wmma && options().native_dsa_wmma;
        let tile_rows = if use_native_wmma && query_rows == 1 {
            128
        } else if use_native_wmma {
            256
        } else {
            128
        };
        let score_stride = context_rows.div_ceil(tile_rows) * tile_rows;
        let key = crate::kernel::rocm::hip::compute_workspace_key(DEVICE_ID);
        let scores = PAGED_DSA_WORKSPACES.with(|workspaces| workspaces.borrow().get(&key).and_then(|workspace| workspace.scores.clone()).ok_or("DSA score oracle 缺少 scratch".to_owned()))?;
        if use_native_wmma && query_rows == 1 {
            let mut specialized_scores = vec![0_u32; score_stride];
            scores.copy_to_host(as_bytes_mut(&mut specialized_scores))?;

            const LEGACY_TILE_ROWS: usize = 256;
            let legacy_stride = context_rows.div_ceil(LEGACY_TILE_ROWS) * LEGACY_TILE_ROWS;
            let legacy_scores = DeviceBuffer::allocate(DEVICE_ID, legacy_stride * 4)?;
            let legacy_histograms = DeviceBuffer::allocate(DEVICE_ID, 256 * 4)?;
            let mut d_histograms = legacy_histograms.pointer;
            let mut histogram_elements = 256u32;
            let mut clear_args = [(&mut d_histograms as *mut *mut c_void).cast(), (&mut histogram_elements as *mut u32).cast()];
            launch_tensor_kernel(functions.dsa_clear, 1, 256, &mut clear_args, "HIP DSA decode score legacy histogram oracle")?;

            let mut d_keys = keys.pointer;
            let mut d_scales = scales.pointer;
            let mut d_table = block_table.pointer;
            let mut d_query = query.pointer;
            let mut d_weights = head_weights.pointer;
            let mut d_legacy_scores = legacy_scores.pointer;
            let mut rows = 1u32;
            let mut context = context_rows as u32;
            let mut start = query_start as u32;
            let mut heads = HEAD_COUNT as u32;
            let mut dim = HEAD_DIM as u32;
            let mut group = KEY_GROUP_SIZE as u32;
            let mut block = BLOCK_SIZE as u32;
            let mut legacy_tile = LEGACY_TILE_ROWS as u32;
            let mut legacy_args = [
                (&mut d_keys as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_weights as *mut *mut c_void).cast(),
                (&mut d_legacy_scores as *mut *mut c_void).cast(),
                (&mut d_histograms as *mut *mut c_void).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut context as *mut u32).cast(),
                (&mut start as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut dim as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut block as *mut u32).cast(),
                (&mut legacy_tile as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_score_native_wmma, context.div_ceil(legacy_tile), 1, 512, 4 * 256 * 4, &mut legacy_args, "HIP DSA decode score legacy bitwise oracle")?;
            super::super::synchronize_device(DEVICE_ID, "HIP DSA decode score bitwise oracle")?;
            let mut legacy_score_host = vec![0_u32; legacy_stride];
            legacy_scores.copy_to_host(as_bytes_mut(&mut legacy_score_host))?;
            if specialized_scores[..context_rows] != legacy_score_host[..context_rows] {
                let mismatch = specialized_scores[..context_rows].iter().zip(&legacy_score_host[..context_rows]).position(|(specialized, legacy)| specialized != legacy).unwrap();
                return Err(format!("DSA decode score bitwise oracle 不一致: context={context_rows} token={mismatch} specialized={} legacy={}", specialized_scores[mismatch], legacy_score_host[mismatch]));
            }
        } else if use_native_wmma && query_rows == 2 {
            let mut specialized_scores = vec![0_u32; query_rows * score_stride];
            scores.copy_to_host(as_bytes_mut(&mut specialized_scores))?;
            let legacy_scores = DeviceBuffer::allocate(DEVICE_ID, query_rows * score_stride * 4)?;
            let legacy_histograms = DeviceBuffer::allocate(DEVICE_ID, query_rows * 256 * 4)?;
            let mut d_histograms = legacy_histograms.pointer;
            let mut histogram_elements = (query_rows * 256) as u32;
            let mut clear_args = [(&mut d_histograms as *mut *mut c_void).cast(), (&mut histogram_elements as *mut u32).cast()];
            launch_tensor_kernel(functions.dsa_clear, histogram_elements.div_ceil(256), 256, &mut clear_args, "HIP DSA rows2 legacy histogram oracle")?;

            let mut d_keys = keys.pointer;
            let mut d_scales = scales.pointer;
            let mut d_table = block_table.pointer;
            let mut d_query = query.pointer;
            let mut d_weights = head_weights.pointer;
            let mut d_legacy_scores = legacy_scores.pointer;
            let mut rows = query_rows as u32;
            let mut context = context_rows as u32;
            let mut start = query_start as u32;
            let mut heads = HEAD_COUNT as u32;
            let mut dim = HEAD_DIM as u32;
            let mut group = KEY_GROUP_SIZE as u32;
            let mut block = BLOCK_SIZE as u32;
            let mut tile = tile_rows as u32;
            let mut stride = score_stride as u32;
            let mut parity = 2_u32;
            let mut legacy_args = [
                (&mut d_keys as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_weights as *mut *mut c_void).cast(),
                (&mut d_legacy_scores as *mut *mut c_void).cast(),
                (&mut d_histograms as *mut *mut c_void).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut context as *mut u32).cast(),
                (&mut start as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut dim as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut block as *mut u32).cast(),
                (&mut tile as *mut u32).cast(),
                (&mut stride as *mut u32).cast(),
                (&mut parity as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_score_native_wmma, context.div_ceil(tile), rows.div_ceil(4), 512, 4 * 256 * 4, &mut legacy_args, "HIP DSA rows2 score legacy bitwise oracle")?;
            super::super::synchronize_device(DEVICE_ID, "HIP DSA rows2 score bitwise oracle")?;
            let mut legacy_score_host = vec![0_u32; query_rows * score_stride];
            legacy_scores.copy_to_host(as_bytes_mut(&mut legacy_score_host))?;
            for row in 0..query_rows {
                let visible = query_start + row + 1;
                let start = row * score_stride;
                let specialized = &specialized_scores[start..start + visible];
                let legacy = &legacy_score_host[start..start + visible];
                if specialized != legacy {
                    let token = specialized.iter().zip(legacy).position(|(specialized, legacy)| specialized != legacy).unwrap();
                    return Err(format!("DSA rows2 score bitwise oracle 不一致: context={context_rows} row={row} token={token} specialized={} legacy={}", specialized[token], legacy[token]));
                }
            }
        }
        if use_native_wmma && query_rows >= 8 && context_rows >= 128 * 1024 {
            let mut d_keys = keys.pointer;
            let mut d_scales = scales.pointer;
            let mut d_table = block_table.pointer;
            let mut d_query = query.pointer;
            let mut d_weights = head_weights.pointer;
            let mut d_scores = scores.pointer;
            let mut d_histograms = ptr::null_mut();
            let mut rows = query_rows as u32;
            let mut context = context_rows as u32;
            let mut start = query_start as u32;
            let mut heads = HEAD_COUNT as u32;
            let mut dim = HEAD_DIM as u32;
            let mut group = KEY_GROUP_SIZE as u32;
            let mut block = BLOCK_SIZE as u32;
            let mut tile = tile_rows as u32;
            let mut stride = score_stride as u32;
            let mut parity = 2_u32;
            // kernel 签名尾部是 score_stride + shard_parity（后加），此前少传
            // 两项导致错位读到栈垃圾——shard_parity<2 时 block_table 折半寻址
            // 越界（131K×8 的 illegal access 根因，§3.4 参数对齐坑重演）。
            let mut score_args = [
                (&mut d_keys as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_weights as *mut *mut c_void).cast(),
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_histograms as *mut *mut c_void).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut context as *mut u32).cast(),
                (&mut start as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut dim as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut block as *mut u32).cast(),
                (&mut tile as *mut u32).cast(),
                (&mut stride as *mut u32).cast(),
                (&mut parity as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_score_native_wmma, context.div_ceil(tile), rows.div_ceil(4), 512, 4 * 256 * 4, &mut score_args, "HIP DSA full score oracle")?;
        }
        let bytes = query_rows * selection_width * std::mem::size_of::<u32>();
        let baseline = DeviceBuffer::allocate(DEVICE_ID, bytes)?;
        let mut d_scores = scores.pointer;
        let mut d_baseline = baseline.pointer;
        let mut rows = query_rows as u32;
        let mut stride = score_stride as u32;
        let mut start = query_start as u32;
        let mut top_k = selection_width as u32;
        let mut args =
            [(&mut d_scores as *mut *mut c_void).cast(), (&mut d_baseline as *mut *mut c_void).cast(), (&mut rows as *mut u32).cast(), (&mut stride as *mut u32).cast(), (&mut start as *mut u32).cast(), (&mut top_k as *mut u32).cast()];
        launch_tensor_kernel(functions.dsa_select, rows, 256, &mut args, "HIP DSA score pipeline baseline oracle")?;
        super::super::synchronize_device(DEVICE_ID, "HIP DSA score pipeline oracle")?;
        let mut baseline_host = vec![0_u32; query_rows * selection_width];
        let mut compact_host = vec![0_u32; query_rows * selection_width];
        baseline.copy_to_host(unsafe { std::slice::from_raw_parts_mut(baseline_host.as_mut_ptr().cast(), bytes) })?;
        compact.copy_to_host(unsafe { std::slice::from_raw_parts_mut(compact_host.as_mut_ptr().cast(), bytes) })?;
        for row in 0..query_rows {
            let target = selection_width.min(query_start + row + 1);
            let range = row * selection_width..row * selection_width + target;
            if baseline_host[range.clone()] != compact_host[range.clone()] {
                return Err(format!("DSA score pipeline oracle 不一致: context={context_rows} row={row}"));
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn dsa_mean_pool_q8_gpu_matches_constant_block_oracle() {
        const DEVICE_ID: i32 = 0;
        const COLUMNS: usize = 128;
        const GROUP_SIZE: usize = 128;
        const POOL_SIZE: usize = 128;
        const BLOCK_SIZE: usize = 128;
        const CONTEXT_ROWS: usize = POOL_SIZE * 2 + 7;
        let expected = [2.0_f32, -4.0, 6.0];
        let keys = (0..CONTEXT_ROWS)
            .flat_map(|row| {
                let code = if row < POOL_SIZE {
                    2_i8
                } else if row < POOL_SIZE * 2 {
                    -4_i8
                } else {
                    6_i8
                };
                std::iter::repeat_n(code, COLUMNS)
            })
            .collect::<Vec<_>>();
        let scales = vec![0x3f80_u16; CONTEXT_ROWS];
        let table = (0..CONTEXT_ROWS.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
        let pool_rows = CONTEXT_ROWS.div_ceil(POOL_SIZE);
        let pool_table = vec![0_u32];
        let keys = DeviceBuffer::upload(DEVICE_ID, as_bytes(&keys)).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scales)).unwrap();
        let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
        let pool_table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&pool_table)).unwrap();
        let pooled_keys = DeviceBuffer::allocate(DEVICE_ID, pool_rows * COLUMNS).unwrap();
        let pooled_scales = DeviceBuffer::allocate(DEVICE_ID, pool_rows * 2).unwrap();

        try_dsa_mean_pool_q8(DEVICE_ID, &keys, &scales, &table, &pooled_keys, &pooled_scales, &pool_table, 0, pool_rows, CONTEXT_ROWS, COLUMNS, GROUP_SIZE, POOL_SIZE, BLOCK_SIZE).unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP DSA HISA mean pool oracle").unwrap();

        let mut codes = vec![0_i8; pool_rows * COLUMNS];
        let mut scale_bits = vec![0_u16; pool_rows];
        pooled_keys.copy_to_host(as_bytes_mut(&mut codes)).unwrap();
        pooled_scales.copy_to_host(as_bytes_mut(&mut scale_bits)).unwrap();
        for pool in 0..pool_rows {
            let scale = half::bf16::from_bits(scale_bits[pool]).to_f32();
            for column in 0..COLUMNS {
                let actual = codes[pool * COLUMNS + column] as f32 * scale;
                assert!((actual - expected[pool]).abs() < 0.03, "pool={pool} column={column} actual={actual} expected={}", expected[pool]);
            }
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn dsa_interval_bounds_gpu_cover_constant_block_oracle() {
        const DEVICE_ID: i32 = 0;
        const HEAD_DIM: usize = 128;
        const HEAD_COUNT: usize = 2;
        const GROUP_SIZE: usize = 128;
        const POOL_SIZE: usize = 128;
        const BLOCK_SIZE: usize = 128;
        const CONTEXT_ROWS: usize = POOL_SIZE * 2 + 7;
        let values = [2_i8, -4, 6];
        let expected_scores = [256.0_f32, -256.0, 768.0];
        let keys = (0..CONTEXT_ROWS)
            .flat_map(|row| {
                let pool = (row / POOL_SIZE).min(values.len() - 1);
                std::iter::repeat_n(values[pool], HEAD_DIM)
            })
            .collect::<Vec<_>>();
        let scales = vec![0x3f80_u16; CONTEXT_ROWS];
        let table = (0..CONTEXT_ROWS.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
        let pool_rows = CONTEXT_ROWS.div_ceil(POOL_SIZE);
        let query = [vec![1.0_f32; HEAD_DIM], vec![-1.0_f32; HEAD_DIM]].concat();
        let weights = [1.0_f32, -0.5];
        let keys = DeviceBuffer::upload(DEVICE_ID, as_bytes(&keys)).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scales)).unwrap();
        let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
        let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query)).unwrap();
        let weights = DeviceBuffer::upload(DEVICE_ID, as_bytes(&weights)).unwrap();
        let lower = DeviceBuffer::allocate(DEVICE_ID, pool_rows * HEAD_DIM * 4).unwrap();
        let upper = DeviceBuffer::allocate(DEVICE_ID, pool_rows * HEAD_DIM * 4).unwrap();
        let bounds = DeviceBuffer::allocate(DEVICE_ID, pool_rows * 4).unwrap();

        try_dsa_interval_pool_q8(DEVICE_ID, &keys, &scales, &table, &lower, &upper, 0, pool_rows, CONTEXT_ROWS, HEAD_DIM, GROUP_SIZE, POOL_SIZE, BLOCK_SIZE).unwrap();
        try_dsa_interval_score_bounds(DEVICE_ID, &lower, &upper, &query, &weights, &bounds, pool_rows, HEAD_COUNT, HEAD_DIM).unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP DSA interval oracle").unwrap();

        let mut lower_host = vec![0.0_f32; pool_rows * HEAD_DIM];
        let mut upper_host = vec![0.0_f32; pool_rows * HEAD_DIM];
        let mut bound_host = vec![0.0_f32; pool_rows];
        lower.copy_to_host(as_bytes_mut(&mut lower_host)).unwrap();
        upper.copy_to_host(as_bytes_mut(&mut upper_host)).unwrap();
        bounds.copy_to_host(as_bytes_mut(&mut bound_host)).unwrap();
        for pool in 0..pool_rows {
            for column in 0..HEAD_DIM {
                assert_eq!(lower_host[pool * HEAD_DIM + column], values[pool] as f32);
                assert_eq!(upper_host[pool * HEAD_DIM + column], values[pool] as f32);
            }
            assert!(bound_host[pool] >= expected_scores[pool], "pool={pool} bound={} expected={}", bound_host[pool], expected_scores[pool]);
            assert!(bound_host[pool] - expected_scores[pool] < 0.2, "pool={pool} bound={} expected={}", bound_host[pool], expected_scores[pool]);
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn dsa_compact_topk_matches_radix_at_long_boundaries() {
        for context_rows in [1023, 1024, 1025, 32_774, 256_006, 400_006] {
            compare_compact_topk(context_rows).unwrap();
            compare_score_pipeline(context_rows, 4, false).unwrap();
        }
        compare_score_pipeline(131_073, 4, true).unwrap();
        compare_score_pipeline(131_078, 8, false).unwrap();
        compare_score_pipeline(131_073, 8, true).unwrap();
        compare_score_pipeline(1025, 3, false).unwrap();
        compare_score_pipeline(32_774, 1, false).unwrap();
    }

    #[test]
    #[ignore = "需要 ROCm gfx11+ GPU"]
    fn dsa_rows2_native_wmma_matches_legacy_bits() {
        compare_score_pipeline(46_160, 2, false).unwrap();
    }

    #[test]
    #[ignore = "需要 ROCm gfx11+ GPU"]
    fn dsa_compact_sequence_shards_match_full_topk_bits() {
        const DEVICE_ID: i32 = 0;
        const CONTEXT_ROWS: usize = 50_013;
        const HEAD_COUNT: usize = 64;
        const HEAD_DIM: usize = 128;
        const KEY_GROUP_SIZE: usize = 128;
        const TOP_K: usize = 2048;
        const BLOCK_SIZE: usize = 64;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let keys = (0..CONTEXT_ROWS * HEAD_DIM).map(|index| ((index.wrapping_mul(17).wrapping_add(index / HEAD_DIM * 13)) % 255) as u8).collect::<Vec<_>>();
        let scales = vec![0x3f80_u16; CONTEXT_ROWS];
        let query = (0..HEAD_COUNT * HEAD_DIM).map(|index| ((index.wrapping_mul(29) % 257) as f32 - 128.0) * (1.0 / 127.0)).collect::<Vec<_>>();
        let weights = (0..HEAD_COUNT).map(|head| (head + 1) as f32 / HEAD_COUNT as f32).collect::<Vec<_>>();
        let full_table = (0..CONTEXT_ROWS.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
        let full_keys = DeviceBuffer::upload(DEVICE_ID, &keys).unwrap();
        let full_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scales)).unwrap();
        let full_table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&full_table)).unwrap();
        let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query)).unwrap();
        let weights = DeviceBuffer::upload(DEVICE_ID, as_bytes(&weights)).unwrap();
        let full = try_dsa_select_paged_q8(DEVICE_ID, &full_keys, &full_scales, KEY_GROUP_SIZE, false, &full_table, &query, &weights, 1, CONTEXT_ROWS, CONTEXT_ROWS - 1, HEAD_COUNT, HEAD_DIM, TOP_K, false, BLOCK_SIZE).unwrap();
        let score_stride = CONTEXT_ROWS.div_ceil(128) * 128;
        let workspace_key = crate::kernel::rocm::hip::compute_workspace_key(DEVICE_ID);
        let full_scores = PAGED_DSA_WORKSPACES.with(|workspaces| workspaces.borrow().get(&workspace_key).unwrap().scores.as_ref().unwrap().clone());
        let mut full_score_host = vec![0_u32; score_stride];
        full_scores.copy_to_host(as_bytes_mut(&mut full_score_host)).unwrap();

        let mut shards = Vec::with_capacity(2);
        for parity in 0..2 {
            let tokens = (0..CONTEXT_ROWS).filter(|token| (token / BLOCK_SIZE) % 2 == parity).collect::<Vec<_>>();
            let shard_keys = tokens.iter().flat_map(|&token| keys[token * HEAD_DIM..(token + 1) * HEAD_DIM].iter().copied()).collect::<Vec<_>>();
            let shard_scales = tokens.iter().map(|&token| scales[token]).collect::<Vec<_>>();
            let shard_table = (0..tokens.len().div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
            let shard_keys = DeviceBuffer::upload(DEVICE_ID, &shard_keys).unwrap();
            let shard_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&shard_scales)).unwrap();
            let shard_table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&shard_table)).unwrap();
            let shard =
                try_dsa_select_paged_q8_sequence_shard(DEVICE_ID, &shard_keys, &shard_scales, KEY_GROUP_SIZE, &shard_table, &query, &weights, 1, CONTEXT_ROWS, CONTEXT_ROWS - 1, HEAD_COUNT, HEAD_DIM, TOP_K, BLOCK_SIZE, parity).unwrap();
            let mut shard_tokens = vec![0_u32; TOP_K];
            let mut shard_scores = vec![0_u32; TOP_K];
            shard.selection.copy_to_host(as_bytes_mut(&mut shard_tokens)).unwrap();
            shard.scores.copy_to_host(as_bytes_mut(&mut shard_scores)).unwrap();
            for rank in 0..TOP_K {
                let token = shard_tokens[rank] as usize;
                assert_eq!((token / BLOCK_SIZE) % 2, parity, "parity={parity} rank={rank} token={token}");
                assert_eq!(shard_scores[rank], full_score_host[token], "parity={parity} rank={rank} token={token}");
            }
            shards.push(shard);
        }
        let merged = try_dsa_merge_sequence_shard_topk(DEVICE_ID, &shards[0], &shards[1], 1, TOP_K).unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP DSA compact sequence shard oracle").unwrap();
        let mut full_host = vec![0_u32; TOP_K];
        let mut merged_host = vec![0_u32; TOP_K];
        full.copy_to_host(as_bytes_mut(&mut full_host)).unwrap();
        merged.copy_to_host(as_bytes_mut(&mut merged_host)).unwrap();
        if merged_host != full_host {
            let rank = merged_host.iter().zip(&full_host).position(|(merged, full)| merged != full).unwrap();
            panic!("DSA sequence shard merge 不一致: rank={rank} merged={} full={}", merged_host[rank], full_host[rank]);
        }
    }

    #[test]
    #[ignore = "需要 ROCm gfx11+ GPU"]
    fn dsa_hadamard_shadow_i8_gpu_matches_integer_oracle() {
        const DEVICE_ID: i32 = 0;
        const CONTEXT_ROWS: usize = 2305;
        const QUERY_ROWS: usize = 4;
        const QUERY_START: usize = CONTEXT_ROWS - QUERY_ROWS;
        const HEAD_COUNT: usize = 64;
        const HEAD_DIM: usize = 128;
        const TOP_K: usize = 256;
        const BLOCK_SIZE: usize = 1024;
        let keys = (0..CONTEXT_ROWS * HEAD_DIM)
            .map(|index| {
                let row = index / HEAD_DIM;
                let column = index % HEAD_DIM;
                if column == 0 { 24.0 + (row as f32 * 0.017).sin() * 0.01 } else { ((index * 17 + row * 13) as f32 * 0.031).sin() * 0.25 }
            })
            .collect::<Vec<_>>();
        let query = (0..QUERY_ROWS * HEAD_COUNT * HEAD_DIM).map(|index| if index % HEAD_DIM == 0 { 24.0 } else { ((index * 29 + 7) as f32 * 0.019).cos() * 0.25 }).collect::<Vec<_>>();
        let head_weights = (0..QUERY_ROWS * HEAD_COUNT).map(|index| ((index % HEAD_COUNT) + 1) as f32 / HEAD_COUNT as f32).collect::<Vec<_>>();
        let table = (0..CONTEXT_ROWS.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
        let key_input = DeviceBuffer::upload(DEVICE_ID, as_bytes(&keys)).unwrap();
        let raw_key_cache = DeviceBuffer::allocate(DEVICE_ID, CONTEXT_ROWS * HEAD_DIM).unwrap();
        let raw_key_scales = DeviceBuffer::allocate(DEVICE_ID, CONTEXT_ROWS * 2).unwrap();
        let key_cache = DeviceBuffer::allocate(DEVICE_ID, CONTEXT_ROWS * HEAD_DIM).unwrap();
        let key_scales = DeviceBuffer::allocate(DEVICE_ID, CONTEXT_ROWS * 2).unwrap();
        let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
        try_paged_cache_append_f32_q8(DEVICE_ID, &key_input, &raw_key_cache, &raw_key_scales, &table, 0, CONTEXT_ROWS, HEAD_DIM, HEAD_DIM, BLOCK_SIZE).unwrap();
        try_paged_cache_transform_q8_hadamard(DEVICE_ID, &raw_key_cache, &raw_key_scales, &key_cache, &key_scales, &table, 0, CONTEXT_ROWS, HEAD_DIM, BLOCK_SIZE).unwrap();
        let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query)).unwrap();
        let weights = DeviceBuffer::upload(DEVICE_ID, as_bytes(&head_weights)).unwrap();
        let raw_selection = try_dsa_select_paged_q8(DEVICE_ID, &raw_key_cache, &raw_key_scales, HEAD_DIM, false, &table, &query, &weights, QUERY_ROWS, CONTEXT_ROWS, QUERY_START, HEAD_COUNT, HEAD_DIM, TOP_K, false, BLOCK_SIZE).unwrap();
        let mut raw_selected = vec![0_u32; QUERY_ROWS * TOP_K];
        raw_selection.copy_to_host(as_bytes_mut(&mut raw_selected)).unwrap();
        // rerank oracle 当前只覆盖单 query row；用 raw exact 成员加稳定 token 补位
        // 构造候选，验证 candidate-local score/select/map 与全量 raw 完全一致。
        const CANDIDATES: usize = TOP_K + 128;
        let mut candidate_tokens = raw_selected[..TOP_K].to_vec();
        let mut present = vec![false; CONTEXT_ROWS];
        for &token in &candidate_tokens {
            present[token as usize] = true;
        }
        for token in 0..CONTEXT_ROWS {
            if candidate_tokens.len() == CANDIDATES {
                break;
            }
            if !present[token] {
                candidate_tokens.push(token as u32);
            }
        }
        candidate_tokens.sort_unstable();
        let candidates = DeviceBuffer::upload(DEVICE_ID, as_bytes(&candidate_tokens)).unwrap();
        let reranked =
            try_dsa_rerank_paged_q8_candidates(DEVICE_ID, &raw_key_cache, &raw_key_scales, HEAD_DIM, &table, &query, &weights, &candidates, 1, CONTEXT_ROWS, CONTEXT_ROWS - 1, HEAD_COUNT, HEAD_DIM, CANDIDATES, TOP_K, BLOCK_SIZE).unwrap();
        let mut reranked_selected = vec![0_u32; TOP_K];
        reranked.copy_to_host(as_bytes_mut(&mut reranked_selected)).unwrap();
        assert_eq!(reranked_selected, raw_selected[..TOP_K], "raw candidate rerank 与全量 exact selection 不一致");
        let selection = try_dsa_select_paged_q8(DEVICE_ID, &key_cache, &key_scales, HEAD_DIM, true, &table, &query, &weights, QUERY_ROWS, CONTEXT_ROWS, QUERY_START, HEAD_COUNT, HEAD_DIM, TOP_K, false, BLOCK_SIZE).unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP DSA Hadamard i8 oracle").unwrap();

        let mut key_codes = vec![0_i8; CONTEXT_ROWS * HEAD_DIM];
        let mut key_scale_bits = vec![0_u16; CONTEXT_ROWS];
        key_cache.copy_to_host(as_bytes_mut(&mut key_codes)).unwrap();
        key_scales.copy_to_host(as_bytes_mut(&mut key_scale_bits)).unwrap();
        let (query_codes, query_scales) = PAGED_DSA_WORKSPACES.with(|workspaces| {
            let workspaces = workspaces.borrow();
            let workspace = workspaces.get(&crate::kernel::rocm::hip::compute_workspace_key(DEVICE_ID)).unwrap();
            (workspace.quantized_query.clone().unwrap(), workspace.query_scales.clone().unwrap())
        });
        let mut query_code_host = vec![0_i8; QUERY_ROWS * HEAD_COUNT * HEAD_DIM];
        let mut query_scale_host = vec![0_f32; QUERY_ROWS * HEAD_COUNT];
        query_codes.copy_to_host(as_bytes_mut(&mut query_code_host)).unwrap();
        query_scales.copy_to_host(as_bytes_mut(&mut query_scale_host)).unwrap();
        let mut actual = vec![0_u32; QUERY_ROWS * TOP_K];
        selection.copy_to_host(as_bytes_mut(&mut actual)).unwrap();

        // compact select 的稳定顺序是“高于阈值的 token 顺序，再接阈值 token”，
        // 不是整个集合按 token 排序。用同一份 GPU score 跑旧 radix oracle 固定顺序，
        // CPU i32 oracle 只负责验证最终 Top-K 集合。
        let score_stride = CONTEXT_ROWS.div_ceil(256) * 256;
        let scores = PAGED_DSA_WORKSPACES.with(|workspaces| workspaces.borrow().get(&crate::kernel::rocm::hip::compute_workspace_key(DEVICE_ID)).unwrap().scores.clone().unwrap());
        let stable_selection = DeviceBuffer::allocate(DEVICE_ID, QUERY_ROWS * TOP_K * 4).unwrap();
        let functions = paged_mla_functions(DEVICE_ID).unwrap();
        let mut d_scores = scores.pointer;
        let mut d_stable = stable_selection.pointer;
        let mut rows = QUERY_ROWS as u32;
        let mut stride = score_stride as u32;
        let mut start = QUERY_START as u32;
        let mut top_k = TOP_K as u32;
        let mut stable_args =
            [(&mut d_scores as *mut *mut c_void).cast(), (&mut d_stable as *mut *mut c_void).cast(), (&mut rows as *mut u32).cast(), (&mut stride as *mut u32).cast(), (&mut start as *mut u32).cast(), (&mut top_k as *mut u32).cast()];
        launch_tensor_kernel(functions.dsa_select, rows, 256, &mut stable_args, "HIP DSA Hadamard i8 stable radix oracle").unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP DSA Hadamard i8 stable radix oracle").unwrap();
        let mut stable = vec![0_u32; QUERY_ROWS * TOP_K];
        stable_selection.copy_to_host(as_bytes_mut(&mut stable)).unwrap();
        assert_eq!(actual, stable, "Hadamard i8 compact selection 顺序与稳定 radix 不一致");
        let mut gpu_score_keys = vec![0_u32; QUERY_ROWS * score_stride];
        scores.copy_to_host(as_bytes_mut(&mut gpu_score_keys)).unwrap();

        for row in 0..QUERY_ROWS {
            let visible = QUERY_START + row + 1;
            let mut scored = (0..visible)
                .map(|token| {
                    let key_scale = half::bf16::from_bits(key_scale_bits[token]).to_f32();
                    let mut score = 0.0_f32;
                    for head in 0..HEAD_COUNT {
                        let query_base = (row * HEAD_COUNT + head) * HEAD_DIM;
                        let key_base = token * HEAD_DIM;
                        let dot = (0..HEAD_DIM).map(|column| query_code_host[query_base + column] as i32 * key_codes[key_base + column] as i32).sum::<i32>();
                        let dot = dot as f32 * query_scale_host[row * HEAD_COUNT + head] * key_scale;
                        score += head_weights[row * HEAD_COUNT + head] * dot.max(0.0);
                    }
                    (token, score)
                })
                .collect::<Vec<_>>();
            let mut max_absolute_error = 0.0_f32;
            let mut max_relative_error = 0.0_f32;
            for &(token, cpu_score) in &scored {
                let ordered = gpu_score_keys[row * score_stride + token];
                let bits = if ordered & 0x8000_0000 != 0 { ordered ^ 0x8000_0000 } else { ordered ^ 0xffff_ffff };
                let gpu_score = f32::from_bits(bits);
                let absolute = (gpu_score - cpu_score).abs();
                let relative = absolute / gpu_score.abs().max(cpu_score.abs()).max(1.0);
                max_absolute_error = max_absolute_error.max(absolute);
                max_relative_error = max_relative_error.max(relative);
            }
            assert!(max_relative_error < 2e-4, "Hadamard i8 score 与 CPU i32 oracle 偏差过大: row={row} max_abs={max_absolute_error} max_rel={max_relative_error}");
            scored.sort_unstable_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
            let threshold = scored[TOP_K - 1].1;
            let next = scored[TOP_K].1;
            let threshold_margin = (threshold - next).abs();
            let threshold_relative_margin = threshold_margin / threshold.abs().max(next.abs()).max(1.0);
            let mut expected = scored.iter().take(TOP_K).map(|(token, _)| *token as u32).collect::<Vec<_>>();
            expected.sort_unstable();
            let mut selected = actual[row * TOP_K..(row + 1) * TOP_K].to_vec();
            selected.sort_unstable();
            let overlap = selected.iter().filter(|token| expected.binary_search(token).is_ok()).count();
            eprintln!(
                "Hadamard i8 oracle row={row} score_max_abs={max_absolute_error:.6} score_max_rel={max_relative_error:.8} threshold_margin={threshold_margin:.8} threshold_rel={threshold_relative_margin:.8} topk_overlap={overlap}/{TOP_K}"
            );
            // CPU 标量归约与 GPU WMMA 的加法顺序不同；边界严格同分时可有多个
            // token 换位。上面已逐项限制 score 误差，并验证 compact 与 GPU 稳定
            // radix 完全一致；这里只拒绝越过可解释边界的集合变化。
            assert!(
                overlap == TOP_K || threshold_relative_margin < 2e-4,
                "Hadamard i8 Top-K 边界翻转超过 score 误差范围: row={row} overlap={overlap}/{TOP_K} threshold_rel={threshold_relative_margin} selected={selected:?} expected={expected:?}"
            );
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn dsa_kpool_gpu_matches_causal_pool_expansion() {
        const DEVICE_ID: i32 = 0;
        const CONTEXT_ROWS: usize = 40;
        const QUERY_ROWS: usize = 8;
        const QUERY_START: usize = CONTEXT_ROWS - QUERY_ROWS;
        const HEAD_COUNT: usize = 32;
        const HEAD_DIM: usize = 128;
        const TOP_K: usize = 8;
        const KPOOL: usize = 4;
        const BLOCK_SIZE: usize = 64;
        let keys = (0..CONTEXT_ROWS).flat_map(|row| std::iter::repeat_n((row + 1) as f32, HEAD_DIM)).collect::<Vec<_>>();
        let gates = vec![0.0_f32; CONTEXT_ROWS * HEAD_DIM];
        let ape = vec![0.0_f32; KPOOL * HEAD_DIM];
        let query = vec![1.0_f32; QUERY_ROWS * HEAD_COUNT * HEAD_DIM];
        let head_weights = vec![1.0_f32 / HEAD_COUNT as f32; QUERY_ROWS * HEAD_COUNT];
        let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&[0_u32])).unwrap();
        let pool_table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&[0_u32])).unwrap();
        let key_input = DeviceBuffer::upload(DEVICE_ID, as_bytes(&keys)).unwrap();
        let gate_input = DeviceBuffer::upload(DEVICE_ID, as_bytes(&gates)).unwrap();
        let ape_input = DeviceBuffer::upload(DEVICE_ID, as_bytes(&ape)).unwrap();
        let key_cache = DeviceBuffer::allocate(DEVICE_ID, CONTEXT_ROWS * HEAD_DIM).unwrap();
        let key_scales = DeviceBuffer::allocate(DEVICE_ID, CONTEXT_ROWS * 2).unwrap();
        let pool_rows = CONTEXT_ROWS / KPOOL;
        let pooled_keys = DeviceBuffer::allocate(DEVICE_ID, pool_rows * HEAD_DIM).unwrap();
        let pooled_scales = DeviceBuffer::allocate(DEVICE_ID, pool_rows * 2).unwrap();
        try_paged_cache_append_f32_q8(DEVICE_ID, &key_input, &key_cache, &key_scales, &table, 0, CONTEXT_ROWS, HEAD_DIM, HEAD_DIM, BLOCK_SIZE).unwrap();
        try_dsa_kpool_compress_q8(DEVICE_ID, &key_cache, &key_scales, &table, &gate_input, &ape_input, &pooled_keys, &pooled_scales, &pool_table, 0, pool_rows, CONTEXT_ROWS, CONTEXT_ROWS, HEAD_DIM, HEAD_DIM, KPOOL, BLOCK_SIZE).unwrap();
        let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query)).unwrap();
        let head_weights = DeviceBuffer::upload(DEVICE_ID, as_bytes(&head_weights)).unwrap();
        let selection = try_dsa_select_paged_q8_kpool(DEVICE_ID, &pooled_keys, &pooled_scales, HEAD_DIM, &pool_table, &query, &head_weights, QUERY_ROWS, CONTEXT_ROWS, QUERY_START, HEAD_COUNT, HEAD_DIM, TOP_K, KPOOL, BLOCK_SIZE).unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP DSA kpool oracle").unwrap();
        let width = TOP_K + KPOOL - 1;
        let mut actual = vec![0_u32; QUERY_ROWS * width];
        selection.copy_to_host(as_bytes_mut(&mut actual)).unwrap();
        for row in 0..QUERY_ROWS {
            let visible = QUERY_START + row + 1;
            let complete = visible / KPOOL;
            let mut expected = ((complete - TOP_K / KPOOL)..complete).flat_map(|pool| (pool * KPOOL..(pool + 1) * KPOOL).map(|token| token as u32)).collect::<Vec<_>>();
            expected.extend((complete * KPOOL..visible).map(|token| token as u32));
            let count = expected.len();
            expected.resize(width, 0);
            let mut selected = actual[row * width..row * width + count].to_vec();
            selected.sort_unstable();
            assert_eq!(selected, expected[..count], "row={row} visible={visible}");
            assert!(actual[row * width + count..(row + 1) * width].iter().all(|&token| token == 0), "row={row} padding 非零");
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn q8_cache_pair_copy_matches_cpu_with_wrap() {
        const DEVICE_ID: i32 = 0;
        const SOURCE_ROWS: usize = 5;
        const TARGET_ROWS: usize = 4;
        const COPY_ROWS: usize = 3;
        const COLUMNS: usize = 64;
        const SCALE_COLUMNS: usize = 2;
        const SOURCE_ROW: usize = 1;
        const TARGET_ROW: usize = 3;

        let _ = super::super::configure(super::super::RocmOptions::default());
        let source_key = (0..SOURCE_ROWS * COLUMNS).map(|index| index.wrapping_mul(17) as u8).collect::<Vec<_>>();
        let source_value = (0..SOURCE_ROWS * COLUMNS).map(|index| index.wrapping_mul(29).wrapping_add(3) as u8).collect::<Vec<_>>();
        let source_key_scales = (0..SOURCE_ROWS * SCALE_COLUMNS).map(|index| 0x3f00_u16.wrapping_add(index as u16)).collect::<Vec<_>>();
        let source_value_scales = (0..SOURCE_ROWS * SCALE_COLUMNS).map(|index| 0x3e00_u16.wrapping_add((index * 3) as u16)).collect::<Vec<_>>();
        let mut expected_key = vec![0xa5_u8; TARGET_ROWS * COLUMNS];
        let mut expected_value = vec![0x5a_u8; TARGET_ROWS * COLUMNS];
        let mut expected_key_scales = vec![0x1111_u16; TARGET_ROWS * SCALE_COLUMNS];
        let mut expected_value_scales = vec![0x2222_u16; TARGET_ROWS * SCALE_COLUMNS];
        for row in 0..COPY_ROWS {
            let source_code = (SOURCE_ROW + row) * COLUMNS;
            let target_code = ((TARGET_ROW + row) % TARGET_ROWS) * COLUMNS;
            expected_key[target_code..target_code + COLUMNS].copy_from_slice(&source_key[source_code..source_code + COLUMNS]);
            expected_value[target_code..target_code + COLUMNS].copy_from_slice(&source_value[source_code..source_code + COLUMNS]);
            let source_scale = (SOURCE_ROW + row) * SCALE_COLUMNS;
            let target_scale = ((TARGET_ROW + row) % TARGET_ROWS) * SCALE_COLUMNS;
            expected_key_scales[target_scale..target_scale + SCALE_COLUMNS].copy_from_slice(&source_key_scales[source_scale..source_scale + SCALE_COLUMNS]);
            expected_value_scales[target_scale..target_scale + SCALE_COLUMNS].copy_from_slice(&source_value_scales[source_scale..source_scale + SCALE_COLUMNS]);
        }
        let source_key = DeviceBuffer::upload(DEVICE_ID, &source_key).unwrap();
        let source_value = DeviceBuffer::upload(DEVICE_ID, &source_value).unwrap();
        let source_key_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&source_key_scales)).unwrap();
        let source_value_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&source_value_scales)).unwrap();
        let target_key = DeviceBuffer::upload(DEVICE_ID, &vec![0xa5_u8; TARGET_ROWS * COLUMNS]).unwrap();
        let target_value = DeviceBuffer::upload(DEVICE_ID, &vec![0x5a_u8; TARGET_ROWS * COLUMNS]).unwrap();
        let target_key_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&vec![0x1111_u16; TARGET_ROWS * SCALE_COLUMNS])).unwrap();
        let target_value_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&vec![0x2222_u16; TARGET_ROWS * SCALE_COLUMNS])).unwrap();
        try_q8_cache_copy_pair(
            DEVICE_ID,
            &source_key,
            &source_key_scales,
            &source_value,
            &source_value_scales,
            &target_key,
            &target_key_scales,
            &target_value,
            &target_value_scales,
            SOURCE_ROW,
            TARGET_ROW,
            COPY_ROWS,
            COLUMNS,
            SCALE_COLUMNS,
            TARGET_ROWS,
        )
        .unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP Q8 cache pair copy oracle").unwrap();
        let mut actual_key = vec![0_u8; TARGET_ROWS * COLUMNS];
        let mut actual_value = vec![0_u8; TARGET_ROWS * COLUMNS];
        let mut actual_key_scales = vec![0_u16; TARGET_ROWS * SCALE_COLUMNS];
        let mut actual_value_scales = vec![0_u16; TARGET_ROWS * SCALE_COLUMNS];
        target_key.copy_to_host(&mut actual_key).unwrap();
        target_value.copy_to_host(&mut actual_value).unwrap();
        target_key_scales.copy_to_host(as_bytes_mut(&mut actual_key_scales)).unwrap();
        target_value_scales.copy_to_host(as_bytes_mut(&mut actual_value_scales)).unwrap();
        assert_eq!(actual_key, expected_key);
        assert_eq!(actual_value, expected_value);
        assert_eq!(actual_key_scales, expected_key_scales);
        assert_eq!(actual_value_scales, expected_value_scales);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_q8_bf16_append_matches_separate_kernels() {
        const DEVICE_ID: i32 = 0;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const GROUP_SIZE: usize = 64;
        const BLOCK_SIZE: usize = 128;

        super::super::configure(super::super::RocmOptions::default()).unwrap();

        for rows in [1_usize, 3, 4096] {
            let latent = (0..rows * LATENT_DIM).map(|index| ((index.wrapping_mul(29).wrapping_add(index / LATENT_DIM * 17)) % 509) as f32 * (1.0 / 97.0) - 2.5).collect::<Vec<_>>();
            let rope = (0..rows * ROPE_DIM).map(|index| ((index.wrapping_mul(13).wrapping_add(index / ROPE_DIM * 7)) % 257) as f32 * (1.0 / 113.0) - 1.0).collect::<Vec<_>>();
            let table = (0..rows.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
            let latent = DeviceBuffer::upload(DEVICE_ID, as_bytes(&latent)).unwrap();
            let rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
            let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
            let latent_bytes = rows * LATENT_DIM;
            let scale_bytes = rows * (LATENT_DIM / GROUP_SIZE) * 2;
            let rope_bytes = rows * ROPE_DIM * 2;
            let baseline_latent = DeviceBuffer::allocate(DEVICE_ID, latent_bytes).unwrap();
            let baseline_scales = DeviceBuffer::allocate(DEVICE_ID, scale_bytes).unwrap();
            let baseline_rope = DeviceBuffer::allocate(DEVICE_ID, rope_bytes).unwrap();
            let fused_latent = DeviceBuffer::allocate(DEVICE_ID, latent_bytes).unwrap();
            let fused_scales = DeviceBuffer::allocate(DEVICE_ID, scale_bytes).unwrap();
            let fused_rope = DeviceBuffer::allocate(DEVICE_ID, rope_bytes).unwrap();

            try_paged_cache_append_f32_q8(DEVICE_ID, &latent, &baseline_latent, &baseline_scales, &table, 0, rows, LATENT_DIM, GROUP_SIZE, BLOCK_SIZE).unwrap();
            try_paged_cache_append_f32_bf16(DEVICE_ID, &rope, &baseline_rope, &table, 0, rows, ROPE_DIM, BLOCK_SIZE).unwrap();
            try_paged_cache_append_mla_f32_q8_bf16(DEVICE_ID, &latent, &fused_latent, &fused_scales, &rope, &fused_rope, &table, 0, rows, LATENT_DIM, ROPE_DIM, GROUP_SIZE, BLOCK_SIZE).unwrap();
            super::super::synchronize_device(DEVICE_ID, "HIP MLA cache append oracle").unwrap();

            for (name, baseline, fused, bytes) in [("latent", &baseline_latent, &fused_latent, latent_bytes), ("scales", &baseline_scales, &fused_scales, scale_bytes), ("rope", &baseline_rope, &fused_rope, rope_bytes)] {
                let mut baseline_host = vec![0_u8; bytes];
                let mut fused_host = vec![0_u8; bytes];
                baseline.copy_to_host(&mut baseline_host).unwrap();
                fused.copy_to_host(&mut fused_host).unwrap();
                assert_eq!(baseline_host, fused_host, "rows={rows} {name} 不一致");
            }
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_selection_split_keeps_global_topk_exactly_once() {
        const DEVICE_ID: i32 = 0;
        const ROWS: usize = 3;
        const WIDTH: usize = 12;
        const BLOCK: usize = 4;
        set_device(DEVICE_ID).unwrap();
        let selection = [
            0_u32, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, // parity 交替
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, // 逆序
            3, 4, 11, 12, 19, 20, 27, 28, 35, 36, 43, 44,
        ];
        let source = DeviceBuffer::upload(DEVICE_ID, as_bytes(&selection)).unwrap();
        let shards = try_split_paged_mla_selection_parity(DEVICE_ID, &source, ROWS, WIDTH, BLOCK).unwrap();
        let mut owner = vec![0_u32; ROWS * WIDTH];
        let mut peer = vec![0_u32; ROWS * WIDTH];
        let mut owner_counts = vec![0_u32; ROWS];
        let mut peer_counts = vec![0_u32; ROWS];
        shards.owner.copy_to_host(as_bytes_mut(&mut owner)).unwrap();
        shards.peer.copy_to_host(as_bytes_mut(&mut peer)).unwrap();
        shards.owner_counts.copy_to_host(as_bytes_mut(&mut owner_counts)).unwrap();
        shards.peer_counts.copy_to_host(as_bytes_mut(&mut peer_counts)).unwrap();
        for row in 0..ROWS {
            let input = &selection[row * WIDTH..(row + 1) * WIDTH];
            let expected_owner = input.iter().copied().filter(|token| ((*token as usize / BLOCK) & 1) == 0).collect::<Vec<_>>();
            let expected_peer = input.iter().copied().filter(|token| ((*token as usize / BLOCK) & 1) == 1).collect::<Vec<_>>();
            let actual_owner = owner[row * WIDTH..row * WIDTH + owner_counts[row] as usize].to_vec();
            let actual_peer = peer[row * WIDTH..row * WIDTH + peer_counts[row] as usize].to_vec();
            assert_eq!(actual_owner, expected_owner);
            assert_eq!(actual_peer, expected_peer);
            assert_eq!(actual_owner.len() + actual_peer.len(), WIDTH);
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_empty_block_parity_shard_is_zero_contribution() {
        const DEVICE_ID: i32 = 0;
        set_device(DEVICE_ID).unwrap();
        let query = DeviceBuffer::upload_f32(DEVICE_ID, &[1.0, -1.0]).unwrap();
        let latent = DeviceBuffer::upload(DEVICE_ID, &[0_u8; 2]).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, &[0_u8; 2]).unwrap();
        let rope = DeviceBuffer::upload(DEVICE_ID, &[0_u8; 4]).unwrap();
        let table = DeviceBuffer::upload(DEVICE_ID, &[0_u8; 4]).unwrap();
        let packed = DeviceBuffer::upload(DEVICE_ID, &[0_u8; 4]).unwrap();
        let weight_scales = DeviceBuffer::upload(DEVICE_ID, &[0_u8; 2]).unwrap();
        let shard = try_paged_mla_attention_ct_shard(
            DEVICE_ID,
            &query,
            &latent,
            Some(&scales),
            2,
            &rope,
            &table,
            None,
            None,
            CtMlaWeightRef { packed: &packed, scales: &weight_scales, rows: 2, cols: 2, group_size: 2, scale_dtype: 0, bits: 16 },
            1,
            33,
            32,
            2,
            1,
            2,
            0,
            64,
            1,
        )
        .unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP MLA empty parity shard oracle").unwrap();
        let mut weighted = [u32::MAX];
        shard.weighted.copy_to_host(as_bytes_mut(&mut weighted)).unwrap();
        assert_eq!(weighted, [0]);
        assert_eq!(shard.stats.download_f32(2).unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_block_parity_shards_match_full_attention() {
        const DEVICE_ID: i32 = 0;
        const CONTEXT_ROWS: usize = 257;
        const HEADS: usize = 32;
        const Q_HEAD: usize = 256;
        const KV_HEAD: usize = 448;
        const LATENT: usize = 512;
        const ROPE: usize = 64;
        const GROUP: usize = 64;
        const BLOCK: usize = 64;
        let q_projection = HEADS * Q_HEAD;
        let kv_projection = HEADS * KV_HEAD;
        super::super::configure(super::super::RocmOptions::default()).unwrap();
        set_device(DEVICE_ID).unwrap();

        let weight_bits = (0..kv_projection * LATENT)
            .map(|index| {
                let value = (((index * 17) % 127) as f32 - 63.0) * (1.0 / 4096.0);
                (value.to_bits() >> 16) as u16
            })
            .collect::<Vec<_>>();
        let weight = DeviceBuffer::upload(DEVICE_ID, as_bytes(&weight_bits)).unwrap();
        let weight_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&[0x3f80_u16])).unwrap();
        let latent = (0..CONTEXT_ROWS * LATENT).map(|index| ((index * 29 + index / LATENT * 7) % 63 + 1) as u8).collect::<Vec<_>>();
        let latent_scales = vec![0x3b80_u16; CONTEXT_ROWS * (LATENT / GROUP)];
        let rope = (0..CONTEXT_ROWS * ROPE)
            .map(|index| {
                let value = (((index * 13) % 127) as f32 - 63.0) * (1.0 / 128.0);
                (value.to_bits() >> 16) as u16
            })
            .collect::<Vec<_>>();
        let compact = |bytes: &[u8], row_bytes: usize, parity: usize| {
            let mut output = Vec::new();
            for row in 0..CONTEXT_ROWS {
                if ((row / BLOCK) & 1) == parity {
                    output.extend_from_slice(&bytes[row * row_bytes..(row + 1) * row_bytes]);
                }
            }
            output
        };
        let latent_device = DeviceBuffer::upload(DEVICE_ID, &latent).unwrap();
        let scales_device = DeviceBuffer::upload(DEVICE_ID, as_bytes(&latent_scales)).unwrap();
        let rope_device = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
        let full_table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&(0..CONTEXT_ROWS.div_ceil(BLOCK) as u32).collect::<Vec<_>>())).unwrap();
        let owner_latent = DeviceBuffer::upload(DEVICE_ID, &compact(&latent, LATENT, 0)).unwrap();
        let peer_latent = DeviceBuffer::upload(DEVICE_ID, &compact(&latent, LATENT, 1)).unwrap();
        let owner_scales = DeviceBuffer::upload(DEVICE_ID, &compact(as_bytes(&latent_scales), LATENT / GROUP * 2, 0)).unwrap();
        let peer_scales = DeviceBuffer::upload(DEVICE_ID, &compact(as_bytes(&latent_scales), LATENT / GROUP * 2, 1)).unwrap();
        let owner_rope = DeviceBuffer::upload(DEVICE_ID, &compact(as_bytes(&rope), ROPE * 2, 0)).unwrap();
        let peer_rope = DeviceBuffer::upload(DEVICE_ID, &compact(as_bytes(&rope), ROPE * 2, 1)).unwrap();
        let owner_rows = (0..CONTEXT_ROWS).filter(|row| ((row / BLOCK) & 1) == 0).count();
        let peer_rows = CONTEXT_ROWS - owner_rows;
        let owner_table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&(0..owner_rows.div_ceil(BLOCK) as u32).collect::<Vec<_>>())).unwrap();
        let peer_table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&(0..peer_rows.div_ceil(BLOCK) as u32).collect::<Vec<_>>())).unwrap();

        for query_rows in [1_usize, 8] {
            let query_host = (0..query_rows * q_projection).map(|index| ((index % 257) as f32 - 128.0) * (1.0 / 256.0)).collect::<Vec<_>>();
            let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query_host)).unwrap();
            let output_bytes = query_rows * q_projection * if query_rows == 1 { 4 } else { 2 };
            let full = DeviceBuffer::allocate(DEVICE_ID, output_bytes).unwrap();
            let weight_ref = || CtMlaWeightRef { packed: &weight, scales: &weight_scales, rows: kv_projection, cols: LATENT, group_size: LATENT, scale_dtype: 0, bits: 16 };
            try_paged_mla_attention_ct_into(
                DEVICE_ID,
                &query,
                &latent_device,
                Some(&scales_device),
                GROUP,
                &rope_device,
                &full_table,
                None,
                weight_ref(),
                query_rows,
                CONTEXT_ROWS,
                CONTEXT_ROWS - query_rows,
                q_projection,
                HEADS,
                ROPE,
                0,
                BLOCK,
                &full,
                Some(true),
            )
            .unwrap();
            let owner = try_paged_mla_attention_ct_shard(
                DEVICE_ID,
                &query,
                &owner_latent,
                Some(&owner_scales),
                GROUP,
                &owner_rope,
                &owner_table,
                None,
                None,
                weight_ref(),
                query_rows,
                CONTEXT_ROWS,
                CONTEXT_ROWS - query_rows,
                q_projection,
                HEADS,
                ROPE,
                0,
                BLOCK,
                0,
            )
            .unwrap();
            let peer = try_paged_mla_attention_ct_shard(
                DEVICE_ID,
                &query,
                &peer_latent,
                Some(&peer_scales),
                GROUP,
                &peer_rope,
                &peer_table,
                None,
                None,
                weight_ref(),
                query_rows,
                CONTEXT_ROWS,
                CONTEXT_ROWS - query_rows,
                q_projection,
                HEADS,
                ROPE,
                0,
                BLOCK,
                1,
            )
            .unwrap();
            let owner_output = DeviceBuffer::allocate(DEVICE_ID, output_bytes).unwrap();
            let peer_output = DeviceBuffer::allocate(DEVICE_ID, output_bytes).unwrap();
            try_paged_mla_shard_scale_project_ct(DEVICE_ID, &owner.weighted, &owner.stats, &peer.stats, weight_ref(), query_rows, q_projection, HEADS, ROPE, &owner_output).unwrap();
            try_paged_mla_shard_scale_project_ct(DEVICE_ID, &peer.weighted, &peer.stats, &owner.stats, weight_ref(), query_rows, q_projection, HEADS, ROPE, &peer_output).unwrap();
            let half_output_bytes = output_bytes / 2;
            let owner_half = DeviceBuffer::allocate(DEVICE_ID, half_output_bytes).unwrap();
            let peer_half = DeviceBuffer::allocate(DEVICE_ID, half_output_bytes).unwrap();
            let merged_full = DeviceBuffer::allocate(DEVICE_ID, output_bytes).unwrap();
            let merged_full_reverse = DeviceBuffer::allocate(DEVICE_ID, output_bytes).unwrap();
            try_paged_mla_shard_merge_project_heads_ct(DEVICE_ID, &owner.weighted, &peer.weighted, &owner.stats, &peer.stats, weight_ref(), query_rows, q_projection, HEADS, 0, HEADS / 2, ROPE, &owner_half).unwrap();
            try_paged_mla_shard_merge_project_heads_ct(DEVICE_ID, &peer.weighted, &owner.weighted, &peer.stats, &owner.stats, weight_ref(), query_rows, q_projection, HEADS, HEADS / 2, HEADS / 2, ROPE, &peer_half).unwrap();
            try_paged_mla_shard_merge_project_heads_ct(DEVICE_ID, &owner.weighted, &peer.weighted, &owner.stats, &peer.stats, weight_ref(), query_rows, q_projection, HEADS, 0, HEADS, ROPE, &merged_full).unwrap();
            try_paged_mla_shard_merge_project_heads_ct(DEVICE_ID, &peer.weighted, &owner.weighted, &peer.stats, &owner.stats, weight_ref(), query_rows, q_projection, HEADS, 0, HEADS, ROPE, &merged_full_reverse).unwrap();
            super::super::synchronize_device(DEVICE_ID, "HIP MLA parity shard oracle").unwrap();
            let decode = |buffer: &DeviceBuffer| {
                if query_rows == 1 {
                    let mut values = vec![0.0_f32; buffer.bytes / std::mem::size_of::<f32>()];
                    buffer.copy_to_host(as_bytes_mut(&mut values)).unwrap();
                    values
                } else {
                    let mut values = vec![0_u16; buffer.bytes / std::mem::size_of::<u16>()];
                    buffer.copy_to_host(as_bytes_mut(&mut values)).unwrap();
                    values.into_iter().map(|bits| f32::from_bits(u32::from(bits) << 16)).collect()
                }
            };
            let full = decode(&full);
            let owner_output = decode(&owner_output);
            let peer_output = decode(&peer_output);
            let max_abs = full.iter().zip(owner_output.iter().zip(&peer_output)).map(|(full, (owner, peer))| (full - (owner + peer)).abs()).fold(0.0_f32, f32::max);
            println!("[mla-parity-shard-oracle] rows={query_rows} context={CONTEXT_ROWS} max_abs={max_abs:.6e}");
            assert!(max_abs <= if query_rows == 1 { 2.0e-2 } else { 5.0e-2 }, "rows={query_rows} max_abs={max_abs}");
            let owner_half = decode(&owner_half);
            let peer_half = decode(&peer_half);
            let half_columns = q_projection / 2;
            let head_reduce_scatter_max_abs = (0..query_rows)
                .flat_map(|row| {
                    let owner = &owner_half[row * half_columns..(row + 1) * half_columns];
                    let peer = &peer_half[row * half_columns..(row + 1) * half_columns];
                    owner.iter().chain(peer).zip(&full[row * q_projection..(row + 1) * q_projection]).map(|(actual, expected)| (actual - expected).abs())
                })
                .fold(0.0_f32, f32::max);
            println!("[mla-head-reduce-scatter-oracle] rows={query_rows} context={CONTEXT_ROWS} max_abs={head_reduce_scatter_max_abs:.6e}");
            assert!(head_reduce_scatter_max_abs <= if query_rows == 1 { 2.0e-2 } else { 5.0e-2 }, "rows={query_rows} head reduce-scatter max_abs={head_reduce_scatter_max_abs}");
            let merged_full = decode(&merged_full);
            let merged_full_reverse = decode(&merged_full_reverse);
            let full_merge_max_abs = merged_full.iter().zip(&full).map(|(actual, expected)| (actual - expected).abs()).fold(0.0_f32, f32::max);
            println!("[mla-full-head-merge-oracle] rows={query_rows} context={CONTEXT_ROWS} max_abs={full_merge_max_abs:.6e}");
            assert!(full_merge_max_abs <= if query_rows == 1 { 2.0e-2 } else { 5.0e-2 }, "rows={query_rows} full-head merge max_abs={full_merge_max_abs}");
            assert!(merged_full.iter().zip(&merged_full_reverse).all(|(owner_first, peer_first)| owner_first.to_bits() == peer_first.to_bits()), "rows={query_rows} 交换 shard 顺序后 full-head merge 必须逐位一致");
        }

        // prefill 的全局候选表先按 parity 紧凑拆分；pair 两半的 WMMA
        // 结果相加必须与未分片标量 attention 一致。候选刻意跨越两个
        // 相邻 block，并保留每行独立 counts，覆盖真实 DSA 数据流。
        const TOP_K: usize = 64;
        for query_rows in [1_usize, 8] {
            let query_start = CONTEXT_ROWS - query_rows;
            let query_host = (0..query_rows * q_projection).map(|index| ((index % 257) as f32 - 128.0) * (1.0 / 256.0)).collect::<Vec<_>>();
            let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query_host)).unwrap();
            let selection = (0..query_rows).flat_map(|row| (0..TOP_K).map(move |index| if index < TOP_K / 2 { index as u32 } else { (BLOCK + index - TOP_K / 2 + row % 3) as u32 })).collect::<Vec<_>>();
            let selection = DeviceBuffer::upload(DEVICE_ID, as_bytes(&selection)).unwrap();
            let shards = try_split_paged_mla_selection_parity(DEVICE_ID, &selection, query_rows, TOP_K, BLOCK).unwrap();
            let output_bytes = query_rows * q_projection * if query_rows == 1 { 4 } else { 2 };
            let full = DeviceBuffer::allocate(DEVICE_ID, output_bytes).unwrap();
            let weight_ref = || CtMlaWeightRef { packed: &weight, scales: &weight_scales, rows: kv_projection, cols: LATENT, group_size: LATENT, scale_dtype: 0, bits: 16 };
            TEST_SPARSE_PREFILL_WMMA.store(false, std::sync::atomic::Ordering::Relaxed);
            try_paged_mla_attention_ct_into(
                DEVICE_ID,
                &query,
                &latent_device,
                Some(&scales_device),
                GROUP,
                &rope_device,
                &full_table,
                Some(&selection),
                weight_ref(),
                query_rows,
                CONTEXT_ROWS,
                query_start,
                q_projection,
                HEADS,
                ROPE,
                TOP_K,
                BLOCK,
                &full,
                Some(false),
            )
            .unwrap();
            TEST_SPARSE_PREFILL_WMMA.store(true, std::sync::atomic::Ordering::Relaxed);
            let owner = try_paged_mla_attention_ct_shard(
                DEVICE_ID,
                &query,
                &owner_latent,
                Some(&owner_scales),
                GROUP,
                &owner_rope,
                &owner_table,
                Some(&shards.owner),
                Some(&shards.owner_counts),
                weight_ref(),
                query_rows,
                CONTEXT_ROWS,
                query_start,
                q_projection,
                HEADS,
                ROPE,
                TOP_K,
                BLOCK,
                0,
            )
            .unwrap();
            let peer = try_paged_mla_attention_ct_shard(
                DEVICE_ID,
                &query,
                &peer_latent,
                Some(&peer_scales),
                GROUP,
                &peer_rope,
                &peer_table,
                Some(&shards.peer),
                Some(&shards.peer_counts),
                weight_ref(),
                query_rows,
                CONTEXT_ROWS,
                query_start,
                q_projection,
                HEADS,
                ROPE,
                TOP_K,
                BLOCK,
                1,
            )
            .unwrap();
            let owner_output = DeviceBuffer::allocate(DEVICE_ID, output_bytes).unwrap();
            let peer_output = DeviceBuffer::allocate(DEVICE_ID, output_bytes).unwrap();
            try_paged_mla_shard_scale_project_ct(DEVICE_ID, &owner.weighted, &owner.stats, &peer.stats, weight_ref(), query_rows, q_projection, HEADS, ROPE, &owner_output).unwrap();
            try_paged_mla_shard_scale_project_ct(DEVICE_ID, &peer.weighted, &peer.stats, &owner.stats, weight_ref(), query_rows, q_projection, HEADS, ROPE, &peer_output).unwrap();
            super::super::synchronize_device(DEVICE_ID, "HIP MLA selected parity shard oracle").unwrap();
            let decode = |buffer: &DeviceBuffer| {
                if query_rows == 1 {
                    let mut values = vec![0.0_f32; query_rows * q_projection];
                    buffer.copy_to_host(as_bytes_mut(&mut values)).unwrap();
                    values
                } else {
                    let mut values = vec![0_u16; query_rows * q_projection];
                    buffer.copy_to_host(as_bytes_mut(&mut values)).unwrap();
                    values.into_iter().map(|bits| f32::from_bits(u32::from(bits) << 16)).collect::<Vec<_>>()
                }
            };
            let full = decode(&full);
            let owner_output = decode(&owner_output);
            let peer_output = decode(&peer_output);
            let max_abs = full.iter().zip(owner_output.iter().zip(&peer_output)).map(|(full, (owner, peer))| (full - (owner + peer)).abs()).fold(0.0_f32, f32::max);
            println!("[mla-selected-parity-shard-oracle] rows={query_rows} context={CONTEXT_ROWS} topk={TOP_K} max_abs={max_abs:.6e}");
            assert!(max_abs <= if query_rows == 1 { 2.0e-2 } else { 5.0e-2 }, "selected parity rows={query_rows} max_abs={max_abs}");
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_rope_q8_bf16_append_matches_separate_kernels_bits() {
        const DEVICE_ID: i32 = 0;
        const ROWS: usize = 5;
        const POSITION: usize = 7;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const GROUP_SIZE: usize = 64;
        const BLOCK_SIZE: usize = 128;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let end = POSITION + ROWS;
        let latent = (0..ROWS * LATENT_DIM).map(|index| ((index * 29 + index / LATENT_DIM * 17) % 509) as f32 * (1.0 / 97.0) - 2.5).collect::<Vec<_>>();
        let rope = (0..ROWS * ROPE_DIM).map(|index| ((index * 13 + index / ROPE_DIM * 7) % 257) as f32 * (1.0 / 113.0) - 1.0).collect::<Vec<_>>();
        let table_elements = end * (ROPE_DIM / 2);
        let cosine = (0..table_elements).map(|index| (index as f32 * 0.017).cos()).collect::<Vec<_>>();
        let sine = (0..table_elements).map(|index| (index as f32 * 0.017).sin()).collect::<Vec<_>>();
        let table = (0..end.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
        let latent = DeviceBuffer::upload(DEVICE_ID, as_bytes(&latent)).unwrap();
        let rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
        let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
        let latent_bytes = end * LATENT_DIM;
        let scale_bytes = end * (LATENT_DIM / GROUP_SIZE) * 2;
        let rope_bytes = end * ROPE_DIM * 2;
        let baseline_latent = DeviceBuffer::upload(DEVICE_ID, &vec![0_u8; latent_bytes]).unwrap();
        let baseline_scales = DeviceBuffer::upload(DEVICE_ID, &vec![0_u8; scale_bytes]).unwrap();
        let baseline_rope = DeviceBuffer::upload(DEVICE_ID, &vec![0_u8; rope_bytes]).unwrap();
        let fused_latent = DeviceBuffer::upload(DEVICE_ID, &vec![0_u8; latent_bytes]).unwrap();
        let fused_scales = DeviceBuffer::upload(DEVICE_ID, &vec![0_u8; scale_bytes]).unwrap();
        let fused_rope = DeviceBuffer::upload(DEVICE_ID, &vec![0_u8; rope_bytes]).unwrap();

        try_paged_cache_append_f32_q8(DEVICE_ID, &latent, &baseline_latent, &baseline_scales, &table, POSITION, ROWS, LATENT_DIM, GROUP_SIZE, BLOCK_SIZE).unwrap();
        let rotated = super::super::try_rope_resident_f32(DEVICE_ID, &rope, ROWS, ROPE_DIM, 1, ROPE_DIM, RotaryLayout::SplitHalf, POSITION, &cosine, &sine, false).unwrap();
        try_paged_cache_append_f32_bf16(DEVICE_ID, &rotated, &baseline_rope, &table, POSITION, ROWS, ROPE_DIM, BLOCK_SIZE).unwrap();
        try_paged_cache_append_mla_rope_f32_q8_bf16(
            DEVICE_ID,
            &latent,
            &fused_latent,
            &fused_scales,
            &rope,
            &fused_rope,
            &table,
            POSITION,
            ROWS,
            LATENT_DIM,
            ROPE_DIM,
            ROPE_DIM,
            RotaryLayout::SplitHalf,
            GROUP_SIZE,
            BLOCK_SIZE,
            &cosine,
            &sine,
        )
        .unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP MLA fused RoPE cache append oracle").unwrap();

        for (name, baseline, fused, bytes) in [("latent", &baseline_latent, &fused_latent, latent_bytes), ("scales", &baseline_scales, &fused_scales, scale_bytes), ("rope", &baseline_rope, &fused_rope, rope_bytes)] {
            let mut baseline_host = vec![0_u8; bytes];
            let mut fused_host = vec![0_u8; bytes];
            baseline.copy_to_host(&mut baseline_host).unwrap();
            fused.copy_to_host(&mut fused_host).unwrap();
            assert_eq!(baseline_host, fused_host, "{name} cache 位级不一致");
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_sparse_prefill_wmma_matches_scalar_and_reports_latency() {
        const DEVICE_ID: i32 = 0;
        const QUERY_ROWS: usize = 2048;
        const CONTEXT_ROWS: usize = 8192;
        const TOP_K: usize = 2048;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256;
        const Q_PROJECTION: usize = HEAD_COUNT * Q_HEAD_DIM;
        const LATENT_DIM: usize = 512;
        const KV_HEAD_DIM: usize = 448;
        const KV_PROJECTION: usize = HEAD_COUNT * KV_HEAD_DIM;
        const ROPE_DIM: usize = 64;
        const LATENT_GROUP: usize = 64;
        const BLOCK_SIZE: usize = 128;

        super::super::configure(super::super::RocmOptions::default()).unwrap();

        let query = (0..QUERY_ROWS * Q_PROJECTION).map(|index| ((index.wrapping_mul(13) % 257) as f32 - 128.0) * (1.0 / 1024.0)).collect::<Vec<_>>();
        let latent = (0..CONTEXT_ROWS * LATENT_DIM).map(|index| (((index.wrapping_mul(29).wrapping_add(index / LATENT_DIM * 7)) % 127) as i16 - 63) as i8 as u8).collect::<Vec<_>>();
        let latent_scales = vec![0x3b80_u16; CONTEXT_ROWS * (LATENT_DIM / LATENT_GROUP)];
        let rope = (0..CONTEXT_ROWS * ROPE_DIM).map(|index| (((index * 13) % 127) as f32 - 63.0) * (1.0 / 128.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
        let table = (0..CONTEXT_ROWS.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
        let query_start = CONTEXT_ROWS - QUERY_ROWS;
        let selection = (0..QUERY_ROWS)
            .flat_map(|row| {
                let visible = query_start + row + 1;
                (0..TOP_K).map(move |index| ((index * 251 + row * 17) % visible) as u32)
            })
            .collect::<Vec<_>>();
        let weight = (0..KV_PROJECTION * LATENT_DIM).map(|index| (((index * 17) % 127) as f32 - 63.0) * (1.0 / 16384.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
        let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query)).unwrap();
        let latent = DeviceBuffer::upload(DEVICE_ID, &latent).unwrap();
        let latent_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&latent_scales)).unwrap();
        let rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
        let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
        let selection = DeviceBuffer::upload(DEVICE_ID, as_bytes(&selection)).unwrap();
        let weight = DeviceBuffer::upload(DEVICE_ID, as_bytes(&weight)).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&[0x3f80_u16])).unwrap();
        let baseline = DeviceBuffer::allocate(DEVICE_ID, QUERY_ROWS * Q_PROJECTION * 2).unwrap();
        let candidate = DeviceBuffer::allocate(DEVICE_ID, QUERY_ROWS * Q_PROJECTION * 2).unwrap();

        let run = |output: &DeviceBuffer, wmma: bool| {
            TEST_SPARSE_PREFILL_WMMA.store(wmma, std::sync::atomic::Ordering::Relaxed);
            let started = std::time::Instant::now();
            try_paged_mla_attention_ct_into(
                DEVICE_ID,
                &query,
                &latent,
                Some(&latent_scales),
                LATENT_GROUP,
                &rope,
                &table,
                Some(&selection),
                CtMlaWeightRef { packed: &weight, scales: &scales, rows: KV_PROJECTION, cols: LATENT_DIM, group_size: LATENT_DIM, scale_dtype: 0, bits: 16 },
                QUERY_ROWS,
                CONTEXT_ROWS,
                query_start,
                Q_PROJECTION,
                HEAD_COUNT,
                ROPE_DIM,
                TOP_K,
                BLOCK_SIZE,
                output,
                Some(false),
            )
            .unwrap();
            super::super::synchronize_device(DEVICE_ID, "HIP MLA sparse prefill oracle").unwrap();
            started.elapsed().as_secs_f64() * 1e3
        };
        let legacy_wmma = DeviceBuffer::allocate(DEVICE_ID, QUERY_ROWS * Q_PROJECTION * 2).unwrap();
        TEST_SPARSE_PREFILL_HEADS32.store(false, std::sync::atomic::Ordering::Relaxed);
        run(&legacy_wmma, true);
        TEST_SPARSE_PREFILL_HEADS32.store(true, std::sync::atomic::Ordering::Relaxed);
        run(&baseline, false);
        run(&candidate, true);
        let baseline_ms = (0..3).map(|_| run(&baseline, false)).sum::<f64>() / 3.0;
        let candidate_ms = (0..3).map(|_| run(&candidate, true)).sum::<f64>() / 3.0;

        let mut baseline_bits = vec![0_u16; QUERY_ROWS * Q_PROJECTION];
        let mut candidate_bits = vec![0_u16; QUERY_ROWS * Q_PROJECTION];
        baseline.copy_to_host(as_bytes_mut(&mut baseline_bits)).unwrap();
        candidate.copy_to_host(as_bytes_mut(&mut candidate_bits)).unwrap();
        let legacy_bits = legacy_wmma.download_u16(QUERY_ROWS * Q_PROJECTION).unwrap();
        let different = legacy_bits.iter().zip(&candidate_bits).filter(|(left, right)| left != right).count();
        assert_eq!(different, 0, "32-head prefill 与原 16-head WMMA 必须逐位一致");
        let baseline_host = baseline_bits.into_iter().map(|bits| f32::from_bits(u32::from(bits) << 16)).collect::<Vec<_>>();
        let candidate_host = candidate_bits.into_iter().map(|bits| f32::from_bits(u32::from(bits) << 16)).collect::<Vec<_>>();
        let mut max_abs = 0.0_f32;
        let mut max_index = 0;
        let mut squared = 0.0_f64;
        for (index, (reference, actual)) in baseline_host.iter().zip(&candidate_host).enumerate() {
            assert!(reference.is_finite() && actual.is_finite(), "index={index} scalar={reference} wmma={actual}");
            let error = (reference - actual).abs();
            if error > max_abs {
                max_abs = error;
                max_index = index;
            }
            squared += f64::from(error) * f64::from(error);
        }
        let rmse = (squared / baseline_host.len() as f64).sqrt();
        println!(
            "[mla-sparse-prefill-oracle] rows={QUERY_ROWS} context={CONTEXT_ROWS} topk={TOP_K} scalar_ms={baseline_ms:.3} wmma_ms={candidate_ms:.3} speedup={:.3} max_abs={max_abs:.6e} max_index={max_index} reference={:.6e} actual={:.6e} rmse={rmse:.6e}",
            baseline_ms / candidate_ms,
            baseline_host[max_index],
            candidate_host[max_index]
        );
        assert!(max_abs <= 5.0e-3, "max_abs={max_abs}");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_project_value_w4g128_matches_cpu_oracle_and_reports_latency() {
        const DEVICE_ID: i32 = 0;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256;
        const KV_HEAD_DIM: usize = 448;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const GROUP_SIZE: usize = 128;
        const REPEATS: usize = 20;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let input_bits = (0..HEAD_COUNT * LATENT_DIM)
            .map(|index| {
                let value = ((index * 29 % 257) as f32 - 128.0) * (1.0 / 512.0);
                (value.to_bits() >> 16) as u16
            })
            .collect::<Vec<_>>();
        let packed_columns = LATENT_DIM / 8;
        let weight_rows = HEAD_COUNT * KV_HEAD_DIM;
        let packed = (0..weight_rows * packed_columns)
            .map(|word_index| {
                let row = word_index / packed_columns;
                let column_base = word_index % packed_columns * 8;
                (0..8).fold(0u32, |word, element| {
                    let code = ((row * 17 + column_base + element * 13) % 16) as u32;
                    word | (code << (element * 4))
                })
            })
            .collect::<Vec<_>>();
        let groups = LATENT_DIM / GROUP_SIZE;
        let scale_bits = (0..weight_rows * groups)
            .map(|index| {
                let value = ((index * 7 % 5 + 1) as f32) * (1.0 / 512.0);
                (value.to_bits() >> 16) as u16
            })
            .collect::<Vec<_>>();
        let input = DeviceBuffer::upload(DEVICE_ID, as_bytes(&input_bits)).unwrap();
        let packed_device = DeviceBuffer::upload(DEVICE_ID, as_bytes(&packed)).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scale_bits)).unwrap();
        let output = DeviceBuffer::allocate(DEVICE_ID, HEAD_COUNT * Q_HEAD_DIM * 4).unwrap();
        let functions = paged_mla_functions(DEVICE_ID).unwrap();
        let mut d_input = input.pointer;
        let mut d_packed = packed_device.pointer;
        let mut d_scales = scales.pointer;
        let mut d_output = output.pointer;
        let mut rows = 1u32;
        let mut heads = HEAD_COUNT as u32;
        let mut q_head = Q_HEAD_DIM as u32;
        let mut kv_head = KV_HEAD_DIM as u32;
        let mut latent = LATENT_DIM as u32;
        let mut rope = ROPE_DIM as u32;
        let mut group = GROUP_SIZE as u32;
        let mut scale_dtype = 0u32;
        let mut bits = 4u32;
        let mut weight_head_start = 0u32;
        let mut args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_packed as *mut *mut c_void).cast(),
            (&mut d_scales as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut heads as *mut u32).cast(),
            (&mut weight_head_start as *mut u32).cast(),
            (&mut q_head as *mut u32).cast(),
            (&mut kv_head as *mut u32).cast(),
            (&mut latent as *mut u32).cast(),
            (&mut rope as *mut u32).cast(),
            (&mut group as *mut u32).cast(),
            (&mut scale_dtype as *mut u32).cast(),
            (&mut bits as *mut u32).cast(),
        ];
        let value_dim = KV_HEAD_DIM - (Q_HEAD_DIM - ROPE_DIM);
        let grid = (HEAD_COUNT * value_dim.div_ceil(16)) as u32;
        launch_moe_kernel(functions.project_value, grid, 1, 256, 0, &mut args, "HIP MLA W4G128 project oracle warmup").unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W4G128 project oracle warmup").unwrap();
        let started = std::time::Instant::now();
        for _ in 0..REPEATS {
            launch_moe_kernel(functions.project_value, grid, 1, 256, 0, &mut args, "HIP MLA W4G128 project oracle").unwrap();
        }
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W4G128 project oracle").unwrap();
        let project_ms = started.elapsed().as_secs_f64() * 1e3 / REPEATS as f64;

        let input_host = input_bits.iter().map(|bits| f32::from_bits(u32::from(*bits) << 16)).collect::<Vec<_>>();
        let scale_host = scale_bits.iter().map(|bits| f32::from_bits(u32::from(*bits) << 16)).collect::<Vec<_>>();
        let mut expected = vec![0.0f32; HEAD_COUNT * Q_HEAD_DIM];
        for head in 0..HEAD_COUNT {
            for value in 0..value_dim {
                let weight_row = head * KV_HEAD_DIM + Q_HEAD_DIM - ROPE_DIM + value;
                let mut sum = 0.0f32;
                for column in 0..LATENT_DIM {
                    let word = packed[weight_row * packed_columns + column / 8];
                    let code = ((word >> ((column & 7) * 4)) & 15) as i32 - 8;
                    sum += input_host[head * LATENT_DIM + column] * code as f32 * scale_host[weight_row * groups + column / GROUP_SIZE];
                }
                expected[head * Q_HEAD_DIM + value] = sum;
            }
        }
        let mut actual = vec![0.0f32; expected.len()];
        output.copy_to_host(as_bytes_mut(&mut actual)).unwrap();
        let mut max_abs = 0.0f32;
        let mut max_index = 0usize;
        for (index, (reference, value)) in expected.iter().zip(&actual).enumerate() {
            let error = (reference - value).abs();
            if error > max_abs {
                max_abs = error;
                max_index = index;
            }
        }
        println!("[mla-project-w4g128-oracle] project_ms={project_ms:.3} max_abs={max_abs:.6e} max_index={max_index} reference={:.6e} actual={:.6e}", expected[max_index], actual[max_index]);
        assert!(max_abs <= 2.0e-5, "max_abs={max_abs} index={max_index}");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_project_value_w8g32_decode_matches_cpu_oracle_and_reports_latency() {
        const DEVICE_ID: i32 = 0;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256;
        const KV_HEAD_DIM: usize = 448;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const GROUP_SIZE: usize = 32;
        const REPEATS: usize = 50;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let input_bits = (0..HEAD_COUNT * LATENT_DIM)
            .map(|index| {
                let value = ((index * 29 % 257) as f32 - 128.0) * (1.0 / 512.0);
                (value.to_bits() >> 16) as u16
            })
            .collect::<Vec<_>>();
        // W8（i8+128 偏置）+ F16 scale，group 32——GLM-5.3 kv_b 的生产布局。
        let weight_rows = HEAD_COUNT * KV_HEAD_DIM;
        let packed = (0..weight_rows * LATENT_DIM)
            .map(|index| {
                let row = index / LATENT_DIM;
                let column = index % LATENT_DIM;
                ((row * 17 + column * 13) % 255) as u8
            })
            .collect::<Vec<_>>();
        let groups = LATENT_DIM / GROUP_SIZE;
        let scale_bytes = (0..weight_rows * groups).map(|index| half::f16::from_f32(((index * 7 % 5 + 1) as f32) * (1.0 / 512.0)).to_le_bytes()).collect::<Vec<_>>();
        let input = DeviceBuffer::upload(DEVICE_ID, as_bytes(&input_bits)).unwrap();
        let packed_device = DeviceBuffer::upload(DEVICE_ID, &packed).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scale_bytes)).unwrap();
        let output = DeviceBuffer::allocate(DEVICE_ID, HEAD_COUNT * Q_HEAD_DIM * 4).unwrap();
        let functions = paged_mla_functions(DEVICE_ID).unwrap();
        let mut d_input = input.pointer;
        let mut d_packed = packed_device.pointer;
        let mut d_scales = scales.pointer;
        let mut d_output = output.pointer;
        let mut rows = 1u32;
        let mut heads = HEAD_COUNT as u32;
        let mut q_head = Q_HEAD_DIM as u32;
        let mut kv_head = KV_HEAD_DIM as u32;
        let mut latent = LATENT_DIM as u32;
        let mut rope = ROPE_DIM as u32;
        let mut group = GROUP_SIZE as u32;
        let mut scale_dtype = 1u32;
        let mut bits = 8u32;
        let mut weight_head_start = 0u32;
        let mut args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_packed as *mut *mut c_void).cast(),
            (&mut d_scales as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut heads as *mut u32).cast(),
            (&mut weight_head_start as *mut u32).cast(),
            (&mut q_head as *mut u32).cast(),
            (&mut kv_head as *mut u32).cast(),
            (&mut latent as *mut u32).cast(),
            (&mut rope as *mut u32).cast(),
            (&mut group as *mut u32).cast(),
            (&mut scale_dtype as *mut u32).cast(),
            (&mut bits as *mut u32).cast(),
        ];
        let value_dim = KV_HEAD_DIM - (Q_HEAD_DIM - ROPE_DIM);
        let grid = (HEAD_COUNT * value_dim.div_ceil(16)) as u32;
        launch_moe_kernel(functions.project_value, grid, 1, 256, 0, &mut args, "HIP MLA W8G32 project oracle warmup").unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W8G32 project oracle warmup").unwrap();
        let started = std::time::Instant::now();
        for _ in 0..REPEATS {
            launch_moe_kernel(functions.project_value, grid, 1, 256, 0, &mut args, "HIP MLA W8G32 project oracle").unwrap();
        }
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W8G32 project oracle").unwrap();
        let project_ms = started.elapsed().as_secs_f64() * 1e3 / REPEATS as f64;

        let input_host = input_bits.iter().map(|bits| f32::from_bits(u32::from(*bits) << 16)).collect::<Vec<_>>();
        let scale_host = scale_bytes.iter().map(|bytes| half::f16::from_le_bytes(*bytes).to_f32()).collect::<Vec<_>>();
        let mut max_abs = 0.0f32;
        let mut max_index = 0usize;
        let mut actual = vec![0.0f32; HEAD_COUNT * Q_HEAD_DIM];
        output.copy_to_host(as_bytes_mut(&mut actual)).unwrap();
        for head in 0..HEAD_COUNT {
            for value in 0..value_dim {
                let weight_row = head * KV_HEAD_DIM + Q_HEAD_DIM - ROPE_DIM + value;
                let mut sum = 0.0f32;
                for column in 0..LATENT_DIM {
                    let code = packed[weight_row * LATENT_DIM + column] as i32 - 128;
                    sum += input_host[head * LATENT_DIM + column] * code as f32 * scale_host[weight_row * groups + column / GROUP_SIZE];
                }
                let error = (sum - actual[head * Q_HEAD_DIM + value]).abs();
                if error > max_abs {
                    max_abs = error;
                    max_index = head * Q_HEAD_DIM + value;
                }
            }
        }
        println!(
            "[mla-project-w8g32-oracle] project_ms={project_ms:.3} max_abs={max_abs:.6e} max_index={max_index} reference={:.6e} actual={:.6e}",
            {
                let head = max_index / Q_HEAD_DIM;
                let value = max_index % Q_HEAD_DIM;
                let weight_row = head * KV_HEAD_DIM + Q_HEAD_DIM - ROPE_DIM + value;
                (0..LATENT_DIM).fold(0.0f32, |sum, column| {
                    let code = packed[weight_row * LATENT_DIM + column] as i32 - 128;
                    sum + f32::from_bits(u32::from(input_bits[head * LATENT_DIM + column]) << 16) * code as f32 * half::f16::from_le_bytes(scale_bytes[weight_row * groups + column / GROUP_SIZE]).to_f32()
                })
            },
            actual[max_index]
        );
        assert!(max_abs <= 1.0e-3, "max_abs={max_abs} index={max_index}");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_w8g32_rows2_edges_match_independent_rows1() {
        const DEVICE_ID: i32 = 0;
        const QUERY_ROWS: usize = 2;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256;
        const KV_HEAD_DIM: usize = 448;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const GROUP_SIZE: usize = 32;
        const REPEATS: usize = 100;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let query_row_elements = HEAD_COUNT * Q_HEAD_DIM;
        let query_host = (0..QUERY_ROWS * query_row_elements).map(|index| ((index * 29 % 257) as f32 - 128.0) * (1.0 / 512.0)).collect::<Vec<_>>();
        let weight_rows = HEAD_COUNT * KV_HEAD_DIM;
        let packed = (0..weight_rows * LATENT_DIM)
            .map(|index| {
                let row = index / LATENT_DIM;
                let column = index % LATENT_DIM;
                ((row * 17 + column * 13) % 255) as u8
            })
            .collect::<Vec<_>>();
        let groups = LATENT_DIM / GROUP_SIZE;
        let scale_bits = (0..weight_rows * groups).map(|index| half::f16::from_f32(((index * 7 % 5 + 1) as f32) * (1.0 / 512.0)).to_bits()).collect::<Vec<_>>();
        let query_pair = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query_host)).unwrap();
        let query_rows = [DeviceBuffer::upload(DEVICE_ID, as_bytes(&query_host[..query_row_elements])).unwrap(), DeviceBuffer::upload(DEVICE_ID, as_bytes(&query_host[query_row_elements..])).unwrap()];
        let packed = DeviceBuffer::upload(DEVICE_ID, &packed).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scale_bits)).unwrap();
        let absorbed_pair = DeviceBuffer::allocate(DEVICE_ID, QUERY_ROWS * HEAD_COUNT * LATENT_DIM * 2).unwrap();
        let absorbed_rows = [DeviceBuffer::allocate(DEVICE_ID, HEAD_COUNT * LATENT_DIM * 2).unwrap(), DeviceBuffer::allocate(DEVICE_ID, HEAD_COUNT * LATENT_DIM * 2).unwrap()];
        let functions = paged_mla_functions(DEVICE_ID).unwrap();
        let launch_absorb = |query: &DeviceBuffer, rows: u32, output: &DeviceBuffer| {
            let mut d_query = query.pointer;
            let mut d_packed = packed.pointer;
            let mut d_scales = scales.pointer;
            let mut d_output = output.pointer;
            let mut rows = rows;
            let mut heads = HEAD_COUNT as u32;
            let mut q_head = Q_HEAD_DIM as u32;
            let mut kv_head = KV_HEAD_DIM as u32;
            let mut latent = LATENT_DIM as u32;
            let mut rope = ROPE_DIM as u32;
            let mut group = GROUP_SIZE as u32;
            let mut scale_dtype = 1u32;
            let mut bits = 8u32;
            let mut args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_packed as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_output as *mut *mut c_void).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut q_head as *mut u32).cast(),
                (&mut kv_head as *mut u32).cast(),
                (&mut latent as *mut u32).cast(),
                (&mut rope as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut scale_dtype as *mut u32).cast(),
                (&mut bits as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.absorb_query, (HEAD_COUNT * (LATENT_DIM / 64)) as u32, 1, 256, 0, &mut args, "HIP MLA W8G32 rows2 absorb oracle").unwrap();
        };
        launch_absorb(&query_pair, 2, &absorbed_pair);
        launch_absorb(&query_rows[0], 1, &absorbed_rows[0]);
        launch_absorb(&query_rows[1], 1, &absorbed_rows[1]);
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W8G32 rows2 absorb oracle").unwrap();
        let started = std::time::Instant::now();
        for _ in 0..REPEATS {
            launch_absorb(&query_pair, 2, &absorbed_pair);
        }
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W8G32 rows2 absorb bench").unwrap();
        let absorb_us = started.elapsed().as_secs_f64() * 1e6 / REPEATS as f64;
        let pair_absorbed = absorbed_pair.download_u16(QUERY_ROWS * HEAD_COUNT * LATENT_DIM).unwrap();
        for row in 0..QUERY_ROWS {
            let reference = absorbed_rows[row].download_u16(HEAD_COUNT * LATENT_DIM).unwrap();
            assert_eq!(&pair_absorbed[row * reference.len()..(row + 1) * reference.len()], reference, "absorb row={row}");
        }

        let output_pair = DeviceBuffer::allocate(DEVICE_ID, QUERY_ROWS * query_row_elements * 2).unwrap();
        let output_rows = [DeviceBuffer::allocate(DEVICE_ID, query_row_elements * 4).unwrap(), DeviceBuffer::allocate(DEVICE_ID, query_row_elements * 4).unwrap()];
        let launch_project = |input: &DeviceBuffer, rows: u32, output: &DeviceBuffer| {
            let mut d_input = input.pointer;
            let mut d_packed = packed.pointer;
            let mut d_scales = scales.pointer;
            let mut d_output = output.pointer;
            let mut rows = rows;
            let mut heads = HEAD_COUNT as u32;
            let mut weight_head_start = 0u32;
            let mut q_head = Q_HEAD_DIM as u32;
            let mut kv_head = KV_HEAD_DIM as u32;
            let mut latent = LATENT_DIM as u32;
            let mut rope = ROPE_DIM as u32;
            let mut group = GROUP_SIZE as u32;
            let mut scale_dtype = 1u32;
            let mut bits = 8u32;
            let mut args = [
                (&mut d_input as *mut *mut c_void).cast(),
                (&mut d_packed as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_output as *mut *mut c_void).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut weight_head_start as *mut u32).cast(),
                (&mut q_head as *mut u32).cast(),
                (&mut kv_head as *mut u32).cast(),
                (&mut latent as *mut u32).cast(),
                (&mut rope as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut scale_dtype as *mut u32).cast(),
                (&mut bits as *mut u32).cast(),
            ];
            let value_dim = KV_HEAD_DIM - (Q_HEAD_DIM - ROPE_DIM);
            launch_moe_kernel(functions.project_value, (HEAD_COUNT * value_dim.div_ceil(16)) as u32, 1, 256, 0, &mut args, "HIP MLA W8G32 rows2 project oracle").unwrap();
        };
        launch_project(&absorbed_pair, 2, &output_pair);
        launch_project(&absorbed_rows[0], 1, &output_rows[0]);
        launch_project(&absorbed_rows[1], 1, &output_rows[1]);
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W8G32 rows2 project oracle").unwrap();
        let started = std::time::Instant::now();
        for _ in 0..REPEATS {
            launch_project(&absorbed_pair, 2, &output_pair);
        }
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W8G32 rows2 project bench").unwrap();
        let project_us = started.elapsed().as_secs_f64() * 1e6 / REPEATS as f64;
        let pair_output = output_pair.download_u16(QUERY_ROWS * query_row_elements).unwrap();
        for row in 0..QUERY_ROWS {
            let reference = output_rows[row].download_f32(query_row_elements).unwrap();
            let reference = reference.into_iter().map(|value| half::bf16::from_f32(value).to_bits()).collect::<Vec<_>>();
            assert_eq!(&pair_output[row * query_row_elements..(row + 1) * query_row_elements], reference, "project row={row}");
        }
        eprintln!("[mla-rows2-edge-oracle] absorb_us={absorb_us:.1} project_us={project_us:.1}");
    }

    /// PV perm 直读臂 oracle：zllm_w8_perm_f16 构造 f16(1024+u) +
    /// zllm_f16x2_dot2 累加，-1152·Σx_g 修偏置。与 W8G32 标量臂同 CPU 参考，
    /// bf16 容差族（Mac 端 numpy 仿真 max_abs≈6e-5），门槛沿用 1e-3。
    /// 生产选择器暂不切换；与 W8G32 oracle 同基准计时可直接 A/B。
    /// `cargo test --release --features with-rocm mla_project_value_w8g32_perm -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_project_value_w8g32_perm_decode_matches_cpu_oracle_and_reports_latency() {
        const DEVICE_ID: i32 = 0;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256;
        const KV_HEAD_DIM: usize = 448;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const GROUP_SIZE: usize = 32;
        const REPEATS: usize = 50;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let input_bits = (0..HEAD_COUNT * LATENT_DIM)
            .map(|index| {
                let value = ((index * 29 % 257) as f32 - 128.0) * (1.0 / 512.0);
                (value.to_bits() >> 16) as u16
            })
            .collect::<Vec<_>>();
        let weight_rows = HEAD_COUNT * KV_HEAD_DIM;
        let packed = (0..weight_rows * LATENT_DIM)
            .map(|index| {
                let row = index / LATENT_DIM;
                let column = index % LATENT_DIM;
                ((row * 17 + column * 13) % 255) as u8
            })
            .collect::<Vec<_>>();
        let groups = LATENT_DIM / GROUP_SIZE;
        let scale_bytes = (0..weight_rows * groups).map(|index| half::f16::from_f32(((index * 7 % 5 + 1) as f32) * (1.0 / 512.0)).to_le_bytes()).collect::<Vec<_>>();
        let input = DeviceBuffer::upload(DEVICE_ID, as_bytes(&input_bits)).unwrap();
        let packed_device = DeviceBuffer::upload(DEVICE_ID, &packed).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&scale_bytes)).unwrap();
        let output = DeviceBuffer::allocate(DEVICE_ID, HEAD_COUNT * Q_HEAD_DIM * 4).unwrap();
        let functions = paged_mla_functions(DEVICE_ID).unwrap();
        let mut d_input = input.pointer;
        let mut d_packed = packed_device.pointer;
        let mut d_scales = scales.pointer;
        let mut d_output = output.pointer;
        let mut rows = 1u32;
        let mut heads = HEAD_COUNT as u32;
        let mut q_head = Q_HEAD_DIM as u32;
        let mut kv_head = KV_HEAD_DIM as u32;
        let mut latent = LATENT_DIM as u32;
        let mut rope = ROPE_DIM as u32;
        let mut group = GROUP_SIZE as u32;
        let mut scale_dtype = 1u32;
        let mut bits = 8u32;
        let mut weight_head_start = 0u32;
        let mut args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_packed as *mut *mut c_void).cast(),
            (&mut d_scales as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut heads as *mut u32).cast(),
            (&mut weight_head_start as *mut u32).cast(),
            (&mut q_head as *mut u32).cast(),
            (&mut kv_head as *mut u32).cast(),
            (&mut latent as *mut u32).cast(),
            (&mut rope as *mut u32).cast(),
            (&mut group as *mut u32).cast(),
            (&mut scale_dtype as *mut u32).cast(),
            (&mut bits as *mut u32).cast(),
        ];
        let value_dim = KV_HEAD_DIM - (Q_HEAD_DIM - ROPE_DIM);
        let grid = (HEAD_COUNT * value_dim.div_ceil(16)) as u32;
        // 动态 LDS：本 head 的 f16 输入行 + 每 group 偏置（512×2 + 16×4 = 1088B）。
        let shared = (LATENT_DIM * 2 + groups * 4) as u32;
        launch_moe_kernel(functions.project_value_perm, grid, 1, 256, shared, &mut args, "HIP MLA W8G32 perm project oracle warmup").unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W8G32 perm project oracle warmup").unwrap();
        let started = std::time::Instant::now();
        for _ in 0..REPEATS {
            launch_moe_kernel(functions.project_value_perm, grid, 1, 256, shared, &mut args, "HIP MLA W8G32 perm project oracle").unwrap();
        }
        super::super::synchronize_device(DEVICE_ID, "HIP MLA W8G32 perm project oracle").unwrap();
        let project_ms = started.elapsed().as_secs_f64() * 1e3 / REPEATS as f64;

        let input_host = input_bits.iter().map(|bits| f32::from_bits(u32::from(*bits) << 16)).collect::<Vec<_>>();
        let scale_host = scale_bytes.iter().map(|bytes| half::f16::from_le_bytes(*bytes).to_f32()).collect::<Vec<_>>();
        let mut max_abs = 0.0f32;
        let mut max_index = 0usize;
        let mut actual = vec![0.0f32; HEAD_COUNT * Q_HEAD_DIM];
        output.copy_to_host(as_bytes_mut(&mut actual)).unwrap();
        for head in 0..HEAD_COUNT {
            for value in 0..value_dim {
                let weight_row = head * KV_HEAD_DIM + Q_HEAD_DIM - ROPE_DIM + value;
                let mut sum = 0.0f32;
                for column in 0..LATENT_DIM {
                    let code = packed[weight_row * LATENT_DIM + column] as i32 - 128;
                    sum += input_host[head * LATENT_DIM + column] * code as f32 * scale_host[weight_row * groups + column / GROUP_SIZE];
                }
                let error = (sum - actual[head * Q_HEAD_DIM + value]).abs();
                if error > max_abs {
                    max_abs = error;
                    max_index = head * Q_HEAD_DIM + value;
                }
            }
        }
        println!("[mla-project-w8g32-perm-oracle] project_ms={project_ms:.3} max_abs={max_abs:.6e} max_index={max_index}");
        assert!(max_abs <= 1.0e-3, "max_abs={max_abs} index={max_index}");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_project_value_f32_decode_matches_cpu_oracle_and_reports_latency() {
        const DEVICE_ID: i32 = 0;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256;
        const KV_HEAD_DIM: usize = 448;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const REPEATS: usize = 50;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let input_bits = (0..HEAD_COUNT * LATENT_DIM)
            .map(|index| {
                let value = ((index * 29 % 257) as f32 - 128.0) * (1.0 / 512.0);
                (value.to_bits() >> 16) as u16
            })
            .collect::<Vec<_>>();
        // GLM-5.3 生产 kv_b：W8A16 解码后的 F32 dense（bits=32）。
        let weight_rows = HEAD_COUNT * KV_HEAD_DIM;
        let weight: Vec<f32> = (0..weight_rows * LATENT_DIM).map(|index| (((index * 13) % 31) as f32 - 15.0) / 64.0).collect();
        let input = DeviceBuffer::upload(DEVICE_ID, as_bytes(&input_bits)).unwrap();
        let weight_device = DeviceBuffer::upload(DEVICE_ID, as_bytes(&weight)).unwrap();
        let output = DeviceBuffer::allocate(DEVICE_ID, HEAD_COUNT * Q_HEAD_DIM * 4).unwrap();
        let functions = paged_mla_functions(DEVICE_ID).unwrap();
        let mut d_input = input.pointer;
        let mut d_packed = weight_device.pointer;
        let mut d_scales = weight_device.pointer;
        let mut d_output = output.pointer;
        let mut rows = 1u32;
        let mut heads = HEAD_COUNT as u32;
        let mut weight_head_start = 0u32;
        let mut q_head = Q_HEAD_DIM as u32;
        let mut kv_head = KV_HEAD_DIM as u32;
        let mut latent = LATENT_DIM as u32;
        let mut rope = ROPE_DIM as u32;
        let mut group = LATENT_DIM as u32;
        let mut scale_dtype = 2u32;
        let mut bits = 32u32;
        let mut args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_packed as *mut *mut c_void).cast(),
            (&mut d_scales as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut heads as *mut u32).cast(),
            (&mut weight_head_start as *mut u32).cast(),
            (&mut q_head as *mut u32).cast(),
            (&mut kv_head as *mut u32).cast(),
            (&mut latent as *mut u32).cast(),
            (&mut rope as *mut u32).cast(),
            (&mut group as *mut u32).cast(),
            (&mut scale_dtype as *mut u32).cast(),
            (&mut bits as *mut u32).cast(),
        ];
        let value_dim = KV_HEAD_DIM - (Q_HEAD_DIM - ROPE_DIM);
        let grid = (HEAD_COUNT * value_dim.div_ceil(16)) as u32;
        launch_moe_kernel(functions.project_value, grid, 1, 256, 0, &mut args, "HIP MLA F32 project oracle warmup").unwrap();
        super::super::synchronize_device(DEVICE_ID, "HIP MLA F32 project oracle warmup").unwrap();
        let started = std::time::Instant::now();
        for _ in 0..REPEATS {
            launch_moe_kernel(functions.project_value, grid, 1, 256, 0, &mut args, "HIP MLA F32 project oracle").unwrap();
        }
        super::super::synchronize_device(DEVICE_ID, "HIP MLA F32 project oracle").unwrap();
        let project_ms = started.elapsed().as_secs_f64() * 1e3 / REPEATS as f64;

        let input_host = input_bits.iter().map(|bits| f32::from_bits(u32::from(*bits) << 16)).collect::<Vec<_>>();
        let mut actual = vec![0.0f32; HEAD_COUNT * Q_HEAD_DIM];
        output.copy_to_host(as_bytes_mut(&mut actual)).unwrap();
        let mut max_abs = 0.0f32;
        let mut max_index = 0usize;
        for head in 0..HEAD_COUNT {
            for value in 0..value_dim {
                let weight_row = head * KV_HEAD_DIM + Q_HEAD_DIM - ROPE_DIM + value;
                let sum: f32 = (0..LATENT_DIM).map(|column| input_host[head * LATENT_DIM + column] * weight[weight_row * LATENT_DIM + column]).sum();
                let error = (sum - actual[head * Q_HEAD_DIM + value]).abs();
                if error > max_abs {
                    max_abs = error;
                    max_index = head * Q_HEAD_DIM + value;
                }
            }
        }
        println!("[mla-project-f32-oracle] project_ms={project_ms:.3} max_abs={max_abs:.6e} max_index={max_index}");
        assert!(max_abs <= 1.0e-3, "max_abs={max_abs} index={max_index}");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn paged_workspace_growth_reuses_storage_and_keeps_live_readers() {
        super::super::configure(super::super::RocmOptions::default()).unwrap();
        let mut buffer = None;
        let mut capacity = 0;
        let original = reserve_paged_dsa_buffer(&mut buffer, &mut capacity, 0, 200000).unwrap();
        original.copy_from_host(&[17, 29, 61, 101]).unwrap();
        for bytes in [200004, 208192, 262144] {
            let next = reserve_paged_dsa_buffer(&mut buffer, &mut capacity, 0, bytes).unwrap();
            assert_eq!(next.device_pointer(), original.device_pointer());
            assert!(next.bytes() >= bytes);
        }
        let larger = reserve_paged_dsa_buffer(&mut buffer, &mut capacity, 0, 262148).unwrap();
        assert_ne!(larger.device_pointer(), original.device_pointer());
        // 扩容后的旧读者仍持有来源，不能把尚在使用的 allocation 提前归还。
        let mut bytes = [0; 4];
        original.copy_to_host(&mut bytes).unwrap();
        assert_eq!(bytes, [17, 29, 61, 101]);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_selected_gather_nonidentity_pages_match_serial() {
        const DEVICE: i32 = 0;
        const CONTEXT: usize = 2176;
        const BLOCK: usize = 128;
        const HEADS: usize = 16;
        const Q_HEAD: usize = 96;
        const KV_HEAD: usize = 64;
        const LATENT: usize = 512;
        const GROUP: usize = 64;
        const ROPE: usize = 64;
        super::super::configure(super::super::RocmOptions::default()).unwrap();
        TEST_MLA_DECODE_WMMA.store(true, std::sync::atomic::Ordering::Relaxed);
        let bf16 = |value: f32| (value.to_bits() >> 16) as u16;
        let query = DeviceBuffer::upload_f32(DEVICE, &(0..HEADS * Q_HEAD).map(|i| ((i % 31) as f32 - 15.0) / 32.0).collect::<Vec<_>>()).unwrap();
        let weight = (0..HEADS * KV_HEAD * LATENT).map(|i| bf16(((i * 17 % 127) as f32 - 63.0) / 4096.0)).collect::<Vec<_>>();
        let weight = DeviceBuffer::upload(DEVICE, as_bytes(&weight)).unwrap();
        let weight_scales = DeviceBuffer::upload(DEVICE, as_bytes(&[bf16(1.0)])).unwrap();
        let latent = (0..CONTEXT * LATENT).map(|i| ((i * 29 + i / LATENT * 7) % 63 + 1) as u8).collect::<Vec<_>>();
        let latent = DeviceBuffer::upload(DEVICE, &latent).unwrap();
        let scales = DeviceBuffer::upload(DEVICE, as_bytes(&vec![bf16(1.0 / 256.0); CONTEXT * (LATENT / GROUP)])).unwrap();
        let rope = (0..CONTEXT * ROPE).map(|i| bf16(((i * 13 % 127) as f32 - 63.0) / 128.0)).collect::<Vec<_>>();
        let rope = DeviceBuffer::upload(DEVICE, as_bytes(&rope)).unwrap();
        // 首个逻辑页位于物理末页；重排后误用源页表会越过紧凑 buffer。
        let table = (0..(CONTEXT / BLOCK) as u32).rev().collect::<Vec<_>>();
        let table = DeviceBuffer::upload(DEVICE, as_bytes(&table)).unwrap();
        let serial = DeviceBuffer::allocate(DEVICE, HEADS * Q_HEAD * 4).unwrap();
        let gathered = DeviceBuffer::allocate(DEVICE, HEADS * Q_HEAD * 4).unwrap();
        // 增长、缩短并改选集，覆盖 workspace 复用和近似 rollback 后的重新读取。
        for (generation, top_k) in [1024, 2048, 1024].into_iter().enumerate() {
            let selection = (0..top_k).map(|i| ((i * 251 + generation * 17) % CONTEXT) as u32).collect::<Vec<_>>();
            let selection = DeviceBuffer::upload(DEVICE, as_bytes(&selection)).unwrap();
            for (output, split) in [(&serial, false), (&gathered, true)] {
                try_paged_mla_attention_ct_into(
                    DEVICE,
                    &query,
                    &latent,
                    Some(&scales),
                    GROUP,
                    &rope,
                    &table,
                    Some(&selection),
                    CtMlaWeightRef { packed: &weight, scales: &weight_scales, rows: HEADS * KV_HEAD, cols: LATENT, group_size: LATENT, scale_dtype: 0, bits: 16 },
                    1,
                    CONTEXT,
                    CONTEXT - 1,
                    HEADS * Q_HEAD,
                    HEADS,
                    ROPE,
                    top_k,
                    BLOCK,
                    output,
                    Some(split),
                )
                .unwrap();
            }
            let mut expected = vec![0.0_f32; HEADS * Q_HEAD];
            let mut actual = expected.clone();
            serial.copy_to_host(as_bytes_mut(&mut expected)).unwrap();
            gathered.copy_to_host(as_bytes_mut(&mut actual)).unwrap();
            for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(actual.is_finite() && (actual - expected).abs() <= 1.0e-2 + 1.0e-2 * expected.abs(), "generation={generation} top_k={top_k} element={i} actual={actual} expected={expected}");
            }
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_decode_split_matches_serial_and_reports_latency() {
        const DEVICE_ID: i32 = 0;
        const Q_HEAD_DIM: usize = 256;
        const LATENT_DIM: usize = 512;
        const KV_HEAD_DIM: usize = 448;
        const ROPE_DIM: usize = 64;
        const LATENT_GROUP: usize = 64;
        const BLOCK_SIZE: usize = 128;

        super::super::configure(super::super::RocmOptions { mla_decode_tile_size: 64, ..Default::default() }).unwrap();

        // head_count=32 复现 cooperative 半头拆分下每卡的 grid.x=1 形态；
        // 131072 上下文让 WMMA split 超过旧 merge 上限 128，验证扩容后的正确性。
        // wmma=off 强制走 fdot2 标量路径，覆盖 Step2 的 staged LDS 流水。
        // 元组末位 weight_bits：16=BF16 dense（历史路径）；8=W8A16 G32 F16-scale
        // （生产新驻留形态，覆盖 absorb W8V 臂 + PV W8 快路径的全链路）。
        for (mode, latent_group, head_count, wmma, weight_bits) in [
            ("q8g64", LATENT_GROUP, 64_usize, true, 16_u32),
            ("q8g64", LATENT_GROUP, 64, true, 8),
            ("q8g64", LATENT_GROUP, 32, true, 16),
            ("q8g64", LATENT_GROUP, 32, true, 8),
            ("f16", 0, 64, true, 16),
            ("q8g64", LATENT_GROUP, 64, false, 16),
            ("q8g64", LATENT_GROUP, 32, false, 16),
            ("f16", 0, 64, false, 16),
        ] {
            TEST_MLA_DECODE_WMMA.store(wmma, std::sync::atomic::Ordering::Relaxed);
            for context_rows in [64_usize, 128, 256, 512, 768, 1357, 2048, 16384, 131072] {
                let q_projection = head_count * Q_HEAD_DIM;
                let kv_projection = head_count * KV_HEAD_DIM;
                let query = (0..q_projection).map(|index| ((index % 257) as f32 - 128.0) * (1.0 / 256.0)).collect::<Vec<_>>();
                let weight = (0..kv_projection * LATENT_DIM).map(|index| (((index * 17) % 127) as f32 - 63.0) * (1.0 / 4096.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
                let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query)).unwrap();
                // W8 臂：偏移二进制 codes（值域收窄到 ±7，配合 0.02 scale 让
                // 输出幅值与 BF16 臂同级——绝对容差断言才不会被幅值放大）。
                let (weight, scales, weight_group, weight_scale_dtype) = if weight_bits == 8 {
                    let codes = (0..kv_projection * LATENT_DIM).map(|index| ((index * 17) % 15 + 121) as u8).collect::<Vec<_>>();
                    let scale_bits = half::f16::from_f32(0.02).to_bits();
                    let scales = vec![scale_bits; kv_projection * (LATENT_DIM / 32)];
                    (DeviceBuffer::upload(DEVICE_ID, &codes).unwrap(), DeviceBuffer::upload(DEVICE_ID, as_bytes(&scales)).unwrap(), 32_usize, 1_u32)
                } else {
                    (DeviceBuffer::upload(DEVICE_ID, as_bytes(&weight)).unwrap(), DeviceBuffer::upload(DEVICE_ID, as_bytes(&[0x3f80_u16])).unwrap(), LATENT_DIM, 0_u32)
                };
                let latent_bytes = if latent_group == 0 {
                    let values = (0..context_rows * LATENT_DIM).map(|index| (((index * 29 + index / LATENT_DIM * 7) % 127) as f32 - 63.0) * (1.0 / 64.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
                    as_bytes(&values).to_vec()
                } else {
                    (0..context_rows * LATENT_DIM).map(|index| ((index * 29 + index / LATENT_DIM * 7) % 63 + 1) as u8).collect::<Vec<_>>()
                };
                let latent_scales = (latent_group != 0).then(|| vec![0x3b80_u16; context_rows * (LATENT_DIM / LATENT_GROUP)]);
                let rope = (0..context_rows * ROPE_DIM).map(|index| (((index * 13) % 127) as f32 - 63.0) * (1.0 / 128.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
                let table = (0..context_rows.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
                let top_k = context_rows.min(2048);
                let selection = (0..top_k).map(|index| ((index * 251 + 17) % context_rows) as u32).collect::<Vec<_>>();
                let latent = DeviceBuffer::upload(DEVICE_ID, &latent_bytes).unwrap();
                let latent_scales = latent_scales.as_ref().map(|scales| DeviceBuffer::upload(DEVICE_ID, as_bytes(scales)).unwrap());
                let rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
                let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
                let selection = DeviceBuffer::upload(DEVICE_ID, as_bytes(&selection)).unwrap();
                let serial = DeviceBuffer::allocate(DEVICE_ID, q_projection * 4).unwrap();
                let split = DeviceBuffer::allocate(DEVICE_ID, q_projection * 4).unwrap();

                let run = |output: &DeviceBuffer, split_decode| {
                    let started = std::time::Instant::now();
                    try_paged_mla_attention_ct_into(
                        DEVICE_ID,
                        &query,
                        &latent,
                        latent_scales.as_ref(),
                        latent_group,
                        &rope,
                        &table,
                        Some(&selection),
                        CtMlaWeightRef { packed: &weight, scales: &scales, rows: kv_projection, cols: LATENT_DIM, group_size: weight_group, scale_dtype: weight_scale_dtype, bits: weight_bits },
                        1,
                        context_rows,
                        context_rows - 1,
                        q_projection,
                        head_count,
                        ROPE_DIM,
                        top_k,
                        BLOCK_SIZE,
                        output,
                        Some(split_decode),
                    )
                    .unwrap();
                    super::super::synchronize_device(DEVICE_ID, "HIP MLA split oracle").unwrap();
                    started.elapsed().as_secs_f64() * 1e3
                };
                run(&serial, false);
                run(&split, true);
                let serial_ms = (0..3).map(|_| run(&serial, false)).sum::<f64>() / 3.0;
                let split_ms = (0..3).map(|_| run(&split, true)).sum::<f64>() / 3.0;

                let mut serial_host = vec![0_f32; q_projection];
                let mut split_host = vec![0_f32; q_projection];
                serial.copy_to_host(as_bytes_mut(&mut serial_host)).unwrap();
                split.copy_to_host(as_bytes_mut(&mut split_host)).unwrap();
                let mut max_abs = 0.0_f32;
                let mut squared = 0.0_f64;
                for (index, (reference, candidate)) in serial_host.iter().zip(&split_host).enumerate() {
                    assert!(reference.is_finite() && candidate.is_finite(), "mode={mode} context={context_rows} index={index} serial={reference} split={candidate}");
                    let error = (reference - candidate).abs();
                    max_abs = max_abs.max(error);
                    squared += f64::from(error) * f64::from(error);
                }
                let rmse = (squared / q_projection as f64).sqrt();
                println!(
                    "[mla-split-oracle] mode={mode} bits={weight_bits} heads={head_count} wmma={wmma} context={context_rows} serial_ms={serial_ms:.3} split_ms={split_ms:.3} speedup={:.3} max_abs={max_abs:.6e} rmse={rmse:.6e}",
                    serial_ms / split_ms
                );
                let tolerance = if latent_group == 0 { 1.0e-2 } else { 5.0e-3 };
                assert!(max_abs <= tolerance, "mode={mode} context={context_rows} max_abs={max_abs}");
            }
        }
        TEST_MLA_DECODE_WMMA.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_rows2_decode_split_matches_sparse_wmma_and_reports_latency() {
        const DEVICE_ID: i32 = 0;
        const QUERY_ROWS: usize = 2;
        const CONTEXT_ROWS: usize = 4_096;
        const TOP_K: usize = 2_048;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256;
        const Q_PROJECTION: usize = HEAD_COUNT * Q_HEAD_DIM;
        const LATENT_DIM: usize = 512;
        const KV_HEAD_DIM: usize = 448;
        const KV_PROJECTION: usize = HEAD_COUNT * KV_HEAD_DIM;
        const ROPE_DIM: usize = 64;
        const LATENT_GROUP: usize = 64;
        const WEIGHT_GROUP: usize = 32;
        const BLOCK_SIZE: usize = 128;

        super::super::configure(super::super::RocmOptions::default()).unwrap();
        TEST_MLA_DECODE_WMMA.store(true, std::sync::atomic::Ordering::Relaxed);
        TEST_SPARSE_PREFILL_WMMA.store(true, std::sync::atomic::Ordering::Relaxed);

        let query = (0..QUERY_ROWS * Q_PROJECTION).map(|index| ((index % 257) as f32 - 128.0) * (1.0 / 256.0)).collect::<Vec<_>>();
        let latent = (0..CONTEXT_ROWS * LATENT_DIM).map(|index| (((index * 29 + index / LATENT_DIM * 7) % 127) as i16 - 63) as i8 as u8).collect::<Vec<_>>();
        let latent_scales = vec![0x3b80_u16; CONTEXT_ROWS * (LATENT_DIM / LATENT_GROUP)];
        let rope = (0..CONTEXT_ROWS * ROPE_DIM).map(|index| (((index * 13) % 127) as f32 - 63.0) * (1.0 / 128.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
        let table = (0..CONTEXT_ROWS.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
        let query_start = CONTEXT_ROWS - QUERY_ROWS;
        let selection = (0..QUERY_ROWS)
            .flat_map(|row| {
                let visible = query_start + row + 1;
                (0..TOP_K).map(move |index| ((index * 251 + row * 17) % visible) as u32)
            })
            .collect::<Vec<_>>();
        let weight = (0..KV_PROJECTION * LATENT_DIM).map(|index| (121 + (index * 17 + index / 7) % 15) as u8).collect::<Vec<_>>();
        let weight_scales = vec![half::f16::from_f32(0.02).to_bits(); KV_PROJECTION * (LATENT_DIM / WEIGHT_GROUP)];
        let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query)).unwrap();
        let latent = DeviceBuffer::upload(DEVICE_ID, &latent).unwrap();
        let latent_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&latent_scales)).unwrap();
        let rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
        let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
        let selection = DeviceBuffer::upload(DEVICE_ID, as_bytes(&selection)).unwrap();
        let weight = DeviceBuffer::upload(DEVICE_ID, &weight).unwrap();
        let weight_scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&weight_scales)).unwrap();
        let serial = DeviceBuffer::allocate(DEVICE_ID, QUERY_ROWS * Q_PROJECTION * 2).unwrap();
        let split = DeviceBuffer::allocate(DEVICE_ID, QUERY_ROWS * Q_PROJECTION * 2).unwrap();

        let run = |output: &DeviceBuffer, split_decode| {
            let started = std::time::Instant::now();
            try_paged_mla_attention_ct_into(
                DEVICE_ID,
                &query,
                &latent,
                Some(&latent_scales),
                LATENT_GROUP,
                &rope,
                &table,
                Some(&selection),
                CtMlaWeightRef { packed: &weight, scales: &weight_scales, rows: KV_PROJECTION, cols: LATENT_DIM, group_size: WEIGHT_GROUP, scale_dtype: 1, bits: 8 },
                QUERY_ROWS,
                CONTEXT_ROWS,
                query_start,
                Q_PROJECTION,
                HEAD_COUNT,
                ROPE_DIM,
                TOP_K,
                BLOCK_SIZE,
                output,
                Some(split_decode),
            )
            .unwrap();
            super::super::synchronize_device(DEVICE_ID, "HIP MLA rows2 split oracle").unwrap();
            started.elapsed().as_secs_f64() * 1e3
        };
        run(&serial, false);
        run(&split, true);
        let serial_ms = (0..3).map(|_| run(&serial, false)).sum::<f64>() / 3.0;
        let split_ms = (0..3).map(|_| run(&split, true)).sum::<f64>() / 3.0;
        let serial = serial.download_u16(QUERY_ROWS * Q_PROJECTION).unwrap();
        let split = split.download_u16(QUERY_ROWS * Q_PROJECTION).unwrap();
        let mut max_abs = 0.0_f32;
        let mut squared = 0.0_f64;
        for (index, (&reference, &candidate)) in serial.iter().zip(&split).enumerate() {
            let reference = f32::from_bits(u32::from(reference) << 16);
            let candidate = f32::from_bits(u32::from(candidate) << 16);
            assert!(reference.is_finite() && candidate.is_finite(), "index={index} serial={reference} split={candidate}");
            let error = (reference - candidate).abs();
            max_abs = max_abs.max(error);
            squared += f64::from(error) * f64::from(error);
        }
        let rmse = (squared / serial.len() as f64).sqrt();
        eprintln!("[mla-rows2-split-oracle] serial_ms={serial_ms:.3} split_ms={split_ms:.3} speedup={:.3} max_abs={max_abs:.6e} rmse={rmse:.6e}", serial_ms / split_ms);
        assert!(max_abs <= 2.0e-2, "max_abs={max_abs}");
    }
}

#[cfg(test)]
mod absorb_bench {
    /// absorb_query decode kernel 微基准：真实形状 64 头、nope=512、latent=576，
    /// F32 dense resident（stride kv_head_dim=576 的读取模式）。
    /// `cargo test --release --features with-rocm absorb_query_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn absorb_query_bench() {
        use super::*;
        const DEVICE: i32 = 0;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 576;
        const KV_HEAD_DIM: usize = 576;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        super::super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();
        // query F32 [head, q_head_dim]
        let query: Vec<f32> = (0..HEAD_COUNT * Q_HEAD_DIM).map(|index| (((index as i32 * 37) % 41 - 20) as f32) / 32.0).collect();
        let query_device = DeviceBuffer::upload(DEVICE, unsafe { std::slice::from_raw_parts(query.as_ptr().cast::<u8>(), query.len() * 4) }).unwrap();
        // dense F32 kv_b [head * kv_head_dim, latent_dim]（与生产同布局）
        let weight: Vec<f32> = (0..HEAD_COUNT * KV_HEAD_DIM * LATENT_DIM).map(|index| (((index as i32 * 13) % 31 - 15) as f32) / 64.0).collect();
        let weight_device = DeviceBuffer::upload(DEVICE, unsafe { std::slice::from_raw_parts(weight.as_ptr().cast::<u8>(), weight.len() * 4) }).unwrap();
        let weight_gib = weight.len() as f64 * 4.0 / (1u64 << 30) as f64;
        let absorbed = DeviceBuffer::allocate(DEVICE, HEAD_COUNT * LATENT_DIM * 2).unwrap();
        let absorbed_bytes = HEAD_COUNT * LATENT_DIM * 2;
        // absorb kernel 的真实 grid：heads * latent_tiles、256 线程
        let functions = paged_mla_functions(DEVICE).unwrap();
        let latent_tiles = LATENT_DIM.div_ceil(16);
        let bench = |label: &str, rounds: usize| {
            let mut d_query = query_device.pointer;
            let mut d_packed = weight_device.pointer;
            let mut d_scales = weight_device.pointer; // bits=32 时不用
            let mut d_absorbed = absorbed.pointer;
            let mut rows = 1_u32;
            let mut heads = HEAD_COUNT as u32;
            let mut q_head = Q_HEAD_DIM as u32;
            let mut kv_head = KV_HEAD_DIM as u32;
            let mut latent = LATENT_DIM as u32;
            let mut rope = ROPE_DIM as u32;
            let mut group = 512_u32;
            let mut scale_dtype = 2_u32;
            let mut bits = 32_u32;
            let mut args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_packed as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_absorbed as *mut *mut c_void).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut q_head as *mut u32).cast(),
                (&mut kv_head as *mut u32).cast(),
                (&mut latent as *mut u32).cast(),
                (&mut rope as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut scale_dtype as *mut u32).cast(),
                (&mut bits as *mut u32).cast(),
            ];
            for _ in 0..3 {
                launch_moe_kernel(functions.absorb_query, (HEAD_COUNT * latent_tiles) as u32, 1, 256, 0, &mut args, "absorb bench").unwrap();
            }
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                launch_moe_kernel(functions.absorb_query, (HEAD_COUNT * latent_tiles) as u32, 1, 256, 0, &mut args, "absorb bench").unwrap();
            }
            super::super::super::synchronize_device(DEVICE, "absorb bench").unwrap();
            let micros = started.elapsed().as_micros() as f64 / rounds as f64;
            eprintln!("[absorb-bench] {label} avg_us={micros:.1} bw_GBps={:.0}", weight_gib * 1024.0 / (micros / 1e6));
        };
        let _ = absorbed_bytes;
        bench("absorb-decode", 100);
    }

    /// absorb W8 直读臂：生产真实形状（64 头、q_head=256/nope=192、kv_head=192、
    /// latent=512、rope=64），kv_b W8A16 G32 F16-scale（= fused_kv_b_w8 的产物形态）
    /// vs 同值 F32 dense。数值上两臂读同一组逻辑权重（F32 值恰为 code*scale），
    /// 预期逐位一致；计时看 W8 直读的带宽收益是否兑现（25.2MB → 6.7MB）。
    /// `cargo test --release --features with-rocm absorb_query_w8_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn absorb_query_w8_bench() {
        use super::*;
        const DEVICE: i32 = 0;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256;
        const KV_HEAD_DIM: usize = 192;
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const GROUP: usize = 32;
        const ROWS: usize = HEAD_COUNT * KV_HEAD_DIM;
        super::super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();

        let query: Vec<f32> = (0..HEAD_COUNT * Q_HEAD_DIM).map(|index| (((index as i32 * 37) % 41 - 20) as f32) / 32.0).collect();
        let query_device = DeviceBuffer::upload_f32(DEVICE, &query).unwrap();

        // W8 量化（与 fused_kv_b_w8 同法：对称 G32，scale=max/127，F16 存储）。
        let raw: Vec<f32> = (0..ROWS * LATENT_DIM).map(|index| (((index as i32 * 13) % 31 - 15) as f32) / 64.0).collect();
        let mut packed = vec![0u8; ROWS * LATENT_DIM];
        let mut scales = vec![0u16; ROWS * (LATENT_DIM / GROUP)];
        let mut dense = vec![0.0f32; ROWS * LATENT_DIM];
        for row in 0..ROWS {
            for group in 0..LATENT_DIM / GROUP {
                let base = row * LATENT_DIM + group * GROUP;
                let max_abs = raw[base..base + GROUP].iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                let scale_f16 = half::f16::from_f32(max_abs / 127.0);
                let scale = scale_f16.to_f32();
                scales[row * (LATENT_DIM / GROUP) + group] = scale_f16.to_bits();
                for k in 0..GROUP {
                    let code = if scale == 0.0 { 0i32 } else { (raw[base + k] / scale).round().clamp(-127.0, 127.0) as i32 };
                    // zllm W8A16 约定是偏移二进制（code+128），与 paged_weight 的 -128 配套。
                    packed[base + k] = (code as u8).wrapping_add(128);
                    // F32 dense 参照值与 kernel 内 dequant 逐位同序（f32(code) * f32(scale)）。
                    dense[base + k] = (code as f32) * scale;
                }
            }
        }
        let packed_device = DeviceBuffer::upload(DEVICE, &packed).unwrap();
        let scales_bytes = unsafe { std::slice::from_raw_parts(scales.as_ptr().cast::<u8>(), scales.len() * 2) };
        let scales_device = DeviceBuffer::upload(DEVICE, scales_bytes).unwrap();
        let dense_bytes = unsafe { std::slice::from_raw_parts(dense.as_ptr().cast::<u8>(), dense.len() * 4) };
        let dense_device = DeviceBuffer::upload(DEVICE, dense_bytes).unwrap();
        let absorbed = DeviceBuffer::allocate(DEVICE, HEAD_COUNT * LATENT_DIM * 2).unwrap();
        let functions = paged_mla_functions(DEVICE).unwrap();

        let run = |packed: &DeviceBuffer, scales_d: &DeviceBuffer, group: u32, scale_dtype: u32, bits: u32, rounds: usize| -> f64 {
            // W8G32 向量化臂每 block 覆盖 64 latent 列（与 kernel 门控同条件）。
            let latent_tiles = if bits == 8 && group == 32 { LATENT_DIM / 64 } else { LATENT_DIM / 16 };
            let mut d_query = query_device.pointer;
            let mut d_packed = packed.pointer;
            let mut d_scales = scales_d.pointer;
            let mut d_absorbed = absorbed.pointer;
            let mut rows = 1_u32;
            let mut heads = HEAD_COUNT as u32;
            let mut q_head = Q_HEAD_DIM as u32;
            let mut kv_head = KV_HEAD_DIM as u32;
            let mut latent = LATENT_DIM as u32;
            let mut rope = ROPE_DIM as u32;
            let mut group = group;
            let mut scale_dtype = scale_dtype;
            let mut bits = bits;
            let mut args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_packed as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_absorbed as *mut *mut c_void).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut q_head as *mut u32).cast(),
                (&mut kv_head as *mut u32).cast(),
                (&mut latent as *mut u32).cast(),
                (&mut rope as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut scale_dtype as *mut u32).cast(),
                (&mut bits as *mut u32).cast(),
            ];
            for _ in 0..3 {
                launch_moe_kernel(functions.absorb_query, (HEAD_COUNT * latent_tiles) as u32, 1, 256, 0, &mut args, "absorb w8 bench").unwrap();
            }
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                for _ in 0..rounds {
                    launch_moe_kernel(functions.absorb_query, (HEAD_COUNT * latent_tiles) as u32, 1, 256, 0, &mut args, "absorb w8 bench").unwrap();
                }
                super::super::super::synchronize_device(DEVICE, "absorb w8 bench").unwrap();
                best = best.min(started.elapsed().as_micros() as f64 / rounds as f64);
            }
            best
        };

        let f32_us = run(&dense_device, &dense_device, 512, 2, 32, 100);
        let f32_out = absorbed.download_u16(HEAD_COUNT * LATENT_DIM).unwrap();
        let w8_us = run(&packed_device, &scales_device, GROUP as u32, 1, 8, 100);
        let w8_out = absorbed.download_u16(HEAD_COUNT * LATENT_DIM).unwrap();
        let diff = f32_out.iter().zip(&w8_out).filter(|(a, b)| a != b).count();
        let f32_mb = ROWS * LATENT_DIM * 4;
        let w8_mb = ROWS * LATENT_DIM + ROWS * (LATENT_DIM / GROUP) * 2;
        eprintln!("[absorb-w8-bench] f32_dense min_us={f32_us:.1} bw_GBps={:.0}", f32_mb as f64 / 1e3 / f32_us);
        eprintln!("[absorb-w8-bench] w8_g32 min_us={w8_us:.1} bw_GBps={:.0}", w8_mb as f64 / 1e3 / w8_us);
        eprintln!("[absorb-w8-bench] bit-diff={diff}/{}", f32_out.len());
        assert!(diff == 0, "W8 直读与同值 F32 dense 应逐位一致");
    }
}

#[cfg(test)]
mod decode_scan_bench {
    /// MLA selected decode scan 探针矩阵：生产形状（64 头、latent 512 Q8G64、
    /// rope 64、top_k 2048、context 4096、paged block 64），对比
    ///   A) 生产 kernel tile=32（数值参照）
    ///   B) 生产 kernel tile=64/128（纯 launch 配置臂）
    ///   C) 列并行 QK kernel tile=32/64（8 wave 分列段 + LDS 归约，容差族）
    /// partial 与 merge 分别计时。
    /// `cargo test --release --features with-rocm mla_decode_scan_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_decode_scan_bench() {
        use super::*;
        const DEVICE: i32 = 0;
        const HEAD_COUNT: usize = 64;
        const Q_HEAD_DIM: usize = 256; // 生产 GLM-5.3 真实形状（nope 192 + rope 64）
        const LATENT_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const GROUP: usize = 64;
        const GROUPS: usize = LATENT_DIM / GROUP;
        const CONTEXT: usize = 50000;
        const TOP_K: usize = 2048;
        const BLOCK_SIZE: usize = 64;
        super::super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();

        let as_bytes = |values: &[u16]| unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) };
        let bf16 = |value: f32| (value.to_bits() >> 16) as u16;

        // query F32 [head_count, q_head_dim]
        let query: Vec<f32> = (0..HEAD_COUNT * Q_HEAD_DIM).map(|i| (((i as i32 * 37) % 41 - 20) as f32) / 32.0).collect();
        let query_device = DeviceBuffer::upload_f32(DEVICE, &query).unwrap();
        // absorbed BF16 [64, 512]
        let absorbed: Vec<u16> = (0..HEAD_COUNT * LATENT_DIM).map(|i| bf16((((i as i32 * 13) % 31 - 15) as f32) / 64.0)).collect();
        let absorbed_device = DeviceBuffer::upload(DEVICE, as_bytes(&absorbed)).unwrap();
        // latent Q8 codes [4096, 512] + scales BF16 [4096, 8]
        let codes: Vec<u8> = (0..CONTEXT * LATENT_DIM).map(|i| ((i as i32 * 29) % 255) as u8).collect();
        let latent_device = DeviceBuffer::upload(DEVICE, &codes).unwrap();
        let scales: Vec<u16> = (0..CONTEXT * GROUPS).map(|i| bf16(((i % 7) + 1) as f32 / 64.0)).collect();
        let scales_device = DeviceBuffer::upload(DEVICE, as_bytes(&scales)).unwrap();
        // rope BF16 [4096, 64]
        let rope: Vec<u16> = (0..CONTEXT * ROPE_DIM).map(|i| bf16((((i as i32 * 17) % 23 - 11) as f32) / 32.0)).collect();
        let rope_device = DeviceBuffer::upload(DEVICE, as_bytes(&rope)).unwrap();
        // block table 恒等
        let table: Vec<u32> = (0..CONTEXT.div_ceil(BLOCK_SIZE)).map(|i| i as u32).collect();
        let table_device = DeviceBuffer::upload(DEVICE, unsafe { std::slice::from_raw_parts(table.as_ptr().cast::<u8>(), table.len() * 4) }).unwrap();
        // selection：2048 个互异 token（7919 与 4096 互质）
        let selection: Vec<u32> = (0..TOP_K).map(|i| ((i * 7919) % CONTEXT) as u32).collect();
        let selection_device = DeviceBuffer::upload(DEVICE, unsafe { std::slice::from_raw_parts(selection.as_ptr().cast::<u8>(), selection.len() * 4) }).unwrap();

        let functions = paged_mla_functions(DEVICE).unwrap();
        let weighted = DeviceBuffer::allocate(DEVICE, HEAD_COUNT * LATENT_DIM * 2).unwrap();

        // 与生产 launcher 相同的 shared 计算（见 try_paged_mla_attention_ct_inner）。
        let baseline_shared = (32 * 16 + 16 * 16 + LATENT_DIM * 16) * 2 + 16 * 4 + 32 * 4 + 32 * LATENT_DIM + 32 * GROUPS * 2;
        let colpar_shared = (8 * 32 * 16 + 16 * 16 + 16 * LATENT_DIM) * 2 + 16 * 4 + 32 * 4 + 32 * LATENT_DIM + 32 * GROUPS * 2;
        assert!(baseline_shared <= 64 * 1024 && colpar_shared <= 64 * 1024);

        let run_arm = |kernel: usize, tile_size: usize, merge_kernel: usize, rounds: usize| -> (f64, f64, f64, f64) {
            let tile_count = TOP_K.div_ceil(tile_size);
            let partial = DeviceBuffer::allocate(DEVICE, tile_count * HEAD_COUNT * LATENT_DIM * 4).unwrap();
            let stats = DeviceBuffer::allocate(DEVICE, tile_count * HEAD_COUNT * 2 * 4).unwrap();
            let mut d_query = query_device.pointer;
            let mut d_absorbed = absorbed_device.pointer;
            let mut d_latent = latent_device.pointer;
            let mut d_scales = scales_device.pointer;
            let mut d_rope = rope_device.pointer;
            let mut d_table = table_device.pointer;
            let mut d_selection = selection_device.pointer;
            let mut d_counts = ptr::null_mut();
            let mut d_partial = partial.pointer;
            let mut d_stats = stats.pointer;
            let mut d_weighted_null = ptr::null_mut();
            let mut context = CONTEXT as u32;
            let mut start = (CONTEXT - 1) as u32;
            let mut heads = HEAD_COUNT as u32;
            let mut q_head = Q_HEAD_DIM as u32;
            let mut latent = LATENT_DIM as u32;
            let mut group = GROUP as u32;
            let mut rope = ROPE_DIM as u32;
            let mut topk = TOP_K as u32;
            let mut block = BLOCK_SIZE as u32;
            let mut selected = 1_u32;
            let mut tile_u32 = tile_size as u32;
            let mut split_tiles = tile_count as u32;
            let mut shard = 2_u32;
            let mut args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_absorbed as *mut *mut c_void).cast(),
                (&mut d_latent as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_rope as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut d_counts as *mut *mut c_void).cast(),
                (&mut d_partial as *mut *mut c_void).cast(),
                (&mut d_stats as *mut *mut c_void).cast(),
                (&mut d_weighted_null as *mut *mut c_void).cast(),
                (&mut context as *mut u32).cast(),
                (&mut start as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut q_head as *mut u32).cast(),
                (&mut latent as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut rope as *mut u32).cast(),
                (&mut topk as *mut u32).cast(),
                (&mut block as *mut u32).cast(),
                (&mut selected as *mut u32).cast(),
                (&mut tile_u32 as *mut u32).cast(),
                (&mut split_tiles as *mut u32).cast(),
                (&mut shard as *mut u32).cast(),
            ];
            let shared = if kernel == functions.decode_partial_wmma_q8_colpar512_shared {
                (8 * 2 * 8 * 32 * 2 + 16 * 16 + 32 * (LATENT_DIM + 8)) * 2 + (16 + 32) * 4
            } else if kernel != functions.decode_partial_wmma_q8 {
                colpar_shared
            } else {
                baseline_shared
            } as u32;
            let grid_y = tile_count as u32;
            for _ in 0..3 {
                launch_moe_kernel(kernel, 4, grid_y, 256, shared, &mut args, "scan bench partial").unwrap();
            }
            super::super::super::synchronize_device(DEVICE, "scan bench warmup").unwrap();
            // 与生产共卡运行， contention 漂移大：3 批计时取 min/avg，min 更接近净 kernel 时间。
            let mut partial_min = f64::MAX;
            let mut partial_sum = 0.0f64;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                for _ in 0..rounds {
                    launch_moe_kernel(kernel, 4, grid_y, 256, shared, &mut args, "scan bench partial").unwrap();
                }
                super::super::super::synchronize_device(DEVICE, "scan bench partial").unwrap();
                let us = started.elapsed().as_micros() as f64 / rounds as f64;
                partial_min = partial_min.min(us);
                partial_sum += us;
            }
            let partial_avg = partial_sum / 3.0;

            let mut d_weighted = weighted.pointer;
            let mut merge_partial = partial.pointer;
            let mut merge_stats = stats.pointer;
            let mut heads2 = HEAD_COUNT as u32;
            let mut latent2 = LATENT_DIM as u32;
            let mut tiles2 = tile_count as u32;
            let mut merged_stats_null = ptr::null_mut();
            let mut merge_args = [
                (&mut merge_partial as *mut *mut c_void).cast(),
                (&mut merge_stats as *mut *mut c_void).cast(),
                (&mut d_weighted as *mut *mut c_void).cast(),
                (&mut heads2 as *mut u32).cast(),
                (&mut latent2 as *mut u32).cast(),
                (&mut tiles2 as *mut u32).cast(),
                (&mut merged_stats_null as *mut *mut c_void).cast(),
            ];
            for _ in 0..3 {
                launch_moe_kernel(merge_kernel, (HEAD_COUNT * (LATENT_DIM / 128)) as u32, 1, 128, 0, &mut merge_args, "scan bench merge").unwrap();
            }
            super::super::super::synchronize_device(DEVICE, "scan bench merge warmup").unwrap();
            let mut merge_min = f64::MAX;
            let mut merge_sum = 0.0f64;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                for _ in 0..rounds {
                    launch_moe_kernel(merge_kernel, (HEAD_COUNT * (LATENT_DIM / 128)) as u32, 1, 128, 0, &mut merge_args, "scan bench merge").unwrap();
                }
                super::super::super::synchronize_device(DEVICE, "scan bench merge").unwrap();
                let us = started.elapsed().as_micros() as f64 / rounds as f64;
                merge_min = merge_min.min(us);
                merge_sum += us;
            }
            let merge_avg = merge_sum / 3.0;
            (partial_min, partial_avg, merge_min, merge_avg)
        };

        let compare = |label: &str, reference: &[u16]| {
            let current = weighted.download_u16(HEAD_COUNT * LATENT_DIM).unwrap();
            let mut mismatch = 0usize;
            let mut max_abs = 0.0f32;
            for (index, (&a, &b)) in reference.iter().zip(current.iter()).enumerate() {
                let fa = f32::from_bits((a as u32) << 16);
                let fb = f32::from_bits((b as u32) << 16);
                if a != b {
                    mismatch += 1;
                    let abs = (fa - fb).abs();
                    max_abs = max_abs.max(abs);
                    if mismatch <= 3 {
                        eprintln!("[scan-bench] {label} first diff index={index} ref={fa} got={fb}");
                    }
                }
            }
            eprintln!("[scan-bench] {label} mismatch={mismatch}/{} max_abs={max_abs:.6}", reference.len());
            // 容差族门禁：bf16 末位差异预期 max_abs ≈ 1 ulp（幅值 1-2 时 0.0078）；
            // 超过 4 ulp（0.031）视为真 bug。
            assert!(max_abs < 0.031, "{label} 数值超容差（max_abs={max_abs}）");
        };

        // 参照臂：生产 kernel tile=32，连跑两次验证 harness 确定性。
        let (a_pmin, a_pavg, a_mmin, a_mavg) = run_arm(functions.decode_partial_wmma_q8, 32, functions.split_merge, 100);
        eprintln!("[scan-bench] baseline-t32 partial min/avg={a_pmin:.1}/{a_pavg:.1}us merge min/avg={a_mmin:.1}/{a_mavg:.1}us");
        let reference = weighted.download_u16(HEAD_COUNT * LATENT_DIM).unwrap();
        let (a_pmin2, _, _, _) = run_arm(functions.decode_partial_wmma_q8, 32, functions.split_merge, 100);
        let rerun = weighted.download_u16(HEAD_COUNT * LATENT_DIM).unwrap();
        assert!(rerun == reference, "harness 非确定：同一 kernel 两次结果不一致");
        eprintln!("[scan-bench] baseline-t32 rerun partial_min={a_pmin2:.1}us deterministic=ok");

        // 臂 B：tile 扫描（占用率/ws 假说）。
        for tile in [64usize, 128] {
            let (pmin, pavg, mmin, mavg) = run_arm(functions.decode_partial_wmma_q8, tile, functions.split_merge, 100);
            eprintln!("[scan-bench] baseline-t{tile} partial min/avg={pmin:.1}/{pavg:.1}us merge min/avg={mmin:.1}/{mavg:.1}us");
            compare(&format!("baseline-t{tile}"), &reference);
        }

        // 臂 C：列并行 QK（runtime 维度）。逐档位保存输出供 512 特化位级对照。
        assert!(functions.decode_partial_wmma_q8_colpar != 0);
        let mut colpar_refs: Vec<(usize, Vec<u16>)> = Vec::new();
        for tile in [32usize, 64, 96] {
            let (pmin, pavg, mmin, mavg) = run_arm(functions.decode_partial_wmma_q8_colpar, tile, functions.split_merge, 100);
            eprintln!("[scan-bench] colpar-t{tile} partial min/avg={pmin:.1}/{pavg:.1}us merge min/avg={mmin:.1}/{mavg:.1}us");
            compare(&format!("colpar-t{tile}"), &reference);
            colpar_refs.push((tile, weighted.download_u16(HEAD_COUNT * LATENT_DIM).unwrap()));
        }

        // 臂 D：512 全常量特化（与 colpar 同档位必须逐位一致）。
        assert!(functions.decode_partial_wmma_q8_colpar512_shared != 0);
        for (tile, colpar_ref) in &colpar_refs {
            let (pmin, pavg, mmin, mavg) = run_arm(functions.decode_partial_wmma_q8_colpar512_shared, *tile, functions.split_merge, 100);
            eprintln!("[scan-bench] colpar512-shared-t{tile} partial min/avg={pmin:.1}/{pavg:.1}us merge min/avg={mmin:.1}/{mavg:.1}us");
            let current = weighted.download_u16(HEAD_COUNT * LATENT_DIM).unwrap();
            let diff = current.iter().zip(colpar_ref.iter()).filter(|(a, b)| a != b).count();
            eprintln!("[scan-bench] colpar512-shared-t{tile} vs colpar bit-diff={diff}/{}", colpar_ref.len());
            assert!(current == *colpar_ref, "colpar512-shared-t{tile} 与 colpar 不逐位一致");
        }

        // 臂 E：merge 软件流水（同输入必须与 merge 逐位一致；在各 tile 档位计时）。
        for tile in [32usize, 64, 96] {
            let (pmin, pavg, mmin, mavg) = run_arm(functions.decode_partial_wmma_q8_colpar512, tile, functions.split_merge_pl, 100);
            eprintln!("[scan-bench] colpar512-t{tile}-mergepl partial min/avg={pmin:.1}/{pavg:.1}us merge_pl min/avg={mmin:.1}/{mavg:.1}us");
            let current = weighted.download_u16(HEAD_COUNT * LATENT_DIM).unwrap();
            let colpar_ref = &colpar_refs.iter().find(|(t, _)| *t == tile).unwrap().1;
            assert!(current == *colpar_ref, "merge_pl-t{tile} 与 merge 不逐位一致");
            eprintln!("[scan-bench] mergepl-t{tile} bit-exact=ok");
        }

        // 臂 F：phase_mask 消融（partial 单 kernel 计时，数值无意义不校验）。
        // bit0 gather / bit1 QK / bit2 softmax / bit3 PV。
        {
            const TILE: usize = 64;
            let tile_count = TOP_K / TILE;
            let partial = DeviceBuffer::allocate(DEVICE, tile_count * HEAD_COUNT * LATENT_DIM * 4).unwrap();
            let stats = DeviceBuffer::allocate(DEVICE, tile_count * HEAD_COUNT * 2 * 4).unwrap();
            let mut d_query = query_device.pointer;
            let mut d_absorbed = absorbed_device.pointer;
            let mut d_latent = latent_device.pointer;
            let mut d_scales = scales_device.pointer;
            let mut d_rope = rope_device.pointer;
            let mut d_table = table_device.pointer;
            let mut d_selection = selection_device.pointer;
            let mut d_counts = ptr::null_mut();
            let mut d_partial = partial.pointer;
            let mut d_stats = stats.pointer;
            let mut d_weighted_null = ptr::null_mut();
            let mut context = CONTEXT as u32;
            let mut start = (CONTEXT - 1) as u32;
            let mut heads = HEAD_COUNT as u32;
            let mut q_head = Q_HEAD_DIM as u32;
            let mut latent = LATENT_DIM as u32;
            let mut group = GROUP as u32;
            let mut rope = ROPE_DIM as u32;
            let mut topk = TOP_K as u32;
            let mut block = BLOCK_SIZE as u32;
            let mut selected = 1_u32;
            let mut tile_u32 = TILE as u32;
            let mut split_tiles = tile_count as u32;
            let mut shard = 2_u32;
            let mut mask_u32 = 15_u32;
            let mut args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_absorbed as *mut *mut c_void).cast(),
                (&mut d_latent as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_rope as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut d_counts as *mut *mut c_void).cast(),
                (&mut d_partial as *mut *mut c_void).cast(),
                (&mut d_stats as *mut *mut c_void).cast(),
                (&mut d_weighted_null as *mut *mut c_void).cast(),
                (&mut context as *mut u32).cast(),
                (&mut start as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut q_head as *mut u32).cast(),
                (&mut latent as *mut u32).cast(),
                (&mut group as *mut u32).cast(),
                (&mut rope as *mut u32).cast(),
                (&mut topk as *mut u32).cast(),
                (&mut block as *mut u32).cast(),
                (&mut selected as *mut u32).cast(),
                (&mut tile_u32 as *mut u32).cast(),
                (&mut split_tiles as *mut u32).cast(),
                (&mut shard as *mut u32).cast(),
                (&mut mask_u32 as *mut u32).cast(),
            ];
            for mask in [15u32, 14, 13, 11, 7, 3, 1] {
                mask_u32 = mask;
                for _ in 0..3 {
                    launch_moe_kernel(functions.decode_partial_wmma_q8_colpar512_abl, 4, tile_count as u32, 256, colpar_shared as u32, &mut args, "scan ablation").unwrap();
                }
                super::super::super::synchronize_device(DEVICE, "scan ablation warmup").unwrap();
                let started = std::time::Instant::now();
                for _ in 0..100 {
                    launch_moe_kernel(functions.decode_partial_wmma_q8_colpar512_abl, 4, tile_count as u32, 256, colpar_shared as u32, &mut args, "scan ablation").unwrap();
                }
                super::super::super::synchronize_device(DEVICE, "scan ablation").unwrap();
                let us = started.elapsed().as_micros() as f64 / 100.0;
                eprintln!("[scan-bench] abl mask={mask:02} partial={us:.1}us");
            }
        }
    }
}

#[cfg(test)]
mod dsa_select_bench {
    /// DSA decode select 50K 分解基准：生产形状（32 头 × 128 维、top_k 2048、
    /// Q8G128 keys、block 64、C1 非 parallel tile）。profile_dsa=true 让
    /// try_dsa_select_paged_q8 自己打印 quant/score/select 分解（内含同步边界）。
    /// `cargo test --release --features with-rocm dsa_select_50k_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn dsa_select_50k_bench() {
        use super::*;
        const DEVICE: i32 = 0;
        const CONTEXT: usize = 50000;
        const HEADS: usize = 32;
        const HEAD_DIM: usize = 128;
        const TOP_K: usize = 2048;
        const BLOCK: usize = 64;
        const GROUP: usize = 128;
        super::super::super::configure(crate::kernel::rocm::hip::RocmOptions { profile_dsa: true, ..Default::default() }).unwrap();

        // splitmix32 伪随机 keys：dot 后 score 近高斯（byte1 桶分布接近真实
        // attention 打分，锯齿周期数据会让 radix 桶病态聚集）。
        let keys: Vec<u8> = (0..CONTEXT * HEAD_DIM)
            .map(|i| {
                let mut z = (i as u32).wrapping_add(0x9e37_79b9);
                z = z.wrapping_add(z << 16).rotate_left(0);
                z = (z ^ (z >> 15)).wrapping_mul(0x2c1b_3c6d);
                z = (z ^ (z >> 12)).wrapping_mul(0x297a_2d39);
                (z ^ (z >> 15)) as u8
            })
            .collect();
        let keys = DeviceBuffer::upload(DEVICE, &keys).unwrap();
        let scales: Vec<u16> = vec![0x3c00u16; CONTEXT * (HEAD_DIM / GROUP)];
        let scales = DeviceBuffer::upload(DEVICE, unsafe { std::slice::from_raw_parts(scales.as_ptr().cast::<u8>(), scales.len() * 2) }).unwrap();
        let table: Vec<u32> = (0..CONTEXT.div_ceil(BLOCK) as u32).collect();
        let table = DeviceBuffer::upload(DEVICE, unsafe { std::slice::from_raw_parts(table.as_ptr().cast::<u8>(), table.len() * 4) }).unwrap();
        let query: Vec<f32> = (0..HEADS * HEAD_DIM).map(|i| (((i * 13) % 37) as f32 - 18.0) / 64.0).collect();
        let query = DeviceBuffer::upload_f32(DEVICE, &query).unwrap();
        let weights: Vec<f32> = (0..HEADS).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
        let weights = DeviceBuffer::upload_f32(DEVICE, &weights).unwrap();

        // 预热 + 正式（profile_dsa 的逐段打印落在 stderr）。
        for round in 0..8 {
            let started = std::time::Instant::now();
            let _selection = try_dsa_select_paged_q8(DEVICE, &keys, &scales, GROUP, false, &table, &query, &weights, 1, CONTEXT, CONTEXT - 1, HEADS, HEAD_DIM, TOP_K, false, BLOCK).unwrap();
            super::super::super::synchronize_device(DEVICE, "dsa select bench").unwrap();
            let total_ms = started.elapsed().as_secs_f64() * 1e3;
            if round >= 3 {
                eprintln!("[dsa-select-bench] round={round} e2e_ms={total_ms:.3}");
            }
        }
    }
}
