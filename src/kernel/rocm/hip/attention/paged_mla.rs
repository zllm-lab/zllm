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
    dsa_kpool_compress: usize,
    cache_copy_q8_pair: usize,
    cache_append_mla_q8_bf16: usize,
    dsa_clear: usize,
    dsa_score: usize,
    dsa_quantize_query_i8: usize,
    dsa_score_i8: usize,
    dsa_score_native_wmma_i8: usize,
    dsa_score_wmma: usize,
    dsa_score_native_wmma: usize,
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
    split_merge: usize,
    project_value: usize,
    project_value_wmma: usize,
    wavefront_size: u32,
}

#[cfg(test)]
static TEST_SPARSE_PREFILL_WMMA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

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
    if rows == 0 || columns == 0 || columns > 256 || !columns.is_power_of_two() || rotary_dim == 0 || rotary_dim > columns || !rotary_dim.is_multiple_of(2) {
        return Err(format!("paged DSA fused prologue rows={rows} columns={columns} rotary_dim={rotary_dim} 非法"));
    }
    let elements = rows.checked_mul(columns).ok_or("paged DSA fused prologue 元素数溢出")?;
    let end = position.checked_add(rows).ok_or("paged DSA fused prologue position 溢出")?;
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("paged DSA fused input 大小溢出")?, "paged DSA fused input")?;
    validate_resident(weight, device_id, columns.checked_mul(4).ok_or("paged DSA fused weight 大小溢出")?, "paged DSA fused weight")?;
    validate_resident(bias, device_id, columns.checked_mul(4).ok_or("paged DSA fused bias 大小溢出")?, "paged DSA fused bias")?;
    validate_resident(cache, device_id, end.checked_mul(columns).ok_or("paged DSA fused cache 大小溢出")?, "paged DSA fused cache")?;
    validate_resident(scales, device_id, end.checked_mul(2).ok_or("paged DSA fused scale 大小溢出")?, "paged DSA fused scales")?;
    validate_resident(block_table, device_id, end.div_ceil(block_size).checked_mul(4).ok_or("paged DSA fused block table 大小溢出")?, "paged DSA fused block table")?;
    let half = rotary_dim / 2;
    let (cosine, sine) = super::super::tensor::resident_rope_tables(device_id, cos, sin, half, position..end)?;
    let functions = paged_mla_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = weight.pointer;
    let mut d_bias = bias.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_cache = cache.pointer;
    let mut d_scales = scales.pointer;
    let mut d_table = block_table.pointer;
    let mut position = u32::try_from(position).map_err(|_| "paged DSA fused position 超过 u32")?;
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
        (&mut position as *mut u32).cast(),
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
    rope_rotation: Option<(usize, RotaryLayout, &[f32], &[f32])>,
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
    validate_resident(latent_input, device_id, latent_elements.checked_mul(4).ok_or("paged MLA Q8 latent input 大小溢出")?, "paged MLA Q8 latent input")?;
    validate_resident(latent_cache, device_id, end.checked_mul(latent_columns).ok_or("paged MLA Q8 latent cache 大小溢出")?, "paged MLA Q8 latent cache")?;
    validate_resident(latent_scales, device_id, end.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or("paged MLA Q8 latent scale 大小溢出")?, "paged MLA Q8 latent scales")?;
    validate_resident(rope_input, device_id, rope_elements.checked_mul(4).ok_or("paged MLA rope input 大小溢出")?, "paged MLA rope input")?;
    validate_resident(rope_cache, device_id, end.checked_mul(rope_columns).and_then(|n| n.checked_mul(2)).ok_or("paged MLA rope cache 大小溢出")?, "paged MLA rope cache")?;
    validate_resident(block_table, device_id, end.div_ceil(block_size).checked_mul(4).ok_or("paged MLA block table 大小溢出")?, "paged MLA block table")?;
    let resident_rotation = match rope_rotation {
        Some((rotary_dim, layout, cos, sin)) => {
            if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > rope_columns {
                return Err(format!("paged MLA fused RoPE dim={rotary_dim} rope_columns={rope_columns} 非法"));
            }
            let half = rotary_dim / 2;
            let tables = super::super::tensor::resident_rope_tables(device_id, cos, sin, half, position..end)?;
            Some((tables, rotary_dim, layout))
        }
        None => None,
    };
    let functions = paged_mla_functions(device_id)?;
    let mut d_latent_input = latent_input.pointer;
    let mut d_latent_cache = latent_cache.pointer;
    let mut d_latent_scales = latent_scales.pointer;
    let mut d_rope_input = rope_input.pointer;
    let mut d_cosine = resident_rotation.as_ref().map_or(std::ptr::null_mut(), |((cosine, _), _, _)| cosine.pointer);
    let mut d_sine = resident_rotation.as_ref().map_or(std::ptr::null_mut(), |((_, sine), _, _)| sine.pointer);
    let mut d_rope_cache = rope_cache.pointer;
    let mut d_table = block_table.pointer;
    let mut position = u32::try_from(position).map_err(|_| "paged MLA cache position 超过 u32")?;
    let mut rows = u32::try_from(rows).map_err(|_| "paged MLA cache rows 超过 u32")?;
    let mut latent_columns = u32::try_from(latent_columns).map_err(|_| "paged MLA latent columns 超过 u32")?;
    let mut rope_columns = u32::try_from(rope_columns).map_err(|_| "paged MLA rope columns 超过 u32")?;
    let mut rotary_dim = u32::try_from(resident_rotation.as_ref().map_or(0, |(_, rotary_dim, _)| *rotary_dim)).map_err(|_| "paged MLA RoPE dim 超过 u32")?;
    let mut split_half = u32::from(resident_rotation.as_ref().is_some_and(|(_, _, layout)| *layout == RotaryLayout::SplitHalf));
    let mut group_size = u32::try_from(group_size).map_err(|_| "paged MLA Q8 group_size 超过 u32")?;
    let mut block_size = u32::try_from(block_size).map_err(|_| "paged MLA block_size 超过 u32")?;
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
        (&mut rows as *mut u32).cast(),
        (&mut latent_columns as *mut u32).cast(),
        (&mut rope_columns as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
        (&mut split_half as *mut u32).cast(),
        (&mut group_size as *mut u32).cast(),
        (&mut block_size as *mut u32).cast(),
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
    try_paged_cache_append_mla_f32_q8_bf16_inner(device_id, latent_input, latent_cache, latent_scales, rope_input, rope_cache, block_table, position, rows, latent_columns, rope_columns, group_size, block_size, None)
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
        Some((rotary_dim, layout, cos, sin)),
    )
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
        *buffer = Some(std::rc::Rc::new(DeviceBuffer::allocate(device_id, bytes)?));
        *capacity = bytes;
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

/// profile-only guarded-rerank cost oracle。候选由 coarse path 给出；本函数只用
/// 生产 raw-Q8 K + BF16 Q 重算候选，并在 candidate-local score 上做稳定 exact top-k。
/// 调用方仍决定是否采用结果，当前生产 selection 不会走这里。
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

    super::synchronize_device(device_id, "hipDeviceSynchronize DSA candidate rerank boundary")?;
    let score_started = std::time::Instant::now();
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
    super::synchronize_device(device_id, "hipDeviceSynchronize DSA candidate rerank score")?;
    let score_ms = score_started.elapsed().as_secs_f64() * 1e3;

    let select_started = std::time::Instant::now();
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
    super::synchronize_device(device_id, "hipDeviceSynchronize DSA candidate rerank select")?;
    let select_ms = select_started.elapsed().as_secs_f64() * 1e3;
    eprintln!("[dsa-rerank-profile] device={device_id} rows={query_rows} context={context_rows} candidates={candidate_count} top_k={} score_ms={score_ms:.3} select_ms={select_ms:.3} total_ms={:.3}", top_k, score_ms + select_ms,);
    Ok(selection)
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
    block_size: usize,
) -> Result<DeviceBuffer, String> {
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
    {
        return Err("paged DSA selection shape 非法".to_owned());
    }

    let functions = paged_mla_functions(device_id)?;
    let profile_dsa = options().profile_dsa;
    let use_wmma = !hadamard_i8 && head_count.is_multiple_of(16) && head_dim.is_multiple_of(16);
    let use_native_wmma = use_wmma && functions.dense_wmma && options().native_dsa_wmma;
    let single_row_native_wmma = use_native_wmma && query_rows == 1;
    let use_native_i8 = hadamard_i8 && head_count.is_multiple_of(16) && functions.dense_wmma && options().native_dsa_wmma;
    let compact_select = top_k <= 4096;
    const PREFIX_CANDIDATE_CAPACITY: usize = 4096;
    // 128K 的 2K-row append 会因 prefix candidate overflow 重算大量 score；
    // 只在更长上下文使用压缩路径，目标档位继续走精确 compact top-k。
    const PREFIX_CONTEXT_THRESHOLD: usize = 256 * 1024;
    let prefix_select = use_native_wmma && compact_select && query_rows >= 8 && context_rows >= PREFIX_CONTEXT_THRESHOLD;
    const PARALLEL_SELECT_CONTEXT_THRESHOLD: usize = 32 * 1024;
    const PARALLEL_SELECT_TILE_ROWS: usize = 1024;
    const PARALLEL_SELECT_LONG_TILE_ROWS: usize = 4096;
    // 4K tile 至少保留 60 个 CTA，覆盖 W7900 的 60 CU；128K 仍走 1K tile。
    const PARALLEL_SELECT_LONG_CONTEXT: usize = 60 * PARALLEL_SELECT_LONG_TILE_ROWS;
    // 当前 stage 不能跨 verify 行等待形成 DSA-only cohort；长上下文 decode
    // 改为单行内部按 history tile 并行，selection 集合与稳定顺序都不变。
    let parallel_select = compact_select && !prefix_select && query_rows <= 4 && context_rows >= PARALLEL_SELECT_CONTEXT_THRESHOLD;
    let score_histogram_shared_bytes: u32 = if compact_select {
        if single_row_native_wmma {
            256 * 4
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
    let max_tile_count = context_rows.div_ceil(score_tile_rows);
    let key_groups = head_dim / key_group_size;
    validate_resident(keys, device_id, context_rows.checked_mul(head_dim).ok_or("DSA Q8 keys 大小溢出")?, "DSA Q8 keys")?;
    validate_resident(key_scales, device_id, context_rows.checked_mul(key_groups).and_then(|n| n.checked_mul(2)).ok_or("DSA Q8 scales 大小溢出")?, "DSA Q8 scales")?;
    validate_resident(query, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).and_then(|n| n.checked_mul(4)).ok_or("DSA query 大小溢出")?, "DSA query")?;
    validate_resident(head_weights, device_id, query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(4)).ok_or("DSA weights 大小溢出")?, "DSA head weights")?;
    validate_resident(block_table, device_id, context_rows.div_ceil(block_size).checked_mul(4).ok_or("DSA block table 大小溢出")?, "DSA block table")?;

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
    let candidate_metadata_bytes = batch_rows.checked_mul(3).and_then(|n| n.checked_add(2)).and_then(|n| n.checked_mul(4)).ok_or("DSA prefix metadata 字节数溢出")?;
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
    let mut dim_u32 = u32::try_from(head_dim).map_err(|_| "DSA head_dim 超过 u32")?;
    let mut profile_quant_ms = 0.0f64;
    let mut profile_score_ms = 0.0f64;
    let mut profile_select_ms = 0.0f64;
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
        let batch_tile_count = batch_context_rows.div_ceil(score_tile_rows);
        let batch_score_stride = batch_tile_count.checked_mul(score_tile_rows).ok_or("DSA batch score stride 溢出")?;

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
        let mut d_coarse_histograms = if compact_select { coarse_histograms.pointer } else { ptr::null_mut() };
        let select_tile_plane_bytes = batch_rows.checked_mul(max_select_tile_count).and_then(|n| n.checked_mul(2)).and_then(|n| n.checked_mul(4)).ok_or("DSA parallel select tile plane offset 溢出")?;
        let mut d_select_tile_counts = select_tiles.pointer;
        let mut d_select_tile_offsets = if parallel_select { unsafe { select_tiles.pointer.cast::<u8>().add(select_tile_plane_bytes).cast::<c_void>() } } else { select_tiles.pointer };
        let mut d_select_tile_histograms = if parallel_select { unsafe { select_tiles.pointer.cast::<u8>().add(select_tile_plane_bytes * 2).cast::<c_void>() } } else { select_tiles.pointer };
        let mut current_rows_u32 = u32::try_from(current_rows).map_err(|_| "DSA batch rows 超过 u32")?;
        let mut context_u32 = u32::try_from(batch_context_rows).map_err(|_| "DSA context 超过 u32")?;
        let mut start_u32 = u32::try_from(batch_start).map_err(|_| "DSA query_start 超过 u32")?;
        let mut heads_u32 = u32::try_from(head_count).map_err(|_| "DSA heads 超过 u32")?;
        let mut key_group_u32 = u32::try_from(key_group_size).map_err(|_| "DSA key_group_size 超过 u32")?;
        let mut block_u32 = u32::try_from(block_size).map_err(|_| "DSA block_size 超过 u32")?;
        let mut score_tile_rows_u32 = u32::try_from(score_tile_rows).map_err(|_| "DSA score tile 超过 u32")?;
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
                } else {
                    functions.dsa_score_native_wmma
                }
            } else if use_wmma {
                functions.dsa_score_wmma
            } else {
                functions.dsa_score
            },
            u32::try_from(batch_tile_count).map_err(|_| "DSA tile grid 超过 u32")?,
            if use_native_wmma || use_native_i8 { current_rows_u32.div_ceil(4) } else { current_rows_u32 },
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
        let mut score_stride_u32 = u32::try_from(batch_score_stride).map_err(|_| "DSA score stride 超过 u32")?;
        let mut topk_u32 = u32::try_from(top_k).map_err(|_| "DSA top_k 超过 u32")?;
        let mut visibility_divisor_u32 = 1u32;
        let mut compact_args = [
            (&mut d_scores as *mut *mut c_void).cast(),
            (&mut d_coarse_histograms as *mut *mut c_void).cast(),
            (&mut d_selection as *mut *mut c_void).cast(),
            (&mut current_rows_u32 as *mut u32).cast(),
            (&mut score_stride_u32 as *mut u32).cast(),
            (&mut start_u32 as *mut u32).cast(),
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
            // 240K 起 4K tile 至少有 60 个 CTA，可填满 60 CU 并减少一半以上
            // 的 block/scan 元数据；较短上下文保持 1K，避免 128K 欠占用。
            let select_tile_rows = if batch_context_rows >= PARALLEL_SELECT_LONG_CONTEXT { PARALLEL_SELECT_LONG_TILE_ROWS } else { PARALLEL_SELECT_TILE_ROWS };
            let mut select_tile_count_u32 = u32::try_from(batch_score_stride.div_ceil(select_tile_rows)).map_err(|_| "DSA parallel select tile count 超过 u32")?;
            let mut select_tile_rows_u32 = u32::try_from(select_tile_rows).map_err(|_| "DSA parallel select tile rows 超过 u32")?;
            let mut clear_rows = current_rows_u32;
            let mut clear_threshold_args = [(&mut d_overflow_rows as *mut *mut c_void).cast(), (&mut clear_rows as *mut u32).cast()];
            launch_tensor_kernel(functions.dsa_clear, 1, 256, &mut clear_threshold_args, "HIP DSA clear radix tile counters")?;
            let mut threshold_args = [
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_coarse_histograms as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut d_threshold_bytes as *mut *mut c_void).cast(),
                (&mut d_select_tile_histograms as *mut *mut c_void).cast(),
                (&mut d_overflow_rows as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut select_tile_count_u32 as *mut u32).cast(),
                (&mut select_tile_rows_u32 as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_select_threshold, select_tile_count_u32, current_rows_u32, 256, 0, &mut threshold_args, "HIP DSA parallel compact radix threshold")?;

            let mut count_args = [
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut d_select_tile_counts as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut select_tile_count_u32 as *mut u32).cast(),
                (&mut select_tile_rows_u32 as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_select_tile_counts, select_tile_count_u32, current_rows_u32, 256, 0, &mut count_args, "HIP DSA count selected tiles")?;

            let mut scan_args = [(&mut d_select_tile_counts as *mut *mut c_void).cast(), (&mut d_select_tile_offsets as *mut *mut c_void).cast(), (&mut current_rows_u32 as *mut u32).cast(), (&mut select_tile_count_u32 as *mut u32).cast()];
            launch_tensor_kernel(functions.dsa_select_tile_scan, current_rows_u32, 256, &mut scan_args, "HIP DSA scan selected tiles")?;

            let mut scatter_args = [
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut d_candidate_counts as *mut *mut c_void).cast(),
                (&mut d_threshold_bytes as *mut *mut c_void).cast(),
                (&mut d_select_tile_offsets as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut current_rows_u32 as *mut u32).cast(),
                (&mut score_stride_u32 as *mut u32).cast(),
                (&mut start_u32 as *mut u32).cast(),
                (&mut topk_u32 as *mut u32).cast(),
                (&mut select_tile_count_u32 as *mut u32).cast(),
                (&mut select_tile_rows_u32 as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.dsa_select_tile_scatter, select_tile_count_u32, current_rows_u32, 256, 0, &mut scatter_args, "HIP DSA scatter selected tiles")?;
        } else if compact_select {
            launch_tensor_kernel(functions.dsa_select_compact, current_rows_u32, 256, &mut compact_args, "HIP DSA compact radix select top-k")?;
        } else {
            super::super::try_stable_radix_topk_u32_into(device_id, &scores, None, &selection, current_rows, batch_score_stride, batch_start, top_k, query_offset)?;
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
            "[dsa-profile] device={device_id} rows={query_rows} context={context_rows} batches={profile_batches} hadamard_i8={hadamard_i8} quant_ms={profile_quant_ms:.3} score_ms={profile_score_ms:.3} select_ms={profile_select_ms:.3} prefix={} parallel_select={} candidate_avg={:.1} candidate_max={} overflow_rows={}",
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
    Ok(selection)
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
    let mut project_args = [
        (&mut d_weighted as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
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
    if query_rows == 0 || query_start + query_rows != context_rows || !q_projection.is_multiple_of(head_count) || !weight.rows.is_multiple_of(head_count) || weight.cols == 0 || !weight.cols.is_multiple_of(weight.group_size) {
        return Err("paged MLA shape 非法".to_owned());
    }
    let q_head_dim = q_projection / head_count;
    let kv_head_dim = weight.rows / head_count;
    let latent_dim = weight.cols;
    if latent_group_size != 0 && (!latent_dim.is_multiple_of(latent_group_size) || latent_scales.is_none()) {
        return Err(format!("paged MLA latent Q8G{latent_group_size} cache 非法"));
    }
    validate_resident(query, device_id, query_rows.checked_mul(q_projection).and_then(|n| n.checked_mul(4)).ok_or("paged MLA query 大小溢出")?, "paged MLA query")?;
    let latent_element_bytes = if latent_group_size == 0 { 2 } else { 1 };
    validate_resident(latent_cache, device_id, context_rows.checked_mul(latent_dim).and_then(|n| n.checked_mul(latent_element_bytes)).ok_or("paged MLA latent 大小溢出")?, "paged MLA latent")?;
    if let Some(scales) = latent_scales {
        validate_resident(scales, device_id, context_rows.checked_mul(latent_dim / latent_group_size).and_then(|n| n.checked_mul(2)).ok_or("paged MLA latent scale 大小溢出")?, "paged MLA latent scales")?;
    }
    validate_resident(rope_cache, device_id, context_rows.checked_mul(rope_dim).and_then(|n| n.checked_mul(2)).ok_or("paged MLA rope 大小溢出")?, "paged MLA rope")?;
    validate_resident(block_table, device_id, context_rows.div_ceil(block_size).checked_mul(4).ok_or("paged MLA table 大小溢出")?, "paged MLA table")?;
    if let Some(selection) = selection {
        validate_resident(selection, device_id, query_rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("paged MLA selection 大小溢出")?, "paged MLA selection")?;
    }
    if options().debug_finite {
        try_validate_finite_resident_range_f32(device_id, query, (query_rows - 1) * q_projection, q_projection).map_err(|error| format!("paged MLA query 包含非有限值或异常幅值: {error}"))?;
        if let Some(scales) = latent_scales {
            try_validate_finite_resident_range_bf16(device_id, scales, 0, context_rows * (latent_dim / latent_group_size)).map_err(|error| format!("paged MLA latent scale 包含非有限值或异常幅值: {error}"))?;
        } else {
            try_validate_finite_resident_range_bf16(device_id, latent_cache, 0, context_rows * latent_dim).map_err(|error| format!("paged MLA latent cache 包含非有限值或异常幅值: {error}"))?;
        }
        try_validate_finite_resident_range_bf16(device_id, rope_cache, 0, context_rows * rope_dim).map_err(|error| format!("paged MLA rope cache 包含非有限值或异常幅值: {error}"))?;
    }
    let absorbed_elements = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(latent_dim)).ok_or("paged MLA absorbed 大小溢出")?;
    let output_elements = query_rows.checked_mul(q_projection).ok_or("paged MLA output 大小溢出")?;
    let intermediate_bytes = absorbed_elements.checked_mul(2).ok_or("paged MLA intermediate 字节数溢出")?;
    let (absorbed, weighted) = PAGED_MLA_WORKSPACES.with(|workspaces| -> Result<_, String> {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        Ok((reserve_paged_dsa_buffer(&mut workspace.absorbed, &mut workspace.absorbed_bytes, device_id, intermediate_bytes)?, reserve_paged_dsa_buffer(&mut workspace.weighted, &mut workspace.weighted_bytes, device_id, intermediate_bytes)?))
    })?;
    let output_element_bytes = if query_rows > 1 { 2 } else { 4 };
    validate_resident(output, device_id, output_elements.checked_mul(output_element_bytes).ok_or("paged MLA output 字节数溢出")?, "paged MLA output")?;
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
    let absorb_wmma = query_rows > 1 && options().mla_absorb_wmma;
    let absorb_tile = if absorb_wmma { 128 } else { 16 };
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
    let mut d_weighted = weighted.pointer;
    let mut d_split_partial = ptr::null_mut();
    let mut d_split_stats = ptr::null_mut();
    let mut d_direct_weighted = ptr::null_mut();
    let mut context_u32 = u32::try_from(context_rows).map_err(|_| "paged MLA context 超过 u32")?;
    let mut start_u32 = u32::try_from(query_start).map_err(|_| "paged MLA query_start 超过 u32")?;
    let mut topk_u32 = u32::try_from(top_k).map_err(|_| "paged MLA top_k 超过 u32")?;
    let mut block_u32 = u32::try_from(block_size).map_err(|_| "paged MLA block_size 超过 u32")?;
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
    let split_decode = split_decode_override.or(options().mla_decode_split).unwrap_or(context_rows >= options().mla_decode_split_threshold);
    let attention_started = profile_mla.then(std::time::Instant::now);
    const WMMA_HEADS_PER_BLOCK: usize = 32;
    const WMMA_OUTPUT_STRIDE: usize = 256;
    const WMMA_MAX_SHARED_BYTES: usize = 64 * 1024;
    let selected_wmma_shape = options().mla_decode_wmma
        && functions.decode_partial_wmma_q8 != 0
        && latent_group_size != 0
        && latent_group_size.is_multiple_of(16)
        && head_count.is_multiple_of(WMMA_HEADS_PER_BLOCK)
        && rope_dim.is_multiple_of(16)
        && latent_dim.is_multiple_of(WMMA_OUTPUT_STRIDE)
        && latent_dim / WMMA_OUTPUT_STRIDE == 2;
    let selected_wmma_shared = if selected_wmma_shape {
        Some(
            (64usize * 16)
                .checked_add(WMMA_HEADS_PER_BLOCK.checked_mul(16).ok_or("paged MLA WMMA probability tile 溢出")?)
                .and_then(|elements| elements.checked_add(latent_dim.checked_mul(16)?))
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                .and_then(|bytes| bytes.checked_add(WMMA_HEADS_PER_BLOCK.checked_mul(std::mem::size_of::<f32>())?))
                .and_then(|bytes| bytes.checked_add(64 * std::mem::size_of::<u32>()))
                .and_then(|bytes| bytes.checked_add(64usize.checked_mul(latent_dim)?))
                .and_then(|bytes| bytes.checked_add(64usize.checked_mul(latent_dim.checked_div(latent_group_size)?)?.checked_mul(std::mem::size_of::<u16>())?))
                .ok_or("paged MLA WMMA shared memory 字节数溢出")?,
        )
    } else {
        None
    };
    // gfx11 每个 workgroup 最多使用 64 KiB LDS；超限形态保留原标量路径。
    let selected_wmma_q8 = selected_wmma_shared.is_some_and(|bytes| bytes <= WMMA_MAX_SHARED_BYTES);
    // 区分 dense / sparse / decode 路径的 profile 标签，仅用于 [mla-profile] 归因。
    let mut attention_kind = "decode_split";
    if query_rows == 1 && split_decode {
        let requested_tile_size = options().mla_decode_tile_size;
        let visible_rows = query_start.checked_add(1).ok_or("paged MLA decode visible rows 溢出")?.min(context_rows);
        // DSA selection 按逻辑候选区间拆分，tile 内再映射到真实分页位置。
        let decode_rows = if selection.is_some() { visible_rows.min(top_k) } else { visible_rows };
        let decode_wmma_q8 = selected_wmma_q8;
        let max_decode_tiles = if decode_wmma_q8 { 128 } else { 64 };
        let tile_alignment = if decode_wmma_q8 { 64 } else { 128 };
        // split 数受 merge kernel 上限约束；长上下文自动放大 tile，避免退化为运行时错误。
        let minimum_tile_size = decode_rows.div_ceil(max_decode_tiles).div_ceil(tile_alignment).checked_mul(tile_alignment).ok_or("paged MLA decode minimum tile size 溢出")?;
        let decode_tile_size = requested_tile_size.max(minimum_tile_size).div_ceil(tile_alignment).checked_mul(tile_alignment).ok_or("paged MLA decode tile size 溢出")?;
        let tile_count = decode_rows.div_ceil(decode_tile_size);
        if tile_count == 0 || tile_count > max_decode_tiles {
            return Err(format!("paged MLA decode tile_count={tile_count} 非法"));
        }
        let partial_elements = tile_count.checked_mul(head_count).and_then(|elements| elements.checked_mul(latent_dim)).ok_or("paged MLA decode partial 元素数溢出")?;
        let stats_elements = tile_count.checked_mul(head_count).and_then(|elements| elements.checked_mul(2)).ok_or("paged MLA decode stats 元素数溢出")?;
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
        let mut partial_args = [
            (&mut d_query as *mut *mut c_void).cast(),
            (&mut d_absorbed as *mut *mut c_void).cast(),
            (&mut d_latent as *mut *mut c_void).cast(),
            (&mut d_latent_scales as *mut *mut c_void).cast(),
            (&mut d_rope as *mut *mut c_void).cast(),
            (&mut d_table as *mut *mut c_void).cast(),
            (&mut d_selection as *mut *mut c_void).cast(),
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
        ];
        let decode_shared = if decode_wmma_q8 {
            selected_wmma_shared.expect("WMMA Q8 路径已经检查 shared memory")
        } else {
            let per_head = latent_dim
                .checked_mul(std::mem::size_of::<u16>())
                .and_then(|latent_bytes| rope_dim.checked_mul(std::mem::size_of::<f32>()).and_then(|rope_bytes| latent_bytes.checked_add(rope_bytes)))
                .ok_or("paged MLA decode query cache 字节数溢出")?;
            4usize.checked_mul(per_head).ok_or("paged MLA decode query cache 字节数溢出")?
        };
        let decode_shared_u32 = u32::try_from(decode_shared).map_err(|_| "paged MLA decode query cache 超过 u32")?;
        launch_moe_kernel(
            if decode_wmma_q8 { functions.decode_partial_wmma_q8 } else { functions.decode_partial },
            if decode_wmma_q8 { heads_u32.div_ceil(WMMA_HEADS_PER_BLOCK as u32) } else { heads_u32.div_ceil(4) },
            tile_count_u32,
            if decode_wmma_q8 { 1024 } else { 256 },
            decode_shared_u32,
            &mut partial_args,
            if decode_wmma_q8 { "HIP paged MLA decode partial Q8 WMMA" } else { "HIP paged MLA decode partial" },
        )?;
        let mut merge_args = [
            (&mut d_partial as *mut *mut c_void).cast(),
            (&mut d_stats as *mut *mut c_void).cast(),
            (&mut d_weighted as *mut *mut c_void).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut latent_u32 as *mut u32).cast(),
            (&mut tile_count_u32 as *mut u32).cast(),
        ];
        launch_moe_kernel(functions.split_merge, heads_u32, 1, 256, 0, &mut merge_args, "HIP paged MLA decode merge")?;
    } else {
        let dense_prefill =
            functions.dense_wmma && (selection.is_none() || force_dense_prefill) && query_rows > 1 && latent_dim.is_multiple_of(16) && rope_dim.is_multiple_of(16) && (latent_group_size == 0 || latent_group_size.is_multiple_of(16));
        let sparse_prefill_wmma = query_rows > 1 && !dense_prefill && selection.is_some() && selected_wmma_q8 && sparse_prefill_wmma_enabled();
        let sparse_prefill_heads4 = query_rows > 1 && !dense_prefill && options().sparse_prefill_heads4;
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
        let attention_rows = if dense_prefill { query_rows.div_ceil(2) } else { query_rows };
        let query_blocks = query_rows.div_ceil(2).checked_mul(head_count.div_ceil(8)).ok_or("paged MLA prefill query block 数溢出")?;
        let target_blocks = options().mla_prefill_target_blocks;
        let max_splits = (context_rows / 2048).clamp(1, 64);
        let requested_splits = if dense_prefill && query_blocks < target_blocks { target_blocks.div_ceil(query_blocks).min(max_splits) } else { 1 };
        let split_size = context_rows.div_ceil(requested_splits).div_ceil(128).checked_mul(128).ok_or("paged MLA prefill split size 溢出")?;
        let split_count = context_rows.div_ceil(split_size);
        let split_buffers = if split_count > 1 {
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
            let mut direct_args = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_absorbed as *mut *mut c_void).cast(),
                (&mut d_latent as *mut *mut c_void).cast(),
                (&mut d_latent_scales as *mut *mut c_void).cast(),
                (&mut d_rope as *mut *mut c_void).cast(),
                (&mut d_table as *mut *mut c_void).cast(),
                (&mut d_selection as *mut *mut c_void).cast(),
                (&mut d_split_partial as *mut *mut c_void).cast(),
                (&mut d_split_stats as *mut *mut c_void).cast(),
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
            ];
            launch_moe_kernel(
                functions.decode_partial_wmma_q8,
                heads_u32.div_ceil(WMMA_HEADS_PER_BLOCK as u32),
                query_rows_u32,
                1024,
                u32::try_from(selected_wmma_shared.expect("sparse WMMA 已检查 shared memory")).map_err(|_| "paged MLA sparse WMMA shared memory 超过 u32")?,
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
        if split_count > 1 {
            let mut merge_args = [
                (&mut d_split_partial as *mut *mut c_void).cast(),
                (&mut d_split_stats as *mut *mut c_void).cast(),
                (&mut d_weighted as *mut *mut c_void).cast(),
                (&mut heads_u32 as *mut u32).cast(),
                (&mut latent_u32 as *mut u32).cast(),
                (&mut split_count_u32 as *mut u32).cast(),
            ];
            launch_moe_kernel(functions.split_merge, heads_u32, query_rows_u32, 256, 0, &mut merge_args, "HIP paged MLA prefill split merge")?;
        }
    }
    if let Some(started) = attention_started {
        super::synchronize_device(device_id, "hipDeviceSynchronize MLA attention profile")?;
        eprintln!("[mla-profile] device={device_id} rows={query_rows} context={context_rows} kind={attention_kind} attention_ms={:.3}", started.elapsed().as_secs_f64() * 1e3,);
    }
    if query_rows != 0 && options().debug_finite {
        let row_elements = head_count * latent_dim;
        try_validate_finite_resident_range_bf16(device_id, &weighted, (query_rows - 1) * row_elements, row_elements).map_err(|error| format!("paged MLA weighted latent 包含非有限值: {error}"))?;
    }

    let mut d_output = output.pointer;
    let mut project_args = [
        (&mut d_weighted as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
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
    let value_dim = kv_head_dim.checked_sub(q_head_dim - rope_dim).ok_or("paged MLA value_dim 下溢")?;
    let project_wmma = query_rows > 1;
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

#[cfg(test)]
mod tests {
    use super::*;

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

    fn compare_score_pipeline(context_rows: usize, query_rows: usize, equal_scores: bool) -> Result<(), String> {
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
        let compact = try_dsa_select_paged_q8(DEVICE_ID, &keys, &scales, KEY_GROUP_SIZE, false, &block_table, &query, &head_weights, query_rows, context_rows, query_start, HEAD_COUNT, HEAD_DIM, selection_width, BLOCK_SIZE)?;

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
            ];
            launch_moe_kernel(functions.dsa_score_native_wmma, context.div_ceil(tile), rows.div_ceil(4), 512, 0, &mut score_args, "HIP DSA full score oracle")?;
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
        let raw_selection = try_dsa_select_paged_q8(DEVICE_ID, &raw_key_cache, &raw_key_scales, HEAD_DIM, false, &table, &query, &weights, QUERY_ROWS, CONTEXT_ROWS, QUERY_START, HEAD_COUNT, HEAD_DIM, TOP_K, BLOCK_SIZE).unwrap();
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
        let selection = try_dsa_select_paged_q8(DEVICE_ID, &key_cache, &key_scales, HEAD_DIM, true, &table, &query, &weights, QUERY_ROWS, CONTEXT_ROWS, QUERY_START, HEAD_COUNT, HEAD_DIM, TOP_K, BLOCK_SIZE).unwrap();
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
        run(&baseline, false);
        run(&candidate, true);
        let baseline_ms = (0..3).map(|_| run(&baseline, false)).sum::<f64>() / 3.0;
        let candidate_ms = (0..3).map(|_| run(&candidate, true)).sum::<f64>() / 3.0;

        let mut baseline_bits = vec![0_u16; QUERY_ROWS * Q_PROJECTION];
        let mut candidate_bits = vec![0_u16; QUERY_ROWS * Q_PROJECTION];
        baseline.copy_to_host(as_bytes_mut(&mut baseline_bits)).unwrap();
        candidate.copy_to_host(as_bytes_mut(&mut candidate_bits)).unwrap();
        let baseline_host = baseline_bits.into_iter().map(|bits| f32::from_bits(u32::from(bits) << 16)).collect::<Vec<_>>();
        let candidate_host = candidate_bits.into_iter().map(|bits| f32::from_bits(u32::from(bits) << 16)).collect::<Vec<_>>();
        let mut max_abs = 0.0_f32;
        let mut max_index = 0;
        let mut squared = 0.0_f64;
        for (index, (reference, actual)) in baseline_host.iter().zip(&candidate_host).enumerate() {
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
    fn mla_decode_split_matches_serial_and_reports_latency() {
        const DEVICE_ID: i32 = 0;
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

        let query = (0..Q_PROJECTION).map(|index| ((index % 257) as f32 - 128.0) * (1.0 / 256.0)).collect::<Vec<_>>();
        let weight = (0..KV_PROJECTION * LATENT_DIM).map(|index| (((index * 17) % 127) as f32 - 63.0) * (1.0 / 4096.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
        let query = DeviceBuffer::upload(DEVICE_ID, as_bytes(&query)).unwrap();
        let weight = DeviceBuffer::upload(DEVICE_ID, as_bytes(&weight)).unwrap();
        let scales = DeviceBuffer::upload(DEVICE_ID, as_bytes(&[0x3f80_u16])).unwrap();

        for (mode, latent_group) in [("q8g64", LATENT_GROUP), ("f16", 0)] {
            for context_rows in [64_usize, 128, 256, 512, 768, 1357, 2048] {
                let latent_bytes = if latent_group == 0 {
                    let values = (0..context_rows * LATENT_DIM).map(|index| (((index * 29 + index / LATENT_DIM * 7) % 127) as f32 - 63.0) * (1.0 / 64.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
                    as_bytes(&values).to_vec()
                } else {
                    (0..context_rows * LATENT_DIM).map(|index| ((index * 29 + index / LATENT_DIM * 7) % 63 + 1) as u8).collect::<Vec<_>>()
                };
                let latent_scales = (latent_group != 0).then(|| vec![0x3b80_u16; context_rows * (LATENT_DIM / LATENT_GROUP)]);
                let rope = (0..context_rows * ROPE_DIM).map(|index| (((index * 13) % 127) as f32 - 63.0) * (1.0 / 128.0)).map(|value| (value.to_bits() >> 16) as u16).collect::<Vec<_>>();
                let table = (0..context_rows.div_ceil(BLOCK_SIZE) as u32).collect::<Vec<_>>();
                let latent = DeviceBuffer::upload(DEVICE_ID, &latent_bytes).unwrap();
                let latent_scales = latent_scales.as_ref().map(|scales| DeviceBuffer::upload(DEVICE_ID, as_bytes(scales)).unwrap());
                let rope = DeviceBuffer::upload(DEVICE_ID, as_bytes(&rope)).unwrap();
                let table = DeviceBuffer::upload(DEVICE_ID, as_bytes(&table)).unwrap();
                let serial = DeviceBuffer::allocate(DEVICE_ID, Q_PROJECTION * 4).unwrap();
                let split = DeviceBuffer::allocate(DEVICE_ID, Q_PROJECTION * 4).unwrap();

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
                        None,
                        CtMlaWeightRef { packed: &weight, scales: &scales, rows: KV_PROJECTION, cols: LATENT_DIM, group_size: LATENT_DIM, scale_dtype: 0, bits: 16 },
                        1,
                        context_rows,
                        context_rows - 1,
                        Q_PROJECTION,
                        HEAD_COUNT,
                        ROPE_DIM,
                        0,
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

                let mut serial_host = vec![0_f32; Q_PROJECTION];
                let mut split_host = vec![0_f32; Q_PROJECTION];
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
                let rmse = (squared / Q_PROJECTION as f64).sqrt();
                println!("[mla-split-oracle] mode={mode} context={context_rows} serial_ms={serial_ms:.3} split_ms={split_ms:.3} speedup={:.3} max_abs={max_abs:.6e} rmse={rmse:.6e}", serial_ms / split_ms);
                let tolerance = if latent_group == 0 { 1.0e-2 } else { 5.0e-3 };
                assert!(max_abs <= tolerance, "mode={mode} context={context_rows} max_abs={max_abs}");
            }
        }
    }
}
