//! DeepSeek-V4 压缩 KV、学习式 indexer 与滑窗注意力 HIP 算子。

use super::tensor::validate_resident;
use super::*;

const INDEX_SCRATCH_BYTES: usize = 64 * 1024 * 1024;

const CSA_TILED_HEAD_DIM: u32 = 16 * 32;

const COMPRESSED_SPARSE_SOURCE: &str = include_str!("compressed_sparse/source.hip");

#[derive(Clone, Copy)]
struct CompressedSparseFunctions {
    gated_compress: usize,
    store_overlap: usize,
    store_pending: usize,
    index_scores: usize,
    index_scores_wmma: usize,
    store_segment_table: usize,
    attention: usize,
    attention_tiled: usize,
    attention_segmented_tiled: usize,
    attention_wmma: usize,
    attention_decode_partial: usize,
    attention_segmented_decode_partial: usize,
    attention_decode_merge: usize,
}

type FunctionCache = Mutex<HashMap<i32, Result<(usize, CompressedSparseFunctions), String>>>;

fn compressed_sparse_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(&[DEVICE_CONVERSIONS_PREAMBLE, COMPRESSED_SPARSE_SOURCE].concat(), "zllm_rocm_compressed_sparse.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

fn compressed_sparse_functions(device_id: i32) -> Result<CompressedSparseFunctions, String> {
    static FUNCTIONS: OnceLock<FunctionCache> = OnceLock::new();
    let cache = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| "ROCm compressed sparse kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = cache.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let get: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = compressed_sparse_code()?;
        let mut module = ptr::null_mut();
        let status = unsafe { load(&mut module, code.as_ptr().cast()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLoadData compressed sparse"));
        }
        let function = |name: &str| -> Result<usize, String> {
            let name = CString::new(name).unwrap();
            let mut handle = ptr::null_mut();
            let status = unsafe { get(&mut handle, module, name.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction compressed sparse"));
            }
            Ok(handle as usize)
        };
        Ok((
            module as usize,
            CompressedSparseFunctions {
                gated_compress: function("csa_gated_compress_f32")?,
                store_overlap: function("csa_store_overlap_f32")?,
                store_pending: function("csa_store_pending_f32")?,
                index_scores: function("csa_index_scores_f32")?,
                index_scores_wmma: function("csa_index_scores_wmma_f32")?,
                store_segment_table: function("csa_store_segment_table")?,
                attention: function("csa_attention_wave_q8")?,
                attention_tiled: function("csa_attention_tiled_q8")?,
                attention_segmented_tiled: function("csa_attention_segmented_tiled_q8")?,
                attention_wmma: function("csa_attention_wmma_q8")?,
                attention_decode_partial: function("csa_attention_decode_partial_q8")?,
                attention_segmented_decode_partial: function("csa_attention_segmented_decode_partial_q8")?,
                attention_decode_merge: function("csa_attention_decode_merge_f32")?,
            },
        ))
    })();
    cache.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

fn launch(function: usize, grid_x: u32, grid_y: u32, block: u32, arguments: &mut [*mut c_void], action: &str) -> Result<(), String> {
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result = unsafe { launch(function as *mut c_void, grid_x, grid_y, 1, block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, action));
    }
    Ok(())
}

fn u32_value(name: &str, value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name}={value} 超过 u32"))
}

#[allow(clippy::too_many_arguments)]
pub fn try_csa_gated_compress_f32(
    device_id: i32,
    pending_key: &DeviceBuffer,
    pending_gate: &DeviceBuffer,
    pending_rows: usize,
    key: &DeviceBuffer,
    gate: &DeviceBuffer,
    input_rows: usize,
    position_bias: &DeviceBuffer,
    norm: &DeviceBuffer,
    overlap_key: &DeviceBuffer,
    overlap_gate: &DeviceBuffer,
    ratio: usize,
    width: usize,
    overlap: bool,
    entry_start: usize,
    rotary_dim: usize,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    table_elements: usize,
    eps: f32,
) -> Result<(DeviceBuffer, usize), String> {
    let channels = if overlap { width.checked_mul(2).ok_or("ROCm compressor channels 溢出")? } else { width };
    if ratio == 0 || width == 0 || width > 1024 || pending_rows >= ratio || input_rows == 0 || rotary_dim == 0 || rotary_dim > width || !rotary_dim.is_multiple_of(2) {
        return Err(format!("ROCm compressor shape 非法: pending={pending_rows} rows={input_rows} ratio={ratio} width={width} rotary={rotary_dim}"));
    }
    let input_elements = input_rows.checked_mul(channels).ok_or("ROCm compressor input 溢出")?;
    let state_elements = ratio.checked_mul(channels).ok_or("ROCm compressor state 溢出")?;
    validate_resident(key, device_id, input_elements * 4, "compressor key")?;
    validate_resident(gate, device_id, input_elements * 4, "compressor gate")?;
    validate_resident(pending_key, device_id, state_elements * 4, "compressor pending key")?;
    validate_resident(pending_gate, device_id, state_elements * 4, "compressor pending gate")?;
    validate_resident(position_bias, device_id, state_elements * 4, "compressor position bias")?;
    validate_resident(norm, device_id, width * 4, "compressor norm")?;
    validate_resident(cos, device_id, table_elements * 4, "compressor cos")?;
    validate_resident(sin, device_id, table_elements * 4, "compressor sin")?;
    if overlap {
        validate_resident(overlap_key, device_id, ratio * width * 4, "compressor overlap key")?;
        validate_resident(overlap_gate, device_id, ratio * width * 4, "compressor overlap gate")?;
    }
    set_device(device_id)?;
    let total_rows = pending_rows.checked_add(input_rows).ok_or("ROCm compressor rows 溢出")?;
    let windows = total_rows / ratio;
    let remaining = total_rows % ratio;
    let output_bytes = windows.checked_mul(width).and_then(|n| n.checked_mul(4)).ok_or("ROCm compressor output 大小溢出")?;
    // windows 可能为 0(总行数不足一个窗口),仍分配最小 buffer 保持契约。
    let output = DeviceBuffer::allocate(device_id, output_bytes.max(4))?;
    let functions = compressed_sparse_functions(device_id)?;
    let mut d_pending_key = pending_key.pointer;
    let mut d_pending_gate = pending_gate.pointer;
    let mut d_key = key.pointer;
    let mut d_gate = gate.pointer;
    let mut d_position_bias = position_bias.pointer;
    let mut d_norm = norm.pointer;
    let mut d_overlap_key = overlap_key.pointer;
    let mut d_overlap_gate = overlap_gate.pointer;
    let mut d_cos = cos.pointer;
    let mut d_sin = sin.pointer;
    let mut d_output = output.pointer;
    let mut pending_rows_u32 = u32_value("compressor pending rows", pending_rows)?;
    let mut input_rows_u32 = u32_value("compressor input rows", input_rows)?;
    let mut windows_u32 = u32_value("compressor windows", windows)?;
    let mut ratio_u32 = u32_value("compressor ratio", ratio)?;
    let mut width_u32 = u32_value("compressor width", width)?;
    let mut channels_u32 = u32_value("compressor channels", channels)?;
    let mut overlap_u32 = u32::from(overlap);
    let mut entry_start_u32 = u32_value("compressor entry", entry_start)?;
    let mut rotary_dim_u32 = u32_value("compressor rotary dim", rotary_dim)?;
    let mut eps_f32 = eps;
    if windows != 0 {
        let mut arguments = [
            (&mut d_pending_key as *mut *mut c_void).cast(),
            (&mut d_pending_gate as *mut *mut c_void).cast(),
            (&mut d_key as *mut *mut c_void).cast(),
            (&mut d_gate as *mut *mut c_void).cast(),
            (&mut d_position_bias as *mut *mut c_void).cast(),
            (&mut d_norm as *mut *mut c_void).cast(),
            (&mut d_overlap_key as *mut *mut c_void).cast(),
            (&mut d_overlap_gate as *mut *mut c_void).cast(),
            (&mut d_cos as *mut *mut c_void).cast(),
            (&mut d_sin as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut pending_rows_u32 as *mut u32).cast(),
            (&mut windows_u32 as *mut u32).cast(),
            (&mut ratio_u32 as *mut u32).cast(),
            (&mut width_u32 as *mut u32).cast(),
            (&mut channels_u32 as *mut u32).cast(),
            (&mut overlap_u32 as *mut u32).cast(),
            (&mut entry_start_u32 as *mut u32).cast(),
            (&mut rotary_dim_u32 as *mut u32).cast(),
            (&mut eps_f32 as *mut f32).cast(),
        ];
        launch(functions.gated_compress, windows_u32, 1, u32_value("compressor block", width.next_power_of_two())?, &mut arguments, "HIP compressed gated pooling")?;
        if overlap {
            let count = ratio.checked_mul(width).ok_or("ROCm overlap count 溢出")?;
            let mut count_u32 = u32_value("compressor overlap count", count)?;
            let mut arguments = [
                (&mut d_pending_key as *mut *mut c_void).cast(),
                (&mut d_pending_gate as *mut *mut c_void).cast(),
                (&mut d_key as *mut *mut c_void).cast(),
                (&mut d_gate as *mut *mut c_void).cast(),
                (&mut d_position_bias as *mut *mut c_void).cast(),
                (&mut d_overlap_key as *mut *mut c_void).cast(),
                (&mut d_overlap_gate as *mut *mut c_void).cast(),
                (&mut pending_rows_u32 as *mut u32).cast(),
                (&mut windows_u32 as *mut u32).cast(),
                (&mut ratio_u32 as *mut u32).cast(),
                (&mut width_u32 as *mut u32).cast(),
                (&mut channels_u32 as *mut u32).cast(),
                (&mut count_u32 as *mut u32).cast(),
            ];
            launch(functions.store_overlap, count_u32.div_ceil(256), 1, 256, &mut arguments, "HIP compressed overlap state")?;
        }
    }
    if remaining != 0 {
        let mut arguments = [
            (&mut d_pending_key as *mut *mut c_void).cast(),
            (&mut d_pending_gate as *mut *mut c_void).cast(),
            (&mut d_key as *mut *mut c_void).cast(),
            (&mut d_gate as *mut *mut c_void).cast(),
            (&mut d_pending_key as *mut *mut c_void).cast(),
            (&mut d_pending_gate as *mut *mut c_void).cast(),
            (&mut pending_rows_u32 as *mut u32).cast(),
            (&mut input_rows_u32 as *mut u32).cast(),
            (&mut ratio_u32 as *mut u32).cast(),
            (&mut channels_u32 as *mut u32).cast(),
        ];
        let elements = u32_value("compressor pending elements", remaining * channels)?;
        launch(functions.store_pending, elements.div_ceil(256), 1, 256, &mut arguments, "HIP compressed pending state")?;
    }
    Ok((output, remaining))
}

#[allow(clippy::too_many_arguments)]
pub fn try_csa_index_select_f32(
    device_id: i32,
    query: &DeviceBuffer,
    keys: &DeviceBuffer,
    head_weights: &DeviceBuffer,
    visible_counts: &DeviceBuffer,
    query_rows: usize,
    compressed_rows: usize,
    head_count: usize,
    head_dim: usize,
    top_k: usize,
) -> Result<DeviceBuffer, String> {
    if query_rows == 0 || compressed_rows == 0 || head_count == 0 || head_dim == 0 || top_k == 0 {
        return Err("ROCm indexer shape 不能包含 0".to_owned());
    }
    validate_resident(query, device_id, query_rows * head_count * head_dim * 4, "index query")?;
    validate_resident(keys, device_id, compressed_rows * head_dim * 4, "index keys")?;
    validate_resident(head_weights, device_id, query_rows * head_count * 4, "index head weights")?;
    validate_resident(visible_counts, device_id, query_rows * 4, "index visible counts")?;
    set_device(device_id)?;
    let use_wmma = head_count <= 64 && head_dim <= 128 && head_count.is_multiple_of(16) && head_dim.is_multiple_of(16);
    let query_elements = query_rows.checked_mul(head_count).and_then(|n| n.checked_mul(head_dim)).ok_or("ROCm index query 溢出")?;
    let query_bf16 = use_wmma.then(|| super::try_cast_f32_to_bf16_resident(device_id, query, query_elements)).transpose()?;
    let selection = DeviceBuffer::allocate(device_id, query_rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("ROCm index selection 溢出")?)?;
    let score_row_bytes = compressed_rows.checked_mul(4).ok_or("ROCm index score row 溢出")?;
    let chunk_rows = (INDEX_SCRATCH_BYTES / score_row_bytes.max(1)).max(1).min(query_rows);
    let score_bytes = chunk_rows.checked_mul(score_row_bytes).ok_or("ROCm index score scratch 溢出")?;
    let functions = compressed_sparse_functions(device_id)?;
    let mut compressed_rows_u32 = u32_value("index compressed rows", compressed_rows)?;
    let mut head_count_u32 = u32_value("index heads", head_count)?;
    let mut head_dim_u32 = u32_value("index head dim", head_dim)?;
    super::tensor::with_attention_workspace(device_id, &[score_bytes], |workspace| {
        let scores = workspace.buffer(0);
        for query_start in (0..query_rows).step_by(chunk_rows) {
            let rows = (query_rows - query_start).min(chunk_rows);
            let query_offset = query_start * head_count * head_dim;
            let mut d_query = if let Some(query_bf16) = &query_bf16 { unsafe { query_bf16.pointer.cast::<u16>().add(query_offset).cast() } } else { unsafe { query.pointer.cast::<f32>().add(query_offset).cast() } };
            let mut d_keys = keys.pointer;
            let mut d_weights = unsafe { head_weights.pointer.cast::<f32>().add(query_start * head_count).cast() };
            let mut d_visible = unsafe { visible_counts.pointer.cast::<u32>().add(query_start).cast() };
            let mut d_scores = scores.pointer;
            let mut rows_u32 = u32_value("index query rows", rows)?;
            let mut score_arguments = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_keys as *mut *mut c_void).cast(),
                (&mut d_weights as *mut *mut c_void).cast(),
                (&mut d_visible as *mut *mut c_void).cast(),
                (&mut d_scores as *mut *mut c_void).cast(),
                (&mut rows_u32 as *mut u32).cast(),
                (&mut compressed_rows_u32 as *mut u32).cast(),
                (&mut head_count_u32 as *mut u32).cast(),
                (&mut head_dim_u32 as *mut u32).cast(),
            ];
            if use_wmma {
                launch(functions.index_scores_wmma, compressed_rows_u32.div_ceil(32), rows_u32.div_ceil(4), 512, &mut score_arguments, "HIP index scores WMMA")?;
            } else {
                launch(functions.index_scores, compressed_rows_u32, rows_u32, 256, &mut score_arguments, "HIP index scores")?;
            }
            try_stable_radix_topk_u32_into(device_id, scores, Some((visible_counts, query_start)), &selection, rows, compressed_rows, 0, top_k, query_start)?;
        }
        Ok(())
    })?;
    Ok(selection)
}

pub struct CsaAttentionQ8Segment<'a> {
    pub compressed_key: &'a DeviceBuffer,
    pub compressed_key_scales: &'a DeviceBuffer,
    pub compressed_value: &'a DeviceBuffer,
    pub compressed_value_scales: &'a DeviceBuffer,
    pub visible_compressed: &'a DeviceBuffer,
    pub selection: Option<(&'a DeviceBuffer, usize)>,
    pub recent_key: &'a DeviceBuffer,
    pub recent_key_scales: &'a DeviceBuffer,
    pub recent_value: &'a DeviceBuffer,
    pub recent_value_scales: &'a DeviceBuffer,
    pub batch_key: &'a DeviceBuffer,
    pub batch_key_scales: &'a DeviceBuffer,
    pub batch_value: &'a DeviceBuffer,
    pub batch_value_scales: &'a DeviceBuffer,
    pub row_start: usize,
    pub query_rows: usize,
    pub recent_start: usize,
    pub recent_len: usize,
    pub recent_first_position: usize,
    pub position_start: usize,
    pub causal_batch: bool,
    pub compressed_capacity: usize,
}

const CSA_SEGMENT_CAPACITY: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CsaAttentionQ8SegmentDescriptor {
    compressed_key: u64,
    compressed_key_scales: u64,
    compressed_value: u64,
    compressed_value_scales: u64,
    visible_compressed: u64,
    selection: u64,
    recent_key: u64,
    recent_key_scales: u64,
    recent_value: u64,
    recent_value_scales: u64,
    batch_key: u64,
    batch_key_scales: u64,
    batch_value: u64,
    batch_value_scales: u64,
    row_start: u32,
    query_rows: u32,
    selection_top_k: u32,
    recent_start: u32,
    recent_len: u32,
    recent_first_position: u32,
    position_start: u32,
    causal_batch: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CsaAttentionQ8SegmentTable {
    segments: [CsaAttentionQ8SegmentDescriptor; CSA_SEGMENT_CAPACITY],
}

#[allow(clippy::too_many_arguments)]
pub fn try_csa_attention_q8_segmented(
    device_id: i32,
    query: &DeviceBuffer,
    segments: &[CsaAttentionQ8Segment<'_>],
    sink: Option<&DeviceBuffer>,
    total_query_rows: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
    q8_group_size: usize,
) -> Result<DeviceBuffer, String> {
    if segments.is_empty()
        || segments.len() > CSA_SEGMENT_CAPACITY
        || total_query_rows == 0
        || num_heads == 0
        || num_kv_heads == 0
        || !num_heads.is_multiple_of(num_kv_heads)
        || head_dim != CSA_TILED_HEAD_DIM as usize
        || window_size == 0
        || q8_group_size == 0
        || q8_group_size > 256
        || !q8_group_size.is_power_of_two()
        || !head_dim.is_multiple_of(q8_group_size)
    {
        return Err(format!("ROCm segmented CSA shape 非法: segments={} rows={total_query_rows} heads={num_heads}/{num_kv_heads} dim={head_dim} window={window_size} q8={q8_group_size}", segments.len()));
    }
    let query_elements = total_query_rows.checked_mul(num_heads).and_then(|n| n.checked_mul(head_dim)).ok_or("ROCm segmented CSA query 溢出")?;
    let kv_width = num_kv_heads.checked_mul(head_dim).ok_or("ROCm segmented CSA KV width 溢出")?;
    let scale_width = kv_width / q8_group_size;
    validate_resident(query, device_id, query_elements * 4, "segmented CSA query")?;
    if let Some(sink) = sink {
        validate_resident(sink, device_id, num_heads * 4, "segmented CSA sink")?;
    }
    let mut table = CsaAttentionQ8SegmentTable { segments: [CsaAttentionQ8SegmentDescriptor::default(); CSA_SEGMENT_CAPACITY] };
    let mut expected_row = 0;
    for (index, segment) in segments.iter().enumerate() {
        if segment.query_rows == 0 || segment.row_start != expected_row {
            return Err(format!("ROCm segmented CSA row range 非连续: segment={index} start={} expected={expected_row} rows={}", segment.row_start, segment.query_rows));
        }
        for (codes, scales, rows, name) in [
            (segment.batch_key, segment.batch_key_scales, segment.query_rows, "batch key"),
            (segment.batch_value, segment.batch_value_scales, segment.query_rows, "batch value"),
            (segment.compressed_key, segment.compressed_key_scales, segment.compressed_capacity, "compressed key"),
            (segment.compressed_value, segment.compressed_value_scales, segment.compressed_capacity, "compressed value"),
            (segment.recent_key, segment.recent_key_scales, segment.recent_len.max(1), "recent key"),
            (segment.recent_value, segment.recent_value_scales, segment.recent_len.max(1), "recent value"),
        ] {
            validate_resident(codes, device_id, rows.checked_mul(kv_width).ok_or("ROCm segmented CSA Q8 codes 溢出")?, name)?;
            validate_resident(scales, device_id, rows.checked_mul(scale_width).and_then(|n| n.checked_mul(2)).ok_or("ROCm segmented CSA Q8 scales 溢出")?, name)?;
        }
        validate_resident(segment.visible_compressed, device_id, segment.query_rows * 4, "segmented CSA visible")?;
        if let Some((selection, top_k)) = segment.selection {
            validate_resident(selection, device_id, segment.query_rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("ROCm segmented CSA selection 溢出")?, "segmented CSA selection")?;
        }
        let pointer = |buffer: &DeviceBuffer| buffer.pointer as usize as u64;
        table.segments[index] = CsaAttentionQ8SegmentDescriptor {
            compressed_key: pointer(segment.compressed_key),
            compressed_key_scales: pointer(segment.compressed_key_scales),
            compressed_value: pointer(segment.compressed_value),
            compressed_value_scales: pointer(segment.compressed_value_scales),
            visible_compressed: pointer(segment.visible_compressed),
            selection: segment.selection.map_or(0, |(buffer, _)| pointer(buffer)),
            recent_key: pointer(segment.recent_key),
            recent_key_scales: pointer(segment.recent_key_scales),
            recent_value: pointer(segment.recent_value),
            recent_value_scales: pointer(segment.recent_value_scales),
            batch_key: pointer(segment.batch_key),
            batch_key_scales: pointer(segment.batch_key_scales),
            batch_value: pointer(segment.batch_value),
            batch_value_scales: pointer(segment.batch_value_scales),
            row_start: u32_value("segmented CSA row start", segment.row_start)?,
            query_rows: u32_value("segmented CSA rows", segment.query_rows)?,
            selection_top_k: u32_value("segmented CSA top_k", segment.selection.map_or(0, |(_, top_k)| top_k))?,
            recent_start: u32_value("segmented CSA recent start", segment.recent_start)?,
            recent_len: u32_value("segmented CSA recent len", segment.recent_len)?,
            recent_first_position: u32_value("segmented CSA recent position", segment.recent_first_position)?,
            position_start: u32_value("segmented CSA position", segment.position_start)?,
            causal_batch: u32::from(segment.causal_batch),
        };
        expected_row = expected_row.checked_add(segment.query_rows).ok_or("ROCm segmented CSA rows 溢出")?;
    }
    if expected_row != total_query_rows {
        return Err(format!("ROCm segmented CSA segment rows={expected_row} != total={total_query_rows}"));
    }

    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, query_elements * 4)?;
    let functions = compressed_sparse_functions(device_id)?;
    let mut d_query = query.pointer;
    let mut d_sink = sink.map_or(ptr::null_mut(), |buffer| buffer.pointer);
    let mut d_output = output.pointer;
    let mut segment_count = u32_value("segmented CSA segments", segments.len())?;
    let mut total_rows = u32_value("segmented CSA total rows", total_query_rows)?;
    let mut heads = u32_value("segmented CSA heads", num_heads)?;
    let mut kv_heads = u32_value("segmented CSA KV heads", num_kv_heads)?;
    let mut dim = u32_value("segmented CSA head dim", head_dim)?;
    let mut window = u32_value("segmented CSA window", window_size)?;
    let mut q8_group = u32_value("segmented CSA Q8 group", q8_group_size)?;
    let mut has_sink = u32::from(sink.is_some());
    let heads_per_kv = num_heads / num_kv_heads;
    let (heads_per_block, block) = if heads_per_kv.is_multiple_of(16) {
        (16usize, 512u32)
    } else if heads_per_kv.is_multiple_of(8) {
        (8, 256)
    } else {
        return Err("ROCm segmented CSA 需要每 KV head 至少 8 个 heads".to_owned());
    };
    let head_groups = num_heads.div_ceil(heads_per_block);
    let estimated_keys = segments.iter().map(|segment| segment.selection.map_or(segment.compressed_capacity, |(_, top_k)| top_k).saturating_add(window_size).saturating_add(segment.query_rows)).max().unwrap_or(0);
    let split_segments = if total_query_rows <= 64 && estimated_keys >= 96 { (256 / total_query_rows.saturating_mul(head_groups).max(1)).clamp(2, 32) } else { 0 };
    super::tensor::with_deferred_tensor_workspace(device_id, &[std::mem::size_of::<CsaAttentionQ8SegmentTable>()], |workspace| {
        let descriptor_buffer = workspace.buffer(0);
        let mut d_descriptors = descriptor_buffer.pointer;
        let mut store_arguments = [(&mut table as *mut CsaAttentionQ8SegmentTable).cast(), (&mut d_descriptors as *mut *mut c_void).cast()];
        launch(functions.store_segment_table, 1, 1, CSA_SEGMENT_CAPACITY as u32, &mut store_arguments, "HIP segmented CSA descriptor store")?;
        if split_segments != 0 {
            let partial_elements = total_query_rows.checked_mul(num_heads).and_then(|n| n.checked_mul(split_segments)).and_then(|n| n.checked_mul(head_dim + 2)).ok_or("ROCm segmented CSA partial 溢出")?;
            let partials = DeviceBuffer::allocate_reusable(device_id, partial_elements.checked_mul(4).ok_or("ROCm segmented CSA partial bytes 溢出")?)?;
            let mut d_partials = partials.pointer;
            let mut split = u32_value("segmented CSA split", split_segments)?;
            let mut arguments = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_descriptors as *mut *mut c_void).cast(),
                (&mut d_sink as *mut *mut c_void).cast(),
                (&mut d_partials as *mut *mut c_void).cast(),
                (&mut segment_count as *mut u32).cast(),
                (&mut total_rows as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut kv_heads as *mut u32).cast(),
                (&mut dim as *mut u32).cast(),
                (&mut window as *mut u32).cast(),
                (&mut q8_group as *mut u32).cast(),
                (&mut has_sink as *mut u32).cast(),
                (&mut split as *mut u32).cast(),
            ];
            let grid = total_rows.checked_mul(u32_value("segmented CSA head groups", head_groups)?).and_then(|n| n.checked_mul(split)).ok_or("ROCm segmented CSA split grid 溢出")?;
            launch(functions.attention_segmented_decode_partial, grid, 1, block, &mut arguments, "HIP segmented compressed sparse split partial")?;
            let mut merge_arguments = [
                (&mut d_partials as *mut *mut c_void).cast(),
                (&mut d_output as *mut *mut c_void).cast(),
                (&mut total_rows as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut split as *mut u32).cast(),
                (&mut dim as *mut u32).cast(),
            ];
            launch(functions.attention_decode_merge, total_rows.checked_mul(heads).ok_or("ROCm segmented CSA merge grid 溢出")?, 1, 256, &mut merge_arguments, "HIP segmented compressed sparse split merge")?;
        } else {
            let mut arguments = [
                (&mut d_query as *mut *mut c_void).cast(),
                (&mut d_descriptors as *mut *mut c_void).cast(),
                (&mut d_sink as *mut *mut c_void).cast(),
                (&mut d_output as *mut *mut c_void).cast(),
                (&mut segment_count as *mut u32).cast(),
                (&mut total_rows as *mut u32).cast(),
                (&mut heads as *mut u32).cast(),
                (&mut kv_heads as *mut u32).cast(),
                (&mut dim as *mut u32).cast(),
                (&mut window as *mut u32).cast(),
                (&mut q8_group as *mut u32).cast(),
                (&mut has_sink as *mut u32).cast(),
            ];
            launch(
                functions.attention_segmented_tiled,
                total_rows.checked_mul(u32_value("segmented CSA head groups", head_groups)?).ok_or("ROCm segmented CSA grid 溢出")?,
                1,
                block,
                &mut arguments,
                "HIP segmented compressed sparse tiled attention",
            )?;
        }
        Ok(())
    })?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_csa_attention_q8(
    device_id: i32,
    query: &DeviceBuffer,
    compressed_key: &DeviceBuffer,
    compressed_key_scales: &DeviceBuffer,
    compressed_value: &DeviceBuffer,
    compressed_value_scales: &DeviceBuffer,
    visible_compressed: &DeviceBuffer,
    selection: Option<(&DeviceBuffer, usize)>,
    recent_key: &DeviceBuffer,
    recent_key_scales: &DeviceBuffer,
    recent_value: &DeviceBuffer,
    recent_value_scales: &DeviceBuffer,
    recent_start: usize,
    recent_len: usize,
    recent_first_position: usize,
    batch_key: &DeviceBuffer,
    batch_key_scales: &DeviceBuffer,
    batch_value: &DeviceBuffer,
    batch_value_scales: &DeviceBuffer,
    position_start: usize,
    causal_batch: bool,
    sink: Option<&DeviceBuffer>,
    query_rows: usize,
    compressed_capacity: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
    q8_group_size: usize,
) -> Result<DeviceBuffer, String> {
    if query_rows == 0
        || num_heads == 0
        || num_kv_heads == 0
        || head_dim == 0
        || head_dim > 1024
        || window_size == 0
        || q8_group_size == 0
        || q8_group_size > 256
        || !q8_group_size.is_power_of_two()
        || !head_dim.is_multiple_of(q8_group_size)
        || !num_heads.is_multiple_of(num_kv_heads)
    {
        return Err("ROCm CSA attention shape 非法".to_owned());
    }
    let query_elements = query_rows.checked_mul(num_heads).and_then(|n| n.checked_mul(head_dim)).ok_or("ROCm CSA query 溢出")?;
    let kv_width = num_kv_heads.checked_mul(head_dim).ok_or("ROCm CSA KV width 溢出")?;
    let scale_width = kv_width / q8_group_size;
    validate_resident(query, device_id, query_elements * 4, "CSA query")?;
    for (codes, scales, rows, name) in [
        (batch_key, batch_key_scales, query_rows, "batch key"),
        (batch_value, batch_value_scales, query_rows, "batch value"),
        (compressed_key, compressed_key_scales, compressed_capacity, "compressed key"),
        (compressed_value, compressed_value_scales, compressed_capacity, "compressed value"),
        // recent_start 只会在 ring 填满后推进，此时 committed capacity 已等于 window；
        // 填满前 kernel 只访问 0..recent_len，不能强迫 lazy cache 提前提交完整窗口。
        (recent_key, recent_key_scales, recent_len.max(1), "recent key"),
        (recent_value, recent_value_scales, recent_len.max(1), "recent value"),
    ] {
        let code_bytes = rows.checked_mul(kv_width).ok_or_else(|| format!("ROCm CSA {name} Q8 溢出"))?;
        let scale_bytes = rows.checked_mul(scale_width).and_then(|n| n.checked_mul(2)).ok_or_else(|| format!("ROCm CSA {name} Q8 scales 溢出"))?;
        validate_resident(codes, device_id, code_bytes, &format!("CSA {name} Q8"))?;
        validate_resident(scales, device_id, scale_bytes, &format!("CSA {name} Q8 scales"))?;
    }
    validate_resident(visible_compressed, device_id, query_rows * 4, "CSA visible counts")?;
    if let Some((selection, top_k)) = selection {
        validate_resident(selection, device_id, query_rows * top_k * 4, "CSA selection")?;
    }
    if let Some(sink) = sink {
        validate_resident(sink, device_id, num_heads * 4, "CSA sink")?;
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, query_elements * 4)?;
    let dummy = DeviceBuffer::allocate(device_id, 4)?;
    let functions = compressed_sparse_functions(device_id)?;
    let mut d_query = query.pointer;
    let mut d_compressed_key = compressed_key.pointer;
    let mut d_compressed_key_scales = compressed_key_scales.pointer;
    let mut d_compressed_value = compressed_value.pointer;
    let mut d_compressed_value_scales = compressed_value_scales.pointer;
    let mut d_visible = visible_compressed.pointer;
    let mut d_selection = selection.map_or(dummy.pointer, |(buffer, _)| buffer.pointer);
    let mut d_recent_key = recent_key.pointer;
    let mut d_recent_key_scales = recent_key_scales.pointer;
    let mut d_recent_value = recent_value.pointer;
    let mut d_recent_value_scales = recent_value_scales.pointer;
    let mut d_batch_key = batch_key.pointer;
    let mut d_batch_key_scales = batch_key_scales.pointer;
    let mut d_batch_value = batch_value.pointer;
    let mut d_batch_value_scales = batch_value_scales.pointer;
    let mut d_sink = sink.map_or(dummy.pointer, |buffer| buffer.pointer);
    let mut d_output = output.pointer;
    let mut query_rows_u32 = u32_value("CSA query rows", query_rows)?;
    let mut selection_top_k = u32_value("CSA selection top_k", selection.map_or(0, |(_, top_k)| top_k))?;
    let mut recent_start = u32_value("CSA recent start", recent_start)?;
    let mut recent_len = u32_value("CSA recent len", recent_len)?;
    let mut recent_first_position = u32_value("CSA recent position", recent_first_position)?;
    let mut position_start = u32_value("CSA position", position_start)?;
    let mut causal_batch = u32::from(causal_batch);
    let mut batch_rows = query_rows_u32;
    let mut num_heads = u32_value("CSA heads", num_heads)?;
    let mut num_kv_heads = u32_value("CSA KV heads", num_kv_heads)?;
    let mut head_dim = u32_value("CSA head dim", head_dim)?;
    let mut window_size = u32_value("CSA window", window_size)?;
    let mut q8_group_size = u32_value("CSA Q8 group", q8_group_size)?;
    let mut has_sink = u32::from(sink.is_some());
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_compressed_key as *mut *mut c_void).cast(),
        (&mut d_compressed_key_scales as *mut *mut c_void).cast(),
        (&mut d_compressed_value as *mut *mut c_void).cast(),
        (&mut d_compressed_value_scales as *mut *mut c_void).cast(),
        (&mut d_visible as *mut *mut c_void).cast(),
        (&mut d_selection as *mut *mut c_void).cast(),
        (&mut d_recent_key as *mut *mut c_void).cast(),
        (&mut d_recent_key_scales as *mut *mut c_void).cast(),
        (&mut d_recent_value as *mut *mut c_void).cast(),
        (&mut d_recent_value_scales as *mut *mut c_void).cast(),
        (&mut d_batch_key as *mut *mut c_void).cast(),
        (&mut d_batch_key_scales as *mut *mut c_void).cast(),
        (&mut d_batch_value as *mut *mut c_void).cast(),
        (&mut d_batch_value_scales as *mut *mut c_void).cast(),
        (&mut d_sink as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut query_rows_u32 as *mut u32).cast(),
        (&mut selection_top_k as *mut u32).cast(),
        (&mut recent_start as *mut u32).cast(),
        (&mut recent_len as *mut u32).cast(),
        (&mut recent_first_position as *mut u32).cast(),
        (&mut position_start as *mut u32).cast(),
        (&mut causal_batch as *mut u32).cast(),
        (&mut batch_rows as *mut u32).cast(),
        (&mut num_heads as *mut u32).cast(),
        (&mut num_kv_heads as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
        (&mut window_size as *mut u32).cast(),
        (&mut q8_group_size as *mut u32).cast(),
        (&mut has_sink as *mut u32).cast(),
    ];
    let heads_per_kv = num_heads / num_kv_heads;
    if head_dim <= 128 && head_dim.is_multiple_of(16) && heads_per_kv.is_multiple_of(16) {
        let head_groups = num_heads / 16;
        let grid = query_rows_u32.checked_mul(head_groups).ok_or("ROCm CSA WMMA grid 溢出")?;
        launch(functions.attention_wmma, grid, 1, 256, &mut arguments, "HIP compressed sparse WMMA attention")?;
    } else if head_dim == CSA_TILED_HEAD_DIM && heads_per_kv >= 8 && heads_per_kv.is_multiple_of(8) {
        let (heads_per_block, block) = if heads_per_kv.is_multiple_of(16) { (16, 512) } else { (8, 256) };
        let head_groups = num_heads.div_ceil(heads_per_block);
        // decode/verify 小批次:单遍 tiled 的 grid 只有 rows*head_groups 个
        // block(DeepSeek 形状为 4-24),96 CU 长上下文下大量闲置;key 数足够时
        // 改走 split-KV 两阶段,把 key 序列切段并行后再归并。
        let estimated_keys = selection.map_or(compressed_capacity, |(_, top_k)| top_k).saturating_add(window_size as usize).saturating_add(query_rows);
        let split_segments = if query_rows <= 8 && estimated_keys >= 96 {
            // speculative verify 与单行 decode 必须复用同一归并树；分段数若随
            // query_rows 改变，row 0 也会因浮点归并顺序不同而改变 target token。
            (256 / (head_groups as usize).max(1)).clamp(4, 32)
        } else {
            0
        };
        if split_segments > 0 {
            let partial_elements = query_rows.checked_mul(num_heads as usize).and_then(|n| n.checked_mul(split_segments)).and_then(|n| n.checked_mul(head_dim as usize + 2)).ok_or("ROCm CSA split partial 溢出")?;
            let partials = DeviceBuffer::allocate_reusable(device_id, partial_elements.checked_mul(4).ok_or("ROCm CSA split partial 字节溢出")?)?;
            let mut segments_u32 = u32::try_from(split_segments).map_err(|_| "ROCm CSA split segments 超过 u32".to_owned())?;
            let mut d_partials = partials.pointer;
            let mut split_arguments = arguments.to_vec();
            // partial kernel 用 partials 指针替换第 17 个参数(原 output 位),
            // segments 追加在末尾;直接追加会把 output 当 partials 写越界。
            split_arguments[16] = (&mut d_partials as *mut *mut c_void).cast();
            split_arguments.push((&mut segments_u32 as *mut u32).cast());
            let grid = query_rows_u32.checked_mul(head_groups as u32).and_then(|g| g.checked_mul(segments_u32)).ok_or("ROCm CSA split partial grid 溢出")?;
            launch(functions.attention_decode_partial, grid, 1, block, &mut split_arguments, "HIP compressed sparse decode split partial")?;
            let mut merge_arguments: Vec<*mut c_void> = vec![
                (&mut d_partials as *mut *mut c_void).cast(),
                (&mut d_output as *mut *mut c_void).cast(),
                (&mut query_rows_u32 as *mut u32).cast(),
                (&mut num_heads as *mut u32).cast(),
                (&mut segments_u32 as *mut u32).cast(),
                (&mut head_dim as *mut u32).cast(),
            ];
            let merge_grid = query_rows_u32.checked_mul(num_heads).ok_or("ROCm CSA split merge grid 溢出")?;
            launch(functions.attention_decode_merge, merge_grid, 1, 256, &mut merge_arguments, "HIP compressed sparse decode split merge")?;
        } else {
            let grid = query_rows_u32.checked_mul(head_groups).ok_or("ROCm CSA tiled grid 溢出")?;
            launch(functions.attention_tiled, grid, 1, block, &mut arguments, "HIP compressed sparse tiled attention")?;
        }
    } else {
        let groups = query_rows_u32.checked_mul(num_heads).ok_or("ROCm CSA grid 溢出")?;
        launch(functions.attention, groups.div_ceil(8), 1, 256, &mut arguments, "HIP compressed sparse attention")?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hip_source_keeps_all_deepseek_kernels_parameterized() {
        for kernel in ["csa_gated_compress_f32", "csa_store_pending_f32", "csa_index_scores_f32", "csa_attention_wave_q8"] {
            assert!(COMPRESSED_SPARSE_SOURCE.contains(kernel));
        }
        for literal in ["6144", "DeepSeek-V4-Flash-0731"] {
            assert!(!COMPRESSED_SPARSE_SOURCE.contains(literal));
        }
    }

    #[test]
    fn csa_tiled_dot2_512_matches_cpu_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[csa-tiled-dot2] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let (device_id, rows, heads, head_dim, window, group) = (0, 9usize, 16usize, 512usize, 128usize, 32usize);
        let scale = 1.0f32 / 64.0;
        let scale_bf16 = (scale.to_bits() >> 16) as u16;
        let query = (0..rows * heads * head_dim).map(|index| ((index % 17) as f32 - 8.0) / 32.0).collect::<Vec<_>>();
        let batch = (0..rows * head_dim).map(|index| ((index * 7 + index / head_dim * 3) % 31 + 1) as u8).collect::<Vec<_>>();
        let batch_scales = vec![scale_bf16; rows * head_dim / group];
        let compressed = vec![0u8; head_dim];
        let compressed_scales = vec![scale_bf16; head_dim / group];
        let recent = vec![0u8; window * head_dim];
        let recent_scales = vec![scale_bf16; window * head_dim / group];
        let visible = vec![0u32; rows];
        let bytes_u16 = |values: &[u16]| unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
        let bytes_u32 = |values: &[u32]| unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };

        let query_buffer = DeviceBuffer::upload_f32(device_id, &query).unwrap();
        let batch_buffer = DeviceBuffer::upload(device_id, &batch).unwrap();
        let batch_scale_buffer = DeviceBuffer::upload(device_id, bytes_u16(&batch_scales)).unwrap();
        let compressed_buffer = DeviceBuffer::upload(device_id, &compressed).unwrap();
        let compressed_scale_buffer = DeviceBuffer::upload(device_id, bytes_u16(&compressed_scales)).unwrap();
        let recent_buffer = DeviceBuffer::upload(device_id, &recent).unwrap();
        let recent_scale_buffer = DeviceBuffer::upload(device_id, bytes_u16(&recent_scales)).unwrap();
        let visible_buffer = DeviceBuffer::upload(device_id, bytes_u32(&visible)).unwrap();
        let output = try_csa_attention_q8(
            device_id,
            &query_buffer,
            &compressed_buffer,
            &compressed_scale_buffer,
            &compressed_buffer,
            &compressed_scale_buffer,
            &visible_buffer,
            None,
            &recent_buffer,
            &recent_scale_buffer,
            &recent_buffer,
            &recent_scale_buffer,
            0,
            0,
            0,
            &batch_buffer,
            &batch_scale_buffer,
            &batch_buffer,
            &batch_scale_buffer,
            0,
            true,
            None,
            rows,
            1,
            heads,
            1,
            head_dim,
            window,
            group,
        )
        .expect("CSA tiled dot2");
        let actual = output.download_f32(rows * heads * head_dim).unwrap();

        let bf16 = |value: f32| {
            let bits = value.to_bits();
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
            f32::from_bits(rounded & 0xffff0000)
        };
        let mut expected = vec![0.0f32; actual.len()];
        for row in 0..rows {
            for head in 0..heads {
                let query_base = (row * heads + head) * head_dim;
                let mut scores = Vec::with_capacity(row + 1);
                for key_row in 0..=row {
                    let mut dot = 0.0f32;
                    for column in 0..head_dim {
                        dot += bf16(query[query_base + column]) * bf16(batch[key_row * head_dim + column] as f32 * scale);
                    }
                    scores.push(dot / (head_dim as f32).sqrt());
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores.iter().map(|score| (*score - maximum).exp()).sum::<f32>();
                for column in 0..head_dim {
                    let mut value = 0.0f32;
                    for key_row in 0..=row {
                        value += (scores[key_row] - maximum).exp() * batch[key_row * head_dim + column] as f32 * scale;
                    }
                    expected[query_base + column] = value / denominator;
                }
            }
        }
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() <= expected.abs() * 2.0e-3 + 2.0e-3, "index={index} actual={actual} expected={expected}");
        }
    }
}
