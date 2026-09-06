//! paged MLA HIPRTC 模块的编译、装载与逐设备函数表缓存。

use super::*;

fn code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(&[super::super::super::hiprtc::DEVICE_CONVERSIONS_PREAMBLE, PAGED_MLA_SOURCE].concat(), "zllm_rocm_paged_mla.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

pub(super) fn paged_mla_functions(device_id: i32) -> Result<PagedMlaFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, PagedMlaFunctions), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm paged MLA kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = load(device_id);
    functions.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

fn load(device_id: i32) -> Result<(usize, PagedMlaFunctions), String> {
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let device_get_attribute: Symbol<HipDeviceGetAttribute> = runtime.symbol(&runtime.hip, b"hipDeviceGetAttribute\0")?;
    let mut wavefront_size = 0i32;
    let status = unsafe { device_get_attribute(&mut wavefront_size, 87, device_id) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipDeviceGetAttribute wavefront size"));
    }
    if !matches!(wavefront_size, 32 | 64) {
        return Err(format!("ROCm wavefront size={wavefront_size} 不受支持"));
    }
    let load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
    let get: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
    let mut module = ptr::null_mut();
    let status = unsafe { load(&mut module, code()?.as_ptr().cast()) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLoadData paged MLA"));
    }
    let function = |name: &str| -> Result<usize, String> {
        let name = CString::new(name).expect("HIP kernel 名不含 NUL");
        let mut handle = ptr::null_mut();
        let status = unsafe { get(&mut handle, module, name.as_ptr()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleGetFunction paged MLA"));
        }
        Ok(handle as usize)
    };
    let supports = |name: &str| {
        let name = CString::new(name).expect("HIP kernel 名不含 NUL");
        let mut handle = ptr::null_mut();
        (unsafe { get(&mut handle, module, name.as_ptr()) }) == HIP_SUCCESS
    };
    let dense_wmma = supports("zllm_paged_dense_wmma_marker");
    let decode_partial_wmma_q8 = if dense_wmma { function("mla_paged_decode_partial_q8_wmma_f32")? } else { 0 };
    let decode_partial_wmma_q8_colpar = if dense_wmma { function("mla_paged_decode_partial_q8_wmma_f32_colpar")? } else { 0 };
    let decode_partial_wmma_q8_colpar512 = if dense_wmma { function("mla_paged_decode_partial_q8_wmma_f32_colpar512")? } else { 0 };
    let decode_partial_wmma_q8_colpar512_abl = if dense_wmma { function("mla_paged_decode_partial_q8_wmma_f32_colpar512_abl")? } else { 0 };
    Ok((
        module as usize,
        PagedMlaFunctions {
            cache_append: function("paged_cache_append_f32_bf16")?,
            cache_append_q8: function("paged_cache_append_f32_q8")?,
            cache_append_dsa_prologue_q8: function("paged_dsa_append_layernorm_rope_q8")?,
            cache_append_q8_hadamard: function("paged_cache_append_f32_q8_hadamard")?,
            cache_transform_q8_hadamard: function("paged_cache_transform_q8_hadamard")?,
            dsa_mean_pool: function("dsa_mean_pool_q8")?,
            dsa_interval_pool: function("dsa_interval_pool_q8")?,
            dsa_interval_bounds: function("dsa_interval_score_bounds")?,
            dsa_kpool_compress: function("dsa_kpool_compress_q8")?,
            cache_copy_q8_pair: function("q8_cache_copy_pair")?,
            cache_append_mla_q8_bf16: function("paged_cache_append_mla_f32_q8_bf16")?,
            cache_append_mla_q8_bf16_indirect: function("paged_cache_append_mla_f32_q8_bf16_indirect")?,
            mla_hot_scatter_q8: function("mla_hot_scatter_q8")?,
            mla_hot_gather_q8: function("mla_hot_gather_q8")?,
            dsa_clear: function("dsa_clear_u32")?,
            dsa_gather_selection_scores: function("dsa_gather_selection_scores")?,
            mla_gather_selected_q8: function("mla_gather_selected_q8")?,
            dsa_merge_sequence_shards: function("dsa_merge_sequence_shard_topk")?,
            dsa_score: function("dsa_score_tiles_q8")?,
            dsa_quantize_query_i8: function("dsa_quantize_query_hadamard_i8")?,
            dsa_score_i8: function("dsa_score_tiles_i8")?,
            dsa_score_native_wmma_i8: function("dsa_score_tiles_native_wmma_i8")?,
            dsa_score_wmma: function("dsa_score_tiles_wmma_q8")?,
            dsa_score_native_wmma: function("dsa_score_tiles_native_wmma_q8")?,
            dsa_score_native_wmma_rows2: function("dsa_score_tiles_native_wmma_q8_rows2")?,
            dsa_score_native_wmma_decode: function("dsa_score_tiles_native_wmma_q8_decode")?,
            dsa_score_native_wmma_kpool: function("dsa_score_tiles_native_wmma_q8_kpool")?,
            dsa_score_prefix_native_wmma: function("dsa_score_prefix_tiles_native_wmma_q8")?,
            dsa_score_selected_native_wmma: function("dsa_score_selected_native_wmma_q8")?,
            dsa_map_candidate_selection: function("dsa_map_candidate_selection")?,
            dsa_compact_prefixes: function("dsa_compact_score_prefixes")?,
            dsa_select_prefix: function("dsa_prefix_radix_select_topk")?,
            dsa_select: function("dsa_radix_select_topk")?,
            dsa_select_compact: function("dsa_compact_radix_select_topk")?,
            dsa_select_threshold: function("dsa_compact_radix_threshold")?,
            dsa_select_radix_stage: function("dsa_compact_radix_stage")?,
            dsa_select_tile_counts: function("dsa_count_selected_tiles")?,
            dsa_select_tile_scan: function("dsa_scan_selected_tiles")?,
            dsa_select_tile_scatter: function("dsa_scatter_selected_tiles")?,
            dsa_expand_kpool_selection: function("dsa_expand_kpool_selection")?,
            absorb_query: function("mla_absorb_query_ct")?,
            absorb_query_wmma: function("mla_absorb_query_ct_wmma")?,
            dense_attention: function("mla_paged_dense_latent_f32")?,
            sparse_attention: function("mla_paged_sparse_latent_f32")?,
            dense_wmma,
            decode_attention: function("mla_paged_decode_latent_f32")?,
            decode_partial: function("mla_paged_decode_partial_f32")?,
            decode_partial_wmma_q8,
            decode_partial_wmma_q8_colpar,
            decode_partial_wmma_q8_colpar512,
            decode_partial_wmma_q8_colpar512_abl,
            split_merge: function("mla_paged_split_merge_f32")?,
            split_merge_pl: function("mla_paged_split_merge_f32_pl")?,
            selection_split: function("mla_split_selection_parity")?,
            shard_scale: function("mla_paged_shard_scale_bf16")?,
            shard_merge_heads: function("mla_paged_shard_merge_heads_bf16")?,
            project_value: function("mla_project_value_ct")?,
            project_value_wmma: function("mla_project_value_ct_wmma")?,
            project_value_perm: function("mla_project_value_ct_perm")?,
            wavefront_size: wavefront_size as u32,
        },
    ))
}
