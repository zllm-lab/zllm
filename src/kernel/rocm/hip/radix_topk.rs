//! ROCm 通用 stable radix top-k：消费 ordered-u32 score，不绑定注意力布局。

use std::{
    collections::HashMap,
    ffi::{CString, c_void},
    ptr,
    sync::{Mutex, OnceLock},
};

use libloading::Symbol;

use super::tensor::validate_resident;
use super::{DeviceBuffer, HIP_SUCCESS, HipModuleGetFunction, HipModuleLoadData, RocmRuntime, compile_hip_source, set_device};

const RADIX_TOPK_SOURCE: &str = include_str!("radix_topk/source.hip");

fn radix_topk_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(RADIX_TOPK_SOURCE, "zllm_rocm_radix_topk.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

type FunctionCache = Mutex<HashMap<i32, Result<(usize, usize), String>>>;

fn radix_topk_function(device_id: i32) -> Result<usize, String> {
    static FUNCTIONS: OnceLock<FunctionCache> = OnceLock::new();
    let cache = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| "ROCm radix top-k kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = cache.get(&device_id) {
        return result.as_ref().map(|(_, function)| *function).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let get: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = radix_topk_code()?;
        let mut module = ptr::null_mut();
        let status = unsafe { load(&mut module, code.as_ptr().cast()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLoadData radix top-k"));
        }
        let name = CString::new("stable_radix_topk_u32").unwrap();
        let mut function = ptr::null_mut();
        let status = unsafe { get(&mut function, module, name.as_ptr()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleGetFunction radix top-k"));
        }
        Ok((module as usize, function as usize))
    })();
    cache.insert(device_id, result.clone());
    result.map(|(_, function)| function)
}

/// 对 ordered-u32 score 做稳定 top-k。`visible_counts=None` 使用标准 causal
/// `query_start + row + 1`；稀疏压缩布局传每行真实可见数量。
#[allow(clippy::too_many_arguments)]
pub fn try_stable_radix_topk_u32_into(
    device_id: i32,
    score_keys: &DeviceBuffer,
    visible_counts: Option<(&DeviceBuffer, usize)>,
    selection: &DeviceBuffer,
    query_rows: usize,
    score_stride: usize,
    query_start: usize,
    top_k: usize,
    selection_row_offset: usize,
) -> Result<(), String> {
    if query_rows == 0 || score_stride == 0 || top_k == 0 {
        return Err(format!("ROCm radix top-k shape 非法: rows={query_rows} stride={score_stride} top_k={top_k}"));
    }
    let score_bytes = query_rows.checked_mul(score_stride).and_then(|n| n.checked_mul(4)).ok_or("ROCm radix top-k score 大小溢出")?;
    let selection_rows = selection_row_offset.checked_add(query_rows).ok_or("ROCm radix top-k output rows 溢出")?;
    let selection_bytes = selection_rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("ROCm radix top-k output 大小溢出")?;
    validate_resident(score_keys, device_id, score_bytes, "radix top-k scores")?;
    validate_resident(selection, device_id, selection_bytes, "radix top-k selection")?;
    if let Some((visible, offset)) = visible_counts {
        let bytes = offset.checked_add(query_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm radix top-k visible 大小溢出")?;
        validate_resident(visible, device_id, bytes, "radix top-k visible counts")?;
    }
    let function = radix_topk_function(device_id)?;
    let mut d_scores = score_keys.pointer;
    let mut d_visible = visible_counts.map_or(ptr::null_mut(), |(buffer, offset)| unsafe { buffer.pointer.cast::<u32>().add(offset).cast() });
    let mut d_selection = unsafe { selection.pointer.cast::<u32>().add(selection_row_offset * top_k).cast() };
    let mut rows = u32::try_from(query_rows).map_err(|_| "ROCm radix top-k rows 超过 u32")?;
    let mut stride = u32::try_from(score_stride).map_err(|_| "ROCm radix top-k stride 超过 u32")?;
    let mut start = u32::try_from(query_start).map_err(|_| "ROCm radix top-k query_start 超过 u32")?;
    let mut top_k = u32::try_from(top_k).map_err(|_| "ROCm radix top-k top_k 超过 u32")?;
    let mut arguments = [
        (&mut d_scores as *mut *mut c_void).cast(),
        (&mut d_visible as *mut *mut c_void).cast(),
        (&mut d_selection as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut stride as *mut u32).cast(),
        (&mut start as *mut u32).cast(),
        (&mut top_k as *mut u32).cast(),
    ];
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = {
        let started = super::hip_api_stats::start();
        let result = unsafe { launch(function as *mut c_void, rows, 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
        result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "HIP stable radix top-k"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hip_source_is_model_independent() {
        assert!(RADIX_TOPK_SOURCE.contains("stable_radix_topk_u32"));
        for name in ["DeepSeek", "GLM", "DSA", "CSA"] {
            assert!(!RADIX_TOPK_SOURCE.contains(name));
        }
    }
}
