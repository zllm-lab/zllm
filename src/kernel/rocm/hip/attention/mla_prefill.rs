use super::super::tensor::{launch_tensor_kernel, validate_resident};
use super::*;

const MLA_ATTENTION_SOURCE: &str = include_str!("mla_prefill/source.hip");

fn mla_attention_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(MLA_ATTENTION_SOURCE, "zllm_rocm_mla.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

fn mla_attention_function(device_id: i32) -> Result<usize, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, usize, usize), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm MLA kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, function, _)| *function).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = mla_attention_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData resident MLA"));
        }
        let name = CString::new("mla_attention_f32").unwrap();
        let mut function = ptr::null_mut();
        let function_status = unsafe { module_get_function(&mut function, module, name.as_ptr()) };
        if function_status != HIP_SUCCESS {
            return Err(runtime.hip_error(function_status, "hipModuleGetFunction resident MLA"));
        }
        let name = CString::new("gqa_prefill_f32").unwrap();
        let mut gqa = ptr::null_mut();
        let function_status = unsafe { module_get_function(&mut gqa, module, name.as_ptr()) };
        if function_status != HIP_SUCCESS {
            return Err(runtime.hip_error(function_status, "hipModuleGetFunction resident GQA"));
        }
        Ok((module as usize, function as usize, gqa as usize))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, function, _)| function)
}

fn gqa_attention_function(device_id: i32) -> Result<usize, String> {
    let _ = mla_attention_function(device_id)?;
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, usize, usize), String>>>> = OnceLock::new();
    let _ = FUNCTIONS;
    let runtime = RocmRuntime::open()?;
    let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
    let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
    static GQA: OnceLock<Mutex<HashMap<i32, Result<(usize, usize), String>>>> = OnceLock::new();
    let functions = GQA.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm GQA kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, function)| *function).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let code = mla_attention_code()?;
        let mut module = ptr::null_mut();
        let status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLoadData resident GQA"));
        }
        let name = CString::new("gqa_prefill_f32").unwrap();
        let mut function = ptr::null_mut();
        let status = unsafe { module_get_function(&mut function, module, name.as_ptr()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleGetFunction resident GQA"));
        }
        Ok((module as usize, function as usize))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, function)| function)
}

#[allow(clippy::too_many_arguments)]
pub fn try_gqa_prefill_resident_f32(device_id: i32, query: &DeviceBuffer, key: &DeviceBuffer, value: &DeviceBuffer, rows: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize, score_scale: f32) -> Result<DeviceBuffer, String> {
    if rows == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return Err("resident GQA shape 非法".to_owned());
    }
    let block_size = head_dim.next_power_of_two();
    if block_size > 1024 {
        return Err(format!("resident GQA head_dim={head_dim} 超过 block 上限"));
    }
    let query_elements = rows.checked_mul(num_heads).and_then(|value| value.checked_mul(head_dim)).ok_or("resident GQA query 大小溢出")?;
    let kv_elements = rows.checked_mul(num_kv_heads).and_then(|value| value.checked_mul(head_dim)).ok_or("resident GQA KV 大小溢出")?;
    let query_bytes = query_elements.checked_mul(4).ok_or("resident GQA query 字节溢出")?;
    let kv_bytes = kv_elements.checked_mul(4).ok_or("resident GQA KV 字节溢出")?;
    validate_resident(query, device_id, query_bytes, "GQA query")?;
    validate_resident(key, device_id, kv_bytes, "GQA key")?;
    validate_resident(value, device_id, kv_bytes, "GQA value")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, query_bytes)?;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_value = value.pointer;
    let mut d_output = output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident GQA rows 超过 u32".to_owned())?;
    let mut num_heads = u32::try_from(num_heads).map_err(|_| "resident GQA heads 超过 u32".to_owned())?;
    let mut num_kv_heads = u32::try_from(num_kv_heads).map_err(|_| "resident GQA KV heads 超过 u32".to_owned())?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "resident GQA head_dim 超过 u32".to_owned())?;
    let mut score_scale = score_scale;
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_value as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut num_heads as *mut u32).cast(),
        (&mut num_kv_heads as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
        (&mut score_scale as *mut f32).cast(),
    ];
    let grid = rows.checked_mul(num_heads).ok_or("resident GQA grid 溢出")?;
    launch_tensor_kernel(gqa_attention_function(device_id)?, grid, u32::try_from(block_size).map_err(|_| "resident GQA block 超过 u32")?, &mut arguments, "HIP resident GQA prefill")?;
    Ok(output)
}
