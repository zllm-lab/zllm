//! AttnRes mix HIP kernel。
//!
//! 对 B+1 个候选(B 个历史 block residual + 当前 prefix)按行做 RMS 打分、
//! softmax 与加权求和,算法与 CPU reference / Metal `attn_res_mix_f16` 一致,
//! 全 F32。候选指针通过 device 端指针数组传入,避免 Metal 版的 pack 拷贝。

use super::tensor::validate_resident;
use super::*;

/// 共享数组按 256 线程 / 32 候选静态分配,与 Metal kernel 上限一致。
const ATTENTION_RES_MAX_CANDIDATES: usize = 32;

const ATTENTION_RES_SOURCE: &str = include_str!("attn_res/source.hip");

#[derive(Clone, Copy)]
pub(super) struct AttnResFunctions {
    mix: usize,
}

fn attn_res_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(ATTENTION_RES_SOURCE, "zllm_rocm_attn_res.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

pub(super) fn attn_res_functions(device_id: i32) -> Result<AttnResFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, AttnResFunctions), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm AttnRes kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = attn_res_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData AttnRes"));
        }
        let name = CString::new("zllm_attn_res_mix_f32").unwrap();
        let mut function = ptr::null_mut();
        let status = unsafe { module_get_function(&mut function, module, name.as_ptr()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleGetFunction AttnRes"));
        }
        Ok((module as usize, AttnResFunctions { mix: function as usize }))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

/// 执行 AttnRes mix kernel。`candidates` 末位是 current,其余为 block residual,
/// 顺序与 CPU/Metal 实现一致。返回 output device buffer。
pub(crate) fn try_attn_res_mix_resident_device_f32(device_id: i32, candidates: &[&DeviceBuffer], norm_weight: &DeviceBuffer, projection_weight: &DeviceBuffer, rows: usize, columns: usize, eps: f32) -> Result<DeviceBuffer, String> {
    set_device(device_id)?;
    if candidates.is_empty() || candidates.len() > ATTENTION_RES_MAX_CANDIDATES {
        return Err(format!("AttnRes candidate 数量 {} 非法,允许 1..={ATTENTION_RES_MAX_CANDIDATES}", candidates.len()));
    }
    if rows == 0 || columns == 0 {
        return Err("AttnRes rows/columns 不能为 0".to_owned());
    }

    let candidate_bytes = rows.checked_mul(columns).ok_or("AttnRes candidate 元素溢出")?.checked_mul(4).ok_or("AttnRes candidate 字节溢出")?;
    let weight_bytes = columns.checked_mul(4).ok_or("AttnRes weight 字节溢出")?;
    let output_bytes = candidate_bytes;
    for (index, candidate) in candidates.iter().enumerate() {
        validate_resident(candidate, device_id, candidate_bytes, &format!("AttnRes candidate {index}"))?;
    }
    validate_resident(norm_weight, device_id, weight_bytes, "AttnRes norm weight")?;
    validate_resident(projection_weight, device_id, weight_bytes, "AttnRes projection weight")?;

    let functions = attn_res_functions(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;

    // 候选指针数组上载到 device,kernel 从 global memory 读指针。
    let pointers = candidates.iter().map(|candidate| candidate.pointer as usize).collect::<Vec<_>>();
    let pointer_bytes = unsafe { std::slice::from_raw_parts(pointers.as_ptr().cast::<u8>(), std::mem::size_of_val(pointers.as_slice())) };
    let pointer_array = DeviceBuffer::upload(device_id, pointer_bytes)?;

    let mut d_pointer_array = pointer_array.pointer;
    let mut d_norm = norm_weight.pointer;
    let mut d_projection = projection_weight.pointer;
    let mut d_output = output.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "AttnRes rows 超过 u32".to_owned())?;
    let mut columns_u32 = u32::try_from(columns).map_err(|_| "AttnRes columns 超过 u32".to_owned())?;
    let mut candidate_count_u32 = u32::try_from(candidates.len()).map_err(|_| "AttnRes candidate 数量超过 u32".to_owned())?;
    let mut eps_f32 = eps;

    let mut arguments = [
        (&mut d_pointer_array as *mut *mut c_void).cast(),
        (&mut d_norm as *mut *mut c_void).cast(),
        (&mut d_projection as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut candidate_count_u32 as *mut u32).cast(),
        (&mut eps_f32 as *mut f32).cast(),
    ];

    // 每行一个 block,256 线程分列 stride 循环
    let block = 256u32;
    let grid_x = u32::try_from(rows).map_err(|_| "AttnRes rows 超过 u32 grid".to_owned())?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result = unsafe { launch(functions.mix as *mut c_void, grid_x, 1, 1, block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel AttnRes mix"));
    }

    // 指针数组 buffer 在 launch 后即不可再引用,同步确保 kernel 已消费完毕。
    synchronize_device(device_id, "AttnRes mix synchronize")?;

    Ok(output)
}
