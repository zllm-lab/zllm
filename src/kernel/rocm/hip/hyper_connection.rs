//! mHC 张量变换 HIP 算子;所有中间结果留在设备端 F32 buffer。
//!
//! 与 Metal 版同源(逐元素 kernel + split 单线程 Sinkhorn);ROCm backend 管线
//! 统一 F32 resident tensor,因此不需要 Metal 版的 dtype 分发。

use std::{
    collections::HashMap,
    ffi::{CString, c_void},
    ptr,
    sync::{Mutex, OnceLock},
};

use libloading::Symbol;

use super::tensor::validate_resident;
use super::{DeviceBuffer, HIP_SUCCESS, HipModuleGetFunction, HipModuleLoadData, RocmRuntime, compile_hip_source, set_device};

const MHC_BLOCK: u32 = 256;
const MHC_SPLIT_BLOCK: u32 = 64;

const HYPER_CONNECTION_SOURCE: &str = include_str!("hyper_connection/source.hip");

#[derive(Clone, Copy)]
struct HyperConnectionFunctions {
    expand: usize,
    reduce: usize,
    mean: usize,
    expand_scaled: usize,
    expand_scaled_add: usize,
    mix: usize,
    split: usize,
    prepare_coefficients: usize,
    prepare_outputs: usize,
    head_reduce: usize,
}

fn hyper_connection_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(HYPER_CONNECTION_SOURCE, "zllm_rocm_hyper_connection.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

type FunctionCache = Mutex<HashMap<i32, Result<(usize, HyperConnectionFunctions), String>>>;

fn hyper_connection_functions(device_id: i32) -> Result<HyperConnectionFunctions, String> {
    static FUNCTIONS: OnceLock<FunctionCache> = OnceLock::new();
    let cache = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| "ROCm hyper connection kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = cache.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let get: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = hyper_connection_code()?;
        let mut module = ptr::null_mut();
        let status = unsafe { load(&mut module, code.as_ptr().cast()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLoadData hyper connection"));
        }
        let function = |name: &str| -> Result<usize, String> {
            let name = CString::new(name).unwrap();
            let mut handle = ptr::null_mut();
            let status = unsafe { get(&mut handle, module, name.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction hyper connection"));
            }
            Ok(handle as usize)
        };
        Ok((
            module as usize,
            HyperConnectionFunctions {
                expand: function("mhc_expand_f32")?,
                reduce: function("mhc_reduce_f32")?,
                mean: function("mhc_mean_f32")?,
                expand_scaled: function("mhc_expand_scaled_f32")?,
                expand_scaled_add: function("mhc_expand_scaled_add_f32")?,
                mix: function("mhc_mix_f32")?,
                split: function("mhc_split_f32")?,
                prepare_coefficients: function("mhc_prepare_coefficients_f32")?,
                prepare_outputs: function("mhc_prepare_outputs_f32")?,
                head_reduce: function("mhc_head_reduce_f32")?,
            },
        ))
    })();
    cache.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

fn launch(function: usize, grid_x: u32, block: u32, arguments: &mut [*mut c_void], action: &str) -> Result<(), String> {
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = {
        let started = super::hip_api_stats::start();
        let result = unsafe { launch(function as *mut c_void, grid_x, 1, 1, block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
        result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, action));
    }
    Ok(())
}

fn u32_value(name: &str, value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name}={value} 超过 u32"))
}

fn grid(count: usize, block: u32) -> Result<u32, String> {
    u32_value("mHC grid", count.div_ceil(block as usize))
}

pub fn try_mhc_expand_f32(device_id: i32, hidden: &DeviceBuffer, rows: usize, columns: usize, copies: usize) -> Result<DeviceBuffer, String> {
    let input_elements = rows.checked_mul(columns).ok_or("ROCm mHC expand 输入溢出")?;
    let output_elements = input_elements.checked_mul(copies).ok_or("ROCm mHC expand 输出溢出")?;
    validate_resident(hidden, device_id, input_elements * 4, "mHC hidden")?;
    let functions = hyper_connection_functions(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements * 4)?;
    let mut d_hidden = hidden.pointer;
    let mut d_output = output.pointer;
    let mut columns_u32 = u32_value("mHC columns", columns)?;
    let mut copies_u32 = u32_value("mHC copies", copies)?;
    let mut count_u64 = output_elements as u64;
    launch(
        functions.expand,
        grid(output_elements, MHC_BLOCK)?,
        MHC_BLOCK,
        &mut [(&mut d_hidden as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut columns_u32 as *mut u32).cast(), (&mut copies_u32 as *mut u32).cast(), (&mut count_u64 as *mut u64).cast()],
        "hip mhc expand",
    )?;
    Ok(output)
}

pub fn try_mhc_reduce_f32(device_id: i32, hidden: &DeviceBuffer, coefficients: &DeviceBuffer, rows: usize, width: usize, copies: usize) -> Result<DeviceBuffer, String> {
    let hidden_elements = rows.checked_mul(width).and_then(|value| value.checked_mul(copies)).ok_or("ROCm mHC reduce hidden 溢出")?;
    let output_elements = rows.checked_mul(width).ok_or("ROCm mHC reduce 输出溢出")?;
    validate_resident(hidden, device_id, hidden_elements * 4, "mHC hidden")?;
    validate_resident(coefficients, device_id, rows * copies * 4, "mHC coefficients")?;
    let functions = hyper_connection_functions(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements * 4)?;
    let mut d_hidden = hidden.pointer;
    let mut d_coefficients = coefficients.pointer;
    let mut d_output = output.pointer;
    let mut width_u32 = u32_value("mHC width", width)?;
    let mut copies_u32 = u32_value("mHC copies", copies)?;
    let mut count_u64 = output_elements as u64;
    launch(
        functions.reduce,
        grid(output_elements, MHC_BLOCK)?,
        MHC_BLOCK,
        &mut [
            (&mut d_hidden as *mut *mut c_void).cast(),
            (&mut d_coefficients as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut width_u32 as *mut u32).cast(),
            (&mut copies_u32 as *mut u32).cast(),
            (&mut count_u64 as *mut u64).cast(),
        ],
        "hip mhc reduce",
    )?;
    Ok(output)
}

pub fn try_mhc_mean_f32(device_id: i32, hidden: &DeviceBuffer, rows: usize, width: usize, copies: usize) -> Result<DeviceBuffer, String> {
    let hidden_elements = rows.checked_mul(width).and_then(|value| value.checked_mul(copies)).ok_or("ROCm mHC mean hidden 溢出")?;
    let output_elements = rows.checked_mul(width).ok_or("ROCm mHC mean 输出溢出")?;
    validate_resident(hidden, device_id, hidden_elements * 4, "mHC hidden")?;
    let functions = hyper_connection_functions(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements * 4)?;
    let mut d_hidden = hidden.pointer;
    let mut d_output = output.pointer;
    let mut width_u32 = u32_value("mHC width", width)?;
    let mut copies_u32 = u32_value("mHC copies", copies)?;
    let mut count_u64 = output_elements as u64;
    launch(
        functions.mean,
        grid(output_elements, MHC_BLOCK)?,
        MHC_BLOCK,
        &mut [(&mut d_hidden as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut width_u32 as *mut u32).cast(), (&mut copies_u32 as *mut u32).cast(), (&mut count_u64 as *mut u64).cast()],
        "hip mhc mean",
    )?;
    Ok(output)
}

pub fn try_mhc_expand_scaled_f32(device_id: i32, hidden: &DeviceBuffer, coefficients: &DeviceBuffer, rows: usize, width: usize, copies: usize) -> Result<DeviceBuffer, String> {
    let input_elements = rows.checked_mul(width).ok_or("ROCm mHC expand_scaled 输入溢出")?;
    let output_elements = input_elements.checked_mul(copies).ok_or("ROCm mHC expand_scaled 输出溢出")?;
    validate_resident(hidden, device_id, input_elements * 4, "mHC hidden")?;
    validate_resident(coefficients, device_id, rows * copies * 4, "mHC coefficients")?;
    let functions = hyper_connection_functions(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements * 4)?;
    let mut d_hidden = hidden.pointer;
    let mut d_coefficients = coefficients.pointer;
    let mut d_output = output.pointer;
    let mut width_u32 = u32_value("mHC width", width)?;
    let mut copies_u32 = u32_value("mHC copies", copies)?;
    let mut count_u64 = output_elements as u64;
    launch(
        functions.expand_scaled,
        grid(output_elements, MHC_BLOCK)?,
        MHC_BLOCK,
        &mut [
            (&mut d_hidden as *mut *mut c_void).cast(),
            (&mut d_coefficients as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut width_u32 as *mut u32).cast(),
            (&mut copies_u32 as *mut u32).cast(),
            (&mut count_u64 as *mut u64).cast(),
        ],
        "hip mhc expand_scaled",
    )?;
    Ok(output)
}

pub fn try_mhc_expand_scaled_add_f32(device_id: i32, hidden: &DeviceBuffer, coefficients: &DeviceBuffer, residual: &DeviceBuffer, rows: usize, width: usize, copies: usize) -> Result<DeviceBuffer, String> {
    let input_elements = rows.checked_mul(width).ok_or("ROCm mHC fused expand_scaled_add 输入溢出")?;
    let output_elements = input_elements.checked_mul(copies).ok_or("ROCm mHC fused expand_scaled_add 输出溢出")?;
    validate_resident(hidden, device_id, input_elements * 4, "mHC fused hidden")?;
    validate_resident(coefficients, device_id, rows * copies * 4, "mHC fused coefficients")?;
    validate_resident(residual, device_id, output_elements * 4, "mHC fused residual")?;
    let functions = hyper_connection_functions(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_elements * 4)?;
    let mut d_hidden = hidden.pointer;
    let mut d_coefficients = coefficients.pointer;
    let mut d_residual = residual.pointer;
    let mut d_output = output.pointer;
    let mut width_u32 = u32_value("mHC fused width", width)?;
    let mut copies_u32 = u32_value("mHC fused copies", copies)?;
    let mut count_u64 = output_elements as u64;
    launch(
        functions.expand_scaled_add,
        grid(output_elements, MHC_BLOCK)?,
        MHC_BLOCK,
        &mut [
            (&mut d_hidden as *mut *mut c_void).cast(),
            (&mut d_coefficients as *mut *mut c_void).cast(),
            (&mut d_residual as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut width_u32 as *mut u32).cast(),
            (&mut copies_u32 as *mut u32).cast(),
            (&mut count_u64 as *mut u64).cast(),
        ],
        "hip mhc fused expand scaled add",
    )?;
    Ok(output)
}

pub fn try_mhc_mix_f32(device_id: i32, hidden: &DeviceBuffer, matrix: &DeviceBuffer, rows: usize, width: usize, copies: usize) -> Result<DeviceBuffer, String> {
    let output_elements = rows.checked_mul(width).and_then(|value| value.checked_mul(copies)).ok_or("ROCm mHC mix 输出溢出")?;
    validate_resident(hidden, device_id, output_elements * 4, "mHC hidden")?;
    validate_resident(matrix, device_id, rows * copies * copies * 4, "mHC matrix")?;
    let functions = hyper_connection_functions(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements * 4)?;
    let mut d_hidden = hidden.pointer;
    let mut d_matrix = matrix.pointer;
    let mut d_output = output.pointer;
    let mut width_u32 = u32_value("mHC width", width)?;
    let mut copies_u32 = u32_value("mHC copies", copies)?;
    let mut count_u64 = output_elements as u64;
    launch(
        functions.mix,
        grid(output_elements, MHC_BLOCK)?,
        MHC_BLOCK,
        &mut [
            (&mut d_hidden as *mut *mut c_void).cast(),
            (&mut d_matrix as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut width_u32 as *mut u32).cast(),
            (&mut copies_u32 as *mut u32).cast(),
            (&mut count_u64 as *mut u64).cast(),
        ],
        "hip mhc mix",
    )?;
    Ok(output)
}

#[allow(clippy::type_complexity)]
pub fn try_mhc_split_f32(device_id: i32, mixes: &DeviceBuffer, base: &DeviceBuffer, scale: &DeviceBuffer, rows: usize, copies: usize, iterations: usize, eps: f32) -> Result<(DeviceBuffer, DeviceBuffer, DeviceBuffer), String> {
    let mix_columns = copies.checked_mul(copies + 2).ok_or("ROCm mHC split mix columns 溢出")?;
    validate_resident(mixes, device_id, rows * mix_columns * 4, "mHC mixes")?;
    validate_resident(base, device_id, mix_columns * 4, "mHC base")?;
    validate_resident(scale, device_id, 3 * 4, "mHC scale")?;
    if rows == 0 || copies == 0 || iterations == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(format!("ROCm mHC split shape 非法: rows={rows} copies={copies} iterations={iterations} eps={eps}"));
    }
    let functions = hyper_connection_functions(device_id)?;
    let pre = DeviceBuffer::allocate(device_id, rows * copies * 4)?;
    let post = DeviceBuffer::allocate(device_id, rows * copies * 4)?;
    let combination = DeviceBuffer::allocate(device_id, rows * copies * copies * 4)?;
    let mut d_mixes = mixes.pointer;
    let mut d_base = base.pointer;
    let mut d_scale = scale.pointer;
    let mut d_pre = pre.pointer;
    let mut d_post = post.pointer;
    let mut d_combination = combination.pointer;
    let mut copies_u32 = u32_value("mHC copies", copies)?;
    let mut iterations_u32 = u32_value("mHC iterations", iterations)?;
    let mut eps_f32 = eps;
    let mut rows_u32 = u32_value("mHC rows", rows)?;
    launch(
        functions.split,
        grid(rows, MHC_SPLIT_BLOCK)?,
        MHC_SPLIT_BLOCK,
        &mut [
            (&mut d_mixes as *mut *mut c_void).cast(),
            (&mut d_base as *mut *mut c_void).cast(),
            (&mut d_scale as *mut *mut c_void).cast(),
            (&mut d_pre as *mut *mut c_void).cast(),
            (&mut d_post as *mut *mut c_void).cast(),
            (&mut d_combination as *mut *mut c_void).cast(),
            (&mut copies_u32 as *mut u32).cast(),
            (&mut iterations_u32 as *mut u32).cast(),
            (&mut eps_f32 as *mut f32).cast(),
            (&mut rows_u32 as *mut u32).cast(),
        ],
        "hip mhc split",
    )?;
    Ok((pre, post, combination))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn try_mhc_prepare_sublayer_f32(
    device_id: i32,
    hidden: &DeviceBuffer,
    mixes: &DeviceBuffer,
    base: &DeviceBuffer,
    scale: &DeviceBuffer,
    rows: usize,
    width: usize,
    copies: usize,
    iterations: usize,
    eps: f32,
) -> Result<(DeviceBuffer, DeviceBuffer, DeviceBuffer), String> {
    let mix_columns = copies.checked_mul(copies + 2).ok_or("ROCm mHC prepare mix columns 溢出")?;
    if rows == 0 || copies == 0 || copies > 16 || iterations == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(format!("ROCm mHC prepare shape 非法: rows={rows} width={width} copies={copies} iterations={iterations} eps={eps}"));
    }
    validate_resident(hidden, device_id, rows.checked_mul(width).and_then(|n| n.checked_mul(copies)).and_then(|n| n.checked_mul(4)).ok_or("ROCm mHC prepare hidden 溢出")?, "mHC prepare hidden")?;
    validate_resident(mixes, device_id, rows.checked_mul(mix_columns).and_then(|n| n.checked_mul(4)).ok_or("ROCm mHC prepare mixes 溢出")?, "mHC prepare mixes")?;
    validate_resident(base, device_id, mix_columns * 4, "mHC prepare base")?;
    validate_resident(scale, device_id, 3 * 4, "mHC prepare scale")?;
    let residual_bytes = rows.checked_mul(width).and_then(|n| n.checked_mul(copies)).and_then(|n| n.checked_mul(4)).ok_or("ROCm mHC prepare residual 大小溢出")?;
    let reduced_bytes = rows.checked_mul(width).and_then(|n| n.checked_mul(4)).ok_or("ROCm mHC prepare reduced 大小溢出")?;
    let output_owner = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, residual_bytes.checked_add(reduced_bytes).ok_or("ROCm mHC prepare output 大小溢出")?)?);
    let residual = DeviceBuffer::view(output_owner.clone(), 0, residual_bytes)?;
    let reduced = DeviceBuffer::view(output_owner, residual_bytes, reduced_bytes)?;
    let coefficient_elements = rows.checked_mul(copies).and_then(|n| n.checked_mul(copies + 2)).ok_or("ROCm mHC prepare coefficient 数量溢出")?;
    let coefficient_owner = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, coefficient_elements.checked_mul(4).ok_or("ROCm mHC prepare coefficient 大小溢出")?)?);
    let post_bytes = rows.checked_mul(copies).and_then(|n| n.checked_mul(4)).ok_or("ROCm mHC prepare post 大小溢出")?;
    let post = DeviceBuffer::view(coefficient_owner.clone(), post_bytes, post_bytes)?;
    let functions = hyper_connection_functions(device_id)?;
    let mut d_hidden = hidden.pointer;
    let mut d_mixes = mixes.pointer;
    let mut d_base = base.pointer;
    let mut d_scale = scale.pointer;
    let mut d_coefficients = coefficient_owner.pointer;
    let mut width_u32 = u32_value("mHC prepare width", width)?;
    let mut copies_u32 = u32_value("mHC prepare copies", copies)?;
    let mut iterations_u32 = u32_value("mHC prepare iterations", iterations)?;
    let mut eps_f32 = eps;
    let mut rows_u32 = u32_value("mHC prepare rows", rows)?;
    launch(
        functions.prepare_coefficients,
        u32_value("mHC prepare coefficient grid", rows)?,
        MHC_BLOCK,
        &mut [
            (&mut d_mixes as *mut *mut c_void).cast(),
            (&mut d_base as *mut *mut c_void).cast(),
            (&mut d_scale as *mut *mut c_void).cast(),
            (&mut d_coefficients as *mut *mut c_void).cast(),
            (&mut copies_u32 as *mut u32).cast(),
            (&mut iterations_u32 as *mut u32).cast(),
            (&mut eps_f32 as *mut f32).cast(),
            (&mut rows_u32 as *mut u32).cast(),
        ],
        "hip mhc prepare coefficients",
    )?;
    let mut d_residual = residual.pointer;
    let mut d_reduced = reduced.pointer;
    let output_elements = rows.checked_mul(width).and_then(|n| n.checked_mul(copies + 1)).ok_or("ROCm mHC prepare output 元素数溢出")?;
    launch(
        functions.prepare_outputs,
        grid(output_elements, MHC_BLOCK)?,
        MHC_BLOCK,
        &mut [
            (&mut d_hidden as *mut *mut c_void).cast(),
            (&mut d_coefficients as *mut *mut c_void).cast(),
            (&mut d_residual as *mut *mut c_void).cast(),
            (&mut d_reduced as *mut *mut c_void).cast(),
            (&mut width_u32 as *mut u32).cast(),
            (&mut copies_u32 as *mut u32).cast(),
            (&mut rows_u32 as *mut u32).cast(),
        ],
        "hip mhc prepare outputs",
    )?;
    Ok((residual, reduced, post))
}

pub fn try_mhc_head_reduce_f32(device_id: i32, hidden: &DeviceBuffer, mixes: &DeviceBuffer, base: &DeviceBuffer, scale: &DeviceBuffer, rows: usize, width: usize, copies: usize, eps: f32) -> Result<DeviceBuffer, String> {
    let hidden_elements = rows.checked_mul(width).and_then(|value| value.checked_mul(copies)).ok_or("ROCm mHC head_reduce hidden 溢出")?;
    let output_elements = rows.checked_mul(width).ok_or("ROCm mHC head_reduce 输出溢出")?;
    validate_resident(hidden, device_id, hidden_elements * 4, "mHC hidden")?;
    validate_resident(mixes, device_id, rows * copies * 4, "mHC mixes")?;
    validate_resident(base, device_id, copies * 4, "mHC base")?;
    validate_resident(scale, device_id, 4, "mHC scale")?;
    let functions = hyper_connection_functions(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements * 4)?;
    let mut d_hidden = hidden.pointer;
    let mut d_mixes = mixes.pointer;
    let mut d_base = base.pointer;
    let mut d_scale = scale.pointer;
    let mut d_output = output.pointer;
    let mut width_u32 = u32_value("mHC width", width)?;
    let mut copies_u32 = u32_value("mHC copies", copies)?;
    let mut eps_f32 = eps;
    let mut count_u64 = output_elements as u64;
    launch(
        functions.head_reduce,
        grid(output_elements, MHC_BLOCK)?,
        MHC_BLOCK,
        &mut [
            (&mut d_hidden as *mut *mut c_void).cast(),
            (&mut d_mixes as *mut *mut c_void).cast(),
            (&mut d_base as *mut *mut c_void).cast(),
            (&mut d_scale as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut width_u32 as *mut u32).cast(),
            (&mut copies_u32 as *mut u32).cast(),
            (&mut eps_f32 as *mut f32).cast(),
            (&mut count_u64 as *mut u64).cast(),
        ],
        "hip mhc head_reduce",
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hip_source_keeps_all_mhc_kernels_parameterized() {
        for kernel in ["mhc_expand_f32", "mhc_reduce_f32", "mhc_mean_f32", "mhc_expand_scaled_f32", "mhc_mix_f32", "mhc_split_f32", "mhc_head_reduce_f32"] {
            assert!(HYPER_CONNECTION_SOURCE.contains(kernel), "缺少 kernel {kernel}");
        }
        for literal in ["16384", "4096", "DeepSeek"] {
            assert!(!HYPER_CONNECTION_SOURCE.contains(literal), "kernel 源码出现模型常量 {literal}");
        }
    }
}
