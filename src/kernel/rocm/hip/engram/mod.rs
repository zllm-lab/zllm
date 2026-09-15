//! V4.1 engram 融合算子:host 查表,device 一次 GEMV + 门控修正。

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::tensor::validate_resident;
use super::*;

const ENGRAM_SOURCE: &str = include_str!("source.hip");

#[derive(Clone, Copy)]
struct EngramFunctions {
    gemv: usize,
    apply: usize,
}

fn engram_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(ENGRAM_SOURCE, "zllm_rocm_engram.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

fn engram_functions(device_id: i32) -> Result<EngramFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, EngramFunctions), String>>>> = OnceLock::new();
    let cache = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| "ROCm engram kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = cache.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = super::ffi::RocmRuntime::open()?;
        let load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let get: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = engram_code()?;
        let mut module = std::ptr::null_mut();
        let status = unsafe { load(&mut module, code.as_ptr().cast()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLoadData engram"));
        }
        let function = |name: &str| -> Result<usize, String> {
            let name = std::ffi::CString::new(name).unwrap();
            let mut handle = std::ptr::null_mut();
            let status = unsafe { get(&mut handle, module, name.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction engram"));
            }
            Ok(handle as usize)
        };
        Ok((module as usize, EngramFunctions { gemv: function("engram_gemv_bf16")?, apply: function("engram_gate_apply_f32")? }))
    })();
    cache.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

/// 就地修正 `h`([rows, hc*dim] 行主序)。`embed` 为 [rows, kv_cols] 的解量 F32;
/// `wkv_bf16` 为 [kv_rows, kv_cols] 的 BF16 位型;`qk_weight` 为 [hc, dim]。
/// 两段 kernel:GEMV 写 scratch,再逐 (row, copy) 门控;scratch 由调用方复用。
#[allow(clippy::too_many_arguments)]
pub fn try_engram_apply_f32(device_id: i32, wkv_bf16: &DeviceBuffer, embed: &DeviceBuffer, qk_weight: &DeviceBuffer, h: &mut DeviceBuffer, rows: usize, hc: usize, dim: usize, kv_rows: usize, kv_cols: usize, eps: f32) -> Result<(), String> {
    if rows == 0 || hc == 0 || dim == 0 || kv_rows != dim * (hc + 1) {
        return Err(format!("ROCm engram shape 非法: rows={rows} hc={hc} dim={dim} kv_rows={kv_rows}"));
    }
    validate_resident(wkv_bf16, device_id, kv_rows * kv_cols * 2, "engram wkv")?;
    validate_resident(embed, device_id, rows * kv_cols * 4, "engram embed")?;
    validate_resident(qk_weight, device_id, hc * dim * 4, "engram qk")?;
    validate_resident(h, device_id, rows * hc * dim * 4, "engram h")?;
    set_device(device_id)?;
    let functions = engram_functions(device_id)?;
    let scratch = DeviceBuffer::allocate(device_id, rows.checked_mul(kv_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm engram scratch 大小溢出")?)?;
    let mut d_wkv = wkv_bf16.pointer;
    let mut d_embed = embed.pointer;
    let mut d_scratch = scratch.pointer;
    let mut kv_rows_u32 = u32::try_from(kv_rows).map_err(|_| format!("engram kv_rows={kv_rows} 超过 u32"))?;
    let mut kv_cols_u32 = u32::try_from(kv_cols).map_err(|_| format!("engram kv_cols={kv_cols} 超过 u32"))?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let runtime = super::ffi::RocmRuntime::open()?;
    {
        let rows_u32 = u32::try_from(rows).map_err(|_| format!("engram rows={rows} 超过 u32"))?;
        let blocks = u32::try_from(kv_rows.div_ceil(256)).map_err(|_| "engram gemv blocks 超过 u32".to_owned())?;
        let mut arguments = [
            (&mut d_wkv as *mut *mut std::ffi::c_void).cast(),
            (&mut d_embed as *mut *mut std::ffi::c_void).cast(),
            (&mut d_scratch as *mut *mut std::ffi::c_void).cast(),
            (&mut kv_rows_u32 as *mut u32).cast(),
            (&mut kv_cols_u32 as *mut u32).cast(),
        ];
        let status = unsafe { launch(functions.gemv as *mut std::ffi::c_void, rows_u32, blocks, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), std::ptr::null_mut()) };
        if status != 0 {
            return Err(runtime.hip_error(status, "HIP engram gemv"));
        }
    }
    let mut d_qk = qk_weight.pointer;
    let mut d_h = h.pointer;
    let mut hc_u32 = u32::try_from(hc).map_err(|_| format!("engram hc={hc} 超过 u32"))?;
    let mut dim_u32 = u32::try_from(dim).map_err(|_| format!("engram dim={dim} 超过 u32"))?;
    let mut eps_f32 = eps;
    let mut arguments = [
        (&mut d_scratch as *mut *mut std::ffi::c_void).cast(),
        (&mut d_qk as *mut *mut std::ffi::c_void).cast(),
        (&mut d_h as *mut *mut std::ffi::c_void).cast(),
        (&mut hc_u32 as *mut u32).cast(),
        (&mut dim_u32 as *mut u32).cast(),
        (&mut kv_rows_u32 as *mut u32).cast(),
        (&mut eps_f32 as *mut f32).cast(),
    ];
    let grid_x = u32::try_from(hc).map_err(|_| format!("engram hc={hc} 超过 u32"))?;
    let grid_y = u32::try_from(rows).map_err(|_| format!("engram rows={rows} 超过 u32"))?;
    let block = 256u32;
    let status = unsafe { launch(functions.apply as *mut std::ffi::c_void, grid_x, grid_y, 1, block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), std::ptr::null_mut()) };
    if status != 0 {
        return Err(runtime.hip_error(status, "HIP engram apply"));
    }
    Ok(())
}

/// CPU 已提前完成 WKV 投影时，只在 GPU 上执行门控并就地修正 resident hidden。
/// projected 生命周期由 device pool 延迟回收覆盖本次 compute stream。
#[allow(clippy::too_many_arguments)]
pub fn try_engram_apply_projected_f32(
    device_id: i32,
    projected: &DeviceBuffer,
    qk_weight: &DeviceBuffer,
    h: &DeviceBuffer,
    rows: usize,
    hc: usize,
    dim: usize,
    eps: f32,
) -> Result<(), String> {
    if rows == 0 || hc == 0 || dim == 0 {
        return Err(format!("ROCm projected engram shape 非法: rows={rows} hc={hc} dim={dim}"));
    }
    let kv_rows = dim.checked_mul(hc + 1).ok_or("ROCm projected engram kv_rows 溢出")?;
    validate_resident(projected, device_id, rows.checked_mul(kv_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm projected engram 大小溢出")?, "engram projected")?;
    validate_resident(qk_weight, device_id, hc.checked_mul(dim).and_then(|n| n.checked_mul(4)).ok_or("ROCm projected engram qk 大小溢出")?, "engram qk")?;
    validate_resident(h, device_id, rows.checked_mul(hc).and_then(|n| n.checked_mul(dim)).and_then(|n| n.checked_mul(4)).ok_or("ROCm projected engram hidden 大小溢出")?, "engram h")?;
    set_device(device_id)?;
    let functions = engram_functions(device_id)?;
    let mut d_projected = projected.pointer;
    let mut d_qk = qk_weight.pointer;
    let mut d_h = h.pointer;
    let mut hc_u32 = u32::try_from(hc).map_err(|_| format!("engram hc={hc} 超过 u32"))?;
    let mut dim_u32 = u32::try_from(dim).map_err(|_| format!("engram dim={dim} 超过 u32"))?;
    let mut kv_rows_u32 = u32::try_from(kv_rows).map_err(|_| format!("engram kv_rows={kv_rows} 超过 u32"))?;
    let mut eps_f32 = eps;
    let mut arguments = [
        (&mut d_projected as *mut *mut std::ffi::c_void).cast(),
        (&mut d_qk as *mut *mut std::ffi::c_void).cast(),
        (&mut d_h as *mut *mut std::ffi::c_void).cast(),
        (&mut hc_u32 as *mut u32).cast(),
        (&mut dim_u32 as *mut u32).cast(),
        (&mut kv_rows_u32 as *mut u32).cast(),
        (&mut eps_f32 as *mut f32).cast(),
    ];
    let grid_x = u32::try_from(hc).map_err(|_| format!("engram hc={hc} 超过 u32"))?;
    let grid_y = u32::try_from(rows).map_err(|_| format!("engram rows={rows} 超过 u32"))?;
    let status = unsafe {
        crate::kernel::rocm::hip::kernel_launch_trampoline(
            functions.apply as *mut std::ffi::c_void,
            grid_x,
            grid_y,
            1,
            256,
            1,
            1,
            0,
            crate::kernel::rocm::hip::active_compute_stream(),
            arguments.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if status != 0 {
        return Err(super::ffi::RocmRuntime::open()?.hip_error(status, "HIP projected engram apply"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// HIPRTC 预热:只触发 engram source 的运行时编译,不执行 kernel;
    /// 无 ROCm 环境自动跳过。
    #[test]
    fn engram_kernel_compiles_on_hip() {
        if !super::super::is_hip_available() {
            eprintln!("[engram] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        assert!(super::engram_functions(0).is_ok(), "engram HIPRTC 编译失败: {:?}", super::engram_functions(0).err());
    }
}
