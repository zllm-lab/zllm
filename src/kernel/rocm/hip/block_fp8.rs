//! Block-scaled FP8 (E4M3 + E8M0) GEMV,ROCm 入口。
//!
//! RDNA3 (gfx1100) **没有 FP8 tensor core**,也不依赖
//! `__builtin_amdgcn_fdot2_f32_e4m3`:kernel 内直接把 E4M3 升到 F32 做 dot,
//! **weight 访存砍半** (1B FP8 vs 2B BF16),收益直接体现在 decode 访存墙上。
//!
//! kernel 模式:
//! - `block_fp8_gemv_f32` (`input_rows == 1`):每个 WG 处理 weight 一行。
//! - `block_fp8_small_n_f32` (`input_rows <= 8`):每个 WG 读取一次 weight，
//!   在寄存器中同时累计多行，供推测验证等小批量路径复用权重带宽。
//!
//! 第一版仅支持 `block_rows == block_cols == 128`(DeepSeek-V4 spec);其他 block
//! 形态返回 `Err`,由 caller 走老 `decode → BF16 WMMA` 路径。

use std::{
    collections::HashMap,
    ffi::{CString, c_void},
    ptr,
    sync::{Mutex, OnceLock},
};

use libloading::Symbol;

use crate::moe::Activation;

use super::{DeviceBuffer, HIP_SUCCESS, HipModuleGetFunction, HipModuleLoadData, RocmRuntime, compile_hip_source, options, set_device, synchronize_device};

const BLOCK_FP8_SOURCE: &str = include_str!("block_fp8/source.hip");

fn block_fp8_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(BLOCK_FP8_SOURCE, "zllm_rocm_block_fp8.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

#[derive(Clone, Copy)]
struct BlockFp8Functions {
    matvec: usize,
    small_n: usize,
    three_segment: usize,
    dual_matvec: usize,
    gated_matvec: usize,
    grouped_columns_matvec: usize,
    decode: usize,
}

fn block_fp8_functions(device_id: i32) -> Result<BlockFp8Functions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<BlockFp8Functions, String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm block-fp8 kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|functions| *functions).map_err(Clone::clone);
    }
    let result = (|| -> Result<BlockFp8Functions, String> {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = block_fp8_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData block-fp8"));
        }
        let mut matvec: *mut c_void = ptr::null_mut();
        let mut small_n: *mut c_void = ptr::null_mut();
        let mut three_segment: *mut c_void = ptr::null_mut();
        let mut dual_matvec: *mut c_void = ptr::null_mut();
        let mut gated_matvec: *mut c_void = ptr::null_mut();
        let mut grouped_columns_matvec: *mut c_void = ptr::null_mut();
        let mut decode: *mut c_void = ptr::null_mut();
        for (name, target) in [
            ("block_fp8_gemv_f32", &mut matvec),
            ("block_fp8_small_n_f32", &mut small_n),
            ("block_fp8_three_segment_bf16_f32", &mut three_segment),
            ("block_fp8_dual_gemv_f32", &mut dual_matvec),
            ("block_fp8_gated_gemv_f32", &mut gated_matvec),
            ("block_fp8_grouped_columns_gemv_f32", &mut grouped_columns_matvec),
            ("block_fp8_decode_bf16", &mut decode),
        ] {
            let cname = CString::new(name).unwrap();
            let status = unsafe { module_get_function(target, module, cname.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction block-fp8"));
            }
        }
        Ok(BlockFp8Functions {
            matvec: matvec as usize,
            small_n: small_n as usize,
            three_segment: three_segment as usize,
            dual_matvec: dual_matvec as usize,
            gated_matvec: gated_matvec as usize,
            grouped_columns_matvec: grouped_columns_matvec as usize,
            decode: decode as usize,
        })
    })();
    functions.insert(device_id, result.clone());
    result
}

/// 把 `[M,K]` E4M3 codes + E8M0 block scales 解码成 `[M,K]` BF16 resident buffer。
///
/// gfx11 没有 FP8 计算单元,prefill 大 GEMM 走"解码 → BF16 WMMA"路径;解码一次
/// 的结果由调用方按权重缓存。kernel 异步提交,消费方在同流上接续 GEMM。
pub fn try_block_fp8_decode_bf16(device_id: i32, codes: &DeviceBuffer, scales: &DeviceBuffer, weight_rows: usize, columns: usize, block_rows: usize, block_cols: usize) -> Result<DeviceBuffer, String> {
    if block_rows != 128 || block_cols != 128 {
        return Err(format!("ROCm block-fp8 decode 只支持 128×128 block，实际 [{block_rows},{block_cols}]"));
    }
    let elements = weight_rows.checked_mul(columns).ok_or("ROCm block-fp8 decode 元素数溢出")?;
    let scale_bytes = weight_rows.div_ceil(block_rows).checked_mul(columns.div_ceil(block_cols)).ok_or("ROCm block-fp8 decode scales 溢出")?;
    if codes.device_id() != device_id || scales.device_id() != device_id {
        return Err("ROCm block-fp8 decode 输入 buffer 跨 device".to_owned());
    }
    if codes.bytes() != elements || scales.bytes() != scale_bytes {
        return Err(format!("ROCm block-fp8 decode bytes codes={}/{} scales={}/{}", codes.bytes(), elements, scales.bytes(), scale_bytes));
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, elements.checked_mul(2).ok_or("ROCm block-fp8 decode 输出字节溢出")?)?;
    let functions = block_fp8_functions(device_id)?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut d_codes = codes.pointer;
    let mut d_scales = scales.pointer;
    let mut d_output = output.pointer;
    let mut elements_u64 = elements as u64;
    let mut columns_u32 = u32::try_from(columns).map_err(|_| "ROCm block-fp8 decode columns 超过 u32")?;
    let mut block_rows_u32 = u32::try_from(block_rows).map_err(|_| "ROCm block-fp8 decode block_rows 超过 u32")?;
    let mut block_cols_u32 = u32::try_from(block_cols).map_err(|_| "ROCm block-fp8 decode block_cols 超过 u32")?;
    let mut arguments = [
        (&mut d_codes as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut elements_u64 as *mut u64).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut block_rows_u32 as *mut u32).cast(),
        (&mut block_cols_u32 as *mut u32).cast(),
    ];
    let grid = u32::try_from(elements.div_ceil(256)).map_err(|_| "ROCm block-fp8 decode grid 超过 u32")?;
    let status = {
        let started = super::hip_api_stats::start();
        let result = unsafe { launch(functions.decode as *mut c_void, grid, 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
        result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel block-fp8 decode"));
    }
    Ok(output)
}

/// 在 RD device 上跑 `output[N,M] = input[N,K] @ weight_fp8[M,K]^T`。
///
/// `codes` 是 `[M, K]` E4M3 行优先,`scales` 是 `[M/BR, K/BC]` E8M0。
/// 仅支持 `BR == BC == 128`(V4 spec);其他 block 形态返回 `Err`。
pub fn try_block_fp8_matmul_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    codes: &DeviceBuffer,
    scales: &DeviceBuffer,
    input_rows: usize,
    input_columns: usize,
    weight_rows: usize,
    block_rows: usize,
    block_cols: usize,
) -> Result<DeviceBuffer, String> {
    if block_rows != 128 || block_cols != 128 {
        return Err(format!("ROCm block-fp8 只支持 128×128 block，实际 [{block_rows},{block_cols}]"));
    }
    if input_rows == 0 || input_columns == 0 || weight_rows == 0 {
        return Err(format!("ROCm block-fp8 shape rows={input_rows} cols={input_columns} weight_rows={weight_rows} 不能为 0"));
    }
    if !input_columns.is_multiple_of(16) {
        return Err(format!("ROCm block-fp8 GEMM input cols={input_columns} 不是 16 的倍数"));
    }
    let input_elements = input_rows.checked_mul(input_columns).ok_or("ROCm block-fp8 input 大小溢出")?;
    let code_bytes = weight_rows.checked_mul(input_columns).ok_or("ROCm block-fp8 codes 大小溢出")?;
    let scale_bytes = weight_rows.div_ceil(block_rows).checked_mul(input_columns.div_ceil(block_cols)).ok_or("ROCm block-fp8 scales 大小溢出")?;
    let output_elements = input_rows.checked_mul(weight_rows).ok_or("ROCm block-fp8 output 大小溢出")?;
    if input.device_id() != device_id || codes.device_id() != device_id || scales.device_id() != device_id {
        return Err("ROCm block-fp8 输入 buffer 跨 device".to_owned());
    }
    let input_bytes = input_elements.checked_mul(4).ok_or("ROCm block-fp8 input 字节数溢出")?;
    if input.bytes() != input_bytes || codes.bytes() != code_bytes || scales.bytes() != scale_bytes {
        return Err(format!("ROCm block-fp8 buffer bytes input={}/{} codes={}/{} scales={}/{}", input.bytes(), input_bytes, codes.bytes(), code_bytes, scales.bytes(), scale_bytes));
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_elements.checked_mul(4).ok_or("ROCm block-fp8 output 字节溢出")?)?;
    let functions = block_fp8_functions(device_id)?;

    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut d_input = input.pointer;
    let mut d_codes = codes.pointer;
    let mut d_scales = scales.pointer;
    let mut d_output = output.pointer;
    let mut input_rows_u32 = u32::try_from(input_rows).map_err(|_| "ROCm block-fp8 input rows 超过 u32")?;
    let mut input_columns_u32 = u32::try_from(input_columns).map_err(|_| "ROCm block-fp8 input columns 超过 u32")?;
    let mut weight_rows_u32 = u32::try_from(weight_rows).map_err(|_| "ROCm block-fp8 weight rows 超过 u32")?;
    let mut block_row_shift_u32 = block_rows.trailing_zeros();
    let mut block_col_shift_u32 = block_cols.trailing_zeros();
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_codes as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut input_rows_u32 as *mut u32).cast(),
        (&mut input_columns_u32 as *mut u32).cast(),
        (&mut weight_rows_u32 as *mut u32).cast(),
        (&mut block_row_shift_u32 as *mut u32).cast(),
        (&mut block_col_shift_u32 as *mut u32).cast(),
    ];

    let profile_started = options().block_fp8_profile.then(std::time::Instant::now);
    if input_rows == 1 {
        // gfx1100 是 wave32；每行只启动一个 wave，避免第二个 wave 的归约结果被丢弃。
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe {
                launch(
                    functions.matvec as *mut c_void,
                    u32::try_from(weight_rows).map_err(|_| "ROCm block-fp8 gemv grid x 超过 u32")?,
                    1,
                    1,
                    32,
                    1,
                    1,
                    0,
                    crate::kernel::rocm::hip::active_compute_stream(),
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel block-fp8 gemv"));
        }
    } else if input_rows <= 8 {
        let status = {
            let started = super::hip_api_stats::start();
            let result = unsafe {
                launch(
                    functions.small_n as *mut c_void,
                    u32::try_from(weight_rows).map_err(|_| "ROCm block-fp8 small-N grid x 超过 u32")?,
                    1,
                    1,
                    32,
                    1,
                    1,
                    0,
                    crate::kernel::rocm::hip::active_compute_stream(),
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
            result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel block-fp8 small-N"));
        }
    } else {
        return Err(format!("ROCm block-fp8 small-N 只支持 1–8 行，实际 {input_rows}"));
    }
    if profile_started.is_some() {
        synchronize_device(device_id, "block-fp8 matmul profile")?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-block-fp8] device={device_id} n={input_rows} m={input_columns} k={weight_rows} wall={:.6}s", started.elapsed().as_secs_f64(),);
    }
    // output 与后继消费者都在同一默认 stream；DeviceBuffer 释放同样按 stream
    // 排序，不能在 resident 热路径插入整卡同步。
    Ok(output)
}

/// 三份等宽 BF16 输入直接对应 `[M, 3K]` Block-FP8 权重的三个列段。
#[allow(clippy::too_many_arguments)]
pub fn try_block_fp8_three_segment_bf16_resident_f32(
    device_id: i32,
    first: &DeviceBuffer,
    second: &DeviceBuffer,
    third: &DeviceBuffer,
    codes: &DeviceBuffer,
    scales: &DeviceBuffer,
    input_rows: usize,
    segment_columns: usize,
    weight_rows: usize,
    block_rows: usize,
    block_cols: usize,
) -> Result<DeviceBuffer, String> {
    if input_rows == 0 || input_rows > 8 || segment_columns == 0 || weight_rows == 0 || block_rows != 128 || block_cols != 128 || !segment_columns.is_multiple_of(16) {
        return Err(format!("ROCm three-segment BlockFP8 shape rows={input_rows} cols={segment_columns} weight_rows={weight_rows} block=[{block_rows},{block_cols}] 非法"));
    }
    let input_columns = segment_columns.checked_mul(3).ok_or("ROCm three-segment BlockFP8 columns 溢出")?;
    let segment_bytes = input_rows.checked_mul(segment_columns).and_then(|n| n.checked_mul(2)).ok_or("ROCm three-segment BlockFP8 input 溢出")?;
    let code_bytes = weight_rows.checked_mul(input_columns).ok_or("ROCm three-segment BlockFP8 codes 溢出")?;
    let scale_bytes = weight_rows.div_ceil(block_rows).checked_mul(input_columns.div_ceil(block_cols)).ok_or("ROCm three-segment BlockFP8 scales 溢出")?;
    if [first, second, third, codes, scales].iter().any(|buffer| buffer.device_id() != device_id) {
        return Err("ROCm three-segment BlockFP8 输入 buffer 跨 device".to_owned());
    }
    if first.bytes() != segment_bytes || second.bytes() != segment_bytes || third.bytes() != segment_bytes || codes.bytes() != code_bytes || scales.bytes() != scale_bytes {
        return Err(format!("ROCm three-segment BlockFP8 bytes input={}/{}/{} expected={} codes={}/{} scales={}/{}", first.bytes(), second.bytes(), third.bytes(), segment_bytes, codes.bytes(), code_bytes, scales.bytes(), scale_bytes,));
    }
    set_device(device_id)?;
    let output_bytes = input_rows.checked_mul(weight_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm three-segment BlockFP8 output 溢出")?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    let functions = block_fp8_functions(device_id)?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut d_first = first.pointer;
    let mut d_second = second.pointer;
    let mut d_third = third.pointer;
    let mut d_codes = codes.pointer;
    let mut d_scales = scales.pointer;
    let mut d_output = output.pointer;
    let mut input_rows_u32 = u32::try_from(input_rows).map_err(|_| "ROCm three-segment BlockFP8 input rows 超过 u32")?;
    let mut segment_columns_u32 = u32::try_from(segment_columns).map_err(|_| "ROCm three-segment BlockFP8 columns 超过 u32")?;
    let mut weight_rows_u32 = u32::try_from(weight_rows).map_err(|_| "ROCm three-segment BlockFP8 weight rows 超过 u32")?;
    let mut block_row_shift_u32 = block_rows.trailing_zeros();
    let mut block_col_shift_u32 = block_cols.trailing_zeros();
    let mut arguments = [
        (&mut d_first as *mut *mut c_void).cast(),
        (&mut d_second as *mut *mut c_void).cast(),
        (&mut d_third as *mut *mut c_void).cast(),
        (&mut d_codes as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut input_rows_u32 as *mut u32).cast(),
        (&mut segment_columns_u32 as *mut u32).cast(),
        (&mut weight_rows_u32 as *mut u32).cast(),
        (&mut block_row_shift_u32 as *mut u32).cast(),
        (&mut block_col_shift_u32 as *mut u32).cast(),
    ];
    let started = super::hip_api_stats::start();
    let status = unsafe {
        launch(
            functions.three_segment as *mut c_void,
            u32::try_from(weight_rows).map_err(|_| "ROCm three-segment BlockFP8 grid 超过 u32")?,
            1,
            1,
            32,
            1,
            1,
            0,
            crate::kernel::rocm::hip::active_compute_stream(),
            arguments.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel three-segment block-fp8"));
    }
    Ok(output)
}

/// 最多 8 个 token、同输入的两路 Block-FP8 GEMV 合并成一次提交。
#[allow(clippy::too_many_arguments)]
pub fn try_block_fp8_dual_gemv_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    input_rows: usize,
    first_codes: &DeviceBuffer,
    first_scales: &DeviceBuffer,
    first_rows: usize,
    second_codes: &DeviceBuffer,
    second_scales: &DeviceBuffer,
    second_rows: usize,
    input_columns: usize,
    block_rows: usize,
    block_cols: usize,
) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    if input_rows == 0 || input_rows > 8 || block_rows != 128 || block_cols != 128 || input_columns == 0 || first_rows == 0 || second_rows == 0 || !input_columns.is_multiple_of(16) {
        return Err(format!("ROCm dual block-fp8 shape input_rows={input_rows} cols={input_columns} rows={first_rows}/{second_rows} block=[{block_rows},{block_cols}] 非法"));
    }
    let input_bytes = input_rows.checked_mul(input_columns).and_then(|n| n.checked_mul(4)).ok_or("ROCm dual block-fp8 input 溢出")?;
    let matrix_bytes = |rows: usize, divisor: usize| rows.checked_mul(input_columns / divisor).ok_or_else(|| "ROCm dual block-fp8 matrix 溢出".to_owned());
    let first_code_bytes = matrix_bytes(first_rows, 1)?;
    let second_code_bytes = matrix_bytes(second_rows, 1)?;
    let scale_columns = input_columns.div_ceil(block_cols);
    let first_scale_bytes = first_rows.div_ceil(block_rows).checked_mul(scale_columns).ok_or("ROCm dual block-fp8 first scales 溢出")?;
    let second_scale_bytes = second_rows.div_ceil(block_rows).checked_mul(scale_columns).ok_or("ROCm dual block-fp8 second scales 溢出")?;
    if input.device_id() != device_id || first_codes.device_id() != device_id || first_scales.device_id() != device_id || second_codes.device_id() != device_id || second_scales.device_id() != device_id {
        return Err("ROCm dual block-fp8 输入 buffer 跨 device".to_owned());
    }
    if input.bytes() != input_bytes || first_codes.bytes() != first_code_bytes || first_scales.bytes() != first_scale_bytes || second_codes.bytes() != second_code_bytes || second_scales.bytes() != second_scale_bytes {
        return Err(format!(
            "ROCm dual block-fp8 bytes input={}/{} first={}/{}+{}/{} second={}/{}+{}/{}",
            input.bytes(),
            input_bytes,
            first_codes.bytes(),
            first_code_bytes,
            first_scales.bytes(),
            first_scale_bytes,
            second_codes.bytes(),
            second_code_bytes,
            second_scales.bytes(),
            second_scale_bytes
        ));
    }
    set_device(device_id)?;
    let first_output_bytes = input_rows.checked_mul(first_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm dual block-fp8 first output 溢出")?;
    let second_output_bytes = input_rows.checked_mul(second_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm dual block-fp8 second output 溢出")?;
    let owner = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, first_output_bytes.checked_add(second_output_bytes).ok_or("ROCm dual block-fp8 output 溢出")?)?);
    let functions = block_fp8_functions(device_id)?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut d_input = input.pointer;
    let mut d_first_codes = first_codes.pointer;
    let mut d_first_scales = first_scales.pointer;
    let mut d_second_codes = second_codes.pointer;
    let mut d_second_scales = second_scales.pointer;
    let mut d_output = owner.pointer;
    let mut input_rows_u32 = u32::try_from(input_rows).map_err(|_| "ROCm dual block-fp8 input rows 超过 u32")?;
    let mut columns_u32 = u32::try_from(input_columns).map_err(|_| "ROCm dual block-fp8 columns 超过 u32")?;
    let mut first_rows_u32 = u32::try_from(first_rows).map_err(|_| "ROCm dual block-fp8 first rows 超过 u32")?;
    let mut second_rows_u32 = u32::try_from(second_rows).map_err(|_| "ROCm dual block-fp8 second rows 超过 u32")?;
    let mut block_row_shift_u32 = block_rows.trailing_zeros();
    let mut block_col_shift_u32 = block_cols.trailing_zeros();
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_first_codes as *mut *mut c_void).cast(),
        (&mut d_first_scales as *mut *mut c_void).cast(),
        (&mut d_second_codes as *mut *mut c_void).cast(),
        (&mut d_second_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut input_rows_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut first_rows_u32 as *mut u32).cast(),
        (&mut second_rows_u32 as *mut u32).cast(),
        (&mut block_row_shift_u32 as *mut u32).cast(),
        (&mut block_col_shift_u32 as *mut u32).cast(),
    ];
    let combined_rows = first_rows.checked_add(second_rows).ok_or("ROCm dual block-fp8 rows 溢出")?;
    let profile_started = options().block_fp8_profile.then(std::time::Instant::now);
    let stats_started = super::hip_api_stats::start();
    let status = unsafe {
        launch(
            functions.dual_matvec as *mut c_void,
            u32::try_from(combined_rows).map_err(|_| "ROCm dual block-fp8 grid 超过 u32")?,
            input_rows_u32,
            1,
            32,
            1,
            1,
            0,
            crate::kernel::rocm::hip::active_compute_stream(),
            arguments.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel dual block-fp8 gemv"));
    }
    if profile_started.is_some() {
        synchronize_device(device_id, "dual block-fp8 profile")?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-block-fp8-dual] device={device_id} n={input_rows} m={input_columns} rows={first_rows}/{second_rows} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    let first = DeviceBuffer::view(owner.clone(), 0, first_output_bytes)?;
    let second = DeviceBuffer::view(owner, first_output_bytes, second_output_bytes)?;
    Ok((first, second))
}

/// 最多 8 个 token、同 shape 的 BlockFP8 gate/up GEMV 与门控激活合并为一次提交。
#[allow(clippy::too_many_arguments)]
pub fn try_block_fp8_gated_gemv_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    input_rows: usize,
    gate_codes: &DeviceBuffer,
    gate_scales: &DeviceBuffer,
    up_codes: &DeviceBuffer,
    up_scales: &DeviceBuffer,
    rows: usize,
    input_columns: usize,
    block_rows: usize,
    block_cols: usize,
    activation: &Activation,
) -> Result<DeviceBuffer, String> {
    if input_rows == 0 || input_rows > 8 || block_rows != 128 || block_cols != 128 || input_columns == 0 || rows == 0 || !input_columns.is_multiple_of(16) {
        return Err(format!("ROCm gated block-fp8 shape input_rows={input_rows} cols={input_columns} rows={rows} block=[{block_rows},{block_cols}] 非法"));
    }
    let input_bytes = input_rows.checked_mul(input_columns).and_then(|n| n.checked_mul(4)).ok_or("ROCm gated block-fp8 input 溢出")?;
    let code_bytes = rows.checked_mul(input_columns).ok_or("ROCm gated block-fp8 codes 溢出")?;
    let scale_bytes = rows.div_ceil(block_rows).checked_mul(input_columns.div_ceil(block_cols)).ok_or("ROCm gated block-fp8 scales 溢出")?;
    if input.device_id() != device_id || gate_codes.device_id() != device_id || gate_scales.device_id() != device_id || up_codes.device_id() != device_id || up_scales.device_id() != device_id {
        return Err("ROCm gated block-fp8 输入 buffer 跨 device".to_owned());
    }
    if input.bytes() != input_bytes || gate_codes.bytes() != code_bytes || gate_scales.bytes() != scale_bytes || up_codes.bytes() != code_bytes || up_scales.bytes() != scale_bytes {
        return Err(format!(
            "ROCm gated block-fp8 bytes input={}/{} gate={}/{}+{}/{} up={}/{}+{}/{}",
            input.bytes(),
            input_bytes,
            gate_codes.bytes(),
            code_bytes,
            gate_scales.bytes(),
            scale_bytes,
            up_codes.bytes(),
            code_bytes,
            up_scales.bytes(),
            scale_bytes
        ));
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, input_rows.checked_mul(rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm gated block-fp8 output 溢出")?)?;
    let functions = block_fp8_functions(device_id)?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let (kind, alpha, beta, limit, linear_beta, has_linear_beta): (u32, f32, f32, f32, f32, u32) = match activation {
        Activation::Silu => (0, 0.0, 0.0, 0.0, 0.0, 0),
        Activation::SiluClamped { limit } => (1, 0.0, 0.0, *limit, 0.0, 0),
        Activation::Situ { beta, linear_beta } => (2, 0.0, *beta, 0.0, linear_beta.unwrap_or(0.0), u32::from(linear_beta.is_some())),
        Activation::SwigluOai { alpha, limit } => (3, *alpha, 0.0, *limit, 0.0, 0),
        Activation::GeluTanh => (4, 0.0, 0.0, 0.0, 0.0, 0),
    };
    let mut d_input = input.pointer;
    let mut d_gate_codes = gate_codes.pointer;
    let mut d_gate_scales = gate_scales.pointer;
    let mut d_up_codes = up_codes.pointer;
    let mut d_up_scales = up_scales.pointer;
    let mut d_output = output.pointer;
    let mut input_rows_u32 = u32::try_from(input_rows).map_err(|_| "ROCm gated block-fp8 input rows 超过 u32")?;
    let mut columns_u32 = u32::try_from(input_columns).map_err(|_| "ROCm gated block-fp8 columns 超过 u32")?;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "ROCm gated block-fp8 rows 超过 u32")?;
    let mut block_row_shift_u32 = block_rows.trailing_zeros();
    let mut block_col_shift_u32 = block_cols.trailing_zeros();
    let mut kind = kind;
    let mut alpha = alpha;
    let mut beta = beta;
    let mut limit = limit;
    let mut linear_beta = linear_beta;
    let mut has_linear_beta = has_linear_beta;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_gate_codes as *mut *mut c_void).cast(),
        (&mut d_gate_scales as *mut *mut c_void).cast(),
        (&mut d_up_codes as *mut *mut c_void).cast(),
        (&mut d_up_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut input_rows_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut block_row_shift_u32 as *mut u32).cast(),
        (&mut block_col_shift_u32 as *mut u32).cast(),
        (&mut kind as *mut u32).cast(),
        (&mut alpha as *mut f32).cast(),
        (&mut beta as *mut f32).cast(),
        (&mut limit as *mut f32).cast(),
        (&mut linear_beta as *mut f32).cast(),
        (&mut has_linear_beta as *mut u32).cast(),
    ];
    let profile_started = options().block_fp8_profile.then(std::time::Instant::now);
    let stats_started = super::hip_api_stats::start();
    let status = unsafe { launch(functions.gated_matvec as *mut c_void, rows_u32, input_rows_u32, 1, 32, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel gated block-fp8 gemv"));
    }
    if profile_started.is_some() {
        synchronize_device(device_id, "gated block-fp8 profile")?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-block-fp8-gated] device={device_id} n={input_rows} m={input_columns} rows={rows} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}

/// 单 token 的按列分组 BlockFP8 GEMV。输入布局为 `[groups, K]`，每组权重
/// 为 `[M, K]`，输出直接按组拼成 `[groups * M]`；连续权重只提交一次 kernel。
pub fn try_block_fp8_grouped_columns_gemv_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weights: &[(&DeviceBuffer, &DeviceBuffer)],
    input_rows: usize,
    rows_per_group: usize,
    group_columns: usize,
    block_rows: usize,
    block_cols: usize,
) -> Result<DeviceBuffer, String> {
    let group_count = weights.len();
    if group_count == 0 || input_rows == 0 || input_rows > 8 || rows_per_group == 0 || group_columns == 0 || !group_columns.is_multiple_of(16) || block_rows != 128 || block_cols != 128 || !rows_per_group.is_multiple_of(block_rows) {
        return Err(format!("ROCm grouped-columns BlockFP8 shape input_rows={input_rows} groups={group_count} rows={rows_per_group} cols={group_columns} block=[{block_rows},{block_cols}] 非法"));
    }
    let input_bytes = input_rows.checked_mul(group_count).and_then(|n| n.checked_mul(group_columns)).and_then(|n| n.checked_mul(4)).ok_or("ROCm grouped-columns BlockFP8 input 溢出")?;
    let code_bytes = rows_per_group.checked_mul(group_columns).ok_or("ROCm grouped-columns BlockFP8 codes 溢出")?;
    let scale_bytes = rows_per_group.div_ceil(block_rows).checked_mul(group_columns.div_ceil(block_cols)).ok_or("ROCm grouped-columns BlockFP8 scales 溢出")?;
    if input.device_id() != device_id || input.bytes() != input_bytes {
        return Err(format!("ROCm grouped-columns BlockFP8 input device/bytes={}/{}，期望 device={device_id} bytes={input_bytes}", input.device_id(), input.bytes()));
    }
    let first_codes = weights[0].0;
    let first_scales = weights[0].1;
    for (group, &(codes, scales)) in weights.iter().enumerate() {
        let expected_codes = (first_codes.pointer as usize).checked_add(group.checked_mul(code_bytes).ok_or("ROCm grouped-columns code offset 溢出")?).ok_or("ROCm grouped-columns code pointer 溢出")?;
        let expected_scales = (first_scales.pointer as usize).checked_add(group.checked_mul(scale_bytes).ok_or("ROCm grouped-columns scale offset 溢出")?).ok_or("ROCm grouped-columns scale pointer 溢出")?;
        if codes.device_id() != device_id || scales.device_id() != device_id || codes.bytes() != code_bytes || scales.bytes() != scale_bytes || codes.pointer as usize != expected_codes || scales.pointer as usize != expected_scales {
            return Err(format!("ROCm grouped-columns BlockFP8 group={group} 权重不是连续等宽 view"));
        }
    }
    set_device(device_id)?;
    let total_rows = group_count.checked_mul(rows_per_group).ok_or("ROCm grouped-columns BlockFP8 output rows 溢出")?;
    let output = DeviceBuffer::allocate_reusable(device_id, input_rows.checked_mul(total_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm grouped-columns BlockFP8 output 溢出")?)?;
    let functions = block_fp8_functions(device_id)?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut d_input = input.pointer;
    let mut d_codes = first_codes.pointer;
    let mut d_scales = first_scales.pointer;
    let mut d_output = output.pointer;
    let mut input_rows_u32 = u32::try_from(input_rows).map_err(|_| "ROCm grouped-columns input rows 超过 u32")?;
    let mut columns_u32 = u32::try_from(group_columns).map_err(|_| "ROCm grouped-columns columns 超过 u32")?;
    let mut rows_u32 = u32::try_from(rows_per_group).map_err(|_| "ROCm grouped-columns rows 超过 u32")?;
    let mut groups_u32 = u32::try_from(group_count).map_err(|_| "ROCm grouped-columns groups 超过 u32")?;
    let mut block_row_shift_u32 = block_rows.trailing_zeros();
    let mut block_col_shift_u32 = block_cols.trailing_zeros();
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_codes as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut input_rows_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut groups_u32 as *mut u32).cast(),
        (&mut block_row_shift_u32 as *mut u32).cast(),
        (&mut block_col_shift_u32 as *mut u32).cast(),
    ];
    let stats_started = super::hip_api_stats::start();
    let status = unsafe {
        launch(
            functions.grouped_columns_matvec as *mut c_void,
            u32::try_from(total_rows).map_err(|_| "ROCm grouped-columns grid 超过 u32")?,
            1,
            1,
            32,
            1,
            1,
            0,
            crate::kernel::rocm::hip::active_compute_stream(),
            arguments.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel grouped-columns block-fp8 gemv"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn skip_if_no_rocm(test_name: &str) -> Option<i32> {
        if !super::super::is_hip_available() {
            eprintln!("[{test_name}] 跳过：本机未检测到 ROCm 运行时");
            return None;
        }
        // 默认走 device 0；多卡时上层显式覆盖 ZLLM_ROCM_DEVICE_ID。
        let device_id = std::env::var("ZLLM_ROCM_DEVICE_ID").ok().and_then(|value| value.parse::<i32>().ok()).unwrap_or(0);
        Some(device_id)
    }

    fn build_block_fp8(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<u8>) {
        // 与 CPU oracle 测试同源合成：确定性 + scale=127 ⇒ scale=1.0。
        // 长行不能混入 NaN code，否则整行两侧都变成 NaN，会把漏算 lane 的 bug 掩盖掉。
        let codes: Vec<u8> = (0..rows * cols).map(|index: usize| ((index.wrapping_mul(37).wrapping_add(13 + seed as usize)) % 256) as u8).map(|code| if code & 0x7f == 0x7f { code - 1 } else { code }).collect();
        let scales = vec![127u8; rows.div_ceil(128) * cols.div_ceil(128)];
        (codes, scales)
    }

    fn upload_codes_scales_input(device_id: i32, codes: &[u8], scales: &[u8], input: &[f32]) -> Result<(DeviceBuffer, DeviceBuffer, DeviceBuffer), String> {
        let codes_buf = DeviceBuffer::upload(device_id, codes)?;
        let scales_buf = DeviceBuffer::upload(device_id, scales)?;
        let input_bytes = unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), input.len() * 4) };
        let input_buf = DeviceBuffer::upload(device_id, input_bytes)?;
        Ok((codes_buf, scales_buf, input_buf))
    }

    fn readback_f32(buffer: &DeviceBuffer, elements: usize) -> Result<Vec<f32>, String> {
        let mut bytes = vec![0u8; elements * 4];
        buffer.copy_to_host(&mut bytes)?;
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), elements) }.to_vec())
    }

    #[test]
    fn block_fp8_decode_bf16_matches_cpu_oracle() {
        let Some(device_id) = skip_if_no_rocm("block_fp8_decode_bf16") else { return };
        let (codes, scales) = build_block_fp8(256, 512, 7);
        // 非单位 scale 才能验证 E8M0 路径:全部块取 2^3。
        let scales: Vec<u8> = scales.iter().map(|_| 130u8).collect();
        let codes_buf = DeviceBuffer::upload(device_id, &codes).expect("upload codes");
        let scales_buf = DeviceBuffer::upload(device_id, &scales).expect("upload scales");
        let decoded = try_block_fp8_decode_bf16(device_id, &codes_buf, &scales_buf, 256, 512, 128, 128).expect("decode");
        let mut bf16_bytes = vec![0u8; 256 * 512 * 2];
        decoded.copy_to_host(&mut bf16_bytes).expect("readback");
        let decoded_f32: Vec<f32> = bf16_bytes.chunks_exact(2).map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect();
        // CPU oracle:BlockFp8Matrix::decode 与 GPU decode(BF16 后)逐元素一致。
        let matrix = crate::weight::format::block_fp8::BlockFp8Matrix::new(codes, scales, 256, 512, 128, 128).expect("matrix");
        let expected = matrix.decode();
        for (index, (actual, expected)) in decoded_f32.iter().zip(expected.iter()).enumerate() {
            // 合成数据含 E4M3 NaN 编码(S.1111.111),两侧一致地解码为 NaN 即通过。
            assert!(actual.is_nan() && expected.is_nan() || (actual - expected).abs() <= expected.abs() * 1.0e-2 + 1.0e-6, "index={index} actual={actual} expected={expected}");
        }
    }

    fn check_gemv_matches_cpu_oracle(device_id: i32, rows: usize, cols: usize, seed: u32) {
        let (codes, scales) = build_block_fp8(rows, cols, seed);
        let input: Vec<f32> = (0..cols).map(|column| (column as f32) * 0.013 + 0.001).collect();

        let (codes_buf, scales_buf, input_buf) = upload_codes_scales_input(device_id, &codes, &scales, &input).expect("upload");

        let started = Instant::now();
        let output = try_block_fp8_matmul_resident_f32(device_id, &input_buf, &codes_buf, &scales_buf, 1, cols, rows, 128, 128).expect("gemv launch");
        eprintln!("[gemv] device={device_id} rows={rows} cols={cols} wall={:.6}s", started.elapsed().as_secs_f64());
        let actual = readback_f32(&output, rows).expect("readback");

        // ground truth = CPU oracle。
        let mut expected = vec![0.0f32; rows];
        crate::kernel::cpu::block_fp8::matvec_block_fp8_matrix(&codes, &scales, 128, 128, rows, cols, &input, &mut expected).expect("cpu oracle");
        assert_eq!(actual.len(), expected.len(), "gemv 长度不一致");
        for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            let diff = (a - e).abs();
            assert!(diff < 5.0e-2, "gemv row={index} actual={a} expected={e} diff={diff}");
        }
    }

    #[test]
    fn rocm_block_fp8_gemv_matches_cpu_oracle() {
        let Some(device_id) = skip_if_no_rocm("gemv") else { return };
        check_gemv_matches_cpu_oracle(device_id, 64, 256, 0);
    }

    #[test]
    fn rocm_block_fp8_small_n_matches_cpu_oracle() {
        let Some(device_id) = skip_if_no_rocm("small-n") else { return };
        let (rows, cols, input_rows) = (64usize, 2048usize, 6usize);
        let (codes, scales) = build_block_fp8(rows, cols, 11);
        let input = (0..input_rows * cols).map(|index| (index as f32 * 0.007).sin() * 0.1).collect::<Vec<_>>();
        let (codes_buffer, scales_buffer, input_buffer) = upload_codes_scales_input(device_id, &codes, &scales, &input).unwrap();
        let output = try_block_fp8_matmul_resident_f32(device_id, &input_buffer, &codes_buffer, &scales_buffer, input_rows, cols, rows, 128, 128).unwrap();
        let actual = readback_f32(&output, input_rows * rows).unwrap();
        for token in 0..input_rows {
            let mut expected = vec![0.0f32; rows];
            crate::kernel::cpu::block_fp8::matvec_block_fp8_matrix(&codes, &scales, 128, 128, rows, cols, &input[token * cols..(token + 1) * cols], &mut expected).unwrap();
            for (row, (&actual, &expected)) in actual[token * rows..(token + 1) * rows].iter().zip(&expected).enumerate() {
                assert!(actual.is_nan() && expected.is_nan() || (actual - expected).abs() < 5.0e-2, "small-N token={token} row={row} actual={actual} expected={expected}");
            }
        }
    }

    #[test]
    fn rocm_block_fp8_three_segment_bf16_matches_concat_bitwise() {
        let Some(device_id) = skip_if_no_rocm("three-segment-bf16") else { return };
        let (rows, segment_columns, input_rows) = (128usize, 2048usize, 6usize);
        let columns = segment_columns * 3;
        let (codes, scales) = build_block_fp8(rows, columns, 23);
        let mut segments = [Vec::new(), Vec::new(), Vec::new()];
        let mut joined = Vec::with_capacity(input_rows * columns);
        for token in 0..input_rows {
            for segment in 0..3 {
                let values = (0..segment_columns).map(|column| half::bf16::from_f32((((token * columns + segment * segment_columns + column) as f32) * 0.0017).sin() * 0.1)).collect::<Vec<_>>();
                joined.extend(values.iter().map(|value| value.to_f32()));
                segments[segment].extend(values.into_iter().map(half::bf16::to_bits));
            }
        }
        let codes = DeviceBuffer::upload(device_id, &codes).unwrap();
        let scales = DeviceBuffer::upload(device_id, &scales).unwrap();
        let joined = DeviceBuffer::upload_f32(device_id, &joined).unwrap();
        let segments = segments.map(|segment| {
            let bytes = unsafe { std::slice::from_raw_parts(segment.as_ptr().cast::<u8>(), std::mem::size_of_val(segment.as_slice())) };
            DeviceBuffer::upload(device_id, bytes).unwrap()
        });
        let expected = try_block_fp8_matmul_resident_f32(device_id, &joined, &codes, &scales, input_rows, columns, rows, 128, 128).unwrap();
        let actual = try_block_fp8_three_segment_bf16_resident_f32(device_id, &segments[0], &segments[1], &segments[2], &codes, &scales, input_rows, segment_columns, rows, 128, 128).unwrap();
        let expected = readback_f32(&expected, input_rows * rows).unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let actual = readback_f32(&actual, input_rows * rows).unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    /// cols=256 时只有 lane 0..15 有非零贡献,掩盖了 warp reduce 起点偏移的 bug;
    /// 本用例 cols=2048 让全部 64 个 lane 都参与归约,覆盖完整归约跨度。
    #[test]
    fn rocm_block_fp8_small_n_first_row_matches_gemv_bitwise() {
        let Some(device_id) = skip_if_no_rocm("small-n-exactness") else { return };
        let (rows, cols, input_rows) = (128usize, 7168usize, 6usize);
        let (codes, scales) = build_block_fp8(rows, cols, 17);
        let first = (0..cols).map(|index| (index as f32 * 0.0037).sin() * 0.1).collect::<Vec<_>>();
        let mut input = Vec::with_capacity(input_rows * cols);
        input.extend_from_slice(&first);
        input.extend((cols..input_rows * cols).map(|index| (index as f32 * 0.0061).cos() * 0.1));
        let (codes_buffer, scales_buffer, batch_buffer) = upload_codes_scales_input(device_id, &codes, &scales, &input).unwrap();
        let single_buffer = DeviceBuffer::upload_f32(device_id, &first).unwrap();
        let single = try_block_fp8_matmul_resident_f32(device_id, &single_buffer, &codes_buffer, &scales_buffer, 1, cols, rows, 128, 128).unwrap();
        let batch = try_block_fp8_matmul_resident_f32(device_id, &batch_buffer, &codes_buffer, &scales_buffer, input_rows, cols, rows, 128, 128).unwrap();
        let single = readback_f32(&single, rows).unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let batch = readback_f32(&batch, input_rows * rows).unwrap()[..rows].iter().copied().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(single, batch);
    }

    #[test]
    fn rocm_block_fp8_gemv_reduction_covers_all_lanes() {
        let Some(device_id) = skip_if_no_rocm("gemv-reduce") else { return };
        check_gemv_matches_cpu_oracle(device_id, 64, 2048, 1);
    }

    #[test]
    fn rocm_block_fp8_gemv_includes_second_wave_columns() {
        let Some(device_id) = skip_if_no_rocm("gemv-second-wave") else { return };
        let (rows, cols) = (1usize, 1024usize);
        let mut codes = vec![0u8; rows * cols];
        codes[512..].fill(0x38); // E4M3 1.0，贡献只放在 block 的第二个 wave。
        let scales = vec![127u8; rows.div_ceil(128) * cols.div_ceil(128)];
        let input = vec![1.0f32; cols];
        let (codes, scales, input) = upload_codes_scales_input(device_id, &codes, &scales, &input).unwrap();
        let output = try_block_fp8_matmul_resident_f32(device_id, &input, &codes, &scales, 1, cols, rows, 128, 128).unwrap();
        assert_eq!(readback_f32(&output, rows).unwrap(), vec![512.0]);
    }

    #[test]
    fn rocm_block_fp8_dual_gemv_matches_cpu_oracle() {
        let Some(device_id) = skip_if_no_rocm("dual-gemv") else { return };
        let cols = 2048;
        let input_rows = 5;
        let (first_codes, first_scales) = build_block_fp8(64, cols, 3);
        let (second_codes, second_scales) = build_block_fp8(96, cols, 9);
        let input: Vec<f32> = (0..input_rows * cols).map(|index| index as f32 * 0.0009 - 0.4).collect();
        let first_codes_buf = DeviceBuffer::upload(device_id, &first_codes).unwrap();
        let first_scales_buf = DeviceBuffer::upload(device_id, &first_scales).unwrap();
        let second_codes_buf = DeviceBuffer::upload(device_id, &second_codes).unwrap();
        let second_scales_buf = DeviceBuffer::upload(device_id, &second_scales).unwrap();
        let input_buf = DeviceBuffer::upload_f32(device_id, &input).unwrap();
        let (first, second) = try_block_fp8_dual_gemv_resident_f32(device_id, &input_buf, input_rows, &first_codes_buf, &first_scales_buf, 64, &second_codes_buf, &second_scales_buf, 96, cols, 128, 128).unwrap();
        for (actual, codes, scales, rows) in [(readback_f32(&first, input_rows * 64).unwrap(), &first_codes, &first_scales, 64), (readback_f32(&second, input_rows * 96).unwrap(), &second_codes, &second_scales, 96)] {
            let mut expected = vec![0.0f32; input_rows * rows];
            for token in 0..input_rows {
                crate::kernel::cpu::block_fp8::matvec_block_fp8_matrix(codes, scales, 128, 128, rows, cols, &input[token * cols..][..cols], &mut expected[token * rows..][..rows]).unwrap();
            }
            for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                assert!(a.is_nan() && e.is_nan() || (a - e).abs() < 5.0e-2, "dual gemv row={index} actual={a} expected={e}");
            }
        }
    }

    #[test]
    fn rocm_block_fp8_gated_gemv_matches_unfused_bitwise() {
        let Some(device_id) = skip_if_no_rocm("gated-gemv") else { return };
        let rows = 64usize;
        let cols = 2048usize;
        let input_rows = 5usize;
        // 逐 bit 对照不能混入 E4M3 NaN 编码，否则两条合法路径可能只差 NaN 符号位。
        let gate_codes = (0..rows * cols).map(|index| 0x20_u8 + (index % 31) as u8).collect::<Vec<_>>();
        let up_codes = (0..rows * cols).map(|index| (0x20_u8 + ((index * 7 + 3) % 31) as u8) | if index.is_multiple_of(3) { 0x80 } else { 0 }).collect::<Vec<_>>();
        let scale_elements = rows.div_ceil(128) * cols.div_ceil(128);
        let gate_scales = vec![120_u8; scale_elements];
        let up_scales = vec![120_u8; scale_elements];
        let input: Vec<f32> = (0..input_rows * cols).map(|index| (index as f32 * 0.0009).sin() * 0.1).collect();
        let gate_codes = DeviceBuffer::upload(device_id, &gate_codes).unwrap();
        let gate_scales = DeviceBuffer::upload(device_id, &gate_scales).unwrap();
        let up_codes = DeviceBuffer::upload(device_id, &up_codes).unwrap();
        let up_scales = DeviceBuffer::upload(device_id, &up_scales).unwrap();
        let input = DeviceBuffer::upload_f32(device_id, &input).unwrap();
        let activations = [Activation::Silu, Activation::SiluClamped { limit: 7.0 }, Activation::Situ { beta: 2.0, linear_beta: Some(1.5) }, Activation::SwigluOai { alpha: 1.702, limit: 7.0 }, Activation::GeluTanh];
        for activation in activations {
            let (gate, up) = try_block_fp8_dual_gemv_resident_f32(device_id, &input, input_rows, &gate_codes, &gate_scales, rows, &up_codes, &up_scales, rows, cols, 128, 128).unwrap();
            let expected = super::super::try_gated_activation_resident_f32(device_id, &gate, &up, input_rows * rows, &activation).unwrap();
            let actual = try_block_fp8_gated_gemv_resident_f32(device_id, &input, input_rows, &gate_codes, &gate_scales, &up_codes, &up_scales, rows, cols, 128, 128, &activation).unwrap();
            let expected = readback_f32(&expected, input_rows * rows).unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
            let actual = readback_f32(&actual, input_rows * rows).unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(actual, expected, "activation={activation:?}");
        }
    }

    #[test]
    fn rocm_block_fp8_grouped_columns_gemv_matches_cpu_oracle() {
        let Some(device_id) = skip_if_no_rocm("grouped-columns-gemv") else { return };
        let (input_rows, groups, rows, cols) = (5usize, 3usize, 128usize, 256usize);
        let (codes, scales) = build_block_fp8(groups * rows, cols, 11);
        let input: Vec<f32> = (0..input_rows * groups * cols).map(|index| index as f32 * 0.007 - 0.3).collect();
        let code_owner = std::sync::Arc::new(DeviceBuffer::upload(device_id, &codes).unwrap());
        let scale_owner = std::sync::Arc::new(DeviceBuffer::upload(device_id, &scales).unwrap());
        let input_buffer = DeviceBuffer::upload_f32(device_id, &input).unwrap();
        let code_bytes = rows * cols;
        let scale_bytes = rows.div_ceil(128) * cols.div_ceil(128);
        let code_views: Vec<DeviceBuffer> = (0..groups).map(|group| DeviceBuffer::view(code_owner.clone(), group * code_bytes, code_bytes).unwrap()).collect();
        let scale_views: Vec<DeviceBuffer> = (0..groups).map(|group| DeviceBuffer::view(scale_owner.clone(), group * scale_bytes, scale_bytes).unwrap()).collect();
        let weights: Vec<(&DeviceBuffer, &DeviceBuffer)> = code_views.iter().zip(&scale_views).collect();
        let output = try_block_fp8_grouped_columns_gemv_resident_f32(device_id, &input_buffer, &weights, input_rows, rows, cols, 128, 128).unwrap();
        let actual = readback_f32(&output, input_rows * groups * rows).unwrap();
        let mut expected = vec![0.0f32; input_rows * groups * rows];
        for token in 0..input_rows {
            for group in 0..groups {
                crate::kernel::cpu::block_fp8::matvec_block_fp8_matrix(
                    &codes[group * code_bytes..(group + 1) * code_bytes],
                    &scales[group * scale_bytes..(group + 1) * scale_bytes],
                    128,
                    128,
                    rows,
                    cols,
                    &input[(token * groups + group) * cols..(token * groups + group + 1) * cols],
                    &mut expected[(token * groups + group) * rows..(token * groups + group + 1) * rows],
                )
                .unwrap();
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(actual.is_nan() && expected.is_nan() || (actual - expected).abs() < 5.0e-2, "grouped gemv index={index} actual={actual} expected={expected}");
        }
    }

    /// GEMM(大批量多行)尚未实现,必须显式报错而非静默输出错误结果;
    /// 1–8 行自 small-N kernel 落地后已是合法路径。
    #[test]
    fn rocm_block_fp8_gemm_prefill_rejects_until_implemented() {
        let Some(device_id) = skip_if_no_rocm("gemm") else { return };
        let rows = 128;
        let cols = 256;
        let n_inputs = 16;
        let (codes, scales) = build_block_fp8(rows, cols, 7);
        let input: Vec<f32> = (0..n_inputs * cols).map(|index| (index as f32 * 0.021) - 0.7).collect();
        let (codes_buf, scales_buf, input_buf) = upload_codes_scales_input(device_id, &codes, &scales, &input).expect("upload");
        let error = try_block_fp8_matmul_resident_f32(device_id, &input_buf, &codes_buf, &scales_buf, n_inputs, cols, rows, 128, 128).expect_err("未支持的多行必须显式报错");
        assert!(error.contains("只支持"), "意外错误: {error}");
    }
}
