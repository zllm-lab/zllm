//! GatedDeltaNet fused HIP kernel。
//!
//! 算法是 token 顺序递归(conv shift-register + delta rule),无法跨 token 并行。
//! kernel 在单个 block 内按 token 顺序执行,利用线程并行加速 delta rule 的
//! value_head × value_head_dim 维度。conv_state 与 recurrent_state 作为持久
//! device buffer 跨 token 保持。

use super::tensor::validate_resident;
use super::*;
use crate::attention::gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetInputs, GatedDeltaNetSpec, GatedDeltaNetWeightsRef};

const GATED_DELTA_NET_SOURCE: &str = include_str!("gated_delta_net/source.hip");

#[derive(Clone, Copy)]
pub(super) struct GatedDeltaNetFunctions {
    fused: usize,
}

fn gated_delta_net_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(GATED_DELTA_NET_SOURCE, "zllm_rocm_gdn.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

pub(super) fn gated_delta_net_functions(device_id: i32) -> Result<GatedDeltaNetFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, GatedDeltaNetFunctions), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm GDN kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = gated_delta_net_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData GDN"));
        }
        let function = |name: &str| -> Result<usize, String> {
            let name = CString::new(name).unwrap();
            let mut function = ptr::null_mut();
            let status = unsafe { module_get_function(&mut function, module, name.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction GDN"));
            }
            Ok(function as usize)
        };
        Ok((module as usize, GatedDeltaNetFunctions { fused: function("zllm_gdn_fused_f32")? }))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

/// 执行 GatedDeltaNet fused kernel。
///
/// conv_state / recurrent_state 是跨 token 持久的 device buffer,本函数原地更新。
/// 返回 output device buffer。
pub(crate) fn try_gated_delta_net_fused_resident_device_f32(
    device_id: i32,
    conv_state: &DeviceBuffer,
    recurrent_state: &DeviceBuffer,
    inputs: GatedDeltaNetInputs<'_, DeviceBuffer>,
    weights: GatedDeltaNetWeightsRef<'_, DeviceBuffer>,
    rows: usize,
    head_layout: GatedDeltaNetHeadLayout,
    spec: &GatedDeltaNetSpec,
) -> Result<DeviceBuffer, String> {
    set_device(device_id)?;
    let GatedDeltaNetInputs { qkv, z, alpha, beta } = inputs;
    let GatedDeltaNetWeightsRef { conv: conv_weight, a_log, dt_bias, norm: norm_weight } = weights;
    let key_dim = spec.key_dim();
    let value_dim = spec.value_dim();
    let conv_dim = spec.conv_dim();

    // kernel 内按 value_head / (value_heads / key_heads) 归组,
    // key_heads 为 0 或不能整除时会在 device 上除零/错组,host 侧先拒绝。
    if spec.key_heads == 0 || spec.value_heads % spec.key_heads != 0 {
        return Err(format!("GDN head 配置非法: key_heads={} value_heads={}，要求 key_heads 非零且 value_heads 是 key_heads 的整数倍", spec.key_heads, spec.value_heads));
    }

    let conv_state_bytes = conv_dim.checked_mul(spec.conv_kernel).ok_or("GDN conv_state 元素溢出")?.checked_mul(4).ok_or("GDN conv_state 字节溢出")?;
    let recurrent_bytes = spec.value_heads.checked_mul(spec.key_head_dim).ok_or("GDN recurrent 元素溢出")?.checked_mul(spec.value_head_dim).ok_or("GDN recurrent 元素溢出")?.checked_mul(4).ok_or("GDN recurrent 字节溢出")?;
    let output_bytes = rows.checked_mul(value_dim).ok_or("GDN output 元素溢出")?.checked_mul(4).ok_or("GDN output 字节溢出")?;

    validate_resident(conv_state, device_id, conv_state_bytes, "GDN conv_state")?;
    validate_resident(recurrent_state, device_id, recurrent_bytes, "GDN recurrent")?;
    validate_resident(qkv, device_id, rows.checked_mul(conv_dim).ok_or("GDN qkv 字节溢出")?.checked_mul(4).unwrap_or(usize::MAX), "GDN qkv")?;
    validate_resident(z, device_id, rows.checked_mul(value_dim).ok_or("GDN z 字节溢出")?.checked_mul(4).unwrap_or(usize::MAX), "GDN z")?;

    let functions = gated_delta_net_functions(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;

    // 共享内存:mixed(conv_dim) + query(key_dim) + key(key_dim)
    let shared_floats = conv_dim + key_dim + key_dim;
    let shared_bytes = u32::try_from(shared_floats.checked_mul(4).ok_or("GDN shared memory 溢出")?).map_err(|_| "GDN shared memory 超过 u32".to_owned())?;

    let mut d_conv = conv_state.pointer;
    let mut d_recurrent = recurrent_state.pointer;
    let mut d_qkv = qkv.pointer;
    let mut d_z = z.pointer;
    let mut d_alpha = alpha.pointer;
    let mut d_beta = beta.pointer;
    let mut d_conv_weight = conv_weight.pointer;
    let mut d_a_log = a_log.pointer;
    let mut d_dt_bias = dt_bias.pointer;
    let mut d_norm_weight = norm_weight.pointer;
    let mut d_output = output.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "GDN rows 超过 u32".to_owned())?;
    let mut key_dim_u32 = u32::try_from(key_dim).map_err(|_| "GDN key_dim 超过 u32".to_owned())?;
    let mut value_dim_u32 = u32::try_from(value_dim).map_err(|_| "GDN value_dim 超过 u32".to_owned())?;
    let mut conv_dim_u32 = u32::try_from(conv_dim).map_err(|_| "GDN conv_dim 超过 u32".to_owned())?;
    let mut conv_kernel_u32 = u32::try_from(spec.conv_kernel).map_err(|_| "GDN conv_kernel 超过 u32".to_owned())?;
    let mut key_heads_u32 = u32::try_from(spec.key_heads).map_err(|_| "GDN key_heads 超过 u32".to_owned())?;
    let mut value_heads_u32 = u32::try_from(spec.value_heads).map_err(|_| "GDN value_heads 超过 u32".to_owned())?;
    let mut key_head_dim_u32 = u32::try_from(spec.key_head_dim).map_err(|_| "GDN key_head_dim 超过 u32".to_owned())?;
    let mut value_head_dim_u32 = u32::try_from(spec.value_head_dim).map_err(|_| "GDN value_head_dim 超过 u32".to_owned())?;
    let mut grouped_heads_u32 = u32::from(matches!(head_layout, GatedDeltaNetHeadLayout::Grouped));
    let mut rms_eps_f32 = spec.rms_eps;

    let mut arguments = [
        (&mut d_conv as *mut *mut c_void).cast(),
        (&mut d_recurrent as *mut *mut c_void).cast(),
        (&mut d_qkv as *mut *mut c_void).cast(),
        (&mut d_z as *mut *mut c_void).cast(),
        (&mut d_alpha as *mut *mut c_void).cast(),
        (&mut d_beta as *mut *mut c_void).cast(),
        (&mut d_conv_weight as *mut *mut c_void).cast(),
        (&mut d_a_log as *mut *mut c_void).cast(),
        (&mut d_dt_bias as *mut *mut c_void).cast(),
        (&mut d_norm_weight as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut key_dim_u32 as *mut u32).cast(),
        (&mut value_dim_u32 as *mut u32).cast(),
        (&mut conv_dim_u32 as *mut u32).cast(),
        (&mut conv_kernel_u32 as *mut u32).cast(),
        (&mut key_heads_u32 as *mut u32).cast(),
        (&mut value_heads_u32 as *mut u32).cast(),
        (&mut key_head_dim_u32 as *mut u32).cast(),
        (&mut value_head_dim_u32 as *mut u32).cast(),
        (&mut grouped_heads_u32 as *mut u32).cast(),
        (&mut rms_eps_f32 as *mut f32).cast(),
    ];

    // 单 block,256 线程;kernel 内部按 token 顺序执行
    let block = 256u32;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result = unsafe { launch(functions.fused as *mut c_void, 1, 1, 1, block, 1, 1, shared_bytes, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel GDN fused"));
    }

    // GDN 是状态递归,必须同步确保 conv/recurrent state 写入完成
    synchronize_device(device_id, "GDN fused synchronize")?;

    Ok(output)
}
