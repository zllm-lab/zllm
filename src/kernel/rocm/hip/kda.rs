//! KDA fused HIP kernel。
//!
//! 算法与 Metal `kda_recurrent_f16` / CPU reference 逐行对应:token 顺序递归
//! (三条 depthwise 短卷积 + SiLU、Q/K L2 归一化、per-(head, key_column) 全秩
//! decay、delta rule 状态更新与读出、输出 RMSNorm + sigmoid 门)。kernel 全 F32。
//! 每个 head 一个 block,lane 并行 value 维度,key 维度在 kernel 内串行。

use super::tensor::validate_resident;
use super::*;
use crate::attention::kda::{KdaInputs, KdaSpec, KdaWeightsRef};

/// 共享数组按 256 上限静态分配,head_dim 超过时 host 侧直接拒绝。
const KDA_MAX_HEAD_DIM: usize = 256;

const KDA_SOURCE: &str = include_str!("kda/source.hip");

#[derive(Clone, Copy)]
pub(super) struct KdaFunctions {
    fused: usize,
    chunked: usize,
    chunked_epilogue: usize,
}

fn kda_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(KDA_SOURCE, "zllm_rocm_kda.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

pub(super) fn kda_functions(device_id: i32) -> Result<KdaFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, KdaFunctions), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm KDA kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = kda_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData KDA"));
        }
        let function = |name: &str| -> Result<usize, String> {
            let name = CString::new(name).unwrap();
            let mut function = ptr::null_mut();
            let status = unsafe { module_get_function(&mut function, module, name.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction KDA"));
            }
            Ok(function as usize)
        };
        Ok((module as usize, KdaFunctions { fused: function("zllm_kda_fused_f32")?, chunked: function("zllm_kda_chunked_f32")?, chunked_epilogue: function("zllm_kda_chunked_epilogue_f32")? }))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

/// 执行 KDA fused kernel。
///
/// conv_state(三段 [q|k|v])/ recurrent_state 是跨 token 持久的 device buffer,
/// 本函数原地更新。返回 output device buffer。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_kda_fused_resident_device_f32(
    device_id: i32,
    conv_state: &DeviceBuffer,
    recurrent_state: &DeviceBuffer,
    inputs: KdaInputs<'_, DeviceBuffer>,
    weights: KdaWeightsRef<'_, DeviceBuffer>,
    rows: usize,
    spec: &KdaSpec,
) -> Result<DeviceBuffer, String> {
    set_device(device_id)?;
    let KdaInputs { query, key, value, decay, beta, output_gate } = inputs;
    let KdaWeightsRef { query_conv, key_conv, value_conv, a_log, dt_bias, output_norm } = weights;
    let projection_size = spec.projection_size();

    if spec.head_dim > KDA_MAX_HEAD_DIM {
        return Err(format!("KDA head_dim={} 超过 kernel 共享内存上限 {KDA_MAX_HEAD_DIM}", spec.head_dim));
    }

    let conv_state_bytes = spec.conv_state_elements().checked_mul(4).ok_or("KDA conv_state 字节溢出")?;
    let recurrent_bytes = spec.recurrent_state_elements().checked_mul(4).ok_or("KDA recurrent 字节溢出")?;
    let output_bytes = rows.checked_mul(projection_size).ok_or("KDA output 元素溢出")?.checked_mul(4).ok_or("KDA output 字节溢出")?;
    let input_bytes = rows.checked_mul(projection_size).ok_or("KDA input 元素溢出")?.checked_mul(4).ok_or("KDA input 字节溢出")?;
    let beta_bytes = rows.checked_mul(spec.num_heads).ok_or("KDA beta 元素溢出")?.checked_mul(4).ok_or("KDA beta 字节溢出")?;
    let conv_weight_bytes = projection_size.checked_mul(spec.short_conv_kernel_size).ok_or("KDA conv weight 元素溢出")?.checked_mul(4).ok_or("KDA conv weight 字节溢出")?;

    validate_resident(conv_state, device_id, conv_state_bytes, "KDA conv_state")?;
    validate_resident(recurrent_state, device_id, recurrent_bytes, "KDA recurrent")?;
    validate_resident(query, device_id, input_bytes, "KDA query")?;
    validate_resident(key, device_id, input_bytes, "KDA key")?;
    validate_resident(value, device_id, input_bytes, "KDA value")?;
    validate_resident(decay, device_id, input_bytes, "KDA decay")?;
    validate_resident(output_gate, device_id, input_bytes, "KDA output_gate")?;
    validate_resident(beta, device_id, beta_bytes, "KDA beta")?;
    validate_resident(query_conv, device_id, conv_weight_bytes, "KDA query conv weight")?;
    validate_resident(key_conv, device_id, conv_weight_bytes, "KDA key conv weight")?;
    validate_resident(value_conv, device_id, conv_weight_bytes, "KDA value conv weight")?;
    validate_resident(a_log, device_id, spec.num_heads.checked_mul(4).ok_or("KDA a_log 字节溢出")?, "KDA a_log")?;
    validate_resident(dt_bias, device_id, projection_size.checked_mul(4).ok_or("KDA dt_bias 字节溢出")?, "KDA dt_bias")?;
    validate_resident(output_norm, device_id, spec.head_dim.checked_mul(4).ok_or("KDA output_norm 字节溢出")?, "KDA output_norm")?;

    let functions = kda_functions(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;

    let mut d_conv = conv_state.pointer;
    let mut d_recurrent = recurrent_state.pointer;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_value = value.pointer;
    let mut d_decay = decay.pointer;
    let mut d_beta = beta.pointer;
    let mut d_output_gate = output_gate.pointer;
    let mut d_query_conv = query_conv.pointer;
    let mut d_key_conv = key_conv.pointer;
    let mut d_value_conv = value_conv.pointer;
    let mut d_a_log = a_log.pointer;
    let mut d_dt_bias = dt_bias.pointer;
    let mut d_output_norm = output_norm.pointer;
    let mut d_output = output.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "KDA rows 超过 u32".to_owned())?;
    let mut num_heads_u32 = u32::try_from(spec.num_heads).map_err(|_| "KDA num_heads 超过 u32".to_owned())?;
    let mut head_dim_u32 = u32::try_from(spec.head_dim).map_err(|_| "KDA head_dim 超过 u32".to_owned())?;
    let mut conv_kernel_u32 = u32::try_from(spec.short_conv_kernel_size).map_err(|_| "KDA conv_kernel 超过 u32".to_owned())?;
    let mut gate_lower_bound_enabled_u32 = u32::from(spec.gate_lower_bound.is_some());
    let mut gate_lower_bound_f32 = spec.gate_lower_bound.unwrap_or(0.0);
    let mut use_qk_l2norm_u32 = u32::from(spec.use_qk_l2norm);
    let mut output_norm_eps_f32 = spec.output_norm_eps;

    let mut arguments = [
        (&mut d_conv as *mut *mut c_void).cast(),
        (&mut d_recurrent as *mut *mut c_void).cast(),
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_value as *mut *mut c_void).cast(),
        (&mut d_decay as *mut *mut c_void).cast(),
        (&mut d_beta as *mut *mut c_void).cast(),
        (&mut d_output_gate as *mut *mut c_void).cast(),
        (&mut d_query_conv as *mut *mut c_void).cast(),
        (&mut d_key_conv as *mut *mut c_void).cast(),
        (&mut d_value_conv as *mut *mut c_void).cast(),
        (&mut d_a_log as *mut *mut c_void).cast(),
        (&mut d_dt_bias as *mut *mut c_void).cast(),
        (&mut d_output_norm as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut num_heads_u32 as *mut u32).cast(),
        (&mut head_dim_u32 as *mut u32).cast(),
        (&mut conv_kernel_u32 as *mut u32).cast(),
        (&mut gate_lower_bound_enabled_u32 as *mut u32).cast(),
        (&mut gate_lower_bound_f32 as *mut f32).cast(),
        (&mut use_qk_l2norm_u32 as *mut u32).cast(),
        (&mut output_norm_eps_f32 as *mut f32).cast(),
    ];

    // 每 head 一个 block,lane 并行 head_dim;head_dim 已限制 <= 256,block 不超上限。
    let block = u32::try_from(spec.head_dim.next_power_of_two().max(32)).map_err(|_| "KDA block 维度超过 u32".to_owned())?;
    let grid_x = u32::try_from(spec.num_heads).map_err(|_| "KDA num_heads 超过 u32 grid".to_owned())?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result = unsafe { launch(functions.fused as *mut c_void, grid_x, 1, 1, block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel KDA fused"));
    }

    // KDA 是状态递归,必须同步确保 conv/recurrent state 写入完成
    synchronize_device(device_id, "KDA fused synchronize")?;

    Ok(output)
}

/// chunked kernel 的 chunk token 数与 value tile 宽,与 kernel 内常量保持一致。
const KDA_CHUNK: usize = 8;
const KDA_VALUE_TILE: usize = 32;

/// chunked kernel 的 shape 适用范围:head_dim ≤ 128 且被 32 整除、conv kernel ≤ 4
/// (LDS state 切片 + 短卷积窗口的静态上限)。范围外沿用 fused 串行 kernel。
pub(crate) fn kda_chunked_prefers(spec: &KdaSpec) -> bool {
    spec.head_dim <= 128 && spec.head_dim % KDA_VALUE_TILE == 0 && spec.short_conv_kernel_size <= 4
}

/// 执行 KDA chunked(WY)kernel:与 fused 串行 kernel 数学等价,chunk 内用三角
/// 线性解并行替代逐 token 递归。conv_state / recurrent_state 原地推进。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_kda_chunked_resident_device_f32(
    device_id: i32,
    conv_state: &DeviceBuffer,
    recurrent_state: &DeviceBuffer,
    inputs: KdaInputs<'_, DeviceBuffer>,
    weights: KdaWeightsRef<'_, DeviceBuffer>,
    rows: usize,
    spec: &KdaSpec,
) -> Result<DeviceBuffer, String> {
    set_device(device_id)?;
    let KdaInputs { query, key, value, decay, beta, output_gate } = inputs;
    let KdaWeightsRef { query_conv, key_conv, value_conv, a_log, dt_bias, output_norm } = weights;
    let projection_size = spec.projection_size();
    if !kda_chunked_prefers(spec) {
        return Err(format!("KDA chunked 不支持 head_dim={} conv_kernel={}", spec.head_dim, spec.short_conv_kernel_size));
    }

    let conv_state_bytes = spec.conv_state_elements().checked_mul(4).ok_or("KDA conv_state 字节溢出")?;
    let recurrent_bytes = spec.recurrent_state_elements().checked_mul(4).ok_or("KDA recurrent 字节溢出")?;
    let output_bytes = rows.checked_mul(projection_size).ok_or("KDA output 元素溢出")?.checked_mul(4).ok_or("KDA output 字节溢出")?;
    let input_bytes = rows.checked_mul(projection_size).ok_or("KDA input 元素溢出")?.checked_mul(4).ok_or("KDA input 字节溢出")?;
    let beta_bytes = rows.checked_mul(spec.num_heads).ok_or("KDA beta 元素溢出")?.checked_mul(4).ok_or("KDA beta 字节溢出")?;
    let conv_weight_bytes = projection_size.checked_mul(spec.short_conv_kernel_size).ok_or("KDA conv weight 元素溢出")?.checked_mul(4).ok_or("KDA conv weight 字节溢出")?;

    validate_resident(conv_state, device_id, conv_state_bytes, "KDA conv_state")?;
    validate_resident(recurrent_state, device_id, recurrent_bytes, "KDA recurrent")?;
    validate_resident(query, device_id, input_bytes, "KDA query")?;
    validate_resident(key, device_id, input_bytes, "KDA key")?;
    validate_resident(value, device_id, input_bytes, "KDA value")?;
    validate_resident(decay, device_id, input_bytes, "KDA decay")?;
    validate_resident(output_gate, device_id, input_bytes, "KDA output_gate")?;
    validate_resident(beta, device_id, beta_bytes, "KDA beta")?;
    validate_resident(query_conv, device_id, conv_weight_bytes, "KDA query conv weight")?;
    validate_resident(key_conv, device_id, conv_weight_bytes, "KDA key conv weight")?;
    validate_resident(value_conv, device_id, conv_weight_bytes, "KDA value conv weight")?;
    validate_resident(a_log, device_id, spec.num_heads.checked_mul(4).ok_or("KDA a_log 字节溢出")?, "KDA a_log")?;
    validate_resident(dt_bias, device_id, projection_size.checked_mul(4).ok_or("KDA dt_bias 字节溢出")?, "KDA dt_bias")?;
    validate_resident(output_norm, device_id, spec.head_dim.checked_mul(4).ok_or("KDA output_norm 字节溢出")?, "KDA output_norm")?;

    let functions = kda_functions(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    let mixed = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;

    // LDS 布局与 kernel 内 carve 逐项对应:key 维 stride = head_dim + 2,value 维 stride = 33。
    let key_stride = spec.head_dim + 2;
    let val_stride = KDA_VALUE_TILE + 1;
    let block = 128usize;
    let shared_floats = spec.head_dim * val_stride + 5 * (KDA_CHUNK * key_stride) + 2 * (KDA_CHUNK * val_stride) + 2 * (KDA_CHUNK * KDA_CHUNK) + 2 * (3 * spec.head_dim) + 3 * KDA_VALUE_TILE + 2 * block + KDA_CHUNK;
    let shared_bytes = u32::try_from(shared_floats.checked_mul(4).ok_or("KDA chunked shared memory 溢出")?).map_err(|_| "KDA chunked shared memory 超过 u32".to_owned())?;

    let mut d_conv = conv_state.pointer;
    let mut d_recurrent = recurrent_state.pointer;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_value = value.pointer;
    let mut d_decay = decay.pointer;
    let mut d_beta = beta.pointer;
    let mut d_output_gate = output_gate.pointer;
    let mut d_query_conv = query_conv.pointer;
    let mut d_key_conv = key_conv.pointer;
    let mut d_value_conv = value_conv.pointer;
    let mut d_a_log = a_log.pointer;
    let mut d_dt_bias = dt_bias.pointer;
    let mut d_output_norm = output_norm.pointer;
    let mut d_mixed = mixed.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "KDA rows 超过 u32".to_owned())?;
    let mut num_heads_u32 = u32::try_from(spec.num_heads).map_err(|_| "KDA num_heads 超过 u32".to_owned())?;
    let mut head_dim_u32 = u32::try_from(spec.head_dim).map_err(|_| "KDA head_dim 超过 u32".to_owned())?;
    let mut conv_kernel_u32 = u32::try_from(spec.short_conv_kernel_size).map_err(|_| "KDA conv_kernel 超过 u32".to_owned())?;
    let mut gate_lower_bound_enabled_u32 = u32::from(spec.gate_lower_bound.is_some());
    let mut gate_lower_bound_f32 = spec.gate_lower_bound.unwrap_or(0.0);
    let mut use_qk_l2norm_u32 = u32::from(spec.use_qk_l2norm);

    let mut arguments = [
        (&mut d_conv as *mut *mut c_void).cast(),
        (&mut d_recurrent as *mut *mut c_void).cast(),
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_value as *mut *mut c_void).cast(),
        (&mut d_decay as *mut *mut c_void).cast(),
        (&mut d_beta as *mut *mut c_void).cast(),
        (&mut d_output_gate as *mut *mut c_void).cast(),
        (&mut d_query_conv as *mut *mut c_void).cast(),
        (&mut d_key_conv as *mut *mut c_void).cast(),
        (&mut d_value_conv as *mut *mut c_void).cast(),
        (&mut d_a_log as *mut *mut c_void).cast(),
        (&mut d_dt_bias as *mut *mut c_void).cast(),
        (&mut d_mixed as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut num_heads_u32 as *mut u32).cast(),
        (&mut head_dim_u32 as *mut u32).cast(),
        (&mut conv_kernel_u32 as *mut u32).cast(),
        (&mut gate_lower_bound_enabled_u32 as *mut u32).cast(),
        (&mut gate_lower_bound_f32 as *mut f32).cast(),
        (&mut use_qk_l2norm_u32 as *mut u32).cast(),
    ];

    let grid_x = u32::try_from(spec.num_heads.checked_mul(spec.head_dim / KDA_VALUE_TILE).ok_or("KDA chunked grid 溢出")?).map_err(|_| "KDA chunked grid 超过 u32".to_owned())?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result = unsafe { launch(functions.chunked as *mut c_void, grid_x, 1, 1, block as u32, 1, 1, shared_bytes, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel KDA chunked"));
    }

    // epilogue:跨 value tile 的 RMSNorm + gate。
    let mut d_output = output.pointer;
    let mut epilogue_heads = num_heads_u32;
    let mut epilogue_head_dim = head_dim_u32;
    let mut epilogue_projection = u32::try_from(projection_size).map_err(|_| "KDA projection 超过 u32".to_owned())?;
    let mut epilogue_eps = spec.output_norm_eps;
    let mut epilogue_arguments = [
        (&mut d_mixed as *mut *mut c_void).cast(),
        (&mut d_output_gate as *mut *mut c_void).cast(),
        (&mut d_output_norm as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut epilogue_heads as *mut u32).cast(),
        (&mut epilogue_head_dim as *mut u32).cast(),
        (&mut epilogue_projection as *mut u32).cast(),
        (&mut epilogue_eps as *mut f32).cast(),
    ];
    let epilogue_grid = u32::try_from(rows.checked_mul(spec.num_heads).ok_or("KDA epilogue grid 溢出")?).map_err(|_| "KDA epilogue grid 超过 u32".to_owned())?;
    let epilogue_block = u32::try_from(spec.head_dim.next_power_of_two()).map_err(|_| "KDA epilogue block 超过 u32".to_owned())?;
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result =
            unsafe { launch(functions.chunked_epilogue as *mut c_void, epilogue_grid, 1, 1, epilogue_block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), epilogue_arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel KDA chunked epilogue"));
    }

    // KDA 是状态递归,必须同步确保 conv/recurrent state 写入完成
    synchronize_device(device_id, "KDA chunked synchronize")?;

    Ok(output)
}
