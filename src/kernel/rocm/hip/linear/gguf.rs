pub(super) const SOURCE: &str = include_str!("gguf/source.hip");

use super::*;

/// GGUF fused decode experts(小批次，route_count ≤ 64)：token-major 两段 kernel，
/// 无路由分组 kernel；activated/output 用按 device 常驻 workspace，跨调用零分配。
#[derive(Default)]
struct GgufFusedWorkspace {
    activated: Option<DeviceBuffer>,
}

thread_local! {
    static GGUF_FUSED_WORKSPACES: std::cell::RefCell<HashMap<(i32, usize), GgufFusedWorkspace>> = std::cell::RefCell::new(HashMap::new());
}

pub(crate) fn release_gguf_fused_workspace(device_id: i32) {
    GGUF_FUSED_WORKSPACES.with(|workspaces| {
        workspaces.borrow_mut().retain(|&(device, _), _| device != device_id);
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gguf_fused_decode_experts(
    device_id: i32,
    input: &DeviceBuffer,
    input_rows: usize,
    hidden_size: usize,
    intermediate_size: usize,
    top_k: usize,
    route_ids: &DeviceBuffer,
    route_weights: &DeviceBuffer,
    route_count: usize,
    metas: &DeviceBuffer,
    output: &DeviceBuffer,
) -> Result<(), String> {
    if input_rows == 0 || route_count != input_rows.checked_mul(top_k).ok_or("GGUF fused route 数溢出")? || route_count > 64 {
        return Err("GGUF fused decode 参数无效".to_owned());
    }
    if !hidden_size.is_multiple_of(256) || !intermediate_size.is_multiple_of(8) || !hidden_size.is_multiple_of(8) {
        return Err("GGUF fused decode 维度不支持".to_owned());
    }
    let input_elements = input_rows * hidden_size;
    let input_bytes_bf16 = input_elements * 2;
    if input.device_id != device_id || (input.bytes != input_bytes_bf16 && input.bytes != input_elements * 4) {
        return Err("GGUF fused decode 输入必须为设备侧 BF16/F32".to_owned());
    }
    let input_is_bf16 = input.bytes == input_bytes_bf16;
    if route_ids.device_id != device_id || route_weights.device_id != device_id || route_ids.bytes < route_count * 4 || route_weights.bytes < route_count * 4 {
        return Err("GGUF fused decode 路由 device 不一致".to_owned());
    }
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let module_launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let functions = ct_quantized_functions(device_id)?;
    let d_metas = metas;

    let output_bytes = input_rows.checked_mul(hidden_size).and_then(|n| n.checked_mul(4)).ok_or("GGUF fused output 溢出")?;
    if output.device_id != device_id || output.bytes() != output_bytes {
        return Err("GGUF fused decode 输出 buffer 尺寸不匹配".to_owned());
    }
    let output_pointer = output.device_pointer();
    // activated 常驻 workspace：只增不减，跨调用零分配。
    let activated_bytes = route_count.checked_mul(intermediate_size).and_then(|n| n.checked_mul(2)).ok_or("GGUF fused activated 溢出")?;
    let activated_pointer = GGUF_FUSED_WORKSPACES.with(|workspaces| {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        if workspace.activated.as_ref().is_none_or(|buffer| buffer.bytes() < activated_bytes) {
            workspace.activated = Some(DeviceBuffer::allocate(device_id, activated_bytes)?);
        }
        Ok::<usize, String>(workspace.activated.as_ref().expect("GGUF fused activated 已初始化").device_pointer())
    })?;

    let mut assignments = u32::try_from(route_count).map_err(|_| "GGUF fused route 数超过 u32".to_owned())?;
    let mut tokens = u32::try_from(input_rows).map_err(|_| "GGUF fused tokens 超过 u32".to_owned())?;
    let mut top_k = u32::try_from(top_k).map_err(|_| "GGUF fused top_k 超过 u32".to_owned())?;
    let mut hidden = u32::try_from(hidden_size).map_err(|_| "GGUF fused hidden 超过 u32".to_owned())?;
    let mut intermediate = u32::try_from(intermediate_size).map_err(|_| "GGUF fused intermediate 超过 u32".to_owned())?;
    let mut is_bf16 = u32::from(input_is_bf16);
    {
        let mut input_pointer = input.pointer;
        let mut metas_pointer = d_metas.pointer;
        let mut route_ids_pointer = route_ids.pointer;
        let mut activated_pointer = activated_pointer;
        let mut arguments = [
            (&mut input_pointer as *mut *mut c_void).cast(),
            (&mut metas_pointer as *mut *mut c_void).cast(),
            (&mut route_ids_pointer as *mut *mut c_void).cast(),
            (&mut route_ids_pointer as *mut *mut c_void).cast(),
            (&mut activated_pointer as *mut usize).cast(),
            (&mut assignments as *mut u32).cast(),
            (&mut top_k as *mut u32).cast(),
            (&mut hidden as *mut u32).cast(),
            (&mut intermediate as *mut u32).cast(),
            (&mut is_bf16 as *mut u32).cast(),
        ];
        let grid_x = u32::try_from(intermediate_size / 8).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let grid_y = u32::try_from(route_count).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let status = unsafe { module_launch(functions.gguf_fused_gate_up as *mut c_void, grid_x, grid_y, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF fused gate_up"));
        }
    }
    {
        let mut activated_pointer = activated_pointer;
        let mut metas_pointer = d_metas.pointer;
        let mut route_ids_pointer = route_ids.pointer;
        let mut route_weights_pointer = route_weights.pointer;
        let mut output_pointer = output_pointer;
        let mut arguments = [
            (&mut activated_pointer as *mut usize).cast(),
            (&mut metas_pointer as *mut *mut c_void).cast(),
            (&mut route_ids_pointer as *mut *mut c_void).cast(),
            (&mut route_ids_pointer as *mut *mut c_void).cast(),
            (&mut route_weights_pointer as *mut *mut c_void).cast(),
            (&mut output_pointer as *mut usize).cast(),
            (&mut tokens as *mut u32).cast(),
            (&mut top_k as *mut u32).cast(),
            (&mut hidden as *mut u32).cast(),
            (&mut intermediate as *mut u32).cast(),
        ];
        let grid_x = u32::try_from(hidden_size / 8).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let status = unsafe { module_launch(functions.gguf_fused_down as *mut c_void, grid_x, tokens, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF fused down"));
        }
    }
    Ok(())
}
