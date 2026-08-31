pub(super) const SOURCE: &str = include_str!("ct_common/source.hip");

use super::*;

pub(super) fn ct_quantized_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| {
        compile_hip_source(
            &[super::super::hiprtc::DEVICE_CONVERSIONS_PREAMBLE, super::ct_common::SOURCE, super::ct_dense::SOURCE, super::ct_grouped::SOURCE, super::convrot::SOURCE, super::gguf::SOURCE].concat(),
            "zllm_rocm_ct_quantized.hip",
        )
    })
    .as_ref()
    .map(Vec::as_slice)
    .map_err(Clone::clone)
}

#[derive(Clone, Copy)]
pub(super) struct CtFunctions {
    pub(super) wavefront_size: u32,
    pub(super) cast: usize,
    pub(super) expand: usize,
    pub(super) dense: usize,
    pub(super) dense_aligned: usize,
    pub(super) scalar: usize,
    pub(super) w8_scalar: usize,
    pub(super) w8_rows2: usize,
    pub(super) _w8_rows3: usize,
    pub(super) _w8_rows4: usize,
    pub(super) _w8_rows5: usize,
    pub(super) w8_rows6: usize,
    pub(super) _w8_rows7: usize,
    pub(super) w8_rows8: usize,
    pub(super) _w4_rows2: usize,
    pub(super) w4_rows8: usize,
    pub(super) w4_dual: usize,
    pub(super) w4_dual_rows8: usize,
    pub(super) w8_dual: usize,
    pub(super) wmma: usize,
    pub(super) wmma_w8_g128: usize,
    pub(super) grouped_linear: usize,
    pub(super) grouped_gate_up_wmma: usize,
    pub(super) grouped_zero: usize,
    pub(super) grouped_down: usize,
    pub(super) grouped_down_wmma: usize,
    pub(super) grouped_down_small: usize,
    pub(super) decode_gate_up: usize,
    pub(super) decode_down: usize,
    pub(super) decode_w8_gate_up: usize,
    pub(super) decode_w8_down: usize,
    pub(super) group_decode_routes: usize,
    pub(super) grouped_down_reduce: usize,
    pub(super) grouped_down_reduce_routes: usize,
    pub(super) convrot_quantize: usize,
    pub(super) convrot_quantize_cached: usize,
    pub(super) convrot_wmma: usize,
    pub(super) convrot_wmma_row16: usize,
    pub(super) convrot_wmma_split4: usize,
    pub(super) convrot_wmma_tiled128: usize,
    pub(super) convrot_wmma_tiled256: usize,
    pub(super) gguf_routes: usize,
    pub(super) gguf_gate_up_wmma: usize,
    pub(super) gguf_grouped_down: usize,
    pub(super) gguf_fused_gate_up: usize,
    pub(super) gguf_fused_down: usize,
    pub(super) cooperative_merge_activation: usize,
    pub(super) cooperative_sharded_down: usize,
    pub(super) cooperative_partial_join: usize,
}

pub(super) fn ct_quantized_functions(device_id: i32) -> Result<CtFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, CtFunctions), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }

    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = ct_quantized_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData CT quantized"));
        }
        let names = [
            "f32_to_bf16",
            "bf16_to_f32",
            "dense_matmul_bf16_f32_wmma",
            "ct_quantized_matmul_bf16_scalar",
            "ct_quantized_matmul_bf16_wmma",
            "ct_grouped_linear_bf16",
            "ct_grouped_zero_f32",
            "ct_grouped_down_scatter_bf16",
            "ct_grouped_down_scatter_wmma_bf16",
            "ct_quantized_matmul_bf16_w8_scalar",
            "ct_decode_gate_up_bf16",
            "ct_decode_down_bf16",
            "ct_decode_gate_up_w8_bf16",
            "ct_decode_down_w8_bf16",
            "ct_quantized_dual_gemv_bf16_w8_scalar",
            "ct_grouped_gate_up_wmma_bf16",
            "dense_matmul_bf16_f32_wmma_aligned",
            "convrot_quantize_i8",
            "convrot_quantize_i8_cached",
            "convrot_int8_matmul_wmma",
            "convrot_int8_matmul_wmma_row16",
            "convrot_int8_matmul_wmma_split4",
            "ct_group_decode_routes",
            "ct_grouped_down_small_bf16",
            "convrot_int8_matmul_wmma_tiled128",
            "convrot_int8_matmul_wmma_tiled256",
            "ct_grouped_down_reduce_bf16",
            "ct_grouped_down_reduce_routes_f32",
            "ct_quantized_matmul_bf16_w8_g128_wmma",
            "gguf_group_decode_routes",
            "gguf_gate_up_wmma_bf16",
            "gguf_grouped_down_f32",
            "gguf_fused_gate_up_f32",
            "gguf_fused_down_f32",
            "ct_quantized_matmul_bf16_w8_rows2",
            "ct_quantized_matmul_bf16_w8_rows3",
            "ct_quantized_matmul_bf16_w8_rows4",
            "ct_quantized_matmul_bf16_w8_rows5",
            "ct_quantized_matmul_bf16_w8_rows6",
            "ct_quantized_matmul_bf16_w8_rows7",
            "ct_quantized_matmul_bf16_w8_rows8",
            "ct_quantized_dual_gemv_bf16_w4_scalar",
            "ct_quantized_matmul_bf16_w4_rows8",
            "ct_quantized_matmul_bf16_w4_rows2",
            "ct_quantized_dual_gemv_bf16_w4_rows8",
            "ct_cooperative_merge_activation_bf16",
            "ct_cooperative_sharded_down_bf16",
            "ct_cooperative_partial_join_f32",
        ];
        let mut handles = [ptr::null_mut(); 48];
        for (handle, name) in handles.iter_mut().zip(names) {
            let name = CString::new(name).unwrap();
            let status = unsafe { module_get_function(handle, module, name.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction CT quantized"));
            }
        }
        let device_get_attribute: Symbol<HipDeviceGetAttribute> = runtime.symbol(&runtime.hip, b"hipDeviceGetAttribute\0")?;
        let mut wavefront_size = 0i32;
        let status = unsafe { device_get_attribute(&mut wavefront_size, 87, device_id) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipDeviceGetAttribute warp size"));
        }
        if !matches!(wavefront_size, 32 | 64) {
            return Err(format!("ROCm wavefront size={wavefront_size} 不受支持"));
        }
        Ok((
            module as usize,
            CtFunctions {
                wavefront_size: wavefront_size as u32,
                cast: handles[0] as usize,
                expand: handles[1] as usize,
                dense: handles[2] as usize,
                dense_aligned: handles[16] as usize,
                scalar: handles[3] as usize,
                w8_scalar: handles[9] as usize,
                w8_rows2: handles[34] as usize,
                _w8_rows3: handles[35] as usize,
                _w8_rows4: handles[36] as usize,
                _w8_rows5: handles[37] as usize,
                w8_rows6: handles[38] as usize,
                _w8_rows7: handles[39] as usize,
                w8_rows8: handles[40] as usize,
                w4_dual: handles[41] as usize,
                w4_dual_rows8: handles[44] as usize,
                w4_rows8: handles[42] as usize,
                _w4_rows2: handles[43] as usize,
                w8_dual: handles[14] as usize,
                wmma: handles[4] as usize,
                wmma_w8_g128: handles[28] as usize,
                grouped_linear: handles[5] as usize,
                grouped_gate_up_wmma: handles[15] as usize,
                grouped_zero: handles[6] as usize,
                grouped_down: handles[7] as usize,
                grouped_down_wmma: handles[8] as usize,
                decode_gate_up: handles[10] as usize,
                decode_down: handles[11] as usize,
                decode_w8_gate_up: handles[12] as usize,
                decode_w8_down: handles[13] as usize,
                group_decode_routes: handles[22] as usize,
                grouped_down_small: handles[23] as usize,
                grouped_down_reduce: handles[26] as usize,
                grouped_down_reduce_routes: handles[27] as usize,
                convrot_quantize: handles[17] as usize,
                convrot_quantize_cached: handles[18] as usize,
                convrot_wmma: handles[19] as usize,
                convrot_wmma_row16: handles[20] as usize,
                convrot_wmma_split4: handles[21] as usize,
                convrot_wmma_tiled128: handles[24] as usize,
                convrot_wmma_tiled256: handles[25] as usize,
                gguf_routes: handles[29] as usize,
                gguf_gate_up_wmma: handles[30] as usize,
                gguf_grouped_down: handles[31] as usize,
                gguf_fused_gate_up: handles[32] as usize,
                gguf_fused_down: handles[33] as usize,
                cooperative_merge_activation: handles[45] as usize,
                cooperative_sharded_down: handles[46] as usize,
                cooperative_partial_join: handles[47] as usize,
            },
        ))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

/// GGUF grouped decode experts 的 WMMA 路径：设备侧路由分组 → gate/up 融合 WMMA(LUT
/// 反量化) → 确定性 down。三个 kernel 替代逐专家 3N 次调用。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gguf_grouped_wmma_experts(
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
    expert_count: usize,
) -> Result<DeviceBuffer, String> {
    if input_rows == 0 || expert_count == 0 || expert_count > 256 || route_count != input_rows.checked_mul(top_k).ok_or("GGUF WMMA route 数溢出")? {
        return Err("GGUF WMMA grouped 参数无效".to_owned());
    }
    if !hidden_size.is_multiple_of(256) || !intermediate_size.is_multiple_of(16) || !hidden_size.is_multiple_of(8) {
        return Err("GGUF WMMA grouped 维度不支持".to_owned());
    }
    let input_elements = input_rows * hidden_size;
    let input_bytes_f32 = input_elements * 4;
    let input_bytes_bf16 = input_elements * 2;
    if input.device_id != device_id || (input.bytes != input_bytes_bf16 && input.bytes != input_bytes_f32) {
        return Err("GGUF WMMA grouped 输入必须为设备侧 BF16/F32".to_owned());
    }
    if route_ids.device_id != device_id || route_weights.device_id != device_id || route_ids.bytes < route_count * 4 || route_weights.bytes < route_count * 4 {
        return Err("GGUF WMMA grouped 路由 device 不一致".to_owned());
    }
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let module_launch = runtime.module_launch()?;
    let functions = ct_quantized_functions(device_id)?;

    // decode 链的 expert 输入常为设备 F32(rmsnorm_f32)；WMMA kernel 只吃 BF16，用 cast kernel 原地转换。
    let d_input_bf16;
    let input_bf16: &DeviceBuffer = if input.bytes == input_bytes_bf16 {
        input
    } else {
        d_input_bf16 = try_cast_f32_to_bf16_resident(device_id, input, input_elements)?;
        &d_input_bf16
    };

    let d_metas = metas;
    // 7 个临时 buffer 走 deferred workspace 复用(参照 try_ct_grouped_experts_bf16);
    // d_output 返回给调用方,保持独立 allocation。
    let grouped_metas_bytes = expert_count * std::mem::size_of::<super::GgufGroupedExpertMeta>();
    let activated_bytes = route_count.checked_mul(intermediate_size).and_then(|n| n.checked_mul(2)).ok_or("GGUF WMMA activated 溢出")?;
    super::tensor::with_deferred_tensor_workspace(device_id, &[route_count * 4, route_count * 4, route_count * 4, route_count * 4, (expert_count + 1) * 4, grouped_metas_bytes, activated_bytes], |workspace| {
        let d_grouped_tokens = workspace.buffer(0);
        let d_grouped_weights = workspace.buffer(1);
        let d_route_to_grouped = workspace.buffer(2);
        let d_grouped_experts = workspace.buffer(3);
        let d_grouped_offsets = workspace.buffer(4);
        let d_grouped_metas = workspace.buffer(5);
        let d_activated = workspace.buffer(6);
        let d_output = DeviceBuffer::allocate(device_id, input_rows.checked_mul(hidden_size).and_then(|n| n.checked_mul(4)).ok_or("GGUF WMMA output 溢出")?)?;

        let mut route_count = u32::try_from(route_count).map_err(|_| "GGUF WMMA route 数超过 u32".to_owned())?;
        let mut top_k = u32::try_from(top_k).map_err(|_| "GGUF WMMA top_k 超过 u32".to_owned())?;
        let mut expert_count = u32::try_from(expert_count).map_err(|_| "GGUF WMMA expert 数超过 u32".to_owned())?;
        {
            let mut route_ids_pointer = route_ids.pointer;
            let mut route_weights_pointer = route_weights.pointer;
            let mut metas_pointer = d_metas.pointer;
            let mut grouped_tokens_pointer = d_grouped_tokens.pointer;
            let mut grouped_weights_pointer = d_grouped_weights.pointer;
            let mut route_to_grouped_pointer = d_route_to_grouped.pointer;
            let mut grouped_offsets_pointer = d_grouped_offsets.pointer;
            let mut grouped_metas_pointer = d_grouped_metas.pointer;
            let mut grouped_experts_pointer = d_grouped_experts.pointer;
            let mut arguments = [
                (&mut route_ids_pointer as *mut *mut c_void).cast(),
                (&mut route_weights_pointer as *mut *mut c_void).cast(),
                (&mut metas_pointer as *mut *mut c_void).cast(),
                (&mut grouped_tokens_pointer as *mut *mut c_void).cast(),
                (&mut grouped_weights_pointer as *mut *mut c_void).cast(),
                (&mut route_to_grouped_pointer as *mut *mut c_void).cast(),
                (&mut grouped_offsets_pointer as *mut *mut c_void).cast(),
                (&mut grouped_metas_pointer as *mut *mut c_void).cast(),
                (&mut grouped_experts_pointer as *mut *mut c_void).cast(),
                (&mut route_count as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut expert_count as *mut u32).cast(),
                (&mut expert_count as *mut u32).cast(),
            ];
            let status = unsafe { module_launch(functions.gguf_routes as *mut c_void, 1, 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF routes"));
            }
        }
        {
            let mut input_pointer = input_bf16.pointer;
            let mut grouped_tokens_pointer = d_grouped_tokens.pointer;
            let mut grouped_metas_pointer = d_grouped_metas.pointer;
            let mut grouped_offsets_pointer = d_grouped_offsets.pointer;
            let mut activated_pointer = d_activated.pointer;
            let mut hidden = u32::try_from(hidden_size).map_err(|_| "GGUF WMMA hidden 超过 u32".to_owned())?;
            let mut intermediate = u32::try_from(intermediate_size).map_err(|_| "GGUF WMMA intermediate 超过 u32".to_owned())?;
            let mut base = 0_u32;
            let mut arguments = [
                (&mut input_pointer as *mut *mut c_void).cast(),
                (&mut grouped_tokens_pointer as *mut *mut c_void).cast(),
                (&mut grouped_metas_pointer as *mut *mut c_void).cast(),
                (&mut grouped_offsets_pointer as *mut *mut c_void).cast(),
                (&mut activated_pointer as *mut *mut c_void).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut base as *mut u32).cast(),
                (&mut expert_count as *mut u32).cast(),
            ];
            let grid_x = u32::try_from(intermediate_size / 64).map_err(|_| "GGUF WMMA grid 超过 u32".to_owned())?;
            let grid_y = u32::try_from(input_rows.div_ceil(128)).map_err(|_| "GGUF WMMA grid 超过 u32".to_owned())?;
            let status = unsafe { module_launch(functions.gguf_gate_up_wmma as *mut c_void, grid_x, grid_y, expert_count, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF gate_up WMMA"));
            }
        }
        {
            let mut activated_pointer = d_activated.pointer;
            let mut grouped_metas_pointer = d_grouped_metas.pointer;
            let mut grouped_experts_pointer = d_grouped_experts.pointer;
            let mut grouped_weights_pointer = d_grouped_weights.pointer;
            let mut route_to_grouped_pointer = d_route_to_grouped.pointer;
            let mut output_pointer = d_output.pointer;
            let mut tokens = u32::try_from(input_rows).map_err(|_| "GGUF WMMA tokens 超过 u32".to_owned())?;
            let mut hidden = u32::try_from(hidden_size).map_err(|_| "GGUF WMMA hidden 超过 u32".to_owned())?;
            let mut intermediate = u32::try_from(intermediate_size).map_err(|_| "GGUF WMMA intermediate 超过 u32".to_owned())?;
            let mut arguments = [
                (&mut activated_pointer as *mut *mut c_void).cast(),
                (&mut grouped_metas_pointer as *mut *mut c_void).cast(),
                (&mut grouped_experts_pointer as *mut *mut c_void).cast(),
                (&mut grouped_weights_pointer as *mut *mut c_void).cast(),
                (&mut route_to_grouped_pointer as *mut *mut c_void).cast(),
                (&mut output_pointer as *mut *mut c_void).cast(),
                (&mut tokens as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
            ];
            let grid_x = u32::try_from(hidden_size / 8).map_err(|_| "GGUF WMMA grid 超过 u32".to_owned())?;
            let status = unsafe { module_launch(functions.gguf_grouped_down as *mut c_void, grid_x, tokens, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF grouped down"));
            }
        }
        // 与 CT grouped 一致：纯异步提交。
        Ok(d_output)
    })
}
