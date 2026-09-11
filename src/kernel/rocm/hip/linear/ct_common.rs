pub(super) const SOURCE: &str = include_str!("ct_common/source.hip");

use super::*;

pub(super) fn ct_quantized_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| {
        compile_hip_source(
            &[super::super::hiprtc::DEVICE_CONVERSIONS_PREAMBLE, super::gguf::QUANT_SOURCE, super::gguf::K_DECODE_SOURCE, super::ct_common::SOURCE, super::ct_dense::SOURCE, super::ct_grouped::SOURCE, super::convrot::SOURCE, super::gguf::SOURCE].concat(),
            "zllm_rocm_ct_quantized.hip",
        )
    })
    .as_ref()
    .map(Vec::as_slice)
    .map_err(Clone::clone)
}

fn gguf_k_decode_wave64_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| {
        super::super::hiprtc::compile_hip_source_with_options(
            &[
                super::super::hiprtc::DEVICE_CONVERSIONS_PREAMBLE,
                super::gguf::QUANT_SOURCE,
                "\n#if defined(__gfx1100__)\n",
                super::gguf::K_DECODE_SOURCE,
                "\n#endif\n",
            ].concat(),
            "zllm_rocm_gguf_k_decode_wave64.hip",
            &["-mwavefrontsize64"],
        )
    }).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

#[derive(Clone, Copy)]
pub(crate) struct CtFunctions {
    pub(super) wavefront_size: u32,
    pub(super) cast: usize,
    pub(super) expand: usize,
    pub(super) gather_bf16_rows: usize,
    pub(super) dense: usize,
    pub(super) dense_aligned: usize,
    pub(super) dense_dsa_query: usize,
    pub(super) scalar: usize,
    pub(super) w8_scalar: usize,
    pub(super) w8_rows2: usize,
    pub(super) w8_rows3: usize,
    pub(super) w8_rows4: usize,
    pub(super) w8_rows5: usize,
    pub(super) w8_rows6: usize,
    pub(super) w8_rows7: usize,
    pub(super) w8_rows8: usize,
    pub(super) w8_g32_rows4: usize,
    pub(super) w8_g32_rows6: usize,
    pub(super) w8_g32_rows8: usize,
    pub(super) _w4_rows2: usize,
    pub(super) w4_rows8: usize,
    pub(super) w4_dual: usize,
    pub(super) w4_dual_rows8: usize,
    pub(super) w8_dual: usize,
    pub(super) w8_dual_g32: usize,
    pub(super) w8_dual_g32_perm: usize,
    pub(super) w8_dual_rows2: usize,
    pub(super) w8_dual_g32_rows4: usize,
    pub(super) w8_dual_g32_rows6: usize,
    pub(super) w8_dual_g32_rows8: usize,
    pub(super) wmma: usize,
    pub(super) wmma_w8_g128: usize,
    pub(super) wmma_w4_g128: usize,
    pub(super) wmma_w4_g128_k64: usize,
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
    pub(super) gguf_scatter_grouped_routes: usize,
    pub(super) gguf_gate_up_wmma_q8_0: usize,
    pub(super) gguf_gate_up_wmma_iq4xs: usize,
    pub(super) gguf_grouped_down_q8_0: usize,
    pub(super) gguf_grouped_down_q5_k: usize,
    pub(super) gguf_grouped_down_q6_k: usize,
    pub(super) gguf_grouped_down: usize,
    pub(super) gguf_grouped_down_iq4xs: usize,
    pub(super) gguf_fused_gate_up: usize,
    pub(super) gguf_fused_gate_up_q4_k: usize,
    pub(super) gguf_fused_gate_up_q5_k: usize,
    pub(super) gguf_fused_down: usize,
    pub(super) gguf_fused_gate_up_iq3s: usize,
    pub(super) gguf_fused_gate_up_iq4xs: usize,
    pub(super) gguf_fused_down_iq4xs: usize,
    pub(super) gguf_fused_down_iq4xs_perm: usize,
    pub(super) gguf_fused_gate_up_q8_0: usize,
    pub(super) gguf_fused_down_q8_0: usize,
    pub(super) gguf_fused_gate_up_q8_0_rows2: usize,
    pub(super) gguf_fused_down_q8_0_rows2: usize,
    pub(super) gguf_fused_gate_up_q3_k: usize,
    pub(super) gguf_fused_down_q4_k: usize,
    pub(super) gguf_fused_down_q5_k: usize,
    pub(super) gguf_fused_down_q6_k: usize,
    pub(super) gguf_fused_gate_up_iq3s_wave32: usize,
    pub(super) gguf_fused_gate_up_iq3s_r112: usize,
    pub(super) gguf_fused_gate_up_iq3s_nogrid: usize,
    pub(super) gguf_fused_gate_up_iq3s_wide: usize,
    pub(super) gguf_fused_gate_up_iq3s_wide_rows2: usize,
    pub(super) gguf_fused_gate_up_iq4xs_rows2: usize,
    pub(super) gguf_fused_gate_up_iq3s_wide_rowsn: usize,
    pub(super) gguf_fused_gate_up_iq4xs_rowsn: usize,
    pub(super) gguf_fused_gate_up_iq3s_split: usize,
    pub(super) gguf_fused_gate_up_split_combine: usize,
    pub(super) cooperative_merge_activation: usize,
    pub(super) cooperative_sharded_down: usize,
    pub(super) cooperative_combine_partial: usize,
    pub(super) cooperative_partial_join: usize,
}

pub(crate) fn ct_quantized_functions(device_id: i32) -> Result<CtFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<([usize; 2], CtFunctions), String>>>> = OnceLock::new();
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
            "ct_cooperative_combine_partial_bf16",
            "ct_cooperative_partial_join_f32",
            "ct_quantized_matmul_bf16_w4_g128_wmma",
            "ct_quantized_matmul_bf16_w4_g128_wmma_k64",
            "dense_matmul_bf16_dsa_query_wmma",
            "gguf_grouped_down_iq4xs_f32",
            "ct_quantized_dual_gemv_bf16_w8_g32",
            "gguf_fused_gate_up_iq4xs_f32",
            "gguf_fused_down_iq4xs_f32",
            "gguf_fused_gate_up_iq3s_f32",
            "gguf_fused_gate_up_q8_0_f32",
            "gguf_fused_down_q8_0_f32",
            "gather_bf16_rows_f32",
            "gguf_fused_gate_up_iq3s_wave32_f32",
            "gguf_fused_gate_up_iq3s_r112_f32",
            "gguf_fused_gate_up_iq3s_nogrid_f32",
            "gguf_fused_gate_up_iq3s_wide_f32",
            "gguf_fused_gate_up_iq3s_split_f32",
            "gguf_fused_gate_up_split_combine_f32",
            "gguf_fused_down_iq4xs_perm_f32",
            "ct_quantized_dual_gemv_bf16_w8_g32_perm",
            "gguf_fused_down_q5_k_f32",
            "gguf_fused_down_q6_k_f32",
            "gguf_fused_gate_up_iq3s_wide_rows2_f32",
            "gguf_fused_gate_up_iq4xs_rows2_f32",
            "gguf_fused_gate_up_q3_k_f32",
            "gguf_fused_down_q4_k_f32",
            "ct_quantized_dual_gemv_bf16_w8_rows2",
            "ct_quantized_matmul_bf16_w8_g32_rows4",
            "ct_quantized_matmul_bf16_w8_g32_rows6",
            "ct_quantized_matmul_bf16_w8_g32_rows8",
            "ct_quantized_dual_gemv_bf16_w8_g32_rows4",
            "ct_quantized_dual_gemv_bf16_w8_g32_rows6",
            "ct_quantized_dual_gemv_bf16_w8_g32_rows8",
            "gguf_fused_gate_up_q8_0_rows2_f32",
            "gguf_fused_down_q8_0_rows2_f32",
            "gguf_fused_gate_up_iq3s_wide_rowsn_f32",
            "gguf_fused_gate_up_iq4xs_rowsn_f32",
            "gguf_scatter_grouped_routes",
            "gguf_gate_up_wmma_q8_0_bf16",
            "gguf_grouped_down_q8_0_f32",
            "gguf_grouped_down_q5_k_f32",
            "gguf_grouped_down_q6_k_f32",
            "gguf_gate_up_wmma_iq4xs_bf16",
            "gguf_fused_gate_up_q4_k_bf16",
            "gguf_fused_gate_up_q5_k_bf16",
        ];
        let mut handles = [ptr::null_mut(); 93];
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
        let mut wave64_module = ptr::null_mut();
        if wavefront_size == 32 {
            // 仅 gfx1100 导出这两个入口；未验证的架构保留原模块。
            let loaded = (|| {
                let code = gguf_k_decode_wave64_code()?;
                let status = unsafe { module_load(&mut wave64_module, code.as_ptr().cast()) };
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "hipModuleLoadData GGUF K wave64"));
                }
                let mut kernels = [ptr::null_mut(); 2];
                for (kernel, name) in kernels.iter_mut().zip(["gguf_fused_gate_up_q4_k_bf16", "gguf_fused_down_q5_k_f32"]) {
                    let name = CString::new(name).unwrap();
                    let status = unsafe { module_get_function(kernel, wave64_module, name.as_ptr()) };
                    if status != HIP_SUCCESS {
                        return Err(runtime.hip_error(status, "hipModuleGetFunction GGUF K wave64"));
                    }
                }
                Ok::<_, String>(kernels)
            })();
            match loaded {
                Ok(kernels) => {
                    handles[91] = kernels[0];
                    handles[68] = kernels[1];
                }
                Err(error) => eprintln!("[rocm-gguf-wave64] device={device_id} 保留默认 kernel: {error}"),
            }
        }
        Ok((
            [module as usize, wave64_module as usize],
            CtFunctions {
                wavefront_size: wavefront_size as u32,
                cast: handles[0] as usize,
                expand: handles[1] as usize,
                dense: handles[2] as usize,
                dense_aligned: handles[16] as usize,
                dense_dsa_query: handles[51] as usize,
                scalar: handles[3] as usize,
                w8_scalar: handles[9] as usize,
                w8_rows2: handles[34] as usize,
                w8_rows3: handles[35] as usize,
                w8_rows4: handles[36] as usize,
                w8_rows5: handles[37] as usize,
                w8_rows6: handles[38] as usize,
                w8_rows7: handles[39] as usize,
                w8_rows8: handles[40] as usize,
                w8_g32_rows4: handles[75] as usize,
                w8_g32_rows6: handles[76] as usize,
                w8_g32_rows8: handles[77] as usize,
                w4_dual: handles[41] as usize,
                w4_dual_rows8: handles[44] as usize,
                w4_rows8: handles[42] as usize,
                _w4_rows2: handles[43] as usize,
                w8_dual: handles[14] as usize,
                w8_dual_g32: handles[53] as usize,
                w8_dual_g32_perm: handles[67] as usize,
                w8_dual_rows2: handles[74] as usize,
                w8_dual_g32_rows4: handles[78] as usize,
                w8_dual_g32_rows6: handles[79] as usize,
                w8_dual_g32_rows8: handles[80] as usize,
                gather_bf16_rows: handles[59] as usize,
                wmma: handles[4] as usize,
                wmma_w8_g128: handles[28] as usize,
                wmma_w4_g128: handles[49] as usize,
                wmma_w4_g128_k64: handles[50] as usize,
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
                gguf_scatter_grouped_routes: handles[85] as usize,
                gguf_gate_up_wmma_q8_0: handles[86] as usize,
                gguf_gate_up_wmma_iq4xs: handles[90] as usize,
                gguf_grouped_down_q8_0: handles[87] as usize,
                gguf_grouped_down_q5_k: handles[88] as usize,
                gguf_grouped_down_q6_k: handles[89] as usize,
                gguf_grouped_down: handles[31] as usize,
                gguf_grouped_down_iq4xs: handles[52] as usize,
                gguf_fused_gate_up: handles[32] as usize,
                gguf_fused_gate_up_q4_k: handles[91] as usize,
                gguf_fused_gate_up_q5_k: handles[92] as usize,
                gguf_fused_down: handles[33] as usize,
                gguf_fused_gate_up_iq3s: handles[56] as usize,
                gguf_fused_gate_up_iq4xs: handles[54] as usize,
                gguf_fused_down_iq4xs: handles[55] as usize,
                gguf_fused_gate_up_q8_0: handles[57] as usize,
                gguf_fused_down_q8_0: handles[58] as usize,
                gguf_fused_gate_up_q8_0_rows2: handles[81] as usize,
                gguf_fused_down_q8_0_rows2: handles[82] as usize,
                gguf_fused_gate_up_q3_k: handles[72] as usize,
                gguf_fused_down_q4_k: handles[73] as usize,
                gguf_fused_down_q5_k: handles[68] as usize,
                gguf_fused_down_q6_k: handles[69] as usize,
                gguf_fused_gate_up_iq3s_wave32: handles[60] as usize,
                gguf_fused_gate_up_iq3s_r112: handles[61] as usize,
                gguf_fused_gate_up_iq3s_nogrid: handles[62] as usize,
                gguf_fused_gate_up_iq3s_wide: handles[63] as usize,
                gguf_fused_gate_up_iq3s_wide_rows2: handles[70] as usize,
                gguf_fused_gate_up_iq4xs_rows2: handles[71] as usize,
                gguf_fused_gate_up_iq3s_wide_rowsn: handles[83] as usize,
                gguf_fused_gate_up_iq4xs_rowsn: handles[84] as usize,
                gguf_fused_gate_up_iq3s_split: handles[64] as usize,
                gguf_fused_gate_up_split_combine: handles[65] as usize,
                gguf_fused_down_iq4xs_perm: handles[66] as usize,
                cooperative_merge_activation: handles[45] as usize,
                cooperative_sharded_down: handles[46] as usize,
                cooperative_combine_partial: handles[47] as usize,
                cooperative_partial_join: handles[48] as usize,
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
    metas: &super::ct_grouped::GgufGroupedMetas,
    expert_count: usize,
) -> Result<DeviceBuffer, String> {
    if input_rows == 0 || top_k == 0 || top_k > 16 || expert_count == 0 || expert_count > 256 || route_count != input_rows.checked_mul(top_k).ok_or("GGUF WMMA route 数溢出")? {
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
    let profile_sample = if options().kernel_profile {
        static SAMPLES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        SAMPLES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16
    } else {
        false
    };
    if profile_sample {
        synchronize_device(device_id, "GGUF grouped profile begin")?;
    }

    // decode 链的 expert 输入常为设备 F32(rmsnorm_f32)；WMMA kernel 只吃 BF16，用 cast kernel 原地转换。
    let cast_started = profile_sample.then(std::time::Instant::now);
    let d_input_bf16;
    let input_bf16: &DeviceBuffer = if input.bytes == input_bytes_bf16 {
        input
    } else {
        d_input_bf16 = try_cast_f32_to_bf16_resident(device_id, input, input_elements)?;
        &d_input_bf16
    };
    let cast_wall = if let Some(started) = cast_started {
        synchronize_device(device_id, "GGUF grouped cast profile")?;
        started.elapsed().as_secs_f64()
    } else {
        0.0
    };

    let d_metas = &metas.buffer;
    let down_function = match metas.uniform_types {
        Some([_, _, 23]) => functions.gguf_grouped_down_iq4xs,
        Some([_, _, 8]) => functions.gguf_grouped_down_q8_0,
        Some([_, _, 13]) => functions.gguf_grouped_down_q5_k,
        Some([_, _, 14]) => functions.gguf_grouped_down_q6_k,
        _ => functions.gguf_grouped_down,
    };
    let gate_function = match metas.uniform_types {
        Some([8, 8, _]) => functions.gguf_gate_up_wmma_q8_0,
        Some([23, 23, _]) => functions.gguf_gate_up_wmma_iq4xs,
        _ => functions.gguf_gate_up_wmma,
    };
    let gate_tile_rows = 128;
    // sum(ceil(expert_rows / tile_rows)) <= ceil(route_count / tile_rows) + E - 1。
    // GPU 分组顺带生成紧凑 tile 前缀；只多预留至多 E-1 个尾部 block，
    // 无需把真实 tile 数下载给 host，也不在 gate/up 内循环处理 token 行。
    let gate_tiles = route_count.div_ceil(gate_tile_rows).checked_add(expert_count.min(route_count) - 1).ok_or("GGUF WMMA gate tile 数溢出")?;
    let gate_grid_y = u32::try_from(gate_tiles).map_err(|_| "GGUF WMMA gate grid 超过 u32".to_owned())?;
    // 临时 buffer 走 deferred workspace 复用(参照 try_ct_grouped_experts_bf16);
    // d_output 返回给调用方,保持独立 allocation。
    let grouped_metas_bytes = expert_count * std::mem::size_of::<super::GgufGroupedExpertMeta>();
    let activated_bytes = route_count.checked_mul(intermediate_size).and_then(|n| n.checked_mul(2)).ok_or("GGUF WMMA activated 溢出")?;
    let bytes_per_output_row = route_count.checked_mul(4).ok_or("GGUF WMMA route output 单列字节溢出")?;
    let output_tile_rows = ((64 * 1024 * 1024 / bytes_per_output_row).max(16).min(hidden_size) / 16) * 16;
    let route_output_bytes = route_count.checked_mul(output_tile_rows).and_then(|n| n.checked_mul(4)).ok_or("GGUF WMMA route output 溢出")?;
    let workspace_sizes = [route_count * 4, route_count * 4, route_count * 4, (expert_count + 1) * 4, (expert_count + 1) * 4, grouped_metas_bytes, activated_bytes, route_output_bytes];
    super::tensor::with_deferred_tensor_workspace(device_id, &workspace_sizes, |workspace| {
        let d_grouped_tokens = workspace.buffer(0);
        let d_grouped_weights = workspace.buffer(1);
        let d_route_to_grouped = workspace.buffer(2);
        let d_gate_tile_offsets = workspace.buffer(3);
        let d_grouped_offsets = workspace.buffer(4);
        let d_grouped_metas = workspace.buffer(5);
        let d_activated = workspace.buffer(6);
        let d_route_output = workspace.buffer(7);
        let d_output = DeviceBuffer::allocate(device_id, input_rows.checked_mul(hidden_size).and_then(|n| n.checked_mul(4)).ok_or("GGUF WMMA output 溢出")?)?;

        let mut route_count = u32::try_from(route_count).map_err(|_| "GGUF WMMA route 数超过 u32".to_owned())?;
        let mut top_k = u32::try_from(top_k).map_err(|_| "GGUF WMMA top_k 超过 u32".to_owned())?;
        let mut expert_count = u32::try_from(expert_count).map_err(|_| "GGUF WMMA expert 数超过 u32".to_owned())?;
        let mut gate_tile_rows = gate_tile_rows as u32;
        let routes_started = profile_sample.then(std::time::Instant::now);
        {
            let mut route_ids_pointer = route_ids.pointer;
            let mut route_weights_pointer = route_weights.pointer;
            let mut metas_pointer = d_metas.pointer;
            let mut grouped_tokens_pointer = d_grouped_tokens.pointer;
            let mut grouped_weights_pointer = d_grouped_weights.pointer;
            let mut route_to_grouped_pointer = d_route_to_grouped.pointer;
            let mut grouped_offsets_pointer = d_grouped_offsets.pointer;
            let mut grouped_metas_pointer = d_grouped_metas.pointer;
            let mut gate_tile_offsets_pointer = d_gate_tile_offsets.pointer;
            let mut arguments = [
                (&mut route_ids_pointer as *mut *mut c_void).cast(),
                (&mut route_weights_pointer as *mut *mut c_void).cast(),
                (&mut metas_pointer as *mut *mut c_void).cast(),
                (&mut grouped_tokens_pointer as *mut *mut c_void).cast(),
                (&mut grouped_weights_pointer as *mut *mut c_void).cast(),
                (&mut route_to_grouped_pointer as *mut *mut c_void).cast(),
                (&mut grouped_offsets_pointer as *mut *mut c_void).cast(),
                (&mut grouped_metas_pointer as *mut *mut c_void).cast(),
                (&mut gate_tile_offsets_pointer as *mut *mut c_void).cast(),
                (&mut route_count as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut expert_count as *mut u32).cast(),
                (&mut expert_count as *mut u32).cast(),
                (&mut gate_tile_rows as *mut u32).cast(),
            ];
            let status = unsafe { module_launch(functions.gguf_routes as *mut c_void, 1, 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF routes"));
            }
        }
        if expert_count > 1 {
            let mut route_ids_pointer = route_ids.pointer;
            let mut route_weights_pointer = route_weights.pointer;
            let mut grouped_offsets_pointer = d_grouped_offsets.pointer;
            let mut grouped_tokens_pointer = d_grouped_tokens.pointer;
            let mut grouped_weights_pointer = d_grouped_weights.pointer;
            let mut route_to_grouped_pointer = d_route_to_grouped.pointer;
            let mut arguments = [
                (&mut route_ids_pointer as *mut *mut c_void).cast(),
                (&mut route_weights_pointer as *mut *mut c_void).cast(),
                (&mut grouped_offsets_pointer as *mut *mut c_void).cast(),
                (&mut grouped_tokens_pointer as *mut *mut c_void).cast(),
                (&mut grouped_weights_pointer as *mut *mut c_void).cast(),
                (&mut route_to_grouped_pointer as *mut *mut c_void).cast(),
                (&mut route_count as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
            ];
            let status = unsafe { module_launch(functions.gguf_scatter_grouped_routes as *mut c_void, expert_count, 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF stable route scatter"));
            }
        }
        let routes_wall = if let Some(started) = routes_started {
            synchronize_device(device_id, "GGUF grouped routes profile")?;
            started.elapsed().as_secs_f64()
        } else {
            0.0
        };
        let gate_started = profile_sample.then(std::time::Instant::now);
        {
            let mut input_pointer = input_bf16.pointer;
            let mut grouped_tokens_pointer = d_grouped_tokens.pointer;
            let mut grouped_metas_pointer = d_grouped_metas.pointer;
            let mut grouped_offsets_pointer = d_grouped_offsets.pointer;
            let mut gate_tile_offsets_pointer = d_gate_tile_offsets.pointer;
            let mut activated_pointer = d_activated.pointer;
            let mut hidden = u32::try_from(hidden_size).map_err(|_| "GGUF WMMA hidden 超过 u32".to_owned())?;
            let mut intermediate = u32::try_from(intermediate_size).map_err(|_| "GGUF WMMA intermediate 超过 u32".to_owned())?;
            let mut arguments = [
                (&mut input_pointer as *mut *mut c_void).cast(),
                (&mut grouped_tokens_pointer as *mut *mut c_void).cast(),
                (&mut grouped_metas_pointer as *mut *mut c_void).cast(),
                (&mut grouped_offsets_pointer as *mut *mut c_void).cast(),
                (&mut gate_tile_offsets_pointer as *mut *mut c_void).cast(),
                (&mut activated_pointer as *mut *mut c_void).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut expert_count as *mut u32).cast(),
            ];
            let grid_x = u32::try_from(intermediate_size / 64).map_err(|_| "GGUF WMMA grid 超过 u32".to_owned())?;
            let status = unsafe { module_launch(gate_function as *mut c_void, grid_x, gate_grid_y, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF gate_up WMMA"));
            }
        }
        let gate_wall = if let Some(started) = gate_started {
            synchronize_device(device_id, "GGUF grouped gate/up profile")?;
            started.elapsed().as_secs_f64()
        } else {
            0.0
        };
        let down_started = profile_sample.then(std::time::Instant::now);
        for output_row_base in (0..hidden_size).step_by(output_tile_rows) {
            let output_slice_rows = (hidden_size - output_row_base).min(output_tile_rows);
            let mut activated_pointer = d_activated.pointer;
            let mut grouped_tokens_pointer = d_grouped_tokens.pointer;
            let mut grouped_weights_pointer = d_grouped_weights.pointer;
            let mut grouped_metas_pointer = d_grouped_metas.pointer;
            let mut grouped_offsets_pointer = d_grouped_offsets.pointer;
            let mut route_output_pointer = d_route_output.pointer;
            let mut hidden = u32::try_from(hidden_size).map_err(|_| "GGUF WMMA hidden 超过 u32".to_owned())?;
            let mut intermediate = u32::try_from(intermediate_size).map_err(|_| "GGUF WMMA intermediate 超过 u32".to_owned())?;
            let mut base = 0_u32;
            let mut output_row_base = u32::try_from(output_row_base).map_err(|_| "GGUF WMMA output_row_base 超过 u32".to_owned())?;
            let mut output_slice_rows = u32::try_from(output_slice_rows).map_err(|_| "GGUF WMMA output_slice_rows 超过 u32".to_owned())?;
            let mut arguments = [
                (&mut activated_pointer as *mut *mut c_void).cast(),
                (&mut grouped_tokens_pointer as *mut *mut c_void).cast(),
                (&mut grouped_weights_pointer as *mut *mut c_void).cast(),
                (&mut grouped_metas_pointer as *mut *mut c_void).cast(),
                (&mut grouped_offsets_pointer as *mut *mut c_void).cast(),
                (&mut route_output_pointer as *mut *mut c_void).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut base as *mut u32).cast(),
                (&mut expert_count as *mut u32).cast(),
                (&mut output_row_base as *mut u32).cast(),
                (&mut output_slice_rows as *mut u32).cast(),
            ];
            let grid_x = output_slice_rows.div_ceil(128);
            let status = unsafe { module_launch(down_function as *mut c_void, grid_x, 1, expert_count, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF grouped down"));
            }

            let mut route_output_pointer = d_route_output.pointer;
            let mut route_to_grouped_pointer = d_route_to_grouped.pointer;
            let mut output_pointer = d_output.pointer;
            let mut shared_output_pointer = ptr::null_mut();
            let mut residual_pointer = ptr::null_mut();
            let mut fused_epilogue = 0_u32;
            let mut reduce_arguments = [
                (&mut route_output_pointer as *mut *mut c_void).cast(),
                (&mut route_to_grouped_pointer as *mut *mut c_void).cast(),
                (&mut output_pointer as *mut *mut c_void).cast(),
                (&mut shared_output_pointer as *mut *mut c_void).cast(),
                (&mut residual_pointer as *mut *mut c_void).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut output_row_base as *mut u32).cast(),
                (&mut output_slice_rows as *mut u32).cast(),
                (&mut route_count as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut fused_epilogue as *mut u32).cast(),
            ];
            let output_slice_elements = u32::try_from(input_rows.checked_mul(output_slice_rows as usize).ok_or("GGUF WMMA output slice 溢出")?).map_err(|_| "GGUF WMMA output slice 超过 u32".to_owned())?;
            let status = unsafe {
                module_launch(functions.grouped_down_reduce_routes as *mut c_void, output_slice_elements.div_ceil(256), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), reduce_arguments.as_mut_ptr(), ptr::null_mut())
            };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF deterministic down reduce"));
            }
        }
        if let Some(started) = down_started {
            synchronize_device(device_id, "GGUF grouped down profile")?;
            eprintln!(
                "[rocm-kernel] gguf-grouped device={device_id} rows={input_rows} routes={} top_k={} experts={} intermediate={intermediate_size} output_tile_rows={output_tile_rows} cast={cast_wall:.6}s compact={routes_wall:.6}s gate_up={gate_wall:.6}s down={:.6}s",
                route_count,
                top_k,
                expert_count,
                started.elapsed().as_secs_f64(),
            );
        }
        // 与 CT grouped 一致：纯异步提交。
        Ok(d_output)
    })
}
