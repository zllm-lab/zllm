pub(super) const QUANT_SOURCE: &str = include_str!("../gguf_quant.hip");
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

#[cfg(test)]
mod tests {
    fn bytes<T>(values: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn iq_rows(tensor_type: u32, rows: usize, columns: usize, seed: usize) -> Vec<u8> {
        let (block_elements, block_bytes) = crate::weight::codec::ggml::block_layout(tensor_type).unwrap();
        let mut packed = vec![0_u8; rows * (columns / block_elements) * block_bytes];
        for (index, byte) in packed.iter_mut().enumerate() {
            *byte = ((index.wrapping_mul(37) + seed * 29 + index / 11) & 255) as u8;
        }
        for block in packed.chunks_exact_mut(block_bytes) {
            block[..2].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
        }
        packed
    }

    #[test]
    fn glm53_iq_expert_source_compiles() {
        super::super::ct_quantized_functions(0).unwrap();
    }

    fn assert_grouped_wmma_matches_cpu(tokens: usize, top_k: usize, expert_count: usize, route_ids_host: &[u32], route_weights_host: &[f32]) {
        use half::bf16;

        const DEVICE: i32 = 0;
        const HIDDEN: usize = 256;
        const INTERMEDIATE: usize = 256;
        assert_eq!(route_ids_host.len(), tokens * top_k);
        assert_eq!(route_weights_host.len(), tokens * top_k);

        let input_bits: Vec<u16> = (0..tokens * HIDDEN).map(|index| bf16::from_f32(((index * 17 % 43) as f32 - 21.0) / 128.0).to_bits()).collect();
        let input_values: Vec<f32> = input_bits.iter().map(|&bits| bf16::from_bits(bits).to_f32()).collect();
        let gate_host = iq_rows(21, INTERMEDIATE, HIDDEN, 1);
        let up_host = iq_rows(21, INTERMEDIATE, HIDDEN, 2);
        let down_host = iq_rows(23, HIDDEN, INTERMEDIATE, 3);
        let gate_values = crate::weight::codec::ggml::dequantize(21, &gate_host, INTERMEDIATE * HIDDEN).unwrap();
        let up_values = crate::weight::codec::ggml::dequantize(21, &up_host, INTERMEDIATE * HIDDEN).unwrap();
        let down_values = crate::weight::codec::ggml::dequantize(23, &down_host, HIDDEN * INTERMEDIATE).unwrap();

        let input = super::DeviceBuffer::upload(DEVICE, bytes(&input_bits)).unwrap();
        let gate = super::DeviceBuffer::upload(DEVICE, &gate_host).unwrap();
        let up = super::DeviceBuffer::upload(DEVICE, &up_host).unwrap();
        let down = super::DeviceBuffer::upload(DEVICE, &down_host).unwrap();
        let route_ids = super::DeviceBuffer::upload(DEVICE, bytes(route_ids_host)).unwrap();
        let route_weights = super::DeviceBuffer::upload(DEVICE, bytes(route_weights_host)).unwrap();
        let meta = super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: down.device_pointer() as u64, gate_type: 21, up_type: 21, down_type: 23 };
        let metas = super::resident_gguf_grouped_metas(DEVICE, &vec![meta; expert_count]).unwrap();
        let output = super::try_gguf_grouped_wmma_experts(DEVICE, &input, tokens, HIDDEN, INTERMEDIATE, top_k, &route_ids, &route_weights, tokens * top_k, &metas, expert_count).unwrap();
        let mut actual = vec![0.0_f32; tokens * HIDDEN];
        output.copy_to_host(bytes_mut(&mut actual)).unwrap();

        let mut expected = vec![0.0_f32; tokens * HIDDEN];
        let mut activated = vec![0.0_f32; INTERMEDIATE];
        for token in 0..tokens {
            let input_row = &input_values[token * HIDDEN..(token + 1) * HIDDEN];
            for row in 0..INTERMEDIATE {
                let gate_row = &gate_values[row * HIDDEN..(row + 1) * HIDDEN];
                let up_row = &up_values[row * HIDDEN..(row + 1) * HIDDEN];
                let gate = gate_row.iter().zip(input_row).map(|(weight, input)| bf16::from_f32(*weight).to_f32() * input).sum::<f32>();
                let up = up_row.iter().zip(input_row).map(|(weight, input)| bf16::from_f32(*weight).to_f32() * input).sum::<f32>();
                activated[row] = bf16::from_f32(gate / (1.0 + (-gate).exp()) * up).to_f32();
            }
            let route_scale = route_weights_host[token * top_k..(token + 1) * top_k].iter().sum::<f32>();
            for row in 0..HIDDEN {
                let down_row = &down_values[row * INTERMEDIATE..(row + 1) * INTERMEDIATE];
                expected[token * HIDDEN + row] = down_row.iter().zip(&activated).map(|(weight, value)| bf16::from_f32(*weight).to_f32() * value).sum::<f32>() * route_scale;
            }
        }
        let mut max_abs = 0.0_f32;
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(actual.is_finite(), "index={index} actual={actual}");
            max_abs = max_abs.max((actual - expected).abs());
            let tolerance = 0.02 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance, "index={index} actual={actual} expected={expected} tolerance={tolerance}");
        }
        eprintln!("[glm53-iq-grouped-wmma-oracle] max_abs={max_abs:.6e}");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_iq_grouped_wmma_matches_cpu() {
        super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();
        let routed_ids = (0..9).flat_map(|_| [1_u32, 0_u32]).collect::<Vec<_>>();
        let routed_weights = (0..9).flat_map(|_| [0.25_f32, 0.5_f32]).collect::<Vec<_>>();
        assert_grouped_wmma_matches_cpu(9, 2, 2, &routed_ids, &routed_weights);

        // 超过一个 128-row tile，覆盖 shared expert 的多 y block 路径。
        let shared_ids = vec![0_u32; 129];
        let shared_weights = vec![1.0_f32; 129];
        assert_grouped_wmma_matches_cpu(129, 1, 1, &shared_ids, &shared_weights);
    }

    fn assert_fused_decode_matches_generic(types: [u32; 3]) {
        use half::bf16;

        const DEVICE: i32 = 0;
        const HIDDEN: usize = 256;
        const INTERMEDIATE: usize = 256;
        let input_bits = (0..HIDDEN).map(|index| bf16::from_f32(((index * 13 % 37) as f32 - 18.0) / 128.0).to_bits()).collect::<Vec<_>>();
        let gate_host = iq_rows(types[0], INTERMEDIATE, HIDDEN, 7);
        let up_host = iq_rows(types[1], INTERMEDIATE, HIDDEN, 11);
        let down_host = iq_rows(types[2], HIDDEN, INTERMEDIATE, 13);
        let input = super::DeviceBuffer::upload(DEVICE, bytes(&input_bits)).unwrap();
        let gate = super::DeviceBuffer::upload(DEVICE, &gate_host).unwrap();
        let up = super::DeviceBuffer::upload(DEVICE, &up_host).unwrap();
        let down = super::DeviceBuffer::upload(DEVICE, &down_host).unwrap();
        let route_ids = super::DeviceBuffer::upload(DEVICE, bytes(&[0_u32])).unwrap();
        let route_weights = super::DeviceBuffer::upload(DEVICE, bytes(&[0.75_f32])).unwrap();
        let meta = super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: down.device_pointer() as u64, gate_type: types[0], up_type: types[1], down_type: types[2] };
        let metas = super::resident_gguf_grouped_metas(DEVICE, &[meta]).unwrap();
        let generic = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
        let packed = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
        super::try_gguf_fused_decode_experts_impl(DEVICE, &input, 1, HIDDEN, INTERMEDIATE, 1, &route_ids, &route_weights, 1, &metas.buffer, &generic, None).unwrap();
        super::try_gguf_fused_decode_experts(DEVICE, &input, 1, HIDDEN, INTERMEDIATE, 1, &route_ids, &route_weights, 1, &metas, &packed).unwrap();
        let mut generic_host = vec![0.0_f32; HIDDEN];
        let mut packed_host = vec![0.0_f32; HIDDEN];
        generic.copy_to_host(bytes_mut(&mut generic_host)).unwrap();
        packed.copy_to_host(bytes_mut(&mut packed_host)).unwrap();
        for (index, (&generic, &packed)) in generic_host.iter().zip(&packed_host).enumerate() {
            assert!(generic.is_finite() && packed.is_finite(), "index={index} generic={generic} packed={packed}");
            let tolerance = 0.02 * generic.abs().max(1.0);
            assert!((generic - packed).abs() <= tolerance, "index={index} generic={generic} packed={packed} tolerance={tolerance}");
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_iq4xs_fused_decode_matches_generic() {
        assert_fused_decode_matches_generic([23, 23, 23]);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_iq3s_iq4xs_fused_decode_matches_generic() {
        assert_fused_decode_matches_generic([21, 21, 23]);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_q8_0_fused_decode_matches_generic() {
        assert_fused_decode_matches_generic([8, 8, 8]);
    }
}

#[allow(clippy::too_many_arguments)]
fn try_gguf_fused_decode_experts_impl(
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
    uniform_types: Option<[u32; 3]>,
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
    let iq3s_gate_up = input_is_bf16 && matches!(uniform_types, Some([21, 21, _]));
    let iq4xs_gate_up = input_is_bf16 && matches!(uniform_types, Some([23, 23, _]));
    let q8_0_gate_up = input_is_bf16 && matches!(uniform_types, Some([8, 8, _]));
    let iq4xs_down = intermediate_size.is_multiple_of(256) && matches!(uniform_types, Some([_, _, 23]));
    let q8_0_down = intermediate_size.is_multiple_of(256) && matches!(uniform_types, Some([_, _, 8]));
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
        let outputs_per_block = if iq3s_gate_up || iq4xs_gate_up { 32 } else { 16 };
        let grid_x = u32::try_from(intermediate_size.div_ceil(outputs_per_block)).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let grid_y = u32::try_from(route_count).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let function = if iq3s_gate_up {
            functions.gguf_fused_gate_up_iq3s
        } else if iq4xs_gate_up {
            functions.gguf_fused_gate_up_iq4xs
        } else if q8_0_gate_up {
            functions.gguf_fused_gate_up_q8_0
        } else {
            functions.gguf_fused_gate_up
        };
        let status = unsafe { module_launch(function as *mut c_void, grid_x, grid_y, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
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
        let outputs_per_block = if iq4xs_down { 32 } else { 16 };
        let grid_x = u32::try_from(hidden_size.div_ceil(outputs_per_block)).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let function = if iq4xs_down {
            functions.gguf_fused_down_iq4xs
        } else if q8_0_down {
            functions.gguf_fused_down_q8_0
        } else {
            functions.gguf_fused_down
        };
        let status = unsafe { module_launch(function as *mut c_void, grid_x, tokens, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF fused down"));
        }
    }
    Ok(())
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
    metas: &super::ct_grouped::GgufGroupedMetas,
    output: &DeviceBuffer,
) -> Result<(), String> {
    try_gguf_fused_decode_experts_impl(device_id, input, input_rows, hidden_size, intermediate_size, top_k, route_ids, route_weights, route_count, &metas.buffer, output, metas.uniform_types)
}

/// GGUF cooperative MoE 直接复用单卡的两条优化算法：小 route 走两段 fused
/// kernel，大 route 走 grouped WMMA。调用方传入的 meta 已是 gate/up 行半片与
/// down-K 半片，因此这里只产生 full-hidden partial，不做 activation all-gather。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gguf_cooperative_routed_bf16(
    device_id: i32,
    input: &DeviceBuffer,
    input_rows: usize,
    hidden_size: usize,
    intermediate_size: usize,
    route_ids: &DeviceBuffer,
    route_weights: &DeviceBuffer,
    route_count: usize,
    top_k: usize,
    metas: &super::ct_grouped::GgufGroupedMetas,
    expert_count: usize,
) -> Result<DeviceBuffer, String> {
    if route_count <= 64 {
        let output = DeviceBuffer::allocate(device_id, input_rows.checked_mul(hidden_size).and_then(|n| n.checked_mul(4)).ok_or("GGUF cooperative output 溢出")?)?;
        try_gguf_fused_decode_experts_impl(device_id, input, input_rows, hidden_size, intermediate_size, top_k, route_ids, route_weights, route_count, &metas.buffer, &output, metas.uniform_types)?;
        Ok(output)
    } else {
        try_gguf_grouped_wmma_experts(device_id, input, input_rows, hidden_size, intermediate_size, top_k, route_ids, route_weights, route_count, metas, expert_count)
    }
}

pub(crate) fn try_gguf_cooperative_shared_bf16(device_id: i32, input: &DeviceBuffer, input_rows: usize, hidden_size: usize, intermediate_size: usize, meta: &super::ct_grouped::GgufGroupedMetas) -> Result<DeviceBuffer, String> {
    let (route_ids, route_weights) = super::ct_grouped::cooperative_single_expert_route(device_id, input_rows)?;
    if input_rows <= 64 {
        let output = DeviceBuffer::allocate_reusable(device_id, input_rows.checked_mul(hidden_size).and_then(|n| n.checked_mul(4)).ok_or("GGUF shared output 溢出")?)?;
        try_gguf_fused_decode_experts(device_id, input, input_rows, hidden_size, intermediate_size, 1, &route_ids, &route_weights, input_rows, meta, &output)?;
        return Ok(output);
    }
    try_gguf_cooperative_routed_bf16(device_id, input, input_rows, hidden_size, intermediate_size, &route_ids, &route_weights, input_rows, 1, meta, 1)
}
