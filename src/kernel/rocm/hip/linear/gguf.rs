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
    pub(crate) fn bytes<T>(values: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values)) }
    }

    pub(crate) fn iq_rows(tensor_type: u32, rows: usize, columns: usize, seed: usize) -> Vec<u8> {
        let (block_elements, block_bytes) = crate::weight::codec::ggml::block_layout(tensor_type).unwrap();
        let mut packed = vec![0_u8; rows * (columns / block_elements) * block_bytes];
        for (index, byte) in packed.iter_mut().enumerate() {
            *byte = ((index.wrapping_mul(37) + seed * 29 + index / 11) & 255) as u8;
        }
        for block in packed.chunks_exact_mut(block_bytes) {
            if tensor_type == 11 {
                block[108..110].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
            } else if tensor_type == 14 {
                block[208..210].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
            } else {
                block[..2].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
                if matches!(tensor_type, 12 | 13) {
                    block[2..4].copy_from_slice(&half::f16::from_f32(0.0005).to_le_bytes());
                }
            }
        }
        packed
    }

    #[test]
    fn glm53_iq_expert_source_compiles() {
        super::super::ct_quantized_functions(0).unwrap();
    }

    /// IQ3_S gate_up 结构对照：现有 8-lane 子组 kernel vs wave32 LDS 探针，
    /// 真实形状（hidden=6144、inter=2048、top_k=8），先比数值再比速度。
    /// `cargo test --release --features with-rocm iq3s_wave32_probe_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn iq3s_wave32_probe_bench() {
        use half::bf16;
        const DEVICE: i32 = 0;
        const HIDDEN: usize = 6144;
        const INTERMEDIATE: usize = 2048;
        const TOP_K: usize = 8;
        const EXPERTS: usize = 8;
        super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();
        let input_bits = (0..HIDDEN).map(|index| bf16::from_f32(((index * 13 % 37) as f32 - 18.0) / 128.0).to_bits()).collect::<Vec<_>>();
        let input = super::DeviceBuffer::upload(DEVICE, bytes(&input_bits)).unwrap();
        let mut metas_host = Vec::with_capacity(EXPERTS);
        let mut weight_bytes = 0_usize;
        for expert in 0..EXPERTS {
            let gate_host = iq_rows(21, INTERMEDIATE, HIDDEN, 7 + expert);
            let up_host = iq_rows(21, INTERMEDIATE, HIDDEN, 37 + expert);
            let down_host = iq_rows(23, HIDDEN, INTERMEDIATE, 67 + expert);
            weight_bytes += gate_host.len() + up_host.len();
            let gate = super::DeviceBuffer::upload(DEVICE, &gate_host).unwrap();
            let up = super::DeviceBuffer::upload(DEVICE, &up_host).unwrap();
            let down = super::DeviceBuffer::upload(DEVICE, &down_host).unwrap();
            metas_host.push(super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: down.device_pointer() as u64, gate_type: 21, up_type: 21, down_type: 23 });
            std::mem::forget((gate, up, down));
        }
        let metas = super::resident_gguf_grouped_metas(DEVICE, &metas_host).unwrap();
        // r112 臂：重排后的 gate/up 副本 + 独立 metas（类型字段只是标签，kernel 由启动点选择）。
        let mut metas_r112 = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let gate_host = iq_rows(21, INTERMEDIATE, HIDDEN, 7 + expert);
            let up_host = iq_rows(21, INTERMEDIATE, HIDDEN, 37 + expert);
            let gate = super::DeviceBuffer::upload(DEVICE, &super::repack_iq3s_r112(&gate_host, INTERMEDIATE, HIDDEN).unwrap()).unwrap();
            let up = super::DeviceBuffer::upload(DEVICE, &super::repack_iq3s_r112(&up_host, INTERMEDIATE, HIDDEN).unwrap()).unwrap();
            metas_r112.push(super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: metas_host[expert].down, gate_type: 121, up_type: 121, down_type: 23 });
            std::mem::forget((gate, up));
        }
        let metas_r112 = super::resident_gguf_grouped_metas(DEVICE, &metas_r112).unwrap();
        let activated_r112 = super::DeviceBuffer::allocate(DEVICE, TOP_K * INTERMEDIATE * 2).unwrap();
        let route_ids = super::DeviceBuffer::upload(DEVICE, bytes(&[0_u32, 1, 2, 3, 4, 5, 6, 7])).unwrap();
        let activated_current = super::DeviceBuffer::allocate(DEVICE, TOP_K * INTERMEDIATE * 2).unwrap();
        let activated_probe = super::DeviceBuffer::allocate(DEVICE, TOP_K * INTERMEDIATE * 2).unwrap();
        let functions = super::super::ct_quantized_functions(DEVICE).unwrap();

        let launch_current = |activated: &super::DeviceBuffer| {
            let mut input_pointer = input.pointer;
            let mut metas_pointer = metas.buffer.pointer;
            let mut route_ids_pointer = route_ids.pointer;
            let mut activated_pointer = activated.pointer as usize;
            let mut assignments = TOP_K as u32;
            let mut top_k = TOP_K as u32;
            let mut hidden = HIDDEN as u32;
            let mut intermediate = INTERMEDIATE as u32;
            let mut is_bf16 = 1u32;
            let mut arguments = [
                (&mut input_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut metas_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut activated_pointer as *mut usize).cast(),
                (&mut assignments as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut is_bf16 as *mut u32).cast(),
            ];
            let status = unsafe {
                crate::kernel::rocm::hip::kernel_launch_trampoline(
                    functions.gguf_fused_gate_up_iq3s as *mut std::ffi::c_void,
                    (INTERMEDIATE / 32) as u32,
                    TOP_K as u32,
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
            assert_eq!(status, crate::kernel::rocm::hip::HIP_SUCCESS);
        };

        launch_current(&activated_current);
        super::try_gguf_fused_gate_up_iq3s_wave32_probe(DEVICE, &input, TOP_K, HIDDEN, INTERMEDIATE, TOP_K, &route_ids, &metas.buffer, &activated_probe).unwrap();
        super::super::synchronize_device(DEVICE, "iq3s wave32 probe 数值").unwrap();
        let mut current_host = vec![0_u16; TOP_K * INTERMEDIATE];
        let mut probe_host = vec![0_u16; TOP_K * INTERMEDIATE];
        activated_current.copy_to_host(bytes_mut(&mut current_host)).unwrap();
        activated_probe.copy_to_host(bytes_mut(&mut probe_host)).unwrap();
        let mut max_abs = 0.0_f32;
        for (index, (left, right)) in current_host.iter().zip(&probe_host).enumerate() {
            let left = bf16::from_bits(*left).to_f32();
            let right = bf16::from_bits(*right).to_f32();
            assert!(left.is_finite() && right.is_finite(), "index={index} current={left} probe={right}");
            max_abs = max_abs.max((left - right).abs());
        }
        let bench = |label: &str, call: &mut dyn FnMut()| {
            for _ in 0..3 {
                call();
            }
            let started = std::time::Instant::now();
            const ROUNDS: usize = 100;
            for _ in 0..ROUNDS {
                call();
            }
            super::super::synchronize_device(DEVICE, "iq3s wave32 probe bench").unwrap();
            let micros = started.elapsed().as_micros() as f64 / ROUNDS as f64;
            eprintln!("[iq3s-wave32-probe] {label} avg_us={micros:.1} bw_GBps={:.0}", weight_bytes as f64 / (1u64 << 30) as f64 * 1024.0 / (micros / 1e6));
        };
        bench("current-8lane", &mut || launch_current(&activated_current));
        bench("wave32-lds", &mut || super::try_gguf_fused_gate_up_iq3s_wave32_probe(DEVICE, &input, TOP_K, HIDDEN, INTERMEDIATE, TOP_K, &route_ids, &metas.buffer, &activated_probe).unwrap());
        let launch_r112 = |activated: &super::DeviceBuffer| {
            let mut input_pointer = input.pointer;
            let mut metas_pointer = metas_r112.buffer.pointer;
            let mut route_ids_pointer = route_ids.pointer;
            let mut activated_pointer = activated.pointer as usize;
            let mut assignments = TOP_K as u32;
            let mut top_k = TOP_K as u32;
            let mut hidden = HIDDEN as u32;
            let mut intermediate = INTERMEDIATE as u32;
            let mut is_bf16 = 1u32;
            let mut arguments = [
                (&mut input_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut metas_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut activated_pointer as *mut usize).cast(),
                (&mut assignments as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut is_bf16 as *mut u32).cast(),
            ];
            let status = unsafe {
                crate::kernel::rocm::hip::kernel_launch_trampoline(
                    functions.gguf_fused_gate_up_iq3s_r112 as *mut std::ffi::c_void,
                    (INTERMEDIATE / 32) as u32,
                    TOP_K as u32,
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
            assert_eq!(status, crate::kernel::rocm::hip::HIP_SUCCESS);
        };
        launch_r112(&activated_r112);
        super::super::synchronize_device(DEVICE, "iq3s r112 数值").unwrap();
        let mut r112_host = vec![0_u16; TOP_K * INTERMEDIATE];
        activated_r112.copy_to_host(bytes_mut(&mut r112_host)).unwrap();
        let mut r112_max_abs = 0.0_f32;
        for (left, right) in current_host.iter().zip(&r112_host) {
            r112_max_abs = r112_max_abs.max((bf16::from_bits(*left).to_f32() - bf16::from_bits(*right).to_f32()).abs());
        }
        // split-K 臂：partial(grid z=splits) + combine。
        let launch_split = |splits: u32, activated: &super::DeviceBuffer, gate_partial: &super::DeviceBuffer, up_partial: &super::DeviceBuffer| {
            let mut input_pointer = input.pointer;
            let mut metas_pointer = metas.buffer.pointer;
            let mut route_ids_pointer = route_ids.pointer;
            let mut gate_pointer = gate_partial.pointer;
            let mut up_pointer = up_partial.pointer;
            let mut assignments = TOP_K as u32;
            let mut top_k = TOP_K as u32;
            let mut hidden = HIDDEN as u32;
            let mut intermediate = INTERMEDIATE as u32;
            let mut is_bf16 = 1u32;
            let mut partial_args = [
                (&mut input_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut metas_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut gate_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut up_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut assignments as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut is_bf16 as *mut u32).cast(),
            ];
            let status = unsafe {
                crate::kernel::rocm::hip::kernel_launch_trampoline(
                    functions.gguf_fused_gate_up_iq3s_split as *mut std::ffi::c_void,
                    (INTERMEDIATE / 32) as u32,
                    TOP_K as u32,
                    splits,
                    256,
                    1,
                    1,
                    0,
                    crate::kernel::rocm::hip::active_compute_stream(),
                    partial_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(status, crate::kernel::rocm::hip::HIP_SUCCESS);
            let mut activated_pointer = activated.pointer;
            let mut splits_u32 = splits;
            let mut combine_args = [
                (&mut gate_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut up_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut activated_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut assignments as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut splits_u32 as *mut u32).cast(),
            ];
            let status = unsafe {
                crate::kernel::rocm::hip::kernel_launch_trampoline(
                    functions.gguf_fused_gate_up_split_combine as *mut std::ffi::c_void,
                    (TOP_K * INTERMEDIATE).div_ceil(256) as u32,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    crate::kernel::rocm::hip::active_compute_stream(),
                    combine_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(status, crate::kernel::rocm::hip::HIP_SUCCESS);
        };
        for splits in [2_u32, 3, 4] {
            let gate_partial = super::DeviceBuffer::allocate(DEVICE, splits as usize * TOP_K * INTERMEDIATE * 4).unwrap();
            let up_partial = super::DeviceBuffer::allocate(DEVICE, splits as usize * TOP_K * INTERMEDIATE * 4).unwrap();
            let activated_split = super::DeviceBuffer::allocate(DEVICE, TOP_K * INTERMEDIATE * 2).unwrap();
            launch_split(splits, &activated_split, &gate_partial, &up_partial);
            super::super::synchronize_device(DEVICE, "iq3s split 数值").unwrap();
            let mut split_host = vec![0_u16; TOP_K * INTERMEDIATE];
            activated_split.copy_to_host(bytes_mut(&mut split_host)).unwrap();
            let mut split_max_abs = 0.0_f32;
            for (left, right) in current_host.iter().zip(&split_host) {
                split_max_abs = split_max_abs.max((bf16::from_bits(*left).to_f32() - bf16::from_bits(*right).to_f32()).abs());
            }
            eprintln!("[iq3s-wave32-probe] split{splits} max_abs={split_max_abs:.6e}");
            assert!(split_max_abs <= 1.0e-2, "split{splits} 数值超差: {split_max_abs}");
            bench(&format!("split{splits}"), &mut || launch_split(splits, &activated_split, &gate_partial, &up_partial));
        }
        bench("r112-align", &mut || launch_r112(&activated_r112));
        // M1 归因臂：跳过码本 LDS 查表（数值无意义，仅计时）。
        let launch_nogrid = |activated: &super::DeviceBuffer| {
            let mut input_pointer = input.pointer;
            let mut metas_pointer = metas.buffer.pointer;
            let mut route_ids_pointer = route_ids.pointer;
            let mut activated_pointer = activated.pointer as usize;
            let mut assignments = TOP_K as u32;
            let mut top_k = TOP_K as u32;
            let mut hidden = HIDDEN as u32;
            let mut intermediate = INTERMEDIATE as u32;
            let mut is_bf16 = 1u32;
            let mut arguments = [
                (&mut input_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut metas_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut activated_pointer as *mut usize).cast(),
                (&mut assignments as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut is_bf16 as *mut u32).cast(),
            ];
            let status = unsafe {
                crate::kernel::rocm::hip::kernel_launch_trampoline(
                    functions.gguf_fused_gate_up_iq3s_nogrid as *mut std::ffi::c_void,
                    (INTERMEDIATE / 32) as u32,
                    TOP_K as u32,
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
            assert_eq!(status, crate::kernel::rocm::hip::HIP_SUCCESS);
        };
        bench("nogrid-m1", &mut || launch_nogrid(&activated_r112));
        // wide-lite 臂：不改布局的宽读版本，要求与现状逐位一致。
        let activated_wide = super::DeviceBuffer::allocate(DEVICE, TOP_K * INTERMEDIATE * 2).unwrap();
        let launch_wide = |activated: &super::DeviceBuffer| {
            let mut input_pointer = input.pointer;
            let mut metas_pointer = metas.buffer.pointer;
            let mut route_ids_pointer = route_ids.pointer;
            let mut activated_pointer = activated.pointer as usize;
            let mut assignments = TOP_K as u32;
            let mut top_k = TOP_K as u32;
            let mut hidden = HIDDEN as u32;
            let mut intermediate = INTERMEDIATE as u32;
            let mut is_bf16 = 1u32;
            let mut arguments = [
                (&mut input_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut metas_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut activated_pointer as *mut usize).cast(),
                (&mut assignments as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut is_bf16 as *mut u32).cast(),
            ];
            let status = unsafe {
                crate::kernel::rocm::hip::kernel_launch_trampoline(
                    functions.gguf_fused_gate_up_iq3s_wide as *mut std::ffi::c_void,
                    (INTERMEDIATE / 32) as u32,
                    TOP_K as u32,
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
            assert_eq!(status, crate::kernel::rocm::hip::HIP_SUCCESS);
        };
        launch_wide(&activated_wide);
        super::super::synchronize_device(DEVICE, "iq3s wide 数值").unwrap();
        let mut wide_host = vec![0_u16; TOP_K * INTERMEDIATE];
        activated_wide.copy_to_host(bytes_mut(&mut wide_host)).unwrap();
        let mut wide_max_abs = 0.0_f32;
        for (left, right) in current_host.iter().zip(&wide_host) {
            wide_max_abs = wide_max_abs.max((bf16::from_bits(*left).to_f32() - bf16::from_bits(*right).to_f32()).abs());
        }
        eprintln!("[iq3s-wave32-probe] wide-lite max_abs={wide_max_abs:.6e}");
        bench("wide-lite", &mut || launch_wide(&activated_wide));
        eprintln!("[iq3s-wave32-probe] max_abs={max_abs:.6e} r112_max_abs={r112_max_abs:.6e}");
        assert!(max_abs <= 1.0e-2, "wave32 探针与现状数值不符: max_abs={max_abs}");
        // r112 与现状逐位一致（纯布局置换+同序归约），要求完全相等。
        assert!(r112_max_abs == 0.0, "r112 重排与现状不逐位一致: {r112_max_abs}");
        assert!(wide_max_abs == 0.0, "wide-lite 与现状不逐位一致: {wide_max_abs}");
    }

    /// fused decode experts 真实形状微基准：IQ3_S gate/up + IQ4_XS down，
    /// hidden=6144、intermediate=2048、top_k=8（GLM-5.3 decode 单层形态）。
    /// `cargo test --release --features with-rocm gguf_fused_decode_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn gguf_fused_decode_bench() {
        use half::bf16;
        const DEVICE: i32 = 0;
        const HIDDEN: usize = 6144;
        const INTERMEDIATE: usize = 2048;
        const TOP_K: usize = 8;
        const EXPERTS: usize = 8;
        super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();
        let input_bits = (0..HIDDEN).map(|index| bf16::from_f32(((index * 13 % 37) as f32 - 18.0) / 128.0).to_bits()).collect::<Vec<_>>();
        let input = super::DeviceBuffer::upload(DEVICE, bytes(&input_bits)).unwrap();
        let mut metas_host = Vec::with_capacity(EXPERTS);
        let mut weight_bytes = 0_usize;
        for expert in 0..EXPERTS {
            let gate_host = iq_rows(21, INTERMEDIATE, HIDDEN, 7 + expert);
            let up_host = iq_rows(21, INTERMEDIATE, HIDDEN, 37 + expert);
            let down_host = iq_rows(23, HIDDEN, INTERMEDIATE, 67 + expert);
            weight_bytes += gate_host.len() + up_host.len() + down_host.len();
            let gate = super::DeviceBuffer::upload(DEVICE, &gate_host).unwrap();
            let up = super::DeviceBuffer::upload(DEVICE, &up_host).unwrap();
            let down = super::DeviceBuffer::upload(DEVICE, &down_host).unwrap();
            metas_host.push(super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: down.device_pointer() as u64, gate_type: 21, up_type: 21, down_type: 23 });
            std::mem::forget((gate, up, down));
        }
        let metas = super::resident_gguf_grouped_metas(DEVICE, &metas_host).unwrap();
        let route_ids_host = [0_u32, 1, 2, 3, 4, 5, 6, 7];
        let route_weights_host = [0.125_f32; TOP_K];
        let route_ids = super::DeviceBuffer::upload(DEVICE, bytes(&route_ids_host)).unwrap();
        let route_weights = super::DeviceBuffer::upload(DEVICE, bytes(&route_weights_host)).unwrap();
        let output = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
        let weight_gib = weight_bytes as f64 / (1u64 << 30) as f64;
        for _ in 0..3 {
            super::try_gguf_fused_decode_experts_impl(DEVICE, &input, 1, HIDDEN, INTERMEDIATE, TOP_K, &route_ids, &route_weights, TOP_K, &metas.buffer, &output, Some([21, 21, 23])).unwrap();
        }
        let started = std::time::Instant::now();
        const ROUNDS: usize = 100;
        for _ in 0..ROUNDS {
            super::try_gguf_fused_decode_experts_impl(DEVICE, &input, 1, HIDDEN, INTERMEDIATE, TOP_K, &route_ids, &route_weights, TOP_K, &metas.buffer, &output, Some([21, 21, 23])).unwrap();
        }
        super::super::synchronize_device(DEVICE, "gguf fused decode bench").unwrap();
        let micros = started.elapsed().as_micros() as f64 / ROUNDS as f64;
        eprintln!("[gguf-fused-bench] top_k={TOP_K} hidden={HIDDEN} inter={INTERMEDIATE} avg_us={micros:.1} bw_GBps={:.0}", weight_gib * 1024.0 / (micros / 1e6));

        // perm 码本臂：down IQ4_XS 直发 A/B（位级一致 + 计时）。
        // down 权重字节 = 8 expert × hidden×intermediate IQ4_XS（136B/256 权重块）。
        let activated_host = (0..TOP_K * INTERMEDIATE).map(|index| bf16::from_f32(((index * 11 % 29) as f32 - 14.0) / 64.0).to_bits()).collect::<Vec<_>>();
        let activated = super::DeviceBuffer::upload(DEVICE, bytes(&activated_host)).unwrap();
        let out_ref = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
        let out_perm = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
        let functions = super::super::ct_quantized_functions(DEVICE).unwrap();
        let module_launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
        let down_gib = (EXPERTS * HIDDEN * (INTERMEDIATE / 256) * 136) as f64 / (1u64 << 30) as f64;
        let mut run_down = |kernel: usize, out: &super::DeviceBuffer, label: &str| {
            let mut activated_pointer = activated.pointer as usize;
            let mut metas_pointer = metas.buffer.pointer;
            let mut route_ids_pointer = route_ids.pointer;
            let mut route_weights_pointer = route_weights.pointer;
            let mut output_pointer = out.pointer as usize;
            let mut tokens = 1_u32;
            let mut top_k_u32 = TOP_K as u32;
            let mut hidden_u32 = HIDDEN as u32;
            let mut intermediate_u32 = INTERMEDIATE as u32;
            let mut arguments = [
                (&mut activated_pointer as *mut usize).cast(),
                (&mut metas_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_weights_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut output_pointer as *mut usize).cast(),
                (&mut tokens as *mut u32).cast(),
                (&mut top_k_u32 as *mut u32).cast(),
                (&mut hidden_u32 as *mut u32).cast(),
                (&mut intermediate_u32 as *mut u32).cast(),
            ];
            let grid_x = (HIDDEN / 32) as u32;
            for _ in 0..3 {
                unsafe { module_launch(kernel as *mut std::ffi::c_void, grid_x, 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), std::ptr::null_mut()) };
            }
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                for _ in 0..ROUNDS {
                    unsafe { module_launch(kernel as *mut std::ffi::c_void, grid_x, 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), std::ptr::null_mut()) };
                }
                super::super::synchronize_device(DEVICE, "down perm bench").unwrap();
                best = best.min(started.elapsed().as_micros() as f64 / ROUNDS as f64);
            }
            eprintln!("[gguf-fused-bench] down-{label} min_us={best:.1} bw_GBps={:.0}", down_gib * 1024.0 / (best / 1e6));
        };
        run_down(functions.gguf_fused_down_iq4xs, &out_ref, "const");
        run_down(functions.gguf_fused_down_iq4xs_perm, &out_perm, "perm");
        let mut ref_host = vec![0.0f32; HIDDEN];
        let mut perm_host = vec![0.0f32; HIDDEN];
        out_ref.copy_to_host(bytes_mut(&mut ref_host)).unwrap();
        out_perm.copy_to_host(bytes_mut(&mut perm_host)).unwrap();
        let diff = ref_host.iter().zip(&perm_host).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("[gguf-fused-bench] down-perm bit-diff={diff}/{HIDDEN}");
        assert!(diff == 0, "perm 码本应与常量表逐位一致");
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
        let mut actual = vec![0.0_f32; tokens * HIDDEN];
        let mut first_bits = None;
        let rounds = if top_k > 1 { 8 } else { 1 };
        for round in 0..rounds {
            let output = super::try_gguf_grouped_wmma_experts(DEVICE, &input, tokens, HIDDEN, INTERMEDIATE, top_k, &route_ids, &route_weights, tokens * top_k, &metas, expert_count).unwrap();
            output.copy_to_host(bytes_mut(&mut actual)).unwrap();
            let bits = actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            if let Some(first_bits) = &first_bits {
                assert_eq!(&bits, first_bits, "GGUF grouped down 第 {round} 轮结果非确定");
            } else {
                first_bits = Some(bits);
            }
        }

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

        // 8 个 expert 同时贡献同一 token，覆盖旧浮点 atomicAdd 的竞争形态。
        let routed_ids = (0..17).flat_map(|token| (0..8).map(move |slot| ((token * 3 + slot * 5) % 8) as u32)).collect::<Vec<_>>();
        let routed_weights = (0..17).flat_map(|_| (1..=8).map(|slot| slot as f32 / 64.0)).collect::<Vec<_>>();
        assert_grouped_wmma_matches_cpu(17, 8, 8, &routed_ids, &routed_weights);

        // 超过一个 128-row tile，覆盖 shared expert 的多 y block 路径。
        let shared_ids = vec![0_u32; 129];
        let shared_weights = vec![1.0_f32; 129];
        assert_grouped_wmma_matches_cpu(129, 1, 1, &shared_ids, &shared_weights);
    }

    fn assert_rows2_gate_up_matches_token_major(tensor_type: u32) {
        use half::bf16;

        const DEVICE: i32 = 0;
        const HIDDEN: usize = 256;
        const INTERMEDIATE: usize = 256;
        const TOP_K: usize = 4;
        const EXPERTS: usize = 6;
        let input_bits = (0..2 * HIDDEN).map(|index| bf16::from_f32(((index * 17 % 47) as f32 - 23.0) / 128.0).to_bits()).collect::<Vec<_>>();
        let input = super::DeviceBuffer::upload(DEVICE, bytes(&input_bits)).unwrap();
        let mut metas_host = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let gate = super::DeviceBuffer::upload(DEVICE, &iq_rows(tensor_type, INTERMEDIATE, HIDDEN, 7 + expert)).unwrap();
            let up = super::DeviceBuffer::upload(DEVICE, &iq_rows(tensor_type, INTERMEDIATE, HIDDEN, 37 + expert)).unwrap();
            metas_host.push(super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: gate.device_pointer() as u64, gate_type: tensor_type, up_type: tensor_type, down_type: tensor_type });
            std::mem::forget((gate, up));
        }
        let metas = super::resident_gguf_grouped_metas(DEVICE, &metas_host).unwrap();
        let route_ids = super::DeviceBuffer::upload(DEVICE, bytes(&[0_u32, 1, 2, 3, 4, 1, 5, 3])).unwrap();
        let token_major = super::DeviceBuffer::allocate(DEVICE, 2 * TOP_K * INTERMEDIATE * 2).unwrap();
        let shared = super::DeviceBuffer::allocate(DEVICE, 2 * TOP_K * INTERMEDIATE * 2).unwrap();
        let functions = super::super::ct_quantized_functions(DEVICE).unwrap();
        let (token_major_function, shared_function) =
            if tensor_type == 21 { (functions.gguf_fused_gate_up_iq3s_wide, functions.gguf_fused_gate_up_iq3s_wide_rows2) } else { (functions.gguf_fused_gate_up_iq4xs, functions.gguf_fused_gate_up_iq4xs_rows2) };
        let launch = |function: usize, output: &super::DeviceBuffer| {
            let mut input_pointer = input.pointer;
            let mut metas_pointer = metas.buffer.pointer;
            let mut route_ids_pointer = route_ids.pointer;
            let mut output_pointer = output.pointer as usize;
            let mut assignments = (2 * TOP_K) as u32;
            let mut top_k = TOP_K as u32;
            let mut hidden = HIDDEN as u32;
            let mut intermediate = INTERMEDIATE as u32;
            let mut input_is_bf16 = 1_u32;
            let mut arguments = [
                (&mut input_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut metas_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut output_pointer as *mut usize).cast(),
                (&mut assignments as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut input_is_bf16 as *mut u32).cast(),
            ];
            let status = unsafe {
                crate::kernel::rocm::hip::kernel_launch_trampoline(
                    function as *mut std::ffi::c_void,
                    (INTERMEDIATE / 32) as u32,
                    (2 * TOP_K) as u32,
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
            assert_eq!(status, crate::kernel::rocm::hip::HIP_SUCCESS);
        };
        launch(token_major_function, &token_major);
        launch(shared_function, &shared);
        super::super::synchronize_device(DEVICE, "GGUF rows2 gate/up 共享 oracle").unwrap();
        let mut token_major_host = vec![0_u16; 2 * TOP_K * INTERMEDIATE];
        let mut shared_host = vec![0_u16; 2 * TOP_K * INTERMEDIATE];
        token_major.copy_to_host(bytes_mut(&mut token_major_host)).unwrap();
        shared.copy_to_host(bytes_mut(&mut shared_host)).unwrap();
        assert_eq!(shared_host, token_major_host, "tensor_type={tensor_type} rows2 共享权重改变 activation");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_rows2_gate_up_weight_sharing_matches_token_major() {
        super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();
        assert_rows2_gate_up_matches_token_major(21);
        assert_rows2_gate_up_matches_token_major(23);
    }

    fn assert_rowsn_gate_up_matches_token_major(tensor_type: u32) {
        use half::bf16;

        const DEVICE: i32 = 0;
        const HIDDEN: usize = 256;
        const INTERMEDIATE: usize = 256;
        const TOP_K: usize = 4;
        const ROWS: usize = 6;
        const EXPERTS: usize = 6;
        let input_bits = (0..ROWS * HIDDEN).map(|index| bf16::from_f32(((index * 17 % 47) as f32 - 23.0) / 128.0).to_bits()).collect::<Vec<_>>();
        let input = super::DeviceBuffer::upload(DEVICE, bytes(&input_bits)).unwrap();
        let mut metas_host = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let gate = super::DeviceBuffer::upload(DEVICE, &iq_rows(tensor_type, INTERMEDIATE, HIDDEN, 7 + expert)).unwrap();
            let up = super::DeviceBuffer::upload(DEVICE, &iq_rows(tensor_type, INTERMEDIATE, HIDDEN, 37 + expert)).unwrap();
            metas_host.push(super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: gate.device_pointer() as u64, gate_type: tensor_type, up_type: tensor_type, down_type: tensor_type });
            std::mem::forget((gate, up));
        }
        let metas = super::resident_gguf_grouped_metas(DEVICE, &metas_host).unwrap();
        // 每行 4 个互异专家,跨行刻意重叠:expert 0/1/3 各成 4 成员组,4/5 各 3 成员。
        let routes: [u32; ROWS * TOP_K] = [
            0, 1, 2, 3, //
            4, 1, 5, 3, //
            0, 2, 4, 1, //
            3, 5, 0, 2, //
            1, 4, 3, 0, //
            2, 5, 1, 4, //
        ];
        let route_ids = super::DeviceBuffer::upload(DEVICE, bytes(&routes)).unwrap();
        let token_major = super::DeviceBuffer::allocate(DEVICE, ROWS * TOP_K * INTERMEDIATE * 2).unwrap();
        let shared = super::DeviceBuffer::allocate(DEVICE, ROWS * TOP_K * INTERMEDIATE * 2).unwrap();
        let functions = super::super::ct_quantized_functions(DEVICE).unwrap();
        let (token_major_function, shared_function) =
            if tensor_type == 21 { (functions.gguf_fused_gate_up_iq3s_wide, functions.gguf_fused_gate_up_iq3s_wide_rowsn) } else { (functions.gguf_fused_gate_up_iq4xs, functions.gguf_fused_gate_up_iq4xs_rowsn) };
        let launch = |function: usize, output: &super::DeviceBuffer| {
            let mut input_pointer = input.pointer;
            let mut metas_pointer = metas.buffer.pointer;
            let mut route_ids_pointer = route_ids.pointer;
            let mut output_pointer = output.pointer as usize;
            let mut assignments = (ROWS * TOP_K) as u32;
            let mut top_k = TOP_K as u32;
            let mut hidden = HIDDEN as u32;
            let mut intermediate = INTERMEDIATE as u32;
            let mut input_is_bf16 = 1_u32;
            let mut arguments = [
                (&mut input_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut metas_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut route_ids_pointer as *mut *mut std::ffi::c_void).cast(),
                (&mut output_pointer as *mut usize).cast(),
                (&mut assignments as *mut u32).cast(),
                (&mut top_k as *mut u32).cast(),
                (&mut hidden as *mut u32).cast(),
                (&mut intermediate as *mut u32).cast(),
                (&mut input_is_bf16 as *mut u32).cast(),
            ];
            let status = unsafe {
                crate::kernel::rocm::hip::kernel_launch_trampoline(
                    function as *mut std::ffi::c_void,
                    (INTERMEDIATE / 32) as u32,
                    (ROWS * TOP_K) as u32,
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
            assert_eq!(status, crate::kernel::rocm::hip::HIP_SUCCESS);
        };
        launch(token_major_function, &token_major);
        launch(shared_function, &shared);
        super::super::synchronize_device(DEVICE, "GGUF rowsN gate/up 共享 oracle").unwrap();
        let mut token_major_host = vec![0_u16; ROWS * TOP_K * INTERMEDIATE];
        let mut shared_host = vec![0_u16; ROWS * TOP_K * INTERMEDIATE];
        token_major.copy_to_host(bytes_mut(&mut token_major_host)).unwrap();
        shared.copy_to_host(bytes_mut(&mut shared_host)).unwrap();
        assert_eq!(shared_host, token_major_host, "tensor_type={tensor_type} rowsN 共享权重改变 activation");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_rowsn_gate_up_weight_sharing_matches_token_major() {
        super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();
        assert_rowsn_gate_up_matches_token_major(21);
        assert_rowsn_gate_up_matches_token_major(23);
    }

    fn assert_fused_decode_matches_generic(types: [u32; 3], bitwise: bool) {
        use half::bf16;

        const DEVICE: i32 = 0;
        const HIDDEN: usize = 256;
        const INTERMEDIATE: usize = 256;
        let input_bits = (0..HIDDEN).map(|index| bf16::from_f32(((index * 13 % 37) as f32 - 18.0) / 128.0).to_bits()).collect::<Vec<_>>();
        let gate_host = iq_rows(types[0], INTERMEDIATE, HIDDEN, 7);
        let up_host = iq_rows(types[1], INTERMEDIATE, HIDDEN, 11);
        let down_host = iq_rows(types[2], HIDDEN, INTERMEDIATE, 13);
        let input_values = input_bits.iter().map(|&bits| bf16::from_bits(bits).to_f32()).collect::<Vec<_>>();
        let gate_values = crate::weight::codec::ggml::dequantize(types[0], &gate_host, INTERMEDIATE * HIDDEN).unwrap();
        let up_values = crate::weight::codec::ggml::dequantize(types[1], &up_host, INTERMEDIATE * HIDDEN).unwrap();
        let down_values = crate::weight::codec::ggml::dequantize(types[2], &down_host, HIDDEN * INTERMEDIATE).unwrap();
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
        let mut activated = vec![0.0_f32; INTERMEDIATE];
        for row in 0..INTERMEDIATE {
            let gate = gate_values[row * HIDDEN..(row + 1) * HIDDEN].iter().zip(&input_values).map(|(weight, input)| bf16::from_f32(*weight).to_f32() * input).sum::<f32>();
            let up = up_values[row * HIDDEN..(row + 1) * HIDDEN].iter().zip(&input_values).map(|(weight, input)| bf16::from_f32(*weight).to_f32() * input).sum::<f32>();
            activated[row] = bf16::from_f32(gate / (1.0 + (-gate).exp()) * up).to_f32();
        }
        let expected = (0..HIDDEN).map(|row| down_values[row * INTERMEDIATE..(row + 1) * INTERMEDIATE].iter().zip(&activated).map(|(weight, value)| bf16::from_f32(*weight).to_f32() * value).sum::<f32>() * 0.75).collect::<Vec<_>>();
        let mut cpu_max_abs = 0.0_f32;
        for (index, (&generic, &packed)) in generic_host.iter().zip(&packed_host).enumerate() {
            assert!(generic.is_finite() && packed.is_finite(), "index={index} generic={generic} packed={packed}");
            if bitwise {
                assert_eq!(packed.to_bits(), generic.to_bits(), "types={types:?} index={index} 专用 kernel 改变结果位模式");
            }
            let tolerance = 0.02 * generic.abs().max(1.0);
            assert!((generic - packed).abs() <= tolerance, "index={index} generic={generic} packed={packed} tolerance={tolerance}");
            let cpu_difference = (packed - expected[index]).abs();
            cpu_max_abs = cpu_max_abs.max(cpu_difference);
            let cpu_tolerance = 0.03 * expected[index].abs().max(1.0);
            assert!(cpu_difference <= cpu_tolerance, "types={types:?} index={index} packed={packed} expected={} difference={cpu_difference} tolerance={cpu_tolerance}", expected[index]);
        }
        eprintln!("[glm53-fused-decode-oracle] types={types:?} cpu_max_abs={cpu_max_abs:.6e}");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_iq4xs_fused_decode_matches_generic() {
        assert_fused_decode_matches_generic([23, 23, 23], false);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_iq3s_iq4xs_fused_decode_matches_generic() {
        assert_fused_decode_matches_generic([21, 21, 23], false);
    }

    #[test]
    fn q8_single_expert_rows2_matches_single_rows() {
        use half::bf16;
        const DEVICE: i32 = 0;
        const HIDDEN: usize = 512;
        for intermediate in [256, 1024, 2048] {
            let bits = (0..2 * HIDDEN).map(|i| bf16::from_f32(((i * 17 % 47) as f32 - 23.0) / 128.0).to_bits()).collect::<Vec<_>>();
            let gate = super::DeviceBuffer::upload(DEVICE, &iq_rows(8, intermediate, HIDDEN, 7)).unwrap();
            let up = super::DeviceBuffer::upload(DEVICE, &iq_rows(8, intermediate, HIDDEN, 37)).unwrap();
            let down = super::DeviceBuffer::upload(DEVICE, &iq_rows(8, HIDDEN, intermediate, 67)).unwrap();
            let meta = super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: down.device_pointer() as u64, gate_type: 8, up_type: 8, down_type: 8 };
            let metas = super::resident_gguf_grouped_metas(DEVICE, &[meta]).unwrap();
            let input = super::DeviceBuffer::upload(DEVICE, bytes(&bits)).unwrap();
            let ids = super::DeviceBuffer::upload(DEVICE, bytes(&[0_u32, 0])).unwrap();
            let weights = super::DeviceBuffer::upload(DEVICE, bytes(&[0.37_f32, 0.81])).unwrap();
            let output = super::DeviceBuffer::allocate(DEVICE, 2 * HIDDEN * 4).unwrap();
            super::try_gguf_fused_decode_experts(DEVICE, &input, 2, HIDDEN, intermediate, 1, &ids, &weights, 2, &metas, &output).unwrap();
            let mut actual = vec![0.0_f32; 2 * HIDDEN];
            output.copy_to_host(bytes_mut(&mut actual)).unwrap();
            for row in 0..2 {
                let input = super::DeviceBuffer::upload(DEVICE, bytes(&bits[row * HIDDEN..(row + 1) * HIDDEN])).unwrap();
                let ids = super::DeviceBuffer::upload(DEVICE, bytes(&[0_u32])).unwrap();
                let weights = super::DeviceBuffer::upload(DEVICE, bytes(&[if row == 0 { 0.37_f32 } else { 0.81 }])).unwrap();
                let output = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
                super::try_gguf_fused_decode_experts(DEVICE, &input, 1, HIDDEN, intermediate, 1, &ids, &weights, 1, &metas, &output).unwrap();
                let mut expected = vec![0.0_f32; HIDDEN];
                output.copy_to_host(bytes_mut(&mut expected)).unwrap();
                for (col, value) in expected.iter().enumerate() {
                    assert!(value.is_finite());
                    assert_eq!(actual[row * HIDDEN + col].to_bits(), value.to_bits(), "intermediate={intermediate} row={row} col={col}");
                }
            }
        }
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_q8_0_fused_decode_matches_generic() {
        assert_fused_decode_matches_generic([8, 8, 8], false);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_iq4xs_q5k_fused_decode_matches_generic() {
        assert_fused_decode_matches_generic([23, 23, 13], false);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_iq3s_q6k_fused_decode_matches_generic() {
        assert_fused_decode_matches_generic([21, 21, 14], false);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn glm53_q3k_q4k_fused_decode_matches_generic_bits() {
        assert_fused_decode_matches_generic([11, 11, 12], true);
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
    try_gguf_fused_decode_experts_with_workspace(device_id, input, input_rows, hidden_size, intermediate_size, top_k, route_ids, route_weights, route_count, metas, output, uniform_types, None)
}

#[allow(clippy::too_many_arguments)]
fn try_gguf_fused_decode_experts_with_workspace(
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
    activated_workspace: Option<&DeviceBuffer>,
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
    let q3_k_gate_up = input_is_bf16 && matches!(uniform_types, Some([11, 11, _]));
    let q8_0_gate_up = input_is_bf16 && matches!(uniform_types, Some([8, 8, _]));
    let iq4xs_down = intermediate_size.is_multiple_of(256) && matches!(uniform_types, Some([_, _, 23]));
    let q8_0_down = intermediate_size.is_multiple_of(256) && matches!(uniform_types, Some([_, _, 8]));
    // 只有一个 metadata 条目时，两行才确定使用同一专家；routed top-1 不满足此前提。
    let single_expert_rows2 = input_rows == 2 && top_k == 1 && metas.bytes() == std::mem::size_of::<GgufGroupedExpertMeta>();
    let q4_k_down = intermediate_size.is_multiple_of(256) && matches!(uniform_types, Some([_, _, 12]));
    let q5_k_down = intermediate_size.is_multiple_of(256) && matches!(uniform_types, Some([_, _, 13]));
    let q6_k_down = intermediate_size.is_multiple_of(256) && matches!(uniform_types, Some([_, _, 14]));
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
    let activated_bytes = route_count.checked_mul(intermediate_size).and_then(|n| n.checked_mul(2)).ok_or("GGUF fused activated 溢出")?;
    let activated_pointer = if let Some(workspace) = activated_workspace {
        if workspace.device_id() != device_id || workspace.bytes() < activated_bytes {
            return Err(format!("GGUF fused activated workspace 不匹配: device={}/{} bytes={}/{}", workspace.device_id(), device_id, workspace.bytes(), activated_bytes,));
        }
        workspace.device_pointer()
    } else {
        // eager 路径继续复用按 device+stream 的常驻 workspace；graph 必须由
        // 调用方持有私有 buffer，否则后续多行 prefill 扩容会使固化指针悬垂。
        GGUF_FUSED_WORKSPACES.with(|workspaces| {
            let mut workspaces = workspaces.borrow_mut();
            let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
            if workspace.activated.as_ref().is_none_or(|buffer| buffer.bytes() < activated_bytes) {
                workspace.activated = Some(DeviceBuffer::allocate(device_id, activated_bytes)?);
            }
            Ok::<usize, String>(workspace.activated.as_ref().expect("GGUF fused activated 已初始化").device_pointer())
        })?
    };

    let mut assignments = u32::try_from(route_count).map_err(|_| "GGUF fused route 数超过 u32".to_owned())?;
    let mut tokens = u32::try_from(input_rows).map_err(|_| "GGUF fused tokens 超过 u32".to_owned())?;
    let mut top_k = u32::try_from(top_k).map_err(|_| "GGUF fused top_k 超过 u32".to_owned())?;
    let mut hidden = u32::try_from(hidden_size).map_err(|_| "GGUF fused hidden 超过 u32".to_owned())?;
    let mut intermediate = u32::try_from(intermediate_size).map_err(|_| "GGUF fused intermediate 超过 u32".to_owned())?;
    let mut is_bf16 = u32::from(input_is_bf16);
    let gate_started = options().kernel_profile.then(std::time::Instant::now);
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
        let outputs_per_block = if iq3s_gate_up || iq4xs_gate_up || q3_k_gate_up { 32 } else { 16 };
        let grid_x = u32::try_from(intermediate_size.div_ceil(outputs_per_block)).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let grid_y = u32::try_from(route_count).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let function = if iq3s_gate_up {
            // wide-lite：同一布局的 u16 宽读版，逐位一致（iq3s_wave32_probe_bench 位级验证），
            // 生产实测 gate_up -8%。原 kernel 保留在模块内作对照。
            if input_rows == 2 {
                functions.gguf_fused_gate_up_iq3s_wide_rows2
            } else if input_rows > 2 {
                // MTP verify 多行:同专家 assignment 由 leader 块一次读权重逐成员复用,
                // 与 token-major 输出逐位一致(oracle 断言)。
                functions.gguf_fused_gate_up_iq3s_wide_rowsn
            } else {
                functions.gguf_fused_gate_up_iq3s_wide
            }
        } else if iq4xs_gate_up {
            if input_rows == 2 {
                functions.gguf_fused_gate_up_iq4xs_rows2
            } else if input_rows > 2 {
                functions.gguf_fused_gate_up_iq4xs_rowsn
            } else {
                functions.gguf_fused_gate_up_iq4xs
            }
        } else if q3_k_gate_up {
            functions.gguf_fused_gate_up_q3_k
        } else if q8_0_gate_up && single_expert_rows2 {
            functions.gguf_fused_gate_up_q8_0_rows2
        } else if q8_0_gate_up {
            functions.gguf_fused_gate_up_q8_0
        } else {
            functions.gguf_fused_gate_up
        };
        let status = unsafe {
            module_launch(function as *mut c_void, grid_x, if q8_0_gate_up && single_expert_rows2 { 1 } else { grid_y }, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut())
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF fused gate_up"));
        }
    }
    if let Some(started) = gate_started {
        synchronize_device(device_id, "GGUF fused gate/up profile")?;
        eprintln!("[rocm-kernel] gguf-fused-gate-up device={device_id} input_rows={input_rows} routes={route_count} top_k={top_k} types={uniform_types:?} wall={:.6}s", started.elapsed().as_secs_f64(),);
        if input_rows > 1 && top_k > 1 {
            let mut routes = vec![0_u32; route_count];
            route_ids.copy_to_host(unsafe { std::slice::from_raw_parts_mut(routes.as_mut_ptr().cast(), route_count * std::mem::size_of::<u32>()) })?;
            let top_k = top_k as usize;
            let overlap = routes[..top_k].iter().filter(|expert| routes[top_k..].contains(expert)).count();
            let unique = routes.iter().enumerate().filter(|(index, expert)| !routes[..*index].contains(expert)).count();
            eprintln!("[rocm-kernel] gguf-route-overlap device={device_id} rows={input_rows} types={uniform_types:?} overlap={overlap} unique={unique} routes={routes:?}");
        }
    }
    let down_started = options().kernel_profile.then(std::time::Instant::now);
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
        let outputs_per_block = if iq4xs_down || q4_k_down { 32 } else { 16 };
        let grid_x = u32::try_from(hidden_size.div_ceil(outputs_per_block)).map_err(|_| "GGUF fused grid 超过 u32".to_owned())?;
        let function = if iq4xs_down {
            functions.gguf_fused_down_iq4xs
        } else if q8_0_down && single_expert_rows2 {
            functions.gguf_fused_down_q8_0_rows2
        } else if q8_0_down {
            functions.gguf_fused_down_q8_0
        } else if q4_k_down {
            functions.gguf_fused_down_q4_k
        } else if q5_k_down {
            functions.gguf_fused_down_q5_k
        } else if q6_k_down {
            functions.gguf_fused_down_q6_k
        } else {
            functions.gguf_fused_down
        };
        let status =
            unsafe { module_launch(function as *mut c_void, grid_x, if q8_0_down && single_expert_rows2 { 1 } else { tokens }, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF fused down"));
        }
    }
    if let Some(started) = down_started {
        synchronize_device(device_id, "GGUF fused down profile")?;
        eprintln!("[rocm-kernel] gguf-fused-down device={device_id} input_rows={input_rows} routes={route_count} top_k={top_k} types={uniform_types:?} wall={:.6}s", started.elapsed().as_secs_f64(),);
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

/// 带量化类型表的变体：graph 录制等场景由调用方持有 output 固定地址。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gguf_fused_decode_experts_typed(
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
    try_gguf_fused_decode_experts_impl(device_id, input, input_rows, hidden_size, intermediate_size, top_k, route_ids, route_weights, route_count, metas, output, uniform_types)
}

/// Graph 录制专用：activated 地址必须由 graph 本身持有，不能引用 eager 的
/// device+stream workspace；后者会在下一次更大 prefill 时扩容并释放旧地址。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gguf_fused_decode_experts_typed_with_workspace(
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
    activated_workspace: &DeviceBuffer,
) -> Result<(), String> {
    try_gguf_fused_decode_experts_with_workspace(device_id, input, input_rows, hidden_size, intermediate_size, top_k, route_ids, route_weights, route_count, metas, output, uniform_types, Some(activated_workspace))
}

/// IQ3_S 110B 块内重排为 112B：[d f16][pad 2B][scales 4B][qs 64B][qh 8B][signs 32B]。
/// 纯字节置换（数值逐位不变），让 decode kernel 的 qs（8B 对齐）与 signs
///（4B 对齐）可以宽加载，替代逐字节标量读。decode 专用；prefill 仍用原始布局。
pub(crate) fn repack_iq3s_r112(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if cols % 256 != 0 || rows == 0 {
        return Err(format!("IQ3_S r112 重排 shape 非法: rows={rows} cols={cols}"));
    }
    let blocks_per_row = cols / 256;
    if bytes.len() != rows * blocks_per_row * 110 {
        return Err(format!("IQ3_S r112 重排字节 {}，期望 {}", bytes.len(), rows * blocks_per_row * 110));
    }
    let mut out = vec![0_u8; rows * blocks_per_row * 112];
    for block in 0..rows * blocks_per_row {
        let source = &bytes[block * 110..(block + 1) * 110];
        let target = &mut out[block * 112..(block + 1) * 112];
        target[0..2].copy_from_slice(&source[0..2]);
        target[4..8].copy_from_slice(&source[106..110]);
        target[8..72].copy_from_slice(&source[2..66]);
        target[72..80].copy_from_slice(&source[66..74]);
        target[80..112].copy_from_slice(&source[74..106]);
    }
    Ok(out)
}

/// wave32 IQ3_S gate_up 探针（结构对照实验用）：与 fused 路径同参数、同
/// activated 输出，仅 kernel 结构不同。仅 bench/oracle 调用。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gguf_fused_gate_up_iq3s_wave32_probe(
    device_id: i32,
    input: &DeviceBuffer,
    route_count: usize,
    hidden_size: usize,
    intermediate_size: usize,
    top_k: usize,
    route_ids: &DeviceBuffer,
    metas: &DeviceBuffer,
    activated: &DeviceBuffer,
) -> Result<(), String> {
    if route_count == 0 || route_count > 64 || hidden_size % 256 != 0 || intermediate_size % 8 != 0 {
        return Err("GGUF wave32 probe 参数非法".to_owned());
    }
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let functions = ct_quantized_functions(device_id)?;
    let mut assignments = u32::try_from(route_count).map_err(|_| "GGUF probe route 数超过 u32".to_owned())?;
    let mut top_k_u32 = u32::try_from(top_k).map_err(|_| "GGUF probe top_k 超过 u32".to_owned())?;
    let mut hidden = u32::try_from(hidden_size).map_err(|_| "GGUF probe hidden 超过 u32".to_owned())?;
    let mut intermediate = u32::try_from(intermediate_size).map_err(|_| "GGUF probe intermediate 超过 u32".to_owned())?;
    let mut is_bf16 = 1u32;
    let mut input_pointer = input.pointer;
    let mut metas_pointer = metas.pointer;
    let mut route_ids_pointer = route_ids.pointer;
    let mut activated_pointer = activated.pointer as usize;
    let mut arguments = [
        (&mut input_pointer as *mut *mut c_void).cast(),
        (&mut metas_pointer as *mut *mut c_void).cast(),
        (&mut route_ids_pointer as *mut *mut c_void).cast(),
        (&mut route_ids_pointer as *mut *mut c_void).cast(),
        (&mut activated_pointer as *mut usize).cast(),
        (&mut assignments as *mut u32).cast(),
        (&mut top_k_u32 as *mut u32).cast(),
        (&mut hidden as *mut u32).cast(),
        (&mut intermediate as *mut u32).cast(),
        (&mut is_bf16 as *mut u32).cast(),
    ];
    let grid_x = u32::try_from(intermediate_size / 8).map_err(|_| "GGUF probe grid 超过 u32".to_owned())?;
    let grid_y = u32::try_from(route_count).map_err(|_| "GGUF probe grid 超过 u32".to_owned())?;
    let status = unsafe {
        crate::kernel::rocm::hip::kernel_launch_trampoline(functions.gguf_fused_gate_up_iq3s_wave32 as *mut c_void, grid_x, grid_y, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut())
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel GGUF wave32 probe"));
    }
    Ok(())
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

#[cfg(test)]
mod graph_replay_probe {
    use super::tests::{bytes, iq_rows};

    /// graph replay 死循环最小复现：录制 [fused decode + add] graph，replay+sync。
    /// `cargo test --release --features with-rocm gguf_fused_graph_replay_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn gguf_fused_graph_replay_probe() {
        use half::bf16;
        const DEVICE: i32 = 7;
        const HIDDEN: usize = 6144;
        const INTERMEDIATE: usize = 2048;
        super::super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();
        let input_bits = (0..HIDDEN).map(|index| bf16::from_f32(((index * 13 % 37) as f32 - 18.0) / 128.0).to_bits()).collect::<Vec<_>>();
        let expert_input = super::DeviceBuffer::upload(DEVICE, bytes(&input_bits)).unwrap();
        let gate_host = iq_rows(21, INTERMEDIATE, HIDDEN, 7);
        let up_host = iq_rows(21, INTERMEDIATE, HIDDEN, 37);
        let down_host = iq_rows(23, HIDDEN, INTERMEDIATE, 67);
        let gate = super::DeviceBuffer::upload(DEVICE, &gate_host).unwrap();
        let up = super::DeviceBuffer::upload(DEVICE, &up_host).unwrap();
        let down = super::DeviceBuffer::upload(DEVICE, &down_host).unwrap();
        let meta = super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: down.device_pointer() as u64, gate_type: 21, up_type: 21, down_type: 23 };
        let metas = super::resident_gguf_grouped_metas(DEVICE, &[meta]).unwrap();
        let (route_ids, route_weights) = super::super::ct_grouped::cooperative_single_expert_route(DEVICE, 1).unwrap();
        let routed_out = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
        let shared_out = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
        let combined = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
        let stream = super::super::active_compute_stream();
        eprintln!("[probe] 开始录制");
        let graph = {
            let recorder = super::super::StaticGraphRecorder::begin(DEVICE, stream).unwrap();
            super::try_gguf_fused_decode_experts_typed(DEVICE, &expert_input, 1, HIDDEN, INTERMEDIATE, 1, &route_ids, &route_weights, 1, &metas.buffer, &routed_out, Some([21, 21, 23])).unwrap();
            super::try_gguf_fused_decode_experts_typed(DEVICE, &expert_input, 1, HIDDEN, INTERMEDIATE, 1, &route_ids, &route_weights, 1, &metas.buffer, &shared_out, Some([21, 21, 23])).unwrap();
            super::super::try_add_resident_f32_into(DEVICE, &routed_out, &shared_out, &combined, HIDDEN, 1.0).unwrap();
            recorder.finish().unwrap().expect("graph 非空")
        };
        eprintln!("[probe] nodes={}，replay 第一次", graph.node_count());
        graph.launch(DEVICE, stream).unwrap();
        super::super::synchronize_device(DEVICE, "probe replay 1").unwrap();
        eprintln!("[probe] replay 1 完成");
        for _ in 0..50 {
            graph.launch(DEVICE, stream).unwrap();
        }
        super::super::synchronize_device(DEVICE, "probe replay 50").unwrap();
        eprintln!("[probe] 51 次 replay 全部完成");
    }

    /// 非 Indexer 层整层 graph 探针：[rmsnorm→dual(q_a/kv_a)→rmsnorm→q_b→rope(间接)
    /// →kv_norm→append(间接)→MLA into→o_proj→add→ffn_norm→route→fused×2→add]×residual
    /// 整链录制一次、replay 与 eager 同输入逐步对照（replay 保真）+ 计时（间隙收益）。
    /// 注意：probe 用 top_k=512 使 latent 预重排 gather 关闭——gather 的 pool 分配在
    /// graph 内是悬垂源，生产接入时必须先把 gather 工作区归 graph 持有。
    /// `cargo test --release --features with-rocm layer_graph_replay_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn layer_graph_replay_probe() {
        use half::{bf16, f16};
        const DEVICE: i32 = 0;
        const HIDDEN: usize = 6144;
        const Q_LORA: usize = 2048;
        const Q_PROJ: usize = 16384;
        const HEADS: usize = 64;
        const KV_LORA: usize = 512;
        const ROPE_DIM: usize = 64;
        const KV_A: usize = KV_LORA + ROPE_DIM;
        const KV_HEAD: usize = 448;
        const INTER: usize = 2048;
        const MOE_TOP_K: usize = 8;
        const EXPERTS: usize = 8;
        const DSA_TOP_K: usize = 512;
        const CONTEXT: usize = 4096;
        const BLOCK: usize = 128;
        const EPS: f32 = 1.0e-5;
        super::super::super::configure(crate::kernel::rocm::hip::RocmOptions::default()).unwrap();

        // ---- 合成权重（形状真实、数值确定）----
        let f32_buf = |values: &[f32]| super::DeviceBuffer::upload(DEVICE, unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }).unwrap();
        let norm_w = |n: usize| f32_buf(&(0..n).map(|i| 1.0 + (i % 7) as f32 * 0.001).collect::<Vec<_>>());
        let w8 = |rows: usize, cols: usize, seed: usize| {
            let packed: Vec<u8> = (0..rows * cols).map(|i| (((i * 31 + seed * 7) % 17) as i32 + 120) as u8).collect();
            let scales: Vec<u16> = vec![f16::from_f32(0.015625).to_bits(); rows * (cols / 32)];
            (super::DeviceBuffer::upload(DEVICE, &packed).unwrap(), super::DeviceBuffer::upload(DEVICE, bytes(&scales)).unwrap())
        };
        let input_norm = norm_w(HIDDEN);
        let (q_a_packed, q_a_scales) = w8(Q_LORA, HIDDEN, 1);
        let (kv_a_packed, kv_a_scales) = w8(KV_A, HIDDEN, 2);
        let q_a_norm = norm_w(Q_LORA);
        let (q_b_packed, q_b_scales) = w8(Q_PROJ, Q_LORA, 3);
        let kv_a_norm = norm_w(KV_LORA);
        let kv_b_host: Vec<u16> = (0..HEADS * KV_HEAD * KV_LORA).map(|i| ((((i * 17 % 127) as f32 - 63.0) * (1.0 / 4096.0)).to_bits() >> 16) as u16).collect();
        let kv_b = super::DeviceBuffer::upload(DEVICE, bytes(&kv_b_host)).unwrap();
        let kv_b_scales = super::DeviceBuffer::upload(DEVICE, bytes(&[0x3f80_u16])).unwrap();
        let (o_packed, o_scales) = w8(HIDDEN, Q_PROJ, 4);
        let ffn_norm = norm_w(HIDDEN);
        let router_w = f32_buf(&(0..EXPERTS * HIDDEN).map(|i| (((i * 13 % 23) as f32 - 11.0) * 1.0e-3)).collect::<Vec<_>>());
        let router_b = f32_buf(&(0..EXPERTS).map(|i| (EXPERTS - i) as f32).collect::<Vec<_>>());

        let mut metas_host = Vec::with_capacity(EXPERTS + 1);
        let mut keep = Vec::new();
        for expert in 0..=EXPERTS {
            let gate = super::DeviceBuffer::upload(DEVICE, &iq_rows(21, INTER, HIDDEN, 7 + expert)).unwrap();
            let up = super::DeviceBuffer::upload(DEVICE, &iq_rows(21, INTER, HIDDEN, 137 + expert)).unwrap();
            let down = super::DeviceBuffer::upload(DEVICE, &iq_rows(23, HIDDEN, INTER, 267 + expert)).unwrap();
            metas_host.push(super::GgufGroupedExpertMeta { gate: gate.device_pointer() as u64, up: up.device_pointer() as u64, down: down.device_pointer() as u64, gate_type: 21, up_type: 21, down_type: 23 });
            keep.push((gate, up, down));
        }
        let routed_metas = super::resident_gguf_grouped_metas(DEVICE, &metas_host[..EXPERTS]).unwrap();
        let shared_metas = super::resident_gguf_grouped_metas(DEVICE, &metas_host[EXPERTS..]).unwrap();

        // ---- KV cache 与 selection（内容固定、地址固定）----
        // cache 留 256 行余量：check(CONTEXT) 时 context_rows=4097 的校验要求容量 > CONTEXT。
        const CACHE_ROWS: usize = CONTEXT + 256;
        let latent_cache = super::DeviceBuffer::upload(DEVICE, &(0..CACHE_ROWS * KV_LORA).map(|i| ((i * 29 + i / KV_LORA * 7) % 63 + 1) as u8).collect::<Vec<_>>()).unwrap();
        let latent_scales = super::DeviceBuffer::upload(DEVICE, &bytes(&vec![0x3b80_u16; CACHE_ROWS * (KV_LORA / 64)]).to_vec()).unwrap();
        let rope_cache = super::DeviceBuffer::upload(DEVICE, bytes(&(0..CACHE_ROWS * ROPE_DIM).map(|i| bf16::from_f32((((i * 13) % 127) as f32 - 63.0) * (1.0 / 128.0)).to_bits()).collect::<Vec<_>>())).unwrap();
        let block_table = super::DeviceBuffer::upload(DEVICE, bytes(&(0..CACHE_ROWS.div_ceil(BLOCK) as u32).collect::<Vec<_>>())).unwrap();
        let selection = super::DeviceBuffer::upload(DEVICE, bytes(&(0..DSA_TOP_K).map(|i| ((i * 7 + 3) % CONTEXT) as u32).collect::<Vec<_>>())).unwrap();

        // rope 全表（行宽 32 = rotary 64/2）
        let table_rows = CONTEXT + 64;
        let cos_host: Vec<f32> = (0..table_rows * 32).map(|i| ((i * 31 % 89) as f32 / 89.0) * 2.0 - 1.0).collect();
        let sin_host: Vec<f32> = (0..table_rows * 32).map(|i| ((i * 17 % 83) as f32 / 83.0) * 2.0 - 1.0).collect();

        // position 间接 buffer
        let rope_pos = super::DeviceBuffer::allocate(DEVICE, 4).unwrap();
        let append_pos = super::DeviceBuffer::allocate(DEVICE, 8).unwrap();
        let hidden = f32_buf(&(0..HIDDEN).map(|i| (((i * 11 % 29) as f32 - 14.0) / 16.0)).collect::<Vec<_>>());

        // ---- 整层链（eager 参照与 graph 录制共用同一驱动）----
        // hold 保活所有中间 buffer（graph 固化其地址，生命周期必须盖住所有 replay）。
        let chain = |pos: usize, hold: &mut Vec<super::DeviceBuffer>| -> super::DeviceBuffer {
            rope_pos.copy_from_host(&(pos as u32).to_le_bytes()).unwrap();
            append_pos.copy_from_host(&[pos as u32, pos as u32].map(|v| v.to_le_bytes()).concat()).unwrap();
            let normed = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
            crate::kernel::rocm::hip::try_rmsnorm_resident_weight_into(DEVICE, &hidden, &input_norm, &normed, None, 1, HIDDEN, EPS, false, crate::kernel::rocm::hip::RmsnormOutput::F32).unwrap();
            let (q_a, kv_a) = crate::kernel::rocm::hip::try_ct_dual_gemv_bf16(DEVICE, 8, &[], Some(&normed), HIDDEN, 1, &q_a_packed, &q_a_scales, 1, 32, Q_LORA, &kv_a_packed, &kv_a_scales, 1, 32, KV_A).unwrap();
            let q_an = super::DeviceBuffer::allocate(DEVICE, Q_LORA * 4).unwrap();
            crate::kernel::rocm::hip::try_rmsnorm_resident_weight_into(DEVICE, &q_a, &q_a_norm, &q_an, None, 1, Q_LORA, EPS, false, crate::kernel::rocm::hip::RmsnormOutput::F32).unwrap();
            let query = crate::kernel::rocm::hip::try_ct_quantized_matmul_bf16(DEVICE, 8, &[], Some(&q_an), &q_b_packed, &q_b_scales, 1, 32, 1, Q_LORA, Q_PROJ).unwrap();
            let query_r = crate::kernel::rocm::hip::try_rope_indirect_resident_f32(DEVICE, &query, 1, Q_PROJ, HEADS, ROPE_DIM, crate::attention::rope::RotaryLayout::Interleaved, &rope_pos, &cos_host, &sin_host, false).unwrap();
            let kv_a = std::sync::Arc::new(kv_a);
            let latent_view = super::DeviceBuffer::view(kv_a.clone(), 0, KV_LORA * 4).unwrap();
            let rope_view = super::DeviceBuffer::view(kv_a, KV_LORA * 4, ROPE_DIM * 4).unwrap();
            let latent_n = super::DeviceBuffer::allocate(DEVICE, KV_LORA * 4).unwrap();
            crate::kernel::rocm::hip::try_rmsnorm_resident_weight_into(DEVICE, &latent_view, &kv_a_norm, &latent_n, None, 1, KV_LORA, EPS, false, crate::kernel::rocm::hip::RmsnormOutput::F32).unwrap();
            crate::kernel::rocm::hip::try_paged_cache_append_mla_rope_indirect_f32_q8_bf16(
                DEVICE,
                &latent_n,
                &latent_cache,
                &latent_scales,
                &rope_view,
                &rope_cache,
                &block_table,
                &append_pos,
                1,
                KV_LORA,
                ROPE_DIM,
                ROPE_DIM,
                crate::attention::rope::RotaryLayout::Interleaved,
                &cos_host,
                &sin_host,
                64,
                BLOCK,
            )
            .unwrap();
            let attention = super::DeviceBuffer::allocate(DEVICE, Q_PROJ * 4).unwrap();
            crate::kernel::rocm::hip::try_paged_mla_attention_ct_into(
                DEVICE,
                &query_r,
                &latent_cache,
                Some(&latent_scales),
                64,
                &rope_cache,
                &block_table,
                Some(&selection),
                crate::kernel::rocm::hip::CtMlaWeightRef { packed: &kv_b, scales: &kv_b_scales, rows: HEADS * KV_HEAD, cols: KV_LORA, group_size: KV_LORA, scale_dtype: 0, bits: 16 },
                1,
                pos + 1,
                pos,
                Q_PROJ,
                HEADS,
                ROPE_DIM,
                DSA_TOP_K,
                BLOCK,
                &attention,
                None,
            )
            .unwrap();
            let proj = crate::kernel::rocm::hip::try_ct_quantized_matmul_bf16(DEVICE, 8, &[], Some(&attention), &o_packed, &o_scales, 1, 32, 1, Q_PROJ, HIDDEN).unwrap();
            let attn_out = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
            crate::kernel::rocm::hip::try_add_resident_f32_into(DEVICE, &hidden, &proj, &attn_out, HIDDEN, 1.0).unwrap();
            let moe_in = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
            crate::kernel::rocm::hip::try_rmsnorm_resident_weight_into(DEVICE, &attn_out, &ffn_norm, &moe_in, None, 1, HIDDEN, EPS, false, crate::kernel::rocm::hip::RmsnormOutput::F32).unwrap();
            let route = crate::kernel::rocm::hip::try_moe_route_resident_device_f32(DEVICE, &moe_in, &router_w, &router_b, 1, HIDDEN, EXPERTS, MOE_TOP_K, 1, 2.5).unwrap();
            let routed_out = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
            crate::kernel::rocm::hip::try_gguf_fused_decode_experts_typed(DEVICE, &moe_in, 1, HIDDEN, INTER, MOE_TOP_K, &route.expert_ids, &route.weights, MOE_TOP_K, &routed_metas.buffer, &routed_out, Some([21, 21, 23])).unwrap();
            let (shared_ids, shared_weights) = super::super::ct_grouped::cooperative_single_expert_route(DEVICE, 1).unwrap();
            let shared_out = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
            crate::kernel::rocm::hip::try_gguf_fused_decode_experts_typed(DEVICE, &moe_in, 1, HIDDEN, INTER, 1, &shared_ids, &shared_weights, 1, &shared_metas.buffer, &shared_out, Some([21, 21, 23])).unwrap();
            let combined = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
            crate::kernel::rocm::hip::try_add_resident_f32_into(DEVICE, &routed_out, &shared_out, &combined, HIDDEN, 1.0).unwrap();
            let final_out = super::DeviceBuffer::allocate(DEVICE, HIDDEN * 4).unwrap();
            crate::kernel::rocm::hip::try_add_resident_f32_into(DEVICE, &attn_out, &combined, &final_out, HIDDEN, 1.0).unwrap();
            hold.push(latent_view);
            hold.push(rope_view);
            hold.push(normed);
            hold.push(q_an);
            hold.push(query);
            hold.push(query_r);
            hold.push(latent_n);
            hold.push(attention);
            hold.push(proj);
            hold.push(attn_out);
            hold.push(moe_in);
            hold.push(route.expert_ids);
            hold.push(route.weights);
            hold.push(routed_out);
            hold.push(shared_out);
            hold.push(combined);
            final_out
        };

        // eager 参照 @CONTEXT-1
        let mut eager_hold = Vec::new();
        let eager_out = chain(CONTEXT - 1, &mut eager_hold);
        super::super::super::synchronize_device(DEVICE, "probe eager 参照").unwrap();

        // graph 录制（录制期 kernel 只录不跑；产出地址被 graph 固化）
        let stream = super::super::active_compute_stream();
        let mut graph_hold = Vec::new();
        let graph = {
            let recorder = crate::kernel::rocm::hip::StaticGraphRecorder::begin(DEVICE, stream).unwrap();
            let out = chain(CONTEXT - 1, &mut graph_hold);
            let graph = recorder.finish().unwrap().expect("整层 graph 非空");
            eprintln!("[layer-graph-probe] nodes={}", graph.node_count());
            graph_hold.push(out);
            graph
        };

        // replay 与 eager 同 position 逐步对照；再换一个 position 证间接生效
        let mut check = |pos: usize| {
            let mut eager2_hold = Vec::new();
            let eager2 = chain(pos, &mut eager2_hold);
            super::super::super::synchronize_device(DEVICE, "probe eager 对照").unwrap();
            rope_pos.copy_from_host(&(pos as u32).to_le_bytes()).unwrap();
            append_pos.copy_from_host(&[pos as u32, pos as u32].map(|v| v.to_le_bytes()).concat()).unwrap();
            graph.launch(DEVICE, stream).unwrap();
            super::super::super::synchronize_device(DEVICE, "probe replay").unwrap();
            let mut actual = vec![0.0_f32; HIDDEN];
            let mut reference = vec![0.0_f32; HIDDEN];
            graph_hold.last().unwrap().copy_to_host(unsafe { std::slice::from_raw_parts_mut(actual.as_mut_ptr().cast(), HIDDEN * 4) }).unwrap();
            eager2.copy_to_host(unsafe { std::slice::from_raw_parts_mut(reference.as_mut_ptr().cast(), HIDDEN * 4) }).unwrap();
            let mut max_abs = 0.0_f32;
            for (left, right) in reference.iter().zip(&actual) {
                assert!(left.is_finite() && right.is_finite(), "pos={pos} 非有限: eager={left} graph={right}");
                max_abs = max_abs.max((left - right).abs());
            }
            eprintln!("[layer-graph-probe] pos={pos} max_abs={max_abs:.6e}");
            assert!(max_abs <= 1.0e-3, "pos={pos} replay 与 eager 不符: {max_abs}");
        };
        check(CONTEXT - 1);
        check(CONTEXT);

        // 计时：eager 整链 vs graph replay
        let mut time = |label: &str, rounds: usize, call: &mut dyn FnMut()| {
            for _ in 0..2 {
                call();
            }
            super::super::super::synchronize_device(DEVICE, "probe 计时预热").unwrap();
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                call();
            }
            super::super::super::synchronize_device(DEVICE, "probe 计时").unwrap();
            let micros = started.elapsed().as_micros() as f64 / rounds as f64;
            eprintln!("[layer-graph-probe] {label} avg_us={micros:.1}");
        };
        let mut eager_hold2 = Vec::new();
        time("eager-chain", 20, &mut || {
            std::hint::black_box(chain(CONTEXT - 1, &mut eager_hold2));
        });
        time("graph-replay", 100, &mut || graph.launch(DEVICE, stream).unwrap());
    }
}
