use super::tensor::{with_deferred_moe_route_workspace, with_tensor_workspace};
use super::*;

/// 将标准逐行 MXFP4 重排为 rocWMMA 16x16 B fragment 的 lane 原生顺序。
/// 每个 tile 仍为 128 字节，scale 也只改变行序，不增加设备权重体积。
pub(crate) fn preshuffle_mxfp4_16x16(packed: &[u8], scales: &[u8], rows: usize, columns: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(rows.is_multiple_of(16) && columns.is_multiple_of(32));
    assert_eq!(packed.len(), rows * columns / 2);
    assert_eq!(scales.len(), rows * columns / 32);

    let mut native_packed = vec![0u8; packed.len()];
    let column_tiles = columns / 16;
    for row_tile in 0..rows / 16 {
        for column_tile in 0..column_tiles {
            let tile_base = (row_tile * column_tiles + column_tile) * 128;
            for lane in 0..32 {
                let row = row_tile * 16 + lane % 16;
                let column = column_tile * 16 + lane / 16 * 8;
                let source = row * (columns / 2) + column / 2;
                let target = tile_base + lane * 4;
                native_packed[target..target + 4].copy_from_slice(&packed[source..source + 4]);
            }
        }
    }

    let mut native_scales = vec![0u8; scales.len()];
    let scale_columns = columns / 32;
    for row_tile in 0..rows / 16 {
        for column in 0..scale_columns {
            let target = (row_tile * scale_columns + column) * 16;
            for row in 0..16 {
                native_scales[target + row] = scales[(row_tile * 16 + row) * scale_columns + column];
            }
        }
    }
    (native_packed, native_scales)
}

const MOE_PREFILL_SOURCE: &str = include_str!("moe/source.hip");

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Fp8GroupedWeightMeta {
    pub(crate) codes: u64,
    pub(crate) scales: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Fp8GroupedExpertMeta {
    pub(crate) gate: Fp8GroupedWeightMeta,
    pub(crate) up: Fp8GroupedWeightMeta,
    pub(crate) down: Fp8GroupedWeightMeta,
}

#[derive(Clone, Copy)]
pub(super) struct MoePrefillFunctions {
    router_logits_decode: usize,
    router_logits_precise: usize,
    router_logits: usize,
    router_topk_256: usize,
    zero: usize,
    gather: usize,
    gather_bf16: usize,
    scatter_add: usize,
    mxfp4_matmul: usize,
    group_routes_small: usize,
    mxfp4_grouped_gate_up: usize,
    mxfp4_grouped_down: usize,
    mxfp4_grouped_down_reduce: usize,
    mxfp4_decode_gate_up: usize,
    mxfp4_decode_down: usize,
    router_selected: usize,
    fp8_grouped_gate_up: usize,
    fp8_grouped_down: usize,
}

fn moe_prefill_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(&[DEVICE_CONVERSIONS_PREAMBLE, MOE_PREFILL_SOURCE].concat(), "zllm_rocm_moe_prefill.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

pub(super) fn moe_prefill_functions(device_id: i32) -> Result<MoePrefillFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, MoePrefillFunctions), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm MoE prefill kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = moe_prefill_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData MoE prefill"));
        }
        let function = |name: &str| -> Result<usize, String> {
            let name = CString::new(name).unwrap();
            let mut function = ptr::null_mut();
            let status = unsafe { module_get_function(&mut function, module, name.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction MoE prefill"));
            }
            Ok(function as usize)
        };
        Ok((
            module as usize,
            MoePrefillFunctions {
                router_logits_decode: function("moe_router_logits_decode_f32")?,
                router_logits_precise: function("moe_router_logits_precise_f32")?,
                router_logits: function("moe_router_logits_f32")?,
                router_topk_256: function("moe_router_topk_256_f32")?,
                zero: function("moe_zero_f32")?,
                gather: function("moe_gather_f32")?,
                gather_bf16: function("moe_gather_bf16")?,
                scatter_add: function("moe_scatter_add_f32")?,
                mxfp4_matmul: function("mxfp4_matmul_f32")?,
                group_routes_small: function("moe_group_routes_small_f32")?,
                mxfp4_grouped_gate_up: function("mxfp4_grouped_gate_up_wmma_f32")?,
                mxfp4_grouped_down: function("mxfp4_grouped_down_scatter_wmma_f32")?,
                mxfp4_grouped_down_reduce: function("mxfp4_grouped_down_reduce_f32")?,
                mxfp4_decode_gate_up: function("mxfp4_decode_gate_up_f32")?,
                mxfp4_decode_down: function("mxfp4_decode_down_reduce_f32")?,
                router_selected: function("moe_router_selected_f32")?,
                fp8_grouped_gate_up: function("fp8_grouped_gate_up_f32")?,
                fp8_grouped_down: function("fp8_grouped_down_f32")?,
            },
        ))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

/// grouped kernel 的第三维是 expert;与 2D 版共享 null stream 与统计。
pub(super) fn launch_moe_kernel_3d(function: usize, grid_x: u32, grid_y: u32, grid_z: u32, block: u32, shared_bytes: u32, arguments: &mut [*mut c_void], action: &str) -> Result<(), String> {
    let started = super::hip_api_stats::start();
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = unsafe { launch(function as *mut c_void, grid_x, grid_y, grid_z, block, 1, 1, shared_bytes, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, action));
    }
    // 与 2D 版一致:kernel_sync 只在显式调试开关下整卡同步,默认路径不变。
    if options().kernel_sync {
        synchronize_device(super::tensor::current_device()?, action)?;
    }
    Ok(())
}

pub(super) fn launch_moe_kernel(function: usize, grid_x: u32, grid_y: u32, block: u32, shared_bytes: u32, arguments: &mut [*mut c_void], action: &str) -> Result<(), String> {
    let started = super::hip_api_stats::start();
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = unsafe { launch(function as *mut c_void, grid_x, grid_y, 1, block, 1, 1, shared_bytes, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, action));
    }
    if options().kernel_sync {
        synchronize_device(super::tensor::current_device()?, action)?;
    }
    Ok(())
}

pub(crate) struct RocmMoeRoute {
    pub(crate) expert_ids: DeviceBuffer,
    pub(crate) weights: DeviceBuffer,
    pub(crate) len: usize,
}

/// 固定地址 graph 使用的 route workspace。普通 eager 路径继续使用 deferred
/// workspace；这里的三块 buffer 由 graph owner 独占并跨 token 保持地址稳定。
pub(crate) struct MoeRouteGraphBuffers {
    logits: DeviceBuffer,
    expert_ids: DeviceBuffer,
    weights: DeviceBuffer,
    len: usize,
}

impl MoeRouteGraphBuffers {
    pub(crate) fn new(device_id: i32, rows: usize, experts: usize, top_k: usize) -> Result<Self, String> {
        if rows == 0 || experts == 0 || top_k == 0 || top_k > experts || top_k > 16 {
            return Err("ROCm graph route shape 非法".to_owned());
        }
        // 录制前完成 HIPRTC，graph 段内不得触发 module load 或 allocation。
        let _ = moe_prefill_functions(device_id)?;
        let logits_bytes = rows.checked_mul(experts).and_then(|n| n.checked_mul(4)).ok_or("ROCm graph logits 字节溢出")?;
        let len = rows.checked_mul(top_k).ok_or("ROCm graph route 数溢出")?;
        let route_bytes = len.checked_mul(4).ok_or("ROCm graph route 字节溢出")?;
        Ok(Self { logits: DeviceBuffer::allocate(device_id, logits_bytes)?, expert_ids: DeviceBuffer::allocate(device_id, route_bytes)?, weights: DeviceBuffer::allocate(device_id, route_bytes)?, len })
    }

    pub(crate) fn expert_ids(&self) -> &DeviceBuffer {
        &self.expert_ids
    }

    pub(crate) fn weights(&self) -> &DeviceBuffer {
        &self.weights
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch(&self, device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, bias: &DeviceBuffer, rows: usize, columns: usize, experts: usize, top_k: usize, scoring: u32, scaling: f32) -> Result<(), String> {
        let len = launch_moe_route_resident_device_f32(device_id, input, weight, bias, rows, columns, experts, top_k, scoring, scaling, &self.logits, &self.expert_ids, &self.weights)?;
        if len != self.len {
            return Err(format!("ROCm graph route 数变化: {len}/{}", self.len));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_moe_route_resident_device_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: &DeviceBuffer,
    rows: usize,
    columns: usize,
    experts: usize,
    top_k: usize,
    scoring: u32,
    scaling: f32,
    logits: &DeviceBuffer,
    ids: &DeviceBuffer,
    weights: &DeviceBuffer,
) -> Result<usize, String> {
    if rows == 0 || columns == 0 || experts == 0 || top_k == 0 || top_k > experts || top_k > 16 || scoring > 2 || !scaling.is_finite() {
        return Err("ROCm resident MoE route 参数非法".to_owned());
    }
    let input_elements = rows.checked_mul(columns).ok_or("ROCm MoE route input 大小溢出")?;
    let weight_elements = experts.checked_mul(columns).ok_or("ROCm MoE route weight 大小溢出")?;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("ROCm MoE route input 字节溢出")?, "MoE route input")?;
    let precise = options().precise_router;
    validate_resident(weight, device_id, weight_elements.checked_mul(if precise { 4 } else { 2 }).ok_or("ROCm MoE route weight 字节溢出")?, "MoE route weight")?;
    validate_resident(bias, device_id, experts.checked_mul(4).ok_or("ROCm MoE route bias 字节溢出")?, "MoE route bias")?;
    let logits_elements = rows.checked_mul(experts).ok_or("ROCm MoE logits 大小溢出")?;
    let route_elements = rows.checked_mul(top_k).ok_or("ROCm MoE route 数量溢出")?;
    validate_resident(logits, device_id, logits_elements.checked_mul(4).ok_or("ROCm MoE logits 字节溢出")?, "MoE route logits")?;
    validate_resident(ids, device_id, route_elements.checked_mul(4).ok_or("ROCm MoE ids 字节溢出")?, "MoE route ids")?;
    validate_resident(weights, device_id, route_elements.checked_mul(4).ok_or("ROCm MoE weights 字节溢出")?, "MoE route weights")?;

    set_device(device_id)?;
    let functions = moe_prefill_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = weight.pointer;
    let mut d_logits = logits.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "ROCm MoE rows 超过 u32".to_owned())?;
    let mut columns_u32 = u32::try_from(columns).map_err(|_| "ROCm MoE columns 超过 u32".to_owned())?;
    let mut experts_u32 = u32::try_from(experts).map_err(|_| "ROCm MoE experts 超过 u32".to_owned())?;
    let mut logits_arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_weight as *mut *mut c_void).cast(),
        (&mut d_logits as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut experts_u32 as *mut u32).cast(),
    ];
    let logits_started = options().kernel_profile.then(std::time::Instant::now);
    if precise {
        launch_moe_kernel(functions.router_logits_precise, experts_u32, rows_u32, 256, 0, &mut logits_arguments, "HIP MoE precise F32 router logits")?;
    } else if rows == 1 {
        let mut decode_arguments =
            [(&mut d_input as *mut *mut c_void).cast(), (&mut d_weight as *mut *mut c_void).cast(), (&mut d_logits as *mut *mut c_void).cast(), (&mut columns_u32 as *mut u32).cast(), (&mut experts_u32 as *mut u32).cast()];
        launch_moe_kernel(functions.router_logits_decode, experts_u32.div_ceil(4), 1, 256, 0, &mut decode_arguments, "HIP MoE decode router logits")?;
    } else {
        // moe_router_logits_f32 按 wave32 硬编码 lane/wave 划分(threadIdx.x & 31),
        // CDNA(wave64)上会算错,launch 前显式拒绝。
        let wavefront_size = device_wavefront_size(device_id)?;
        if wavefront_size != 32 {
            return Err(format!("ROCm MoE router logits kernel 仅支持 wave32,device={device_id} wavefront size={wavefront_size}"));
        }
        launch_moe_kernel(functions.router_logits, experts_u32.div_ceil(128), rows_u32.div_ceil(16), 256, 0, &mut logits_arguments, "HIP MoE router logits")?;
    }
    if let Some(started) = logits_started {
        synchronize_device(device_id, "MoE router logits profile")?;
        eprintln!("[rocm-kernel] moe-router-{}logits device={device_id} rows={rows} wall={:.6}s", if precise { "precise-" } else { "" }, started.elapsed().as_secs_f64());
    }

    let mut d_bias = bias.pointer;
    let mut d_ids = ids.pointer;
    let mut d_route_weights = weights.pointer;
    let mut top_k_u32 = u32::try_from(top_k).map_err(|_| "ROCm MoE top_k 超过 u32".to_owned())?;
    let mut scoring_u32 = scoring;
    let mut scaling_f32 = scaling;
    let mut topk_arguments = [
        (&mut d_logits as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_ids as *mut *mut c_void).cast(),
        (&mut d_route_weights as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut experts_u32 as *mut u32).cast(),
        (&mut top_k_u32 as *mut u32).cast(),
        (&mut scoring_u32 as *mut u32).cast(),
        (&mut scaling_f32 as *mut f32).cast(),
    ];
    let topk_started = options().kernel_profile.then(std::time::Instant::now);
    launch_moe_kernel(functions.router_topk_256, rows_u32, 1, 256, 0, &mut topk_arguments, "HIP MoE router top-k")?;
    if let Some(started) = topk_started {
        synchronize_device(device_id, "MoE router top-k profile")?;
        eprintln!("[rocm-kernel] moe-router-topk device={device_id} rows={rows} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    if options().trace_moe_route {
        static TRACE_INDEX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let trace_index = TRACE_INDEX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let logits_len = rows * experts;
        let mut host_logits = vec![0.0_f32; logits_len];
        let mut host_ids = vec![0_u32; route_elements];
        let mut host_weights = vec![0.0_f32; route_elements];
        logits.copy_to_host(unsafe { std::slice::from_raw_parts_mut(host_logits.as_mut_ptr().cast(), logits_len * 4) })?;
        ids.copy_to_host(unsafe { std::slice::from_raw_parts_mut(host_ids.as_mut_ptr().cast(), route_elements * 4) })?;
        weights.copy_to_host(unsafe { std::slice::from_raw_parts_mut(host_weights.as_mut_ptr().cast(), route_elements * 4) })?;
        let hash = |values: &[u8]| values.iter().fold(0xcbf29ce484222325_u64, |hash, &value| (hash ^ u64::from(value)).wrapping_mul(0x100000001b3));
        let logits_bytes = unsafe { std::slice::from_raw_parts(host_logits.as_ptr().cast(), logits_len * 4) };
        let id_bytes = unsafe { std::slice::from_raw_parts(host_ids.as_ptr().cast(), route_elements * 4) };
        let weight_bytes = unsafe { std::slice::from_raw_parts(host_weights.as_ptr().cast(), route_elements * 4) };
        let first = route_elements.min(top_k);
        eprintln!(
            "rocm-moe-route-kernel index={} rows={} logits={:016x} ids={:016x} weights={:016x} first_ids={:?} first_weights={:?}",
            trace_index,
            rows,
            hash(logits_bytes),
            hash(id_bytes),
            hash(weight_bytes),
            &host_ids[..first],
            &host_weights[..first],
        );
    }
    Ok(route_elements)
}

/// 官方 F8_E4M3 + F32 block scale 的 resident routed expert。权重、路由、
/// activation 与 epilogue 都停留在 device，只返回最终 hidden。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fp8_grouped_experts_f32(
    device_id: i32,
    input: &DeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    top_k: usize,
    expert_count: usize,
    route_ids: &DeviceBuffer,
    route_weights: &DeviceBuffer,
    route_count: usize,
    metas: &DeviceBuffer,
    shared: Option<&DeviceBuffer>,
    residual: Option<&DeviceBuffer>,
) -> Result<DeviceBuffer, String> {
    if rows == 0 || hidden == 0 || intermediate == 0 || top_k == 0 || route_count != rows.checked_mul(top_k).ok_or("FP8 grouped route 数量溢出")? || expert_count == 0 {
        return Err(format!("FP8 grouped shape 非法: rows={rows} hidden={hidden} intermediate={intermediate} top_k={top_k} routes={route_count} experts={expert_count}"));
    }
    let input_bytes = rows.checked_mul(hidden).and_then(|n| n.checked_mul(4)).ok_or("FP8 grouped input 大小溢出")?;
    let route_bytes = route_count.checked_mul(4).ok_or("FP8 grouped route 大小溢出")?;
    let meta_bytes = expert_count.checked_mul(std::mem::size_of::<Fp8GroupedExpertMeta>()).ok_or("FP8 grouped meta 大小溢出")?;
    validate_resident(input, device_id, input_bytes, "FP8 grouped input")?;
    validate_resident(route_ids, device_id, route_bytes, "FP8 grouped route ids")?;
    validate_resident(route_weights, device_id, route_bytes, "FP8 grouped route weights")?;
    validate_resident(metas, device_id, meta_bytes, "FP8 grouped metas")?;
    for (label, tensor) in [("shared", shared), ("residual", residual)] {
        if let Some(tensor) = tensor {
            validate_resident(tensor, device_id, input_bytes, &format!("FP8 grouped {label}"))?;
        }
    }
    set_device(device_id)?;
    let functions = moe_prefill_functions(device_id)?;
    let output = DeviceBuffer::allocate(device_id, input_bytes)?;
    let activated_bytes = route_count.checked_mul(intermediate).and_then(|n| n.checked_mul(2)).ok_or("FP8 grouped activated 大小溢出")?;
    let activated = DeviceBuffer::allocate(device_id, activated_bytes)?;
    let wavefront = usize::try_from(device_wavefront_size(device_id)?).map_err(|_| "FP8 grouped wavefront 无效")?;
    if wavefront == 0 || !256_usize.is_multiple_of(wavefront) {
        return Err(format!("FP8 grouped 不支持 wavefront={wavefront}"));
    }
    let waves = 256 / wavefront;
    let mut d_input = input.pointer;
    let mut d_route_ids = route_ids.pointer;
    let mut d_route_weights = route_weights.pointer;
    let mut d_metas = metas.pointer;
    let mut d_activated = activated.pointer;
    let mut d_shared = shared.map_or(ptr::null_mut(), |buffer| buffer.pointer);
    let mut d_residual = residual.map_or(ptr::null_mut(), |buffer| buffer.pointer);
    let mut d_output = output.pointer;
    let mut rows_u32 = moe_u32("FP8 grouped rows", rows)?;
    let mut hidden_u32 = moe_u32("FP8 grouped hidden", hidden)?;
    let mut intermediate_u32 = moe_u32("FP8 grouped intermediate", intermediate)?;
    let mut route_count_u32 = moe_u32("FP8 grouped routes", route_count)?;
    let mut top_k_u32 = moe_u32("FP8 grouped top_k", top_k)?;
    let mut expert_count_u32 = moe_u32("FP8 grouped experts", expert_count)?;
    let mut gate_arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_route_ids as *mut *mut c_void).cast(),
        (&mut d_metas as *mut *mut c_void).cast(),
        (&mut d_activated as *mut *mut c_void).cast(),
        (&mut hidden_u32 as *mut u32).cast(),
        (&mut intermediate_u32 as *mut u32).cast(),
        (&mut route_count_u32 as *mut u32).cast(),
        (&mut top_k_u32 as *mut u32).cast(),
        (&mut expert_count_u32 as *mut u32).cast(),
    ];
    let gate_started = options().kernel_profile.then(std::time::Instant::now);
    launch_moe_kernel(functions.fp8_grouped_gate_up, moe_u32("FP8 grouped gate grid", intermediate.div_ceil(waves))?, route_count_u32, 256, 0, &mut gate_arguments, "HIP FP8 grouped gate/up")?;
    if let Some(started) = gate_started {
        synchronize_device(device_id, "FP8 grouped gate/up profile")?;
        eprintln!("[rocm-kernel] fp8-grouped-gate-up device={device_id} rows={rows} routes={route_count} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    let mut down_arguments = [
        (&mut d_activated as *mut *mut c_void).cast(),
        (&mut d_route_ids as *mut *mut c_void).cast(),
        (&mut d_route_weights as *mut *mut c_void).cast(),
        (&mut d_metas as *mut *mut c_void).cast(),
        (&mut d_shared as *mut *mut c_void).cast(),
        (&mut d_residual as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut hidden_u32 as *mut u32).cast(),
        (&mut intermediate_u32 as *mut u32).cast(),
        (&mut top_k_u32 as *mut u32).cast(),
        (&mut expert_count_u32 as *mut u32).cast(),
    ];
    let down_started = options().kernel_profile.then(std::time::Instant::now);
    launch_moe_kernel(functions.fp8_grouped_down, moe_u32("FP8 grouped down grid", hidden.div_ceil(waves))?, rows_u32, 256, 0, &mut down_arguments, "HIP FP8 grouped down")?;
    if let Some(started) = down_started {
        synchronize_device(device_id, "FP8 grouped down profile")?;
        eprintln!("[rocm-kernel] fp8-grouped-down device={device_id} rows={rows} routes={route_count} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    if options().debug_finite {
        try_validate_finite_resident_range_f32(device_id, &output, 0, rows * hidden).map_err(|error| format!("FP8 grouped output 包含非有限值: {error}"))?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn moe_u32(name: &str, value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("ROCm MoE {name}={value} 超过 u32"))
}

#[allow(clippy::too_many_arguments)]
/// MXFP4 grouped routed experts:整层一次装填。
///
/// 权重为层级大 buffer(gate 段 [E*I,H] 在前、up 段 [E*I,H] 在后拼成
/// `[2*E*I, H]` packed;down 为 `[E*H, I]`)。路由 CSR(offsets+token_ids+
/// weights)由 host 组装上传;输出先清零,down 段 atomicAdd 散射,每 token 的
/// top_k 路由自然叠加。两次 launch 替代逐专家 gather/GEMM/scatter。
#[allow(clippy::too_many_arguments)]
pub fn try_mxfp4_grouped_experts_f32(
    device_id: i32,
    input: &DeviceBuffer,
    token_count: usize,
    route_tokens: &[u32],
    route_weights: &[f32],
    route_offsets: &[u32],
    gate_up_packed: &DeviceBuffer,
    gate_up_scales: &DeviceBuffer,
    down_packed: &DeviceBuffer,
    down_scales: &DeviceBuffer,
    hidden: usize,
    intermediate: usize,
    expert_count: usize,
    activation_limit: f32,
) -> Result<DeviceBuffer, String> {
    let total = route_tokens.len();
    let total_u32 = moe_u32("grouped route rows", total)?;
    if token_count == 0 || total != route_weights.len() || route_offsets.len() != expert_count + 1 || route_offsets.last().copied() != Some(total_u32) || hidden % 64 != 0 || intermediate % 64 != 0 {
        return Err(format!("ROCm mxfp4 grouped shape 非法: tokens={token_count} total={total} experts={expert_count} hidden={hidden} intermediate={intermediate}"));
    }
    let input_bytes = token_count.checked_mul(hidden).and_then(|n| n.checked_mul(4)).ok_or("grouped input 溢出")?;
    // 层级权重 buffer 布局:packed [2*E*I, H/2] / scales [2*E*I, H/32];down [E*H, I/2]。
    let gate_up_rows = expert_count.checked_mul(intermediate).and_then(|n| n.checked_mul(2)).ok_or("grouped gate/up rows 溢出")?;
    let down_rows = expert_count.checked_mul(hidden).ok_or("grouped down rows 溢出")?;
    let matrix_bytes = |rows: usize, columns: usize, divisor: usize, name: &str| rows.checked_mul(columns / divisor).ok_or_else(|| format!("grouped {name} bytes 溢出"));
    let gate_up_packed_bytes = matrix_bytes(gate_up_rows, hidden, 2, "gate/up packed")?;
    let gate_up_scale_bytes = matrix_bytes(gate_up_rows, hidden, 32, "gate/up scales")?;
    let down_packed_bytes = matrix_bytes(down_rows, intermediate, 2, "down packed")?;
    let down_scale_bytes = matrix_bytes(down_rows, intermediate, 32, "down scales")?;
    validate_resident(input, device_id, input_bytes, "mxfp4 grouped input")?;
    validate_resident(gate_up_packed, device_id, gate_up_packed_bytes, "mxfp4 grouped gate/up packed")?;
    validate_resident(gate_up_scales, device_id, gate_up_scale_bytes, "mxfp4 grouped gate/up scales")?;
    validate_resident(down_packed, device_id, down_packed_bytes, "mxfp4 grouped down packed")?;
    validate_resident(down_scales, device_id, down_scale_bytes, "mxfp4 grouped down scales")?;
    set_device(device_id)?;
    let functions = moe_prefill_functions(device_id)?;
    let output = DeviceBuffer::allocate(device_id, input_bytes)?;
    let activated_bytes = total.checked_mul(intermediate).and_then(|n| n.checked_mul(4)).ok_or("grouped activated 溢出")?;
    let token_bytes = total.checked_mul(4).ok_or("grouped token ids 溢出")?;
    let offset_bytes = route_offsets.len().checked_mul(4).ok_or("grouped offsets 溢出")?;
    let weight_bytes = route_weights.len().checked_mul(4).ok_or("grouped route weights 溢出")?;
    // gate/up 与 down 都提交到同一默认 stream；下一层写入排在 down consumer 之后，
    // 因而每个 stage 线程只需一份 workspace，不能让长 prefill 保留八份大 activation。
    with_tensor_workspace(device_id, &[activated_bytes, token_bytes, offset_bytes, weight_bytes], |workspace| {
        let activated = workspace.buffer(0);
        let token_ids_buffer = workspace.buffer(1);
        let offsets_buffer = workspace.buffer(2);
        let weights_buffer = workspace.buffer(3);
        token_ids_buffer.copy_from_host(unsafe { std::slice::from_raw_parts(route_tokens.as_ptr().cast::<u8>(), token_bytes) })?;
        offsets_buffer.copy_from_host(unsafe { std::slice::from_raw_parts(route_offsets.as_ptr().cast::<u8>(), offset_bytes) })?;
        weights_buffer.copy_from_host(unsafe { std::slice::from_raw_parts(route_weights.as_ptr().cast::<u8>(), weight_bytes) })?;
        // 输出清零(复用 moe_zero)。
        {
            let mut d_output = output.pointer;
            let mut elements = moe_u32("grouped zero elements", input_bytes / 4)?;
            let mut zero_args = [(&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
            launch_moe_kernel(functions.zero, elements.div_ceil(256), 1, 256, 0, &mut zero_args, "hip mxfp4 grouped zero")?;
        }
        // gate+up+SiluClamped:grid (intermediate/32, 路由行上限, E)。
        let route_rows_max = route_offsets.windows(2).map(|pair| pair[1] - pair[0]).max().unwrap_or(1).max(1);
        let mut d_input = input.pointer;
        let mut d_token_ids = token_ids_buffer.pointer;
        let mut d_offsets = offsets_buffer.pointer;
        let mut d_active_experts = ptr::null_mut();
        let mut d_active_count = ptr::null_mut();
        let mut d_activated = activated.pointer;
        let mut d_gate_up_packed = gate_up_packed.pointer;
        let mut d_gate_up_scales = gate_up_scales.pointer;
        let mut hidden_u32 = moe_u32("hidden", hidden)?;
        let mut intermediate_u32 = moe_u32("intermediate", intermediate)?;
        let mut expert_count_u32 = moe_u32("expert count", expert_count)?;
        let mut limit_f32 = activation_limit;
        let mut gate_args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_token_ids as *mut *mut c_void).cast(),
            (&mut d_offsets as *mut *mut c_void).cast(),
            (&mut d_active_experts as *mut *mut c_void).cast(),
            (&mut d_active_count as *mut *mut c_void).cast(),
            (&mut d_activated as *mut *mut c_void).cast(),
            (&mut d_gate_up_packed as *mut *mut c_void).cast(),
            (&mut d_gate_up_scales as *mut *mut c_void).cast(),
            (&mut hidden_u32 as *mut u32).cast(),
            (&mut intermediate_u32 as *mut u32).cast(),
            (&mut expert_count_u32 as *mut u32).cast(),
            (&mut limit_f32 as *mut f32).cast(),
        ];
        launch_moe_kernel_3d(functions.mxfp4_grouped_gate_up, moe_u32("gate/up grid", intermediate.div_ceil(64))?, route_rows_max.div_ceil(128), expert_count_u32, 256, 0, &mut gate_args, "hip mxfp4 grouped gate_up WMMA")?;
        // down + scatter:grid (ceil(hidden/128), ceil(route_rows/128), E)。
        let mut d_route_weights = weights_buffer.pointer;
        let mut d_output = output.pointer;
        let mut d_down_packed = down_packed.pointer;
        let mut d_down_scales = down_scales.pointer;
        let mut route_major_output = 0u32;
        let mut down_args = [
            (&mut d_activated as *mut *mut c_void).cast(),
            (&mut d_token_ids as *mut *mut c_void).cast(),
            (&mut d_offsets as *mut *mut c_void).cast(),
            (&mut d_active_experts as *mut *mut c_void).cast(),
            (&mut d_active_count as *mut *mut c_void).cast(),
            (&mut d_route_weights as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut d_down_packed as *mut *mut c_void).cast(),
            (&mut d_down_scales as *mut *mut c_void).cast(),
            (&mut hidden_u32 as *mut u32).cast(),
            (&mut intermediate_u32 as *mut u32).cast(),
            (&mut expert_count_u32 as *mut u32).cast(),
            (&mut route_major_output as *mut u32).cast(),
        ];
        launch_moe_kernel_3d(functions.mxfp4_grouped_down, moe_u32("down grid", hidden.div_ceil(128))?, route_rows_max.div_ceil(128), expert_count_u32, 256, 0, &mut down_args, "hip mxfp4 grouped down WMMA")?;
        Ok(())
    })?;
    Ok(output)
}

/// 小批量路由全程留在设备端：先按 expert 压成 active CSR，再让同一 expert 的
/// 跨 session activation 共享一次 WMMA 权重读取。grid-z 取 route 上界，空组
/// 由设备 active_count 直接退出，不需要下载路由或同步 CPU。
#[allow(clippy::too_many_arguments)]
pub fn try_mxfp4_grouped_decode_experts_f32(
    device_id: i32,
    input: &DeviceBuffer,
    rows: usize,
    top_k: usize,
    route_ids: &DeviceBuffer,
    route_weights: &DeviceBuffer,
    gate_up_packed: &DeviceBuffer,
    gate_up_scales: &DeviceBuffer,
    down_packed: &DeviceBuffer,
    down_scales: &DeviceBuffer,
    hidden: usize,
    intermediate: usize,
    expert_count: usize,
    activation_limit: f32,
    shared: Option<&DeviceBuffer>,
) -> Result<DeviceBuffer, String> {
    if rows == 0 || top_k == 0 || top_k > 16 || hidden == 0 || intermediate == 0 || expert_count == 0 || !hidden.is_multiple_of(64) || !intermediate.is_multiple_of(64) {
        return Err(format!("ROCm MXFP4 grouped decode shape 非法: rows={rows} top_k={top_k} experts={expert_count} hidden={hidden} intermediate={intermediate}"));
    }
    let route_count = rows.checked_mul(top_k).ok_or("MXFP4 grouped decode route 数溢出")?;
    let input_bytes = rows.checked_mul(hidden).and_then(|n| n.checked_mul(4)).ok_or("MXFP4 grouped decode input 溢出")?;
    validate_resident(input, device_id, input_bytes, "MXFP4 grouped decode input")?;
    validate_resident(route_ids, device_id, route_count.checked_mul(4).ok_or("MXFP4 grouped decode ids 溢出")?, "MXFP4 grouped decode ids")?;
    validate_resident(route_weights, device_id, route_count.checked_mul(4).ok_or("MXFP4 grouped decode weights 溢出")?, "MXFP4 grouped decode weights")?;
    let gate_up_rows = expert_count.checked_mul(intermediate).and_then(|n| n.checked_mul(2)).ok_or("MXFP4 grouped decode gate/up rows 溢出")?;
    let down_rows = expert_count.checked_mul(hidden).ok_or("MXFP4 grouped decode down rows 溢出")?;
    validate_resident(gate_up_packed, device_id, gate_up_rows.checked_mul(hidden / 2).ok_or("MXFP4 grouped decode gate/up packed 溢出")?, "MXFP4 grouped decode gate/up packed")?;
    validate_resident(gate_up_scales, device_id, gate_up_rows.checked_mul(hidden / 32).ok_or("MXFP4 grouped decode gate/up scales 溢出")?, "MXFP4 grouped decode gate/up scales")?;
    validate_resident(down_packed, device_id, down_rows.checked_mul(intermediate / 2).ok_or("MXFP4 grouped decode down packed 溢出")?, "MXFP4 grouped decode down packed")?;
    validate_resident(down_scales, device_id, down_rows.checked_mul(intermediate / 32).ok_or("MXFP4 grouped decode down scales 溢出")?, "MXFP4 grouped decode down scales")?;
    if let Some(shared) = shared {
        validate_resident(shared, device_id, input_bytes, "MXFP4 grouped decode shared")?;
    }
    set_device(device_id)?;
    let functions = moe_prefill_functions(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, input_bytes)?;
    let route_bytes = route_count.checked_mul(4).ok_or("MXFP4 grouped decode route workspace 溢出")?;
    let expert_bytes = expert_count.checked_mul(4).ok_or("MXFP4 grouped decode expert workspace 溢出")?;
    let offset_bytes = expert_count.checked_add(1).and_then(|n| n.checked_mul(4)).ok_or("MXFP4 grouped decode offset workspace 溢出")?;
    let activated_bytes = route_count.checked_mul(intermediate).and_then(|n| n.checked_mul(4)).ok_or("MXFP4 grouped decode activation 溢出")?;
    let route_output_bytes = route_count.checked_mul(hidden).and_then(|n| n.checked_mul(4)).ok_or("MXFP4 grouped decode route output 溢出")?;
    with_deferred_tensor_workspace(device_id, &[activated_bytes, route_bytes, route_bytes, offset_bytes, expert_bytes, 4, expert_bytes, expert_bytes, route_bytes, route_output_bytes], |workspace| {
        let activated = workspace.buffer(0);
        let token_ids = workspace.buffer(1);
        let grouped_weights = workspace.buffer(2);
        let offsets = workspace.buffer(3);
        let active_experts = workspace.buffer(4);
        let active_count = workspace.buffer(5);
        let counts = workspace.buffer(6);
        let cursors = workspace.buffer(7);
        let route_to_grouped = workspace.buffer(8);
        let route_output = workspace.buffer(9);
        let mut d_route_ids = route_ids.pointer;
        let mut d_route_weights = route_weights.pointer;
        let mut d_token_ids = token_ids.pointer;
        let mut d_grouped_weights = grouped_weights.pointer;
        let mut d_offsets = offsets.pointer;
        let mut d_active_experts = active_experts.pointer;
        let mut d_active_count = active_count.pointer;
        let mut d_counts = counts.pointer;
        let mut d_cursors = cursors.pointer;
        let mut d_route_to_grouped = route_to_grouped.pointer;
        let mut rows_u32 = moe_u32("grouped decode rows", rows)?;
        let mut top_k_u32 = moe_u32("grouped decode top-k", top_k)?;
        let mut expert_count_u32 = moe_u32("grouped decode experts", expert_count)?;
        let mut group_args = [
            (&mut d_route_ids as *mut *mut c_void).cast(),
            (&mut d_route_weights as *mut *mut c_void).cast(),
            (&mut d_token_ids as *mut *mut c_void).cast(),
            (&mut d_grouped_weights as *mut *mut c_void).cast(),
            (&mut d_offsets as *mut *mut c_void).cast(),
            (&mut d_active_experts as *mut *mut c_void).cast(),
            (&mut d_active_count as *mut *mut c_void).cast(),
            (&mut d_counts as *mut *mut c_void).cast(),
            (&mut d_cursors as *mut *mut c_void).cast(),
            (&mut d_route_to_grouped as *mut *mut c_void).cast(),
            (&mut rows_u32 as *mut u32).cast(),
            (&mut top_k_u32 as *mut u32).cast(),
            (&mut expert_count_u32 as *mut u32).cast(),
        ];
        launch_moe_kernel(functions.group_routes_small, 1, 1, 1, 0, &mut group_args, "HIP MXFP4 group device routes")?;
        let mut d_input = input.pointer;
        let mut d_activated = activated.pointer;
        let mut d_gate_up_packed = gate_up_packed.pointer;
        let mut d_gate_up_scales = gate_up_scales.pointer;
        let mut hidden_u32 = moe_u32("grouped decode hidden", hidden)?;
        let mut intermediate_u32 = moe_u32("grouped decode intermediate", intermediate)?;
        let mut limit_f32 = activation_limit;
        let mut gate_args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_token_ids as *mut *mut c_void).cast(),
            (&mut d_offsets as *mut *mut c_void).cast(),
            (&mut d_active_experts as *mut *mut c_void).cast(),
            (&mut d_active_count as *mut *mut c_void).cast(),
            (&mut d_activated as *mut *mut c_void).cast(),
            (&mut d_gate_up_packed as *mut *mut c_void).cast(),
            (&mut d_gate_up_scales as *mut *mut c_void).cast(),
            (&mut hidden_u32 as *mut u32).cast(),
            (&mut intermediate_u32 as *mut u32).cast(),
            (&mut expert_count_u32 as *mut u32).cast(),
            (&mut limit_f32 as *mut f32).cast(),
        ];
        let group_grid = moe_u32("grouped decode group grid", route_count.min(expert_count))?;
        launch_moe_kernel_3d(
            functions.mxfp4_grouped_gate_up,
            moe_u32("grouped decode gate grid", intermediate.div_ceil(64))?,
            moe_u32("grouped decode route grid", rows.div_ceil(128))?,
            group_grid,
            256,
            0,
            &mut gate_args,
            "HIP MXFP4 grouped decode gate/up",
        )?;
        let mut d_route_output = route_output.pointer;
        let mut d_down_packed = down_packed.pointer;
        let mut d_down_scales = down_scales.pointer;
        let mut route_major_output = 1u32;
        let mut down_args = [
            (&mut d_activated as *mut *mut c_void).cast(),
            (&mut d_token_ids as *mut *mut c_void).cast(),
            (&mut d_offsets as *mut *mut c_void).cast(),
            (&mut d_active_experts as *mut *mut c_void).cast(),
            (&mut d_active_count as *mut *mut c_void).cast(),
            (&mut d_grouped_weights as *mut *mut c_void).cast(),
            (&mut d_route_output as *mut *mut c_void).cast(),
            (&mut d_down_packed as *mut *mut c_void).cast(),
            (&mut d_down_scales as *mut *mut c_void).cast(),
            (&mut hidden_u32 as *mut u32).cast(),
            (&mut intermediate_u32 as *mut u32).cast(),
            (&mut expert_count_u32 as *mut u32).cast(),
            (&mut route_major_output as *mut u32).cast(),
        ];
        launch_moe_kernel_3d(
            functions.mxfp4_grouped_down,
            moe_u32("grouped decode down grid", hidden.div_ceil(128))?,
            moe_u32("grouped decode down route grid", rows.div_ceil(128))?,
            group_grid,
            256,
            0,
            &mut down_args,
            "HIP MXFP4 grouped decode down",
        )?;
        let mut d_shared = shared.map_or(ptr::null_mut(), |buffer| buffer.pointer);
        let mut d_output = output.pointer;
        let mut reduce_args = [
            (&mut d_route_output as *mut *mut c_void).cast(),
            (&mut d_route_to_grouped as *mut *mut c_void).cast(),
            (&mut d_route_weights as *mut *mut c_void).cast(),
            (&mut d_shared as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows_u32 as *mut u32).cast(),
            (&mut top_k_u32 as *mut u32).cast(),
            (&mut hidden_u32 as *mut u32).cast(),
        ];
        let elements = rows.checked_mul(hidden).ok_or("MXFP4 grouped decode reduce elements 溢出")?;
        launch_moe_kernel(functions.mxfp4_grouped_down_reduce, moe_u32("grouped decode reduce grid", elements.div_ceil(256))?, 1, 256, 0, &mut reduce_args, "HIP MXFP4 grouped decode fixed-order reduce")
    })?;
    Ok(output)
}

/// MXFP4 小批量融合专家路径：设备侧 token-major top-k route 直接驱动 gate/up，
/// down 按 token 完成加权归并。权重布局与 grouped prefill 完全共用。
#[allow(clippy::too_many_arguments)]
pub fn try_mxfp4_decode_experts_f32(
    device_id: i32,
    input: &DeviceBuffer,
    rows: usize,
    top_k: usize,
    route_ids: &DeviceBuffer,
    route_weights: &DeviceBuffer,
    gate_up_packed: &DeviceBuffer,
    gate_up_scales: &DeviceBuffer,
    down_packed: &DeviceBuffer,
    down_scales: &DeviceBuffer,
    hidden: usize,
    intermediate: usize,
    expert_count: usize,
    activation_limit: f32,
    shared: Option<&DeviceBuffer>,
) -> Result<DeviceBuffer, String> {
    if rows == 0 || top_k == 0 || top_k > 16 || hidden == 0 || intermediate == 0 || expert_count == 0 || !hidden.is_multiple_of(32) || !intermediate.is_multiple_of(32) {
        return Err(format!("ROCm MXFP4 小批量 shape 非法: rows={rows} top_k={top_k} experts={expert_count} hidden={hidden} intermediate={intermediate}"));
    }
    let route_count = rows.checked_mul(top_k).ok_or("MXFP4 小批量 route 数溢出")?;
    validate_resident(input, device_id, rows.checked_mul(hidden).and_then(|n| n.checked_mul(4)).ok_or("MXFP4 小批量 input 溢出")?, "MXFP4 decode input")?;
    validate_resident(route_ids, device_id, route_count.checked_mul(4).ok_or("MXFP4 decode route ids 溢出")?, "MXFP4 decode route ids")?;
    validate_resident(route_weights, device_id, route_count.checked_mul(4).ok_or("MXFP4 decode route weights 溢出")?, "MXFP4 decode route weights")?;
    let gate_up_rows = expert_count.checked_mul(intermediate).and_then(|n| n.checked_mul(2)).ok_or("MXFP4 decode gate/up rows 溢出")?;
    let down_rows = expert_count.checked_mul(hidden).ok_or("MXFP4 decode down rows 溢出")?;
    validate_resident(gate_up_packed, device_id, gate_up_rows.checked_mul(hidden / 2).ok_or("MXFP4 decode gate/up packed 溢出")?, "MXFP4 decode gate/up packed")?;
    validate_resident(gate_up_scales, device_id, gate_up_rows.checked_mul(hidden / 32).ok_or("MXFP4 decode gate/up scales 溢出")?, "MXFP4 decode gate/up scales")?;
    validate_resident(down_packed, device_id, down_rows.checked_mul(intermediate / 2).ok_or("MXFP4 decode down packed 溢出")?, "MXFP4 decode down packed")?;
    validate_resident(down_scales, device_id, down_rows.checked_mul(intermediate / 32).ok_or("MXFP4 decode down scales 溢出")?, "MXFP4 decode down scales")?;
    if let Some(shared) = shared {
        validate_resident(shared, device_id, rows.checked_mul(hidden).and_then(|n| n.checked_mul(4)).ok_or("MXFP4 小批量 shared 溢出")?, "MXFP4 decode shared")?;
    }
    set_device(device_id)?;
    let functions = moe_prefill_functions(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, rows.checked_mul(hidden).and_then(|n| n.checked_mul(4)).ok_or("MXFP4 小批量 output 溢出")?)?;
    let activated_bytes = route_count.checked_mul(intermediate).and_then(|n| n.checked_mul(4)).ok_or("MXFP4 decode activation 溢出")?;
    with_deferred_tensor_workspace(device_id, &[activated_bytes], |workspace| {
        let activated = workspace.buffer(0);
        let mut d_input = input.pointer;
        let mut d_route_ids = route_ids.pointer;
        let mut d_route_weights = route_weights.pointer;
        let mut d_activated = activated.pointer;
        let mut d_output = output.pointer;
        let mut d_gate_up_packed = gate_up_packed.pointer;
        let mut d_gate_up_scales = gate_up_scales.pointer;
        let mut d_down_packed = down_packed.pointer;
        let mut d_down_scales = down_scales.pointer;
        let mut d_shared = shared.map_or(ptr::null_mut(), |buffer| buffer.pointer);
        let mut hidden_u32 = moe_u32("decode hidden", hidden)?;
        let mut intermediate_u32 = moe_u32("decode intermediate", intermediate)?;
        let mut expert_count_u32 = moe_u32("decode expert count", expert_count)?;
        let mut top_k_u32 = moe_u32("decode top-k", top_k)?;
        let rows_u32 = moe_u32("decode rows", rows)?;
        let mut has_shared_u32 = u32::from(shared.is_some());
        let mut limit_f32 = activation_limit;
        let mut gate_args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_route_ids as *mut *mut c_void).cast(),
            (&mut d_activated as *mut *mut c_void).cast(),
            (&mut d_gate_up_packed as *mut *mut c_void).cast(),
            (&mut d_gate_up_scales as *mut *mut c_void).cast(),
            (&mut hidden_u32 as *mut u32).cast(),
            (&mut intermediate_u32 as *mut u32).cast(),
            (&mut expert_count_u32 as *mut u32).cast(),
            (&mut top_k_u32 as *mut u32).cast(),
            (&mut limit_f32 as *mut f32).cast(),
        ];
        let gate_block = top_k.checked_mul(32).ok_or("MXFP4 decode gate/up block 溢出")?;
        launch_moe_kernel(functions.mxfp4_decode_gate_up, intermediate_u32, rows_u32, moe_u32("decode gate/up block", gate_block)?, 0, &mut gate_args, "HIP MXFP4 decode fused gate/up")?;
        let mut down_args = [
            (&mut d_activated as *mut *mut c_void).cast(),
            (&mut d_route_ids as *mut *mut c_void).cast(),
            (&mut d_route_weights as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut d_down_packed as *mut *mut c_void).cast(),
            (&mut d_down_scales as *mut *mut c_void).cast(),
            (&mut d_shared as *mut *mut c_void).cast(),
            (&mut hidden_u32 as *mut u32).cast(),
            (&mut intermediate_u32 as *mut u32).cast(),
            (&mut expert_count_u32 as *mut u32).cast(),
            (&mut top_k_u32 as *mut u32).cast(),
            (&mut has_shared_u32 as *mut u32).cast(),
        ];
        launch_moe_kernel(functions.mxfp4_decode_down, moe_u32("decode down grid", hidden.div_ceil(16))?, rows_u32, 512, 0, &mut down_args, "HIP MXFP4 decode fused down/reduce")
    })?;
    Ok(output)
}

/// 固定专家集合路由(V4 哈希层):对预选 top_k 专家算 sqrt(softplus) 分数并
/// 行内归一;输出 `[N, top_k]` 权重,expert_ids 由调用方透传。
#[allow(clippy::too_many_arguments)]
fn launch_moe_router_selected_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    selected: &DeviceBuffer,
    output: &DeviceBuffer,
    rows: usize,
    columns: usize,
    expert_count: usize,
    top_k: usize,
    scaling: f32,
) -> Result<(), String> {
    if rows == 0 || columns == 0 || expert_count == 0 || top_k == 0 || top_k > 16 {
        return Err(format!("ROCm selected router shape rows={rows} cols={columns} experts={expert_count} top_k={top_k} 非法(top_k<=16)"));
    }
    let input_bytes = rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("ROCm selected router input 溢出")?;
    let weight_bytes = expert_count.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("ROCm selected router weight 溢出")?;
    let selected_bytes = rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("ROCm selected router selected 溢出")?;
    if input.bytes() < input_bytes || weight.bytes() < weight_bytes || selected.bytes() < selected_bytes || output.bytes() < selected_bytes {
        return Err(format!("ROCm selected router bytes input={}/{} weight={}/{} selected={}/{} output={}/{}", input.bytes(), input_bytes, weight.bytes(), weight_bytes, selected.bytes(), selected_bytes, output.bytes(), selected_bytes));
    }
    set_device(device_id)?;
    let functions = moe_prefill_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = weight.pointer;
    let mut d_selected = selected.pointer;
    let mut d_output = output.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "ROCm selected router rows 超过 u32")?;
    let mut columns_u32 = u32::try_from(columns).map_err(|_| "ROCm selected router columns 超过 u32")?;
    let mut expert_count_u32 = u32::try_from(expert_count).map_err(|_| "ROCm selected router experts 超过 u32")?;
    let mut top_k_u32 = u32::try_from(top_k).map_err(|_| "ROCm selected router top_k 超过 u32")?;
    let mut scaling_f32 = scaling;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_weight as *mut *mut c_void).cast(),
        (&mut d_selected as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut expert_count_u32 as *mut u32).cast(),
        (&mut top_k_u32 as *mut u32).cast(),
        (&mut scaling_f32 as *mut f32).cast(),
    ];
    launch_moe_kernel(
        functions.router_selected,
        u32::try_from(rows).map_err(|_| "ROCm selected router grid 超过 u32")?,
        1,
        u32::try_from(top_k.checked_mul(32).ok_or("ROCm selected router block 溢出")?).map_err(|_| "ROCm selected router block 超过 u32")?,
        0,
        &mut arguments,
        "HIP MoE selected router",
    )?;
    Ok(())
}

pub fn try_moe_router_selected_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    selected: &DeviceBuffer,
    rows: usize,
    columns: usize,
    expert_count: usize,
    top_k: usize,
    scaling: f32,
) -> Result<DeviceBuffer, String> {
    let output = DeviceBuffer::allocate(device_id, rows.checked_mul(top_k).and_then(|n| n.checked_mul(4)).ok_or("ROCm selected router output 溢出")?)?;
    launch_moe_router_selected_resident_f32(device_id, input, weight, selected, &output, rows, columns, expert_count, top_k, scaling)?;
    synchronize_device(device_id, "selected router synchronize")?;
    Ok(output)
}

/// cooperative MoE 会在路由 consumer 内切换设备，不能把 route buffer 的
/// 生命周期交给只在单卡 stream 尾部记录 event 的 deferred workspace。
/// 这里返回拥有所有权的 device buffer，由调用方在双卡消费全部提交后释放。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_moe_route_selected_resident_device_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    selected_experts: &[u32],
    rows: usize,
    columns: usize,
    expert_count: usize,
    top_k: usize,
    scaling: f32,
) -> Result<RocmMoeRoute, String> {
    let route_count = rows.checked_mul(top_k).ok_or("ROCm selected route 数溢出")?;
    if selected_experts.len() != route_count {
        return Err(format!("ROCm selected route ids={} 期望 {route_count}", selected_experts.len()));
    }
    let route_bytes = route_count.checked_mul(4).ok_or("ROCm selected route 字节溢出")?;
    let ids = DeviceBuffer::allocate_peer(device_id, route_bytes)?;
    let weights = DeviceBuffer::allocate_peer(device_id, route_bytes)?;
    ids.copy_from_host(unsafe { std::slice::from_raw_parts(selected_experts.as_ptr().cast(), route_bytes) })?;
    launch_moe_router_selected_resident_f32(device_id, input, weight, &ids, &weights, rows, columns, expert_count, top_k, scaling)?;
    Ok(RocmMoeRoute { expert_ids: ids, weights, len: route_count })
}

/// 固定专家 decode 路由保持在设备侧；selected ids 与 route weights 共用
/// deferred route workspace，生命周期覆盖后续 consumer kernel。
#[allow(clippy::too_many_arguments)]
pub(crate) fn with_moe_route_selected_resident_device_f32<R>(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    selected_experts: &[u32],
    rows: usize,
    columns: usize,
    expert_count: usize,
    top_k: usize,
    scaling: f32,
    run: impl FnOnce(&DeviceBuffer, &DeviceBuffer, usize) -> Result<R, String>,
) -> Result<R, String> {
    let route_count = rows.checked_mul(top_k).ok_or("ROCm selected route 数溢出")?;
    if selected_experts.len() != route_count {
        return Err(format!("ROCm selected route ids={} 期望 {route_count}", selected_experts.len()));
    }
    let route_bytes = route_count.checked_mul(4).ok_or("ROCm selected route 字节溢出")?;
    with_deferred_moe_route_workspace(device_id, &[route_bytes, route_bytes], |workspace| {
        let ids = workspace.buffer(0);
        let weights = workspace.buffer(1);
        ids.copy_from_host(unsafe { std::slice::from_raw_parts(selected_experts.as_ptr().cast(), route_bytes) })?;
        launch_moe_router_selected_resident_f32(device_id, input, weight, ids, weights, rows, columns, expert_count, top_k, scaling)?;
        run(ids, weights, route_count)
    })
}

/// MXFP4 专家矩阵 in-kernel 反量化 matmul:`output[N,M] = input[N,K] @ w[M,K]^T`。
///
/// `packed` 为 `[M, K/2]` E2M1 行优先,`scales` 为 `[M, K/32]` E8M0;位序与
/// `weight::format::mxfp4` 的 CPU 解码一致。decode(N=1)与 prefill 多行共用
/// 同一 kernel,权重以 4bit 原始形态驻留,显存为 BF16 的 1/4。
pub fn try_mxfp4_matmul_resident_f32(device_id: i32, input: &DeviceBuffer, packed: &DeviceBuffer, scales: &DeviceBuffer, input_rows: usize, columns: usize, weight_rows: usize) -> Result<DeviceBuffer, String> {
    if input_rows == 0 || columns == 0 || weight_rows == 0 || !columns.is_multiple_of(32) {
        return Err(format!("ROCm mxfp4 shape rows={input_rows} cols={columns} weight_rows={weight_rows} 非法(列须为 32 的倍数)"));
    }
    let input_bytes = input_rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("ROCm mxfp4 input 溢出")?;
    let packed_bytes = weight_rows.checked_mul(columns / 2).ok_or("ROCm mxfp4 packed 溢出")?;
    let scale_bytes = weight_rows.checked_mul(columns / 32).ok_or("ROCm mxfp4 scales 溢出")?;
    if input.device_id() != device_id || packed.device_id() != device_id || scales.device_id() != device_id {
        return Err("ROCm mxfp4 输入 buffer 跨 device".to_owned());
    }
    if input.bytes() != input_bytes || packed.bytes() != packed_bytes || scales.bytes() != scale_bytes {
        return Err(format!("ROCm mxfp4 bytes input={}/{} packed={}/{} scales={}/{}", input.bytes(), input_bytes, packed.bytes(), packed_bytes, scales.bytes(), scale_bytes));
    }
    set_device(device_id)?;
    let output_elements = input_rows.checked_mul(weight_rows).ok_or("ROCm mxfp4 output 溢出")?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("ROCm mxfp4 output 字节溢出")?)?;
    let functions = moe_prefill_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_packed = packed.pointer;
    let mut d_scales = scales.pointer;
    let mut d_output = output.pointer;
    let mut input_rows_u32 = moe_u32("mxfp4 input rows", input_rows)?;
    let mut columns_u32 = moe_u32("mxfp4 columns", columns)?;
    let mut weight_rows_u32 = moe_u32("mxfp4 weight rows", weight_rows)?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_packed as *mut *mut c_void).cast(),
        (&mut d_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut input_rows_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut weight_rows_u32 as *mut u32).cast(),
    ];
    launch_moe_kernel(functions.mxfp4_matmul, weight_rows_u32, 1, 32, 0, &mut arguments, "hip mxfp4 matmul")?;
    Ok(output)
}

pub(crate) fn try_moe_route_resident_device_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: &DeviceBuffer,
    rows: usize,
    columns: usize,
    experts: usize,
    top_k: usize,
    scoring: u32,
    scaling: f32,
) -> Result<RocmMoeRoute, String> {
    let logits_bytes = rows.checked_mul(experts).and_then(|elements| elements.checked_mul(4)).ok_or("ROCm MoE logits 字节溢出")?;
    let route_bytes = rows.checked_mul(top_k).and_then(|elements| elements.checked_mul(4)).ok_or("ROCm MoE route 字节溢出")?;
    let logits = DeviceBuffer::allocate(device_id, logits_bytes)?;
    // cooperative consumer 会跨卡读取路由；直接写入显式池，避免先写
    // stream-ordered 临时块再 deferred D2D 时源生命周期与 peer event 交错。
    let ids = DeviceBuffer::allocate_peer(device_id, route_bytes)?;
    let weights = DeviceBuffer::allocate_peer(device_id, route_bytes)?;
    let len = launch_moe_route_resident_device_f32(device_id, input, weight, bias, rows, columns, experts, top_k, scoring, scaling, &logits, &ids, &weights)?;
    Ok(RocmMoeRoute { expert_ids: ids, weights, len })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn with_moe_route_resident_device_f32<R>(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: &DeviceBuffer,
    rows: usize,
    columns: usize,
    experts: usize,
    top_k: usize,
    scoring: u32,
    scaling: f32,
    run: impl FnOnce(&DeviceBuffer, &DeviceBuffer, usize) -> Result<R, String>,
) -> Result<R, String> {
    let logits_bytes = rows.checked_mul(experts).and_then(|elements| elements.checked_mul(4)).ok_or("ROCm MoE logits 字节溢出")?;
    let route_bytes = rows.checked_mul(top_k).and_then(|elements| elements.checked_mul(4)).ok_or("ROCm MoE route 字节溢出")?;
    with_deferred_moe_route_workspace(device_id, &[logits_bytes, route_bytes, route_bytes], |workspace| {
        let logits = workspace.buffer(0);
        let ids = workspace.buffer(1);
        let weights = workspace.buffer(2);
        let len = launch_moe_route_resident_device_f32(device_id, input, weight, bias, rows, columns, experts, top_k, scoring, scaling, logits, ids, weights)?;
        run(ids, weights, len)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn try_moe_route_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: &DeviceBuffer,
    rows: usize,
    columns: usize,
    experts: usize,
    top_k: usize,
    scoring: u32,
    scaling: f32,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let route = try_moe_route_resident_device_f32(device_id, input, weight, bias, rows, columns, experts, top_k, scoring, scaling)?;
    let mut expert_ids = vec![0_u32; route.len];
    route.expert_ids.copy_to_host(unsafe { std::slice::from_raw_parts_mut(expert_ids.as_mut_ptr().cast(), route.len * 4) })?;
    let mut route_weights = vec![0.0_f32; route.len];
    route.weights.copy_to_host(unsafe { std::slice::from_raw_parts_mut(route_weights.as_mut_ptr().cast(), route.len * 4) })?;
    Ok((expert_ids, route_weights))
}

pub fn try_moe_zeros_resident_f32(device_id: i32, elements: usize) -> Result<DeviceBuffer, String> {
    if elements == 0 {
        return Err("ROCm MoE zero 元素数不能为 0".to_owned());
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, elements.checked_mul(4).ok_or("ROCm MoE zero 字节溢出")?)?;
    let functions = moe_prefill_functions(device_id)?;
    let mut d_output = output.pointer;
    let mut elements_u32 = u32::try_from(elements).map_err(|_| "ROCm MoE zero 元素数超过 u32".to_owned())?;
    let mut arguments = [(&mut d_output as *mut *mut c_void).cast(), (&mut elements_u32 as *mut u32).cast()];
    launch_moe_kernel(functions.zero, elements_u32.div_ceil(256), 1, 256, 0, &mut arguments, "HIP MoE zero")?;
    Ok(output)
}

pub fn try_moe_gather_resident(device_id: i32, input: &DeviceBuffer, input_rows: usize, columns: usize, rows: &[u32]) -> Result<DeviceBuffer, String> {
    if rows.is_empty() || rows.iter().any(|&row| row as usize >= input_rows) {
        return Err("ROCm resident MoE gather rows 非法".to_owned());
    }
    let input_elements = input_rows.checked_mul(columns).ok_or("ROCm MoE gather input 大小溢出")?;
    let bf16_bytes = input_elements.checked_mul(2).ok_or("ROCm MoE gather BF16 input 大小溢出")?;
    let f32_bytes = input_elements.checked_mul(4).ok_or("ROCm MoE gather F32 input 大小溢出")?;
    let bf16 = input.bytes() == bf16_bytes;
    let element_bytes = if bf16 { 2 } else { 4 };
    validate_resident(input, device_id, if bf16 { bf16_bytes } else { f32_bytes }, "MoE gather input")?;
    let elements = rows.len().checked_mul(columns).ok_or("ROCm MoE gather output 大小溢出")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, elements.checked_mul(element_bytes).ok_or("ROCm MoE gather output 字节溢出")?)?;
    let functions = moe_prefill_functions(device_id)?;
    let row_bytes = std::mem::size_of_val(rows);
    with_tensor_workspace(device_id, &[row_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(rows.as_ptr().cast(), row_bytes) })?;
        let mut d_input = input.pointer;
        let mut d_rows = workspace.buffer(0).pointer;
        let mut d_output = output.pointer;
        let mut gathered_rows = u32::try_from(rows.len()).map_err(|_| "ROCm MoE gather rows 超过 u32".to_owned())?;
        let mut columns = u32::try_from(columns).map_err(|_| "ROCm MoE gather columns 超过 u32".to_owned())?;
        let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_rows as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut gathered_rows as *mut u32).cast(), (&mut columns as *mut u32).cast()];
        let function = if bf16 { functions.gather_bf16 } else { functions.gather };
        launch_moe_kernel(function, u32::try_from(elements.div_ceil(256)).map_err(|_| "ROCm MoE gather grid 超过 u32".to_owned())?, 1, 256, 0, &mut arguments, if bf16 { "HIP MoE gather BF16" } else { "HIP MoE gather F32" })
    })?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_moe_scatter_add_resident_f32(device_id: i32, output: &DeviceBuffer, output_rows: usize, input: &DeviceBuffer, source_rows: usize, columns: usize, rows: &[u32], weights: &[f32]) -> Result<(), String> {
    if source_rows == 0 || rows.len() != source_rows || weights.len() != source_rows || rows.iter().any(|&row| row as usize >= output_rows) {
        return Err("ROCm resident MoE scatter rows/weights 非法".to_owned());
    }
    validate_resident(output, device_id, output_rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("ROCm MoE scatter output 大小溢出")?, "MoE scatter output")?;
    let elements = source_rows.checked_mul(columns).ok_or("ROCm MoE scatter input 大小溢出")?;
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("ROCm MoE scatter input 字节溢出")?, "MoE scatter input")?;
    set_device(device_id)?;
    let functions = moe_prefill_functions(device_id)?;
    let row_bytes = std::mem::size_of_val(rows);
    let weight_bytes = std::mem::size_of_val(weights);
    with_tensor_workspace(device_id, &[row_bytes, weight_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(rows.as_ptr().cast(), row_bytes) })?;
        workspace.buffer(1).copy_from_host(unsafe { std::slice::from_raw_parts(weights.as_ptr().cast(), weight_bytes) })?;
        let mut d_output = output.pointer;
        let mut d_input = input.pointer;
        let mut d_rows = workspace.buffer(0).pointer;
        let mut d_weights = workspace.buffer(1).pointer;
        let mut source_rows = u32::try_from(source_rows).map_err(|_| "ROCm MoE scatter rows 超过 u32".to_owned())?;
        let mut columns = u32::try_from(columns).map_err(|_| "ROCm MoE scatter columns 超过 u32".to_owned())?;
        let mut arguments = [
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_rows as *mut *mut c_void).cast(),
            (&mut d_weights as *mut *mut c_void).cast(),
            (&mut source_rows as *mut u32).cast(),
            (&mut columns as *mut u32).cast(),
        ];
        launch_moe_kernel(functions.scatter_add, u32::try_from(elements.div_ceil(256)).map_err(|_| "ROCm MoE scatter grid 超过 u32".to_owned())?, 1, 256, 0, &mut arguments, "HIP MoE scatter add")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_grouped_experts_matches_cpu_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[fp8-grouped] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        // hidden=128/intermediate=128 走 gate/up 与 down 的向量快路径;
        // 70 非 8 对齐只触发 down 慢路径,66 非 16 对齐只触发 gate/up 慢路径。
        fp8_grouped_experts_oracle_case(128, 128);
        fp8_grouped_experts_oracle_case(128, 70);
        fp8_grouped_experts_oracle_case(66, 128);
    }

    /// 权重解码语义与向量快路径一致:E4M3 在 BF16 中精确,scale 乘法发生在
    /// F32 dot 之后;输入与 activated 仍按 kernel 行为量化到 BF16。
    fn fp8_grouped_experts_oracle_case(hidden: usize, intermediate: usize) {
        let (device_id, rows, experts, top_k) = (0, 2usize, 3usize, 2usize);
        // 覆盖零、normal、负数与 E4M3 subnormal(0x01/0x03/0x07/0x83)。
        let codes = [0x00_u8, 0x01, 0x03, 0x07, 0x20, 0x28, 0x30, 0x38, 0x40, 0x83, 0xa8, 0xb0, 0xb8];
        let matrix = |expert: usize, kind: usize, matrix_rows: usize, columns: usize| {
            let values = (0..matrix_rows * columns).map(|index| codes[(index * 5 + expert * 3 + kind) % codes.len()]).collect::<Vec<_>>();
            let scale_values = (0..matrix_rows.div_ceil(128) * columns.div_ceil(128)).map(|index| 0.015625_f32 * (1.0 + expert as f32 * 0.25 + kind as f32 * 0.125 + index as f32 * 0.0625)).collect::<Vec<_>>();
            let scales = scale_values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            let mut decoded = vec![0.0_f32; matrix_rows * columns];
            crate::weight::codec::fp8::decode_fp8_matrix(&values, &scales, matrix_rows, columns, &mut decoded);
            (values, scales, decoded)
        };
        let host = (0..experts).map(|expert| (matrix(expert, 0, intermediate, hidden), matrix(expert, 1, intermediate, hidden), matrix(expert, 2, hidden, intermediate))).collect::<Vec<_>>();
        let resident = host
            .iter()
            .map(|(gate, up, down)| {
                Ok::<_, String>((
                    (DeviceBuffer::upload(device_id, &gate.0)?, DeviceBuffer::upload(device_id, &gate.1)?),
                    (DeviceBuffer::upload(device_id, &up.0)?, DeviceBuffer::upload(device_id, &up.1)?),
                    (DeviceBuffer::upload(device_id, &down.0)?, DeviceBuffer::upload(device_id, &down.1)?),
                ))
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("上传 FP8 experts");
        let metas = resident
            .iter()
            .map(|(gate, up, down)| {
                let weight = |weight: &(DeviceBuffer, DeviceBuffer)| Fp8GroupedWeightMeta { codes: weight.0.device_pointer() as u64, scales: weight.1.device_pointer() as u64 };
                Fp8GroupedExpertMeta { gate: weight(gate), up: weight(up), down: weight(down) }
            })
            .collect::<Vec<_>>();
        let meta_bytes = unsafe { std::slice::from_raw_parts(metas.as_ptr().cast::<u8>(), std::mem::size_of_val(metas.as_slice())) };
        let metas = DeviceBuffer::upload(device_id, meta_bytes).expect("上传 FP8 metas");
        let input = (0..rows * hidden).map(|index| index as f32 * 0.0007 - 0.04).collect::<Vec<f32>>();
        let route_ids = [0_u32, 2, 1, 0];
        let route_weights = [0.35_f32, 0.65, 0.2, 0.8];
        let shared = (0..rows * hidden).map(|index| index as f32 * 0.0001 - 0.01).collect::<Vec<f32>>();
        let residual = (0..rows * hidden).map(|index| 0.02 - index as f32 * 0.00003).collect::<Vec<f32>>();
        let mut expected = shared.iter().zip(&residual).map(|(left, right)| left + right).collect::<Vec<f32>>();
        for route in 0..rows * top_k {
            let token = route / top_k;
            let expert = route_ids[route] as usize;
            let (gate, up, down) = &host[expert];
            let mut activated = vec![0.0_f32; intermediate];
            for neuron in 0..intermediate {
                let mut gate_sum = 0.0_f32;
                let mut up_sum = 0.0_f32;
                for column in 0..hidden {
                    let value = half::bf16::from_f32(input[token * hidden + column]).to_f32();
                    gate_sum += value * gate.2[neuron * hidden + column];
                    up_sum += value * up.2[neuron * hidden + column];
                }
                activated[neuron] = half::bf16::from_f32(gate_sum / (1.0 + (-gate_sum).exp()) * up_sum).to_f32();
            }
            for column in 0..hidden {
                let mut value = 0.0_f32;
                for inner in 0..intermediate {
                    value += activated[inner] * down.2[column * intermediate + inner];
                }
                expected[token * hidden + column] += value * route_weights[route];
            }
        }
        let input = DeviceBuffer::upload_f32(device_id, &input).expect("上传 FP8 input");
        let route_ids_bytes = unsafe { std::slice::from_raw_parts(route_ids.as_ptr().cast::<u8>(), std::mem::size_of_val(&route_ids)) };
        let route_ids = DeviceBuffer::upload(device_id, route_ids_bytes).expect("上传 route ids");
        let route_weights = DeviceBuffer::upload_f32(device_id, &route_weights).expect("上传 route weights");
        let shared = DeviceBuffer::upload_f32(device_id, &shared).expect("上传 shared");
        let residual = DeviceBuffer::upload_f32(device_id, &residual).expect("上传 residual");
        let actual = try_fp8_grouped_experts_f32(device_id, &input, rows, hidden, intermediate, top_k, experts, &route_ids, &route_weights, rows * top_k, &metas, Some(&shared), Some(&residual))
            .expect("FP8 grouped")
            .download_f32(rows * hidden)
            .expect("下载 FP8 grouped output");
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() <= 1.0e-4 + expected.abs() * 2.0e-3, "hidden={hidden} intermediate={intermediate} index={index} actual={actual} expected={expected}");
        }
    }

    /// GPU in-kernel 反量化 matmul 必须与 Mxfp4Matrix CPU 解码逐元素一致。
    #[test]
    fn mxfp4_grouped_experts_matches_per_expert_reference() {
        if !super::super::is_hip_available() {
            eprintln!("[mxfp4-grouped] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let device_id = 0;
        let experts = 4usize;
        let hidden = 128usize;
        let intermediate = 64usize;
        let tokens = 6usize;
        let limit = 10.0f32;
        // 权重:每专家 gate/up 各 [I,H]、down [H,I];非平凡 scale。
        let mut gate_up_packed = vec![0u8; 2 * experts * intermediate * hidden / 2];
        let mut gate_up_scales = vec![0u8; 2 * experts * intermediate * hidden / 32];
        let mut down_packed = vec![0u8; experts * hidden * intermediate / 2];
        let mut down_scales = vec![0u8; experts * hidden * intermediate / 32];
        for index in 0..gate_up_packed.len() {
            gate_up_packed[index] = ((index * 31 + 7) % 256) as u8;
        }
        for index in 0..gate_up_scales.len() {
            gate_up_scales[index] = (120 + (index % 13)) as u8;
        }
        for index in 0..down_packed.len() {
            down_packed[index] = ((index * 17 + 3) % 256) as u8;
        }
        for index in 0..down_scales.len() {
            down_scales[index] = (118 + (index % 19)) as u8;
        }
        let dequant = |packed: &[u8], scales: &[u8], _rows: usize, cols: usize, row: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; cols];
            for col in 0..cols {
                let byte = packed[row * (cols / 2) + col / 2];
                let nibble = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let magnitude = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nibble & 7) as usize];
                let signed = if nibble & 8 == 0 { magnitude } else { -magnitude };
                out[col] = signed * 2.0f32.powi(scales[row * (cols / 32) + col / 32] as i32 - 127);
            }
            out
        };
        let input: Vec<f32> = (0..tokens * hidden).map(|index: usize| (index as f32) * 0.017 - 0.4).collect();
        // 路由:token 0-1 → expert 0,2;token 2-3 → 1,3;token 4-5 → 0,3。
        let route_pairs = [(0u32, 0.3f32), (0, 0.7), (1, 0.4), (1, 0.6), (2, 0.5), (2, 0.5), (3, 0.2), (3, 0.8), (0, 0.6), (3, 0.4), (1, 0.1), (2, 0.9)];
        let mut route_tokens = Vec::new();
        let mut route_weights = Vec::new();
        let mut route_offsets = vec![0u32; experts + 1];
        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); experts];
        for (slot, &(expert, weight)) in route_pairs.iter().enumerate() {
            let token = (slot / 2) as u32;
            per_expert[expert as usize].push((token, weight));
        }
        for expert in 0..experts {
            for &(token, weight) in &per_expert[expert] {
                route_tokens.push(token);
                route_weights.push(weight);
            }
            route_offsets[expert + 1] = route_tokens.len() as u32;
        }
        // reference:逐 route 行 gate/up/silu*up/down,按 token 累加。
        let mut expected = vec![0.0f32; tokens * hidden];
        for expert in 0..experts {
            let gate_rows: Vec<Vec<f32>> = (0..intermediate).map(|row| dequant(&gate_up_packed[expert * intermediate * hidden / 2..], &gate_up_scales[expert * intermediate * hidden / 32..], intermediate, hidden, row)).collect();
            let up_rows: Vec<Vec<f32>> =
                (0..intermediate).map(|row| dequant(&gate_up_packed[(experts + expert) * intermediate * hidden / 2..], &gate_up_scales[(experts + expert) * intermediate * hidden / 32..], intermediate, hidden, row)).collect();
            let down_rows: Vec<Vec<f32>> = (0..hidden).map(|row| dequant(&down_packed[expert * hidden * intermediate / 2..], &down_scales[expert * hidden * intermediate / 32..], hidden, intermediate, row)).collect();
            for &(token, weight) in &per_expert[expert] {
                let token = token as usize;
                let mut activated = vec![0.0f32; intermediate];
                for col in 0..intermediate {
                    let mut gate = 0.0f32;
                    let mut up = 0.0f32;
                    for k in 0..hidden {
                        gate += gate_rows[col][k] * input[token * hidden + k];
                        up += up_rows[col][k] * input[token * hidden + k];
                    }
                    let gate = gate.min(limit);
                    let up = up.max(-limit).min(limit);
                    activated[col] = gate * (1.0 / (1.0 + (-gate).exp())) * up;
                }
                for col in 0..hidden {
                    let mut value = 0.0f32;
                    for k in 0..intermediate {
                        value += down_rows[col][k] * activated[k];
                    }
                    expected[token * hidden + col] += value * weight;
                }
            }
        }
        let input_buffer = DeviceBuffer::upload_f32(device_id, &input).unwrap();
        let (gate_up_packed, gate_up_scales) = preshuffle_mxfp4_16x16(&gate_up_packed, &gate_up_scales, 2 * experts * intermediate, hidden);
        let (down_packed, down_scales) = preshuffle_mxfp4_16x16(&down_packed, &down_scales, experts * hidden, intermediate);
        let gate_up_packed_buffer = DeviceBuffer::upload(device_id, &gate_up_packed).unwrap();
        let gate_up_scales_buffer = DeviceBuffer::upload(device_id, &gate_up_scales).unwrap();
        let down_packed_buffer = DeviceBuffer::upload(device_id, &down_packed).unwrap();
        let down_scales_buffer = DeviceBuffer::upload(device_id, &down_scales).unwrap();
        let output = try_mxfp4_grouped_experts_f32(
            device_id,
            &input_buffer,
            tokens,
            &route_tokens,
            &route_weights,
            &route_offsets,
            &gate_up_packed_buffer,
            &gate_up_scales_buffer,
            &down_packed_buffer,
            &down_scales_buffer,
            hidden,
            intermediate,
            experts,
            limit,
        )
        .expect("grouped");
        let actual = output.download_f32(tokens * hidden).unwrap();
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            // gfx11 MXFP4 prefill 先把 activation/weight tile 转为 BF16，再进入
            // WMMA；与 F32 reference 按项目统一 BF16 跨后端阈值验收。
            assert!((actual - expected).abs() <= expected.abs() * 1.0e-2 + 1.0e-2, "index={index} actual={actual} expected={expected}");
        }
    }

    #[test]
    fn mxfp4_decode_experts_matches_cpu_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[mxfp4-decode] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let (device_id, rows, top_k, experts, hidden, intermediate) = (0, 2usize, 3usize, 4usize, 128usize, 64usize);
        let route_ids = [3u32, 0, 2, 1, 3, 0];
        let route_weights = [0.2f32, 0.5, 0.3, 0.4, 0.35, 0.25];
        let limit = 8.0f32;
        let gate_up_packed: Vec<u8> = (0..2 * experts * intermediate * hidden / 2).map(|index| (index * 31 + 7) as u8).collect();
        let gate_up_scales: Vec<u8> = (0..2 * experts * intermediate * hidden / 32).map(|index| 120 + (index % 11) as u8).collect();
        let down_packed: Vec<u8> = (0..experts * hidden * intermediate / 2).map(|index| (index * 17 + 3) as u8).collect();
        let down_scales: Vec<u8> = (0..experts * hidden * intermediate / 32).map(|index| 119 + (index % 13) as u8).collect();
        let input: Vec<f32> = (0..rows * hidden).map(|index| index as f32 * 0.007 - 0.6).collect();
        let value = |packed: &[u8], scales: &[u8], columns: usize, row: usize, column: usize| {
            let byte = packed[row * (columns / 2) + column / 2];
            let code = if column % 2 == 0 { byte & 15 } else { byte >> 4 };
            let magnitude = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
            let signed = if code & 8 == 0 { magnitude } else { -magnitude };
            signed * 2.0f32.powi(scales[row * (columns / 32) + column / 32] as i32 - 127)
        };
        let mut expected = vec![0.0f32; rows * hidden];
        for route in 0..rows * top_k {
            let expert = route_ids[route] as usize;
            let route_weight = route_weights[route];
            let token = route / top_k;
            let input = &input[token * hidden..(token + 1) * hidden];
            let mut activated = vec![0.0f32; intermediate];
            for row in 0..intermediate {
                let gate_row = expert * intermediate + row;
                let up_row = (experts + expert) * intermediate + row;
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for column in 0..hidden {
                    gate += value(&gate_up_packed, &gate_up_scales, hidden, gate_row, column) * input[column];
                    up += value(&gate_up_packed, &gate_up_scales, hidden, up_row, column) * input[column];
                }
                let gate = gate.min(limit);
                let up = up.clamp(-limit, limit);
                activated[row] = gate * (1.0 / (1.0 + (-gate).exp())) * up;
            }
            for row in 0..hidden {
                let mut sum = 0.0f32;
                for column in 0..intermediate {
                    sum += value(&down_packed, &down_scales, intermediate, expert * hidden + row, column) * activated[column];
                }
                expected[token * hidden + row] += sum * route_weight;
            }
        }
        let input = DeviceBuffer::upload_f32(device_id, &input).unwrap();
        let route_ids_bytes = unsafe { std::slice::from_raw_parts(route_ids.as_ptr().cast(), std::mem::size_of_val(&route_ids)) };
        let route_ids = DeviceBuffer::upload(device_id, route_ids_bytes).unwrap();
        let route_weights = DeviceBuffer::upload_f32(device_id, &route_weights).unwrap();
        let (gate_up_packed, gate_up_scales) = preshuffle_mxfp4_16x16(&gate_up_packed, &gate_up_scales, 2 * experts * intermediate, hidden);
        let (down_packed, down_scales) = preshuffle_mxfp4_16x16(&down_packed, &down_scales, experts * hidden, intermediate);
        let gate_up_packed = DeviceBuffer::upload(device_id, &gate_up_packed).unwrap();
        let gate_up_scales = DeviceBuffer::upload(device_id, &gate_up_scales).unwrap();
        let down_packed = DeviceBuffer::upload(device_id, &down_packed).unwrap();
        let down_scales = DeviceBuffer::upload(device_id, &down_scales).unwrap();
        let output = try_mxfp4_decode_experts_f32(device_id, &input, rows, top_k, &route_ids, &route_weights, &gate_up_packed, &gate_up_scales, &down_packed, &down_scales, hidden, intermediate, experts, limit, None).expect("fused decode");
        let actual = output.download_f32(rows * hidden).unwrap();
        let grouped = try_mxfp4_grouped_decode_experts_f32(device_id, &input, rows, top_k, &route_ids, &route_weights, &gate_up_packed, &gate_up_scales, &down_packed, &down_scales, hidden, intermediate, experts, limit, None)
            .expect("grouped decode")
            .download_f32(rows * hidden)
            .unwrap();
        for repeat in 0..8 {
            let repeated = try_mxfp4_grouped_decode_experts_f32(device_id, &input, rows, top_k, &route_ids, &route_weights, &gate_up_packed, &gate_up_scales, &down_packed, &down_scales, hidden, intermediate, experts, limit, None)
                .expect("grouped decode repeat")
                .download_f32(rows * hidden)
                .unwrap();
            assert_eq!(repeated, grouped, "grouped decode 第 {repeat} 次输出发生漂移");
        }
        let shared = (0..rows * hidden).map(|index| index as f32 * 0.001 - 0.03).collect::<Vec<_>>();
        let shared = DeviceBuffer::upload_f32(device_id, &shared).unwrap();
        let expected_shared = super::super::try_add_resident_f32(device_id, &shared, &output, rows * hidden, 1.0).unwrap().download_f32(rows * hidden).unwrap();
        let actual_shared = try_mxfp4_decode_experts_f32(device_id, &input, rows, top_k, &route_ids, &route_weights, &gate_up_packed, &gate_up_scales, &down_packed, &down_scales, hidden, intermediate, experts, limit, Some(&shared))
            .expect("fused decode + shared")
            .download_f32(rows * hidden)
            .unwrap();
        assert_eq!(actual_shared, expected_shared);
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() <= expected.abs() * 2.0e-4 + 2.0e-4, "index={index} actual={actual} expected={expected}");
        }
        for (index, (actual, expected)) in grouped.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() <= expected.abs() * 1.0e-2 + 1.0e-2, "grouped index={index} actual={actual} expected={expected}");
        }
    }

    #[test]
    fn selected_router_real_shape_matches_cpu() {
        if !super::super::is_hip_available() {
            eprintln!("[selected-router-real] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        // 真实 DeepSeek hash 层形状:rows=1、columns=4096、experts=256、top_k=6。
        let (device_id, rows, columns, experts, top_k) = (0, 1usize, 4096usize, 256usize, 6usize);
        let input: Vec<f32> = (0..rows * columns).map(|index: usize| (index as f32 * 0.003) - 0.5).collect();
        let weight: Vec<f32> = (0..experts * columns).map(|index: usize| (index as f32 * 0.0007) - 0.3).collect();
        let selected: Vec<u32> = vec![255, 0, 128, 7, 200, 33];
        let expected = crate::moe::routing::route_sqrt_softplus_selected(&input, rows, columns, &weight, experts, &selected, top_k, 1.5).expect("oracle");
        let input_buf = DeviceBuffer::upload_f32(device_id, &input).unwrap();
        let weight_buf = DeviceBuffer::upload_f32(device_id, &weight).unwrap();
        let selected_bytes = unsafe { std::slice::from_raw_parts(selected.as_ptr().cast::<u8>(), selected.len() * 4) };
        let selected_buf = DeviceBuffer::upload(device_id, selected_bytes).unwrap();
        let output = try_moe_router_selected_resident_f32(device_id, &input_buf, &weight_buf, &selected_buf, rows, columns, experts, top_k, 1.5).unwrap();
        let actual = output.download_f32(rows * top_k).unwrap();
        for (index, (actual, expected)) in actual.iter().zip(expected.weights.iter()).enumerate() {
            assert!((actual - expected).abs() <= 1.0e-5 + expected.abs() * 1.0e-4, "index={index} actual={actual} expected={expected}");
        }
    }

    #[test]
    fn selected_router_matches_cpu_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[selected-router] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let device_id = 0;
        let rows = 5;
        let columns = 128;
        let experts = 64;
        let top_k = 6;
        let input: Vec<f32> = (0..rows * columns).map(|index: usize| (index as f32) * 0.011 - 1.0).collect();
        let weight: Vec<f32> = (0..experts * columns).map(|index: usize| (index as f32) * 0.003 - 0.5).collect();
        let selected: Vec<u32> = (0..rows * top_k).map(|index: usize| ((index * 7 + 3) % experts) as u32).collect();
        let expected = crate::moe::routing::route_sqrt_softplus_selected(&input, rows, columns, &weight, experts, &selected, top_k, 1.5).expect("oracle");

        let input_buf = DeviceBuffer::upload_f32(device_id, &input).unwrap();
        let weight_buf = DeviceBuffer::upload_f32(device_id, &weight).unwrap();
        let selected_bytes = unsafe { std::slice::from_raw_parts(selected.as_ptr().cast::<u8>(), selected.len() * 4) };
        let selected_buf = DeviceBuffer::upload(device_id, selected_bytes).unwrap();
        let output = try_moe_router_selected_resident_f32(device_id, &input_buf, &weight_buf, &selected_buf, rows, columns, experts, top_k, 1.5).unwrap();
        let actual = output.download_f32(rows * top_k).unwrap();
        assert_eq!(actual.len(), expected.weights.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.weights.iter()).enumerate() {
            assert!((actual - expected).abs() <= 1.0e-5 + expected.abs() * 1.0e-4, "index={index} actual={actual} expected={expected}");
        }
    }

    /// GLM5.3 decode 路由形状:288 专家(合作版 lane 多轮扫描)、top_k=8。
    /// 默认 precise router 的权重是 F32,CPU 参考同精度计算 logits。
    #[test]
    fn router_topk_288_experts_matches_cpu_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[router-topk-288] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let (device_id, rows, columns, top_k, scaling) = (0, 1usize, 512usize, 8usize, 2.5_f32);
        for experts in [288usize, 96usize] {
            let input: Vec<f32> = (0..rows * columns).map(|index: usize| (index as f32 * 0.021).sin()).collect();
            let weight: Vec<f32> = (0..experts * columns).map(|index: usize| (index as f32 * 0.017).cos()).collect();
            let bias: Vec<f32> = (0..experts).map(|index: usize| (index as f32 * 0.05).sin()).collect();
            let logits: Vec<f32> = (0..experts).map(|expert| input.iter().zip(&weight[expert * columns..(expert + 1) * columns]).map(|(&x, &w)| x * w).sum()).collect();
            let expected = crate::moe::routing::route_sqrt_softplus_bias_logits(&logits, &bias, top_k, scaling).expect("oracle");
            let input_buf = DeviceBuffer::upload_f32(device_id, &input).unwrap();
            let weight_buf = DeviceBuffer::upload_f32(device_id, &weight).unwrap();
            let bias_bytes = unsafe { std::slice::from_raw_parts(bias.as_ptr().cast::<u8>(), bias.len() * 4) };
            let bias_buf = DeviceBuffer::upload(device_id, bias_bytes).unwrap();
            let (ids, weights) = try_moe_route_resident_f32(device_id, &input_buf, &weight_buf, &bias_buf, rows, columns, experts, top_k, 2, scaling).expect("router");
            assert_eq!(ids, expected.experts, "experts={experts} ids {ids:?} != {:?}", expected.experts);
            for (index, (actual, expected)) in weights.iter().zip(expected.weights.iter()).enumerate() {
                assert!((actual - expected).abs() <= 1.0e-5 + expected.abs() * 1.0e-3, "experts={experts} index={index} actual={actual} expected={expected}");
            }
        }
    }

    #[test]
    fn mxfp4_matmul_matches_cpu_decode_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[mxfp4_matmul] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let device_id = 0;
        let rows = 96;
        let columns = 256;
        let inputs = 3;
        // 随机字节做 packed/scales,含符号 nibble 与非平凡 E8M0 指数。
        let packed: Vec<u8> = (0..rows * columns / 2).map(|index: usize| (index.wrapping_mul(197).wrapping_add(51)) as u8).collect();
        let scales: Vec<u8> = (0..rows * columns / 32).map(|index: usize| 118 + (index % 17) as u8).collect();
        let input: Vec<f32> = (0..inputs * columns).map(|index: usize| (index as f32) * 0.007 - 0.5).collect();

        let matrix = crate::weight::format::mxfp4::Mxfp4Matrix::new(rows, columns, packed.clone(), scales.clone()).expect("matrix");
        let decoded = matrix.decode().expect("oracle");
        let mut expected = vec![0.0f32; inputs * rows];
        for r in 0..inputs {
            for m in 0..rows {
                let mut sum = 0.0;
                for k in 0..columns {
                    sum += decoded[m * columns + k] * input[r * columns + k];
                }
                expected[r * rows + m] = sum;
            }
        }

        let input_buf = DeviceBuffer::upload_f32(device_id, &input).expect("upload input");
        let packed_buf = DeviceBuffer::upload(device_id, &packed).expect("upload packed");
        let scales_buf = DeviceBuffer::upload(device_id, &scales).expect("upload scales");
        let output = try_mxfp4_matmul_resident_f32(device_id, &input_buf, &packed_buf, &scales_buf, inputs, columns, rows).expect("matmul");
        let actual = output.download_f32(inputs * rows).expect("readback");
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() <= expected.abs() * 1.0e-5 + 1.0e-6, "index={index} actual={actual} expected={expected}");
        }
    }
}
