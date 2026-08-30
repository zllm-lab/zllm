pub(super) const SOURCE: &str = include_str!("ct_grouped/source.hip");

use super::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct CtGroupedWeightMeta {
    pub(super) packed: u64,
    pub(super) scales: u64,
    pub(super) group_size: u32,
    pub(super) scale_dtype: u32,
    pub(super) format: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct CtGroupedExpertMeta {
    pub(super) gate: CtGroupedWeightMeta,
    pub(super) up: CtGroupedWeightMeta,
    pub(super) down: CtGroupedWeightMeta,
}

pub(super) fn upload_pod<T>(device_id: i32, values: &[T]) -> Result<DeviceBuffer, String> {
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) };
    // 路由 POD 只活到本次 grouped kernel 结束；走 stream-ordered 临时分配，避免逐层全设备同步。
    let buffer = DeviceBuffer::allocate(device_id, bytes.len())?;
    buffer.copy_from_host(bytes)?;
    Ok(buffer)
}

type CtGroupedMetaCache = std::collections::HashMap<(i32, Vec<u64>), std::sync::Arc<DeviceBuffer>>;

/// GGUF expert metas 的内容寻址设备缓存(对齐 CT resident_grouped_metas)：
/// 常驻 expert 指针稳定，同一层重复调用零上传。
pub(crate) fn resident_gguf_grouped_metas(device_id: i32, metas: &[super::GgufGroupedExpertMeta]) -> Result<std::sync::Arc<DeviceBuffer>, String> {
    let mut identity = Vec::with_capacity(metas.len() * 4);
    for expert in metas {
        identity.push(expert.gate);
        identity.push(expert.up);
        identity.push(expert.down);
        identity.push((u64::from(expert.gate_type) << 40) | (u64::from(expert.up_type) << 20) | u64::from(expert.down_type));
    }
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(i32, Vec<u64>), std::sync::Arc<DeviceBuffer>>>> = std::sync::OnceLock::new();
    let mut cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new())).lock().map_err(|_| "GGUF grouped expert metadata cache 已损坏".to_owned())?;
    let key = (device_id, identity);
    if let Some(buffer) = cache.get(&key) {
        return Ok(buffer.clone());
    }
    let buffer = std::sync::Arc::new(upload_pod(device_id, metas)?);
    cache.insert(key, buffer.clone());
    Ok(buffer)
}

fn resident_grouped_metas(device_id: i32, metas: &[CtGroupedExpertMeta]) -> Result<std::sync::Arc<DeviceBuffer>, String> {
    let mut identity = Vec::with_capacity(metas.len() * 9);
    for expert in metas {
        for weight in [&expert.gate, &expert.up, &expert.down] {
            identity.push(weight.packed);
            identity.push(weight.scales);
            identity.push(((weight.group_size as u64) << 32) | ((weight.format as u64) << 16) | weight.scale_dtype as u64);
        }
    }
    static CACHE: std::sync::OnceLock<std::sync::Mutex<CtGroupedMetaCache>> = std::sync::OnceLock::new();
    let mut cache = CACHE.get_or_init(|| std::sync::Mutex::new(CtGroupedMetaCache::new())).lock().map_err(|_| "ROCm grouped expert metadata cache 已损坏".to_owned())?;
    let key = (device_id, identity);
    if let Some(buffer) = cache.get(&key) {
        return Ok(buffer.clone());
    }
    let buffer = std::sync::Arc::new(upload_pod(device_id, metas)?);
    cache.insert(key, buffer.clone());
    Ok(buffer)
}

fn ct_grouped_expert_metas(device_id: i32, experts: &[CtGroupedExpertRef<'_>]) -> Result<Vec<CtGroupedExpertMeta>, String> {
    let meta = |weight: &CtGroupedWeightRef<'_>| -> Result<CtGroupedWeightMeta, String> {
        if weight.packed.device_id != device_id || weight.scales.device_id != device_id || weight.group_size == 0 {
            return Err("ROCm grouped expert 权重 device/group 不一致".to_owned());
        }
        Ok(CtGroupedWeightMeta {
            packed: weight.packed.pointer as usize as u64,
            scales: weight.scales.pointer as usize as u64,
            group_size: u32::try_from(weight.group_size).map_err(|_| "group_size 超过 u32")?,
            scale_dtype: weight.scale_dtype,
            format: weight.format,
        })
    };
    experts.iter().map(|expert| Ok(CtGroupedExpertMeta { gate: meta(&expert.gate)?, up: meta(&expert.up)?, down: meta(&expert.down)? })).collect()
}

/// 模型加载阶段预热 grouped WMMA 的 device pointer 表；正式 forward 不得
/// 因首次命中某层而再产生 host-to-device metadata 上传。
pub(crate) fn preload_ct_grouped_expert_metas(device_id: i32, experts: &[CtGroupedExpertRef<'_>]) -> Result<(), String> {
    let metas = ct_grouped_expert_metas(device_id, experts)?;
    resident_grouped_metas(device_id, &metas).map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_ct_decode_experts_bf16(
    device_id: i32,
    input: &DeviceBuffer,
    input_rows: usize,
    hidden_size: usize,
    intermediate_size: usize,
    route_ids: &DeviceBuffer,
    route_weights: &DeviceBuffer,
    route_count: usize,
    expert_count: usize,
    all_metas: &DeviceBuffer,
    d_input_bf16: &DeviceBuffer,
    d_activated: &DeviceBuffer,
    epilogue: Option<(&DeviceBuffer, &DeviceBuffer)>,
    integrated_shared: Option<(usize, &DeviceBuffer)>,
    w8: bool,
) -> Result<DeviceBuffer, String> {
    if input_rows == 0 || route_count == 0 || !route_count.is_multiple_of(input_rows) {
        return Err("decode MoE route batch 非法".to_owned());
    }
    let top_k = route_count / input_rows;
    if top_k == 0 || top_k > 16 {
        return Err("decode MoE top-k 非法".to_owned());
    }
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let functions = ct_quantized_functions(device_id)?;
    let input_elements = input_rows.checked_mul(hidden_size).ok_or("decode MoE input 大小溢出")?;
    let output = DeviceBuffer::allocate_reusable(device_id, input_elements.checked_mul(4).ok_or("decode MoE output 大小溢出")?)?;
    let input_is_bf16 = input.bytes() == input_elements.checked_mul(2).ok_or("decode MoE BF16 input 大小溢出")?;

    let gate_started = options().kernel_profile.then(std::time::Instant::now);
    if !input_is_bf16 {
        let mut cast_input = input.pointer;
        let mut cast_output = d_input_bf16.pointer;
        let mut cast_elements = u32::try_from(input_elements).map_err(|_| "decode MoE input 元素超过 u32")?;
        let mut cast_args = [(&mut cast_input as *mut *mut c_void).cast(), (&mut cast_output as *mut *mut c_void).cast(), (&mut cast_elements as *mut u32).cast()];
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe { launch(functions.cast as *mut c_void, cast_elements.div_ceil(256), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), cast_args.as_mut_ptr(), ptr::null_mut()) };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "decode MoE input f32_to_bf16"));
        }
    }
    if options().debug_finite {
        let input_bf16 = if input_is_bf16 { input } else { d_input_bf16 };
        try_validate_finite_resident_range_bf16(device_id, input_bf16, 0, hidden_size).map_err(|error| format!("decode MoE BF16 input 包含非有限值: {error}"))?;
        eprintln!("[decode-moe-finite] device={device_id} stage=input routes={route_count}");
    }

    let mut gate_input = if input_is_bf16 { input.pointer } else { d_input_bf16.pointer };
    let mut gate_route_ids = route_ids.pointer;
    let mut gate_metas = all_metas.pointer;
    let mut gate_output = d_activated.pointer;
    let mut hidden_size_u32 = u32::try_from(hidden_size).map_err(|_| "decode MoE hidden_size 超过 u32")?;
    let mut intermediate_size_u32 = u32::try_from(intermediate_size).map_err(|_| "decode MoE intermediate_size 超过 u32")?;
    let mut route_count_u32 = u32::try_from(route_count).map_err(|_| "decode MoE route_count 超过 u32")?;
    let mut top_k_u32 = u32::try_from(top_k).map_err(|_| "decode MoE top_k 超过 u32")?;
    let mut expert_count_u32 = u32::try_from(expert_count).map_err(|_| "decode MoE expert_count 超过 u32")?;
    let mut shared_expert_u32 = integrated_shared.map_or(Ok(u32::MAX), |(expert, _)| u32::try_from(expert).map_err(|_| "decode MoE shared expert 超过 u32"))?;
    let mut gate_args = [
        (&mut gate_input as *mut *mut c_void).cast(),
        (&mut gate_route_ids as *mut *mut c_void).cast(),
        (&mut gate_metas as *mut *mut c_void).cast(),
        (&mut gate_output as *mut *mut c_void).cast(),
        (&mut hidden_size_u32 as *mut u32).cast(),
        (&mut intermediate_size_u32 as *mut u32).cast(),
        (&mut route_count_u32 as *mut u32).cast(),
        (&mut top_k_u32 as *mut u32).cast(),
        (&mut expert_count_u32 as *mut u32).cast(),
        (&mut shared_expert_u32 as *mut u32).cast(),
    ];
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let gate = if w8 { functions.decode_w8_gate_up } else { functions.decode_gate_up };
        let gate_routes = route_count.checked_add(usize::from(integrated_shared.is_some()) * input_rows).ok_or("decode MoE gate route 数溢出")?;
        let gate_routes = u32::try_from(gate_routes).map_err(|_| "decode MoE gate route 数超过 u32")?;
        let __hip_launch_result = unsafe { launch(gate as *mut c_void, intermediate_size_u32.div_ceil(16), 1, gate_routes, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), gate_args.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "decode MoE fused gate/up"));
    }
    if let Some(started) = gate_started {
        synchronize_device(device_id, "decode MoE gate/up profile")?;
        eprintln!("[rocm-kernel] decode-{}gate-up device={device_id} input_rows={input_rows} routes={route_count} wall={:.6}s", if w8 { "w8-" } else { "" }, started.elapsed().as_secs_f64());
    }
    if options().debug_finite {
        let activated_routes = route_count.checked_add(usize::from(integrated_shared.is_some()) * input_rows).ok_or("decode MoE activated route 数溢出")?;
        let activated_elements = activated_routes.checked_mul(intermediate_size).ok_or("decode MoE activated 元素数溢出")?;
        try_validate_finite_resident_range_bf16(device_id, d_activated, 0, activated_elements).map_err(|error| format!("decode MoE gate/up activated 包含非有限值: {error}"))?;
        eprintln!("[decode-moe-finite] device={device_id} stage=gate_up routes={route_count}");
    }

    let mut down_input = d_activated.pointer;
    let mut down_route_ids = route_ids.pointer;
    let mut down_route_weights = route_weights.pointer;
    let mut down_metas = all_metas.pointer;
    let mut down_output = output.pointer;
    let mut down_shared = epilogue.map_or(ptr::null_mut(), |(shared, _)| shared.pointer);
    let mut down_residual = epilogue.map_or_else(|| integrated_shared.map_or(ptr::null_mut(), |(_, residual)| residual.pointer), |(_, residual)| residual.pointer);
    let mut down_input_columns = intermediate_size_u32;
    let mut down_output_rows = hidden_size_u32;
    let mut down_route_count = route_count_u32;
    let mut down_top_k = top_k_u32;
    let mut down_expert_count = expert_count_u32;
    let mut down_shared_expert = shared_expert_u32;
    let mut down_fused_epilogue = if integrated_shared.is_some() { 2 } else { u32::from(epilogue.is_some()) };
    let mut down_args = [
        (&mut down_input as *mut *mut c_void).cast(),
        (&mut down_route_ids as *mut *mut c_void).cast(),
        (&mut down_route_weights as *mut *mut c_void).cast(),
        (&mut down_metas as *mut *mut c_void).cast(),
        (&mut down_output as *mut *mut c_void).cast(),
        (&mut down_shared as *mut *mut c_void).cast(),
        (&mut down_residual as *mut *mut c_void).cast(),
        (&mut down_input_columns as *mut u32).cast(),
        (&mut down_output_rows as *mut u32).cast(),
        (&mut down_route_count as *mut u32).cast(),
        (&mut down_top_k as *mut u32).cast(),
        (&mut down_expert_count as *mut u32).cast(),
        (&mut down_shared_expert as *mut u32).cast(),
        (&mut down_fused_epilogue as *mut u32).cast(),
    ];
    let down_started = options().kernel_profile.then(std::time::Instant::now);
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let down = if w8 { functions.decode_w8_down } else { functions.decode_down };
        let __hip_launch_result = unsafe {
            launch(
                down as *mut c_void,
                down_output_rows.div_ceil(16),
                u32::try_from(input_rows).map_err(|_| "decode MoE rows 超过 u32")?,
                1,
                256,
                1,
                1,
                0,
                crate::kernel::rocm::hip::active_compute_stream(),
                down_args.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "decode MoE fused down"));
    }
    if let Some(started) = down_started {
        synchronize_device(device_id, "decode MoE down profile")?;
        eprintln!("[rocm-kernel] decode-{}down device={device_id} input_rows={input_rows} routes={route_count} wall={:.6}s", if w8 { "w8-" } else { "" }, started.elapsed().as_secs_f64());
    }
    if options().debug_finite {
        try_validate_finite_resident_range_f32(device_id, &output, 0, input_elements).map_err(|error| format!("decode MoE down output 包含非有限值: {error}"))?;
        eprintln!("[decode-moe-finite] device={device_id} stage=down routes={route_count}");
    }
    // 临时 BF16 buffer 由 producer event 持有，完成前不会回到 available 队列。
    Ok(output)
}

/// W8 单行 expert 快路径：路由已在设备侧，gate/up 与 down 各一次 launch。
/// 仅服务 compressed-tensors MTP 层；主干 W4 grouped 路径保持不变。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_ct_w8_decode_experts_bf16(
    device_id: i32,
    input: &DeviceBuffer,
    hidden_size: usize,
    intermediate_size: usize,
    route_ids: &DeviceBuffer,
    route_weights: &DeviceBuffer,
    route_count: usize,
    experts: &[CtGroupedExpertRef<'_>],
) -> Result<DeviceBuffer, String> {
    if route_count == 0
        || route_count > 16
        || experts.is_empty()
        || experts.len() > 256
        || input.device_id != device_id
        || !matches!(input.bytes, bytes if bytes == hidden_size * 2 || bytes == hidden_size * 4)
        || route_ids.device_id != device_id
        || route_weights.device_id != device_id
        || route_ids.bytes < route_count * 4
        || route_weights.bytes < route_count * 4
    {
        return Err("ROCm W8 decode expert 参数无效".to_owned());
    }
    let scale_bytes = |dtype| match dtype {
        0 | 1 => Some(2usize),
        2 => Some(4usize),
        _ => None,
    };
    let validate = |weight: &CtGroupedWeightRef<'_>, rows: usize, columns: usize| -> Result<(), String> {
        let scale_bytes = scale_bytes(weight.scale_dtype).ok_or("ROCm W8 expert scale dtype 无效")?;
        if weight.packed.device_id != device_id
            || weight.scales.device_id != device_id
            || weight.group_size == 0
            || !columns.is_multiple_of(weight.group_size)
            || weight.packed.bytes < rows.checked_mul(columns).ok_or("ROCm W8 expert packed 大小溢出")?
            || weight.scales.bytes < rows.checked_mul(columns / weight.group_size).and_then(|n| n.checked_mul(scale_bytes)).ok_or("ROCm W8 expert scales 大小溢出")?
        {
            return Err("ROCm W8 expert 权重 shape/device 不兼容".to_owned());
        }
        Ok(())
    };
    for expert in experts {
        validate(&expert.gate, intermediate_size, hidden_size)?;
        validate(&expert.up, intermediate_size, hidden_size)?;
        validate(&expert.down, hidden_size, intermediate_size)?;
    }
    let meta = |weight: &CtGroupedWeightRef<'_>| -> Result<CtGroupedWeightMeta, String> {
        Ok(CtGroupedWeightMeta {
            packed: weight.packed.pointer as usize as u64,
            scales: weight.scales.pointer as usize as u64,
            group_size: u32::try_from(weight.group_size).map_err(|_| "ROCm W8 expert group_size 超过 u32")?,
            scale_dtype: weight.scale_dtype,
            format: weight.format,
        })
    };
    let metas = experts.iter().map(|expert| Ok(CtGroupedExpertMeta { gate: meta(&expert.gate)?, up: meta(&expert.up)?, down: meta(&expert.down)? })).collect::<Result<Vec<_>, String>>()?;
    set_device(device_id)?;
    let all_metas = resident_grouped_metas(device_id, &metas)?;
    let input_workspace_bytes = if input.bytes == hidden_size * 2 { 1 } else { hidden_size * 2 };
    let activated_bytes = route_count.checked_mul(intermediate_size).and_then(|n| n.checked_mul(2)).ok_or("ROCm W8 expert activation 大小溢出")?;
    with_deferred_tensor_workspace(device_id, &[input_workspace_bytes, activated_bytes], |workspace| {
        launch_ct_decode_experts_bf16(device_id, input, 1, hidden_size, intermediate_size, route_ids, route_weights, route_count, experts.len(), &all_metas, workspace.buffer(0), workspace.buffer(1), None, None, true)
    })
}

/// 所有活跃 expert 共享少量 launch；W4 只在 dot 内解码，不建立展开权重。
#[allow(clippy::too_many_arguments)]
fn launch_ct_grouped_experts_bf16(
    device_id: i32,
    input: &DeviceBuffer,
    input_rows: usize,
    hidden_size: usize,
    intermediate_size: usize,
    input_elements: usize,
    max_rows: u32,
    wmma_expert_count: usize,
    grouped_expert_count: usize,
    dynamic_rows: bool,
    d_tokens: &DeviceBuffer,
    d_weights: &DeviceBuffer,
    d_offsets: &DeviceBuffer,
    d_metas: &DeviceBuffer,
    d_input_bf16: &DeviceBuffer,
    d_gate: &DeviceBuffer,
    d_activated: &DeviceBuffer,
    dynamic_route: Option<(&DeviceBuffer, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer, usize)>,
    epilogue: Option<(&DeviceBuffer, &DeviceBuffer)>,
) -> Result<DeviceBuffer, String> {
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let functions = ct_quantized_functions(device_id)?;
    let output_elements = input_rows.checked_mul(hidden_size).ok_or("grouped output 大小溢出")?;
    let d_output = DeviceBuffer::allocate_reusable(device_id, output_elements * 4)?;
    let input_is_bf16 = input.bytes() == input_elements.checked_mul(2).ok_or("grouped BF16 input 大小溢出")?;
    let linear_input_buffer = if input_is_bf16 { input } else { d_input_bf16 };
    let grouped_profile = input_rows > 1 && options().kernel_profile;
    let gate_started = grouped_profile.then(std::time::Instant::now);
    if options().log_expert_pointers_device == Some(device_id) {
        eprintln!(
            "[grouped-buffers] device={device_id} input_rows={input_rows} hidden={hidden_size} intermediate={intermediate_size} max_rows={max_rows} input={:p}+{} tokens={:p}+{} weights={:p}+{} offsets={:p}+{} metas={:p}+{} input_bf16={:p}+{} gate={:p}+{} activated={:p}+{} output={:p}+{}",
            input.pointer,
            input.bytes,
            d_tokens.pointer,
            d_tokens.bytes,
            d_weights.pointer,
            d_weights.bytes,
            d_offsets.pointer,
            d_offsets.bytes,
            d_metas.pointer,
            d_metas.bytes,
            linear_input_buffer.pointer,
            linear_input_buffer.bytes,
            d_gate.pointer,
            d_gate.bytes,
            d_activated.pointer,
            d_activated.bytes,
            d_output.pointer,
            d_output.bytes
        );
    }

    if !input_is_bf16 {
        let mut cast_input = input.pointer;
        let mut cast_output = d_input_bf16.pointer;
        let mut cast_elements = u32::try_from(input_elements).map_err(|_| "grouped input 元素超过 u32")?;
        let mut cast_args = [(&mut cast_input as *mut *mut c_void).cast(), (&mut cast_output as *mut *mut c_void).cast(), (&mut cast_elements as *mut u32).cast()];
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe { launch(functions.cast as *mut c_void, cast_elements.div_ceil(256), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), cast_args.as_mut_ptr(), ptr::null_mut()) };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "grouped input f32_to_bf16"));
        }
    }
    if input_rows == 1 && options().debug_decode_moe_finite {
        try_validate_finite_resident_range_bf16(device_id, linear_input_buffer, 0, input_elements).map_err(|error| format!("rows=1 grouped MoE BF16 input 包含非有限值: {error}"))?;
        eprintln!("[decode-moe-finite] device={device_id} path=grouped stage=input");
    }

    let wmma_expert_count = if input_rows == 1 { 0 } else { wmma_expert_count };
    if wmma_expert_count > 0 {
        let mut linear_input = linear_input_buffer.pointer;
        let mut tokens = d_tokens.pointer;
        let mut expert_meta = d_metas.pointer;
        let mut offsets = d_offsets.pointer;
        let mut activated = d_activated.pointer;
        let mut input_columns = u32::try_from(hidden_size).map_err(|_| "hidden_size 超过 u32")?;
        let mut output_rows = u32::try_from(intermediate_size).map_err(|_| "intermediate_size 超过 u32")?;
        let mut expert_base = 0_u32;
        let mut expert_count = u32::try_from(wmma_expert_count).map_err(|_| "WMMA expert_count 超过 u32")?;
        let mut args = [
            (&mut linear_input as *mut *mut c_void).cast(),
            (&mut tokens as *mut *mut c_void).cast(),
            (&mut expert_meta as *mut *mut c_void).cast(),
            (&mut offsets as *mut *mut c_void).cast(),
            (&mut activated as *mut *mut c_void).cast(),
            (&mut input_columns as *mut u32).cast(),
            (&mut output_rows as *mut u32).cast(),
            (&mut expert_base as *mut u32).cast(),
            (&mut expert_count as *mut u32).cast(),
        ];
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let route_tiles = if dynamic_rows { 1 } else { max_rows.div_ceil(128) };
            let __hip_launch_result = unsafe {
                launch(
                    functions.grouped_gate_up_wmma as *mut c_void,
                    output_rows.div_ceil(16).div_ceil(4),
                    route_tiles,
                    expert_count,
                    functions.wavefront_size * 8,
                    1,
                    1,
                    0,
                    crate::kernel::rocm::hip::active_compute_stream(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "grouped WMMA fused gate/up"));
        }
    }
    let launch_linear = |projection: u32| -> Result<(), String> {
        let mut linear_input = linear_input_buffer.pointer;
        let mut tokens = d_tokens.pointer;
        let mut expert_meta = d_metas.pointer;
        let mut offsets = d_offsets.pointer;
        let mut linear_output = d_gate.pointer;
        let mut gate = d_gate.pointer;
        let mut activated = d_activated.pointer;
        let mut input_columns = u32::try_from(hidden_size).map_err(|_| "hidden_size 超过 u32")?;
        let mut output_rows = u32::try_from(intermediate_size).map_err(|_| "intermediate_size 超过 u32")?;
        let mut projection = projection;
        // 动态 route 无法在 host 侧知道哪些 slot 少于 4 行。WMMA 处理其余
        // slot 后，scalar 遍历全部 slot，但只补齐 WMMA 主动跳过的小 expert。
        let mut dynamic_rows_u32 = u32::from(dynamic_rows && wmma_expert_count == 0);
        let scalar_expert_count = if dynamic_rows { grouped_expert_count } else { grouped_expert_count - wmma_expert_count };
        if scalar_expert_count > 0 {
            let scalar_expert_base = if dynamic_rows { 0 } else { wmma_expert_count };
            let mut expert_base = u32::try_from(scalar_expert_base).map_err(|_| "scalar expert_base 超过 u32")?;
            let mut expert_count = u32::try_from(scalar_expert_count).map_err(|_| "scalar expert_count 超过 u32")?;
            let mut args = [
                (&mut linear_input as *mut *mut c_void).cast(),
                (&mut tokens as *mut *mut c_void).cast(),
                (&mut expert_meta as *mut *mut c_void).cast(),
                (&mut offsets as *mut *mut c_void).cast(),
                (&mut linear_output as *mut *mut c_void).cast(),
                (&mut gate as *mut *mut c_void).cast(),
                (&mut activated as *mut *mut c_void).cast(),
                (&mut input_columns as *mut u32).cast(),
                (&mut output_rows as *mut u32).cast(),
                (&mut expert_base as *mut u32).cast(),
                (&mut expert_count as *mut u32).cast(),
                (&mut projection as *mut u32).cast(),
                (&mut dynamic_rows_u32 as *mut u32).cast(),
            ];
            // WMMA 已覆盖不少于 4 行的 slot；补写 kernel 只处理 1..3 行，
            // 固定一个 route tile，避免按整段 input_rows 发射大量必然返回的 block。
            let singleton_status = {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result =
                    unsafe { launch(functions.grouped_linear as *mut c_void, output_rows.div_ceil(16), 1, expert_count, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), args.as_mut_ptr(), ptr::null_mut()) };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            };
            if singleton_status != HIP_SUCCESS {
                return Err(runtime.hip_error(singleton_status, "grouped singleton gate/up"));
            }
            // singleton pass 只覆盖恰好 1 条 route；动态 WMMA 会跳过 2-3 条
            // route 的 expert，仍须进入下面的 scalar pass 补齐这些 activation。
            if input_rows == 1 {
                return Ok(());
            }
            let route_tiles = if dynamic_rows && wmma_expert_count > 0 { 1 } else { max_rows.div_ceil(16) };
            let status = {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result =
                    unsafe { launch(functions.grouped_linear as *mut c_void, output_rows, route_tiles, expert_count, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), args.as_mut_ptr(), ptr::null_mut()) };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "grouped scalar gate/up"));
            }
        }
        Ok(())
    };
    launch_linear(0)?;
    if input_rows != 1 {
        launch_linear(1)?;
    }
    if let Some(started) = gate_started {
        synchronize_device(device_id, "grouped gate/up profile")?;
        eprintln!(
            "[rocm-kernel] grouped-gate-up device={device_id} input_rows={input_rows} max_rows={max_rows} wmma_experts={wmma_expert_count} scalar_experts={} wall={:.6}s",
            if dynamic_rows { grouped_expert_count } else { grouped_expert_count - wmma_expert_count },
            started.elapsed().as_secs_f64(),
        );
    }
    let down_started = grouped_profile.then(std::time::Instant::now);
    if options().debug_finite || input_rows == 1 && options().debug_decode_moe_finite {
        let activated_elements = dynamic_route.map_or(d_activated.bytes() / 2, |(_, _, _, _, top_k)| input_rows * top_k * intermediate_size);
        try_validate_finite_resident_range_bf16(device_id, d_activated, 0, activated_elements)
            .map_err(|error| format!("grouped MoE gate/up activated 包含非有限值: rows={input_rows} max_rows={max_rows} wmma_experts={wmma_expert_count} grouped_experts={grouped_expert_count} dynamic={dynamic_rows}: {error}"))?;
        eprintln!("[decode-moe-finite] device={device_id} path=grouped stage=gate_up");
    }
    if let Some((route_ids, source_weights, route_to_grouped, source_metas, top_k)) = dynamic_route {
        if options().grouped_down_route_buffer && input_rows > 1 && wmma_expert_count > 0 {
            let route_count = input_rows.checked_mul(top_k).ok_or("route-major down route 数溢出")?;
            let bytes_per_output_row = route_count.checked_mul(4).ok_or("route-major down 单列字节数溢出")?;
            let output_tile_rows = ((64 * 1024 * 1024 / bytes_per_output_row).max(16).min(hidden_size) / 16) * 16;
            if output_tile_rows == 0 {
                return Err("route-major down 无法建立 16 行输出 tile".to_owned());
            }
            let route_output_elements = route_count.checked_mul(output_tile_rows).ok_or("route-major down output 大小溢出")?;
            let d_route_output = DeviceBuffer::allocate(device_id, route_output_elements.checked_mul(4).ok_or("route-major down output 字节溢出")?)?;
            let mut down_input = d_activated.pointer;
            let mut tokens = d_tokens.pointer;
            let mut weights = d_weights.pointer;
            let mut expert_meta = d_metas.pointer;
            let mut offsets = d_offsets.pointer;
            let mut route_output = d_route_output.pointer;
            let mut input_columns = u32::try_from(intermediate_size).map_err(|_| "intermediate_size 超过 u32")?;
            let mut output_rows = u32::try_from(hidden_size).map_err(|_| "hidden_size 超过 u32")?;
            let mut route_major_output = 1_u32;
            let scalar_expert_count = if dynamic_rows { grouped_expert_count } else { grouped_expert_count - wmma_expert_count };
            for output_row_base in (0..hidden_size).step_by(output_tile_rows) {
                let output_slice_rows = (hidden_size - output_row_base).min(output_tile_rows);
                let mut output_row_base = u32::try_from(output_row_base).map_err(|_| "route-major down output_row_base 超过 u32")?;
                let mut output_slice_rows = u32::try_from(output_slice_rows).map_err(|_| "route-major down output_slice_rows 超过 u32")?;
                let mut expert_base = 0_u32;
                let mut expert_count = u32::try_from(wmma_expert_count).map_err(|_| "WMMA expert_count 超过 u32")?;
                let mut wmma_args = [
                    (&mut down_input as *mut *mut c_void).cast(),
                    (&mut tokens as *mut *mut c_void).cast(),
                    (&mut weights as *mut *mut c_void).cast(),
                    (&mut expert_meta as *mut *mut c_void).cast(),
                    (&mut offsets as *mut *mut c_void).cast(),
                    (&mut route_output as *mut *mut c_void).cast(),
                    (&mut input_columns as *mut u32).cast(),
                    (&mut output_rows as *mut u32).cast(),
                    (&mut expert_base as *mut u32).cast(),
                    (&mut expert_count as *mut u32).cast(),
                    (&mut output_row_base as *mut u32).cast(),
                    (&mut output_slice_rows as *mut u32).cast(),
                    (&mut route_major_output as *mut u32).cast(),
                ];
                let status = {
                    let __hip_stats_started = super::hip_api_stats::start();
                    let route_tiles = if dynamic_rows { 1 } else { max_rows.div_ceil(128) };
                    let __hip_launch_result = unsafe {
                        launch(functions.grouped_down_wmma as *mut c_void, output_slice_rows.div_ceil(128), route_tiles, expert_count, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), wmma_args.as_mut_ptr(), ptr::null_mut())
                    };
                    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                    __hip_launch_result
                };
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "grouped WMMA route-major down"));
                }

                if scalar_expert_count > 0 {
                    let scalar_expert_base = if dynamic_rows { 0 } else { wmma_expert_count };
                    let mut expert_base = u32::try_from(scalar_expert_base).map_err(|_| "scalar expert_base 超过 u32")?;
                    let mut expert_count = u32::try_from(scalar_expert_count).map_err(|_| "scalar expert_count 超过 u32")?;
                    let mut small_args = [
                        (&mut down_input as *mut *mut c_void).cast(),
                        (&mut tokens as *mut *mut c_void).cast(),
                        (&mut weights as *mut *mut c_void).cast(),
                        (&mut expert_meta as *mut *mut c_void).cast(),
                        (&mut offsets as *mut *mut c_void).cast(),
                        (&mut route_output as *mut *mut c_void).cast(),
                        (&mut input_columns as *mut u32).cast(),
                        (&mut output_rows as *mut u32).cast(),
                        (&mut expert_base as *mut u32).cast(),
                        (&mut expert_count as *mut u32).cast(),
                        (&mut output_row_base as *mut u32).cast(),
                        (&mut output_slice_rows as *mut u32).cast(),
                        (&mut route_major_output as *mut u32).cast(),
                    ];
                    let status = {
                        let __hip_stats_started = super::hip_api_stats::start();
                        let __hip_launch_result = unsafe {
                            launch(functions.grouped_down_small as *mut c_void, output_slice_rows.div_ceil(16), 1, expert_count, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), small_args.as_mut_ptr(), ptr::null_mut())
                        };
                        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                        __hip_launch_result
                    };
                    if status != HIP_SUCCESS {
                        return Err(runtime.hip_error(status, "grouped small route-major down"));
                    }
                }

                let mut route_output = d_route_output.pointer;
                let mut grouped = route_to_grouped.pointer;
                let mut down_output = d_output.pointer;
                let mut shared_output = epilogue.map_or(ptr::null_mut(), |(shared, _)| shared.pointer);
                let mut residual = epilogue.map_or(ptr::null_mut(), |(_, residual)| residual.pointer);
                let mut route_count = u32::try_from(route_count).map_err(|_| "route-major down route_count 超过 u32")?;
                let mut top_k = u32::try_from(top_k).map_err(|_| "route-major down top_k 超过 u32")?;
                let mut fused_epilogue = u32::from(epilogue.is_some());
                let mut reduce_args = [
                    (&mut route_output as *mut *mut c_void).cast(),
                    (&mut grouped as *mut *mut c_void).cast(),
                    (&mut down_output as *mut *mut c_void).cast(),
                    (&mut shared_output as *mut *mut c_void).cast(),
                    (&mut residual as *mut *mut c_void).cast(),
                    (&mut output_rows as *mut u32).cast(),
                    (&mut output_row_base as *mut u32).cast(),
                    (&mut output_slice_rows as *mut u32).cast(),
                    (&mut route_count as *mut u32).cast(),
                    (&mut top_k as *mut u32).cast(),
                    (&mut fused_epilogue as *mut u32).cast(),
                ];
                let output_slice_elements = u32::try_from(input_rows.checked_mul(output_slice_rows as usize).ok_or("route-major down slice elements 溢出")?).map_err(|_| "route-major down slice elements 超过 u32")?;
                let status = {
                    let __hip_stats_started = super::hip_api_stats::start();
                    let __hip_launch_result = unsafe {
                        launch(functions.grouped_down_reduce_routes as *mut c_void, output_slice_elements.div_ceil(256), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), reduce_args.as_mut_ptr(), ptr::null_mut())
                    };
                    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                    __hip_launch_result
                };
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "grouped route-major deterministic down reduce"));
                }
            }
            if let Some(started) = down_started {
                synchronize_device(device_id, "grouped route-major down profile")?;
                eprintln!("[rocm-kernel] grouped-down-route-buffer device={device_id} input_rows={input_rows} tile_rows={output_tile_rows} bytes={} wall={:.6}s", d_route_output.bytes(), started.elapsed().as_secs_f64());
            }
            return Ok(d_output);
        }
        if epilogue.is_some() {
            return Err("grouped fused epilogue 需要 deterministic route-major down".to_owned());
        }
        let mut down_input = d_activated.pointer;
        let mut ids = route_ids.pointer;
        let mut weights = source_weights.pointer;
        let mut grouped = route_to_grouped.pointer;
        let mut expert_meta = source_metas.pointer;
        let mut down_output = d_output.pointer;
        let mut input_columns = u32::try_from(intermediate_size).map_err(|_| "intermediate_size 超过 u32")?;
        let mut output_rows = u32::try_from(hidden_size).map_err(|_| "hidden_size 超过 u32")?;
        let mut route_count = u32::try_from(input_rows.checked_mul(top_k).ok_or("dynamic route 数溢出")?).map_err(|_| "dynamic route_count 超过 u32")?;
        let mut top_k = u32::try_from(top_k).map_err(|_| "dynamic top_k 超过 u32")?;
        let mut expert_count = u32::try_from(grouped_expert_count).map_err(|_| "dynamic expert_count 超过 u32")?;
        let mut args = [
            (&mut down_input as *mut *mut c_void).cast(),
            (&mut ids as *mut *mut c_void).cast(),
            (&mut weights as *mut *mut c_void).cast(),
            (&mut grouped as *mut *mut c_void).cast(),
            (&mut expert_meta as *mut *mut c_void).cast(),
            (&mut down_output as *mut *mut c_void).cast(),
            (&mut input_columns as *mut u32).cast(),
            (&mut output_rows as *mut u32).cast(),
            (&mut route_count as *mut u32).cast(),
            (&mut top_k as *mut u32).cast(),
            (&mut expert_count as *mut u32).cast(),
        ];
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe {
                launch(
                    functions.grouped_down_reduce as *mut c_void,
                    output_rows.div_ceil(16),
                    u32::try_from(input_rows).map_err(|_| "dynamic rows 超过 u32")?,
                    1,
                    256,
                    1,
                    1,
                    0,
                    crate::kernel::rocm::hip::active_compute_stream(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "grouped deterministic down reduce"));
        }
        if let Some(started) = down_started {
            synchronize_device(device_id, "grouped down profile")?;
            eprintln!("[rocm-kernel] grouped-down-reduce device={device_id} input_rows={input_rows} wall={:.6}s", started.elapsed().as_secs_f64(),);
        }
        if options().debug_finite {
            try_validate_finite_resident_range_f32(device_id, &d_output, 0, d_output.bytes() / 4).map_err(|error| format!("grouped MoE deterministic down output 包含非有限值: {error}"))?;
        }
        return Ok(d_output);
    }
    if input_rows != 1 {
        let mut zero_output = d_output.pointer;
        let mut zero_elements = u32::try_from(output_elements).map_err(|_| "grouped output 元素超过 u32")?;
        let mut zero_args = [(&mut zero_output as *mut *mut c_void).cast(), (&mut zero_elements as *mut u32).cast()];
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe { launch(functions.grouped_zero as *mut c_void, zero_elements.div_ceil(256), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), zero_args.as_mut_ptr(), ptr::null_mut()) };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "grouped output zero"));
        }
    }

    let mut down_input = d_activated.pointer;
    let mut tokens = d_tokens.pointer;
    let mut weights = d_weights.pointer;
    let mut expert_meta = d_metas.pointer;
    let mut offsets = d_offsets.pointer;
    let mut down_output = d_output.pointer;
    let mut input_columns = u32::try_from(intermediate_size).map_err(|_| "intermediate_size 超过 u32")?;
    let mut output_rows = u32::try_from(hidden_size).map_err(|_| "hidden_size 超过 u32")?;
    if wmma_expert_count > 0 {
        let mut expert_base = 0_u32;
        let mut expert_count = u32::try_from(wmma_expert_count).map_err(|_| "WMMA expert_count 超过 u32")?;
        let mut output_row_base = 0_u32;
        let mut output_slice_rows = output_rows;
        let mut route_major_output = 0_u32;
        let mut down_args = [
            (&mut down_input as *mut *mut c_void).cast(),
            (&mut tokens as *mut *mut c_void).cast(),
            (&mut weights as *mut *mut c_void).cast(),
            (&mut expert_meta as *mut *mut c_void).cast(),
            (&mut offsets as *mut *mut c_void).cast(),
            (&mut down_output as *mut *mut c_void).cast(),
            (&mut input_columns as *mut u32).cast(),
            (&mut output_rows as *mut u32).cast(),
            (&mut expert_base as *mut u32).cast(),
            (&mut expert_count as *mut u32).cast(),
            (&mut output_row_base as *mut u32).cast(),
            (&mut output_slice_rows as *mut u32).cast(),
            (&mut route_major_output as *mut u32).cast(),
        ];
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe {
                launch(
                    functions.grouped_down_wmma as *mut c_void,
                    output_rows.div_ceil(16).div_ceil(8),
                    max_rows.div_ceil(128),
                    expert_count,
                    256,
                    1,
                    1,
                    0,
                    crate::kernel::rocm::hip::active_compute_stream(),
                    down_args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "grouped WMMA down scatter"));
        }
    }
    let scalar_expert_count = if dynamic_rows { grouped_expert_count } else { grouped_expert_count - wmma_expert_count };
    if scalar_expert_count > 0 {
        let scalar_expert_base = if dynamic_rows { 0 } else { wmma_expert_count };
        let mut expert_base = u32::try_from(scalar_expert_base).map_err(|_| "scalar expert_base 超过 u32")?;
        let mut expert_count = u32::try_from(scalar_expert_count).map_err(|_| "scalar expert_count 超过 u32")?;
        let mut down_args = [
            (&mut down_input as *mut *mut c_void).cast(),
            (&mut tokens as *mut *mut c_void).cast(),
            (&mut weights as *mut *mut c_void).cast(),
            (&mut expert_meta as *mut *mut c_void).cast(),
            (&mut offsets as *mut *mut c_void).cast(),
            (&mut down_output as *mut *mut c_void).cast(),
            (&mut input_columns as *mut u32).cast(),
            (&mut output_rows as *mut u32).cast(),
            (&mut expert_base as *mut u32).cast(),
            (&mut expert_count as *mut u32).cast(),
        ];
        let status = if input_rows == 1 {
            {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result =
                    unsafe { launch(functions.grouped_down as *mut c_void, output_rows.div_ceil(16), max_rows.div_ceil(16), 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), down_args.as_mut_ptr(), ptr::null_mut()) };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            }
        } else {
            let mut output_row_base = 0_u32;
            let mut output_slice_rows = output_rows;
            let mut route_major_output = 0_u32;
            let mut small_args = [
                down_args[0],
                down_args[1],
                down_args[2],
                down_args[3],
                down_args[4],
                down_args[5],
                down_args[6],
                down_args[7],
                down_args[8],
                down_args[9],
                (&mut output_row_base as *mut u32).cast(),
                (&mut output_slice_rows as *mut u32).cast(),
                (&mut route_major_output as *mut u32).cast(),
            ];
            {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result =
                    unsafe { launch(functions.grouped_down_small as *mut c_void, output_rows.div_ceil(16), 1, expert_count, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), small_args.as_mut_ptr(), ptr::null_mut()) };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            }
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "grouped scalar down scatter"));
        }
    }
    if let Some(started) = down_started {
        synchronize_device(device_id, "grouped down profile")?;
        eprintln!("[rocm-kernel] grouped-down device={device_id} input_rows={input_rows} max_rows={max_rows} wmma_experts={wmma_expert_count} scalar_experts={scalar_expert_count} wall={:.6}s", started.elapsed().as_secs_f64(),);
    }
    if input_rows == 1 && options().debug_decode_moe_finite {
        try_validate_finite_resident_range_f32(device_id, &d_output, 0, d_output.bytes() / 4).map_err(|error| format!("rows=1 grouped MoE down output 包含非有限值: {error}"))?;
        eprintln!("[decode-moe-finite] device={device_id} path=grouped stage=down");
    }
    Ok(d_output)
}

fn deterministic_grouped_routes(input_rows: usize, route_tokens: &[u32], route_weights: &[f32], route_offsets: &[u32], expert_count: usize) -> Result<(Vec<u32>, Vec<f32>, Vec<u32>, usize), String> {
    let route_count = route_tokens.len();
    if input_rows == 0 || route_count == 0 || route_weights.len() != route_count || route_offsets.len() != expert_count + 1 || route_offsets.last().copied() != u32::try_from(route_count).ok() || !route_count.is_multiple_of(input_rows) {
        return Err("ROCm deterministic grouped route 参数无效".to_owned());
    }
    let top_k = route_count / input_rows;
    if top_k == 0 || top_k > 16 {
        return Err(format!("ROCm deterministic grouped route top_k={top_k} 无效"));
    }

    let mut per_token = (0..input_rows).map(|_| Vec::with_capacity(top_k)).collect::<Vec<_>>();
    for expert in 0..expert_count {
        let begin = route_offsets[expert] as usize;
        let end = route_offsets[expert + 1] as usize;
        if begin > end || end > route_count {
            return Err(format!("ROCm deterministic grouped route expert={expert} offset={begin}..{end}/{route_count} 越界"));
        }
        let expert = u32::try_from(expert).map_err(|_| "ROCm deterministic grouped route expert 超过 u32")?;
        for route in begin..end {
            let token = route_tokens[route] as usize;
            if token >= input_rows {
                return Err(format!("ROCm deterministic grouped route token={token}/{input_rows} 越界"));
            }
            per_token[token].push((expert, u32::try_from(route).map_err(|_| "ROCm deterministic grouped route index 超过 u32")?, route_weights[route]));
        }
    }
    if let Some((token, routes)) = per_token.iter().enumerate().find(|(_, routes)| routes.len() != top_k) {
        return Err(format!("ROCm deterministic grouped route token={token} routes={}，期望 top_k={top_k}", routes.len()));
    }

    let mut route_ids = Vec::with_capacity(route_count);
    let mut weights = Vec::with_capacity(route_count);
    let mut route_to_grouped = Vec::with_capacity(route_count);
    for routes in per_token {
        for (expert, grouped, weight) in routes {
            route_ids.push(expert);
            weights.push(weight);
            route_to_grouped.push(grouped);
        }
    }
    Ok((route_ids, weights, route_to_grouped, top_k))
}

pub(crate) fn try_ct_grouped_experts_bf16(
    device_id: i32,
    input: &DeviceBuffer,
    input_rows: usize,
    hidden_size: usize,
    intermediate_size: usize,
    route_tokens: &[u32],
    route_weights: &[f32],
    route_offsets: &[u32],
    experts: &[CtGroupedExpertRef<'_>],
    device_route: Option<(&DeviceBuffer, &DeviceBuffer, usize)>,
    epilogue: Option<(&DeviceBuffer, &DeviceBuffer)>,
    integrated_shared: Option<(&CtGroupedExpertRef<'_>, &DeviceBuffer)>,
) -> Result<DeviceBuffer, String> {
    let host_route_valid = device_route.is_some()
        || (route_offsets.len() == experts.len() + 1
            && !route_tokens.is_empty()
            && route_tokens.len() == route_weights.len()
            && route_offsets.last().copied() == Some(u32::try_from(route_tokens.len()).map_err(|_| "grouped route 数超过 u32")?));
    let input_elements = input_rows.checked_mul(hidden_size).ok_or("grouped input 大小溢出")?;
    let input_f32_bytes = input_elements.checked_mul(4).ok_or("grouped F32 input 大小溢出")?;
    let input_bf16_bytes = input_elements.checked_mul(2).ok_or("grouped BF16 input 大小溢出")?;
    if input.device_id != device_id
        || !matches!(input.bytes, bytes if bytes == input_f32_bytes || bytes == input_bf16_bytes)
        || experts.is_empty()
        || !host_route_valid
        || device_route.is_some_and(|(ids, weights, count)| {
            count == 0 || !count.is_multiple_of(input_rows) || count / input_rows > 16 || ids.device_id != device_id || weights.device_id != device_id || ids.bytes < count * 4 || weights.bytes < count * 4
        })
    {
        return Err("ROCm grouped expert 参数无效".to_owned());
    }
    if epilogue.is_some() && integrated_shared.is_some() {
        return Err("ROCm grouped expert 不能同时使用预计算与融合 shared epilogue".to_owned());
    }
    if let Some((shared, residual)) = epilogue {
        validate_resident(shared, device_id, input_f32_bytes, "grouped shared epilogue")?;
        validate_resident(residual, device_id, input_f32_bytes, "grouped residual epilogue")?;
    }
    if let Some((_, residual)) = integrated_shared {
        validate_resident(residual, device_id, input_f32_bytes, "grouped integrated residual")?;
    }
    let mut metas = ct_grouped_expert_metas(device_id, experts)?;
    let integrated_shared_index = if let Some((shared, _)) = integrated_shared {
        let index = metas.len();
        metas.extend(ct_grouped_expert_metas(device_id, std::slice::from_ref(shared))?);
        Some(index)
    } else {
        None
    };
    if options().log_expert_pointers_device == Some(device_id) {
        for (slot, expert) in experts.iter().enumerate() {
            eprintln!(
                "[grouped-pointer] device={device_id} slot={slot} gate={:p}+{} gate_scales={:p}+{} up={:p}+{} up_scales={:p}+{} down={:p}+{} down_scales={:p}+{}",
                expert.gate.packed.pointer,
                expert.gate.packed.bytes,
                expert.gate.scales.pointer,
                expert.gate.scales.bytes,
                expert.up.packed.pointer,
                expert.up.packed.bytes,
                expert.up.scales.pointer,
                expert.up.scales.bytes,
                expert.down.packed.pointer,
                expert.down.packed.bytes,
                expert.down.scales.pointer,
                expert.down.scales.bytes
            );
        }
    }
    set_device(device_id)?;
    let input_workspace_bytes = if input.bytes == input_bf16_bytes { 1 } else { input_bf16_bytes };
    if let Some((route_ids, route_weights_device, route_count)) = device_route {
        let all_metas = resident_grouped_metas(device_id, &metas)?;
        let activated_routes = route_count.checked_add(usize::from(integrated_shared.is_some()) * input_rows).ok_or("decode MoE route 数溢出")?;
        let intermediate_elements = activated_routes.checked_mul(intermediate_size).ok_or("decode MoE intermediate 大小溢出")?;
        let decode_fused_compatible = metas.iter().all(|expert| {
            let group_size = expert.gate.group_size as usize;
            expert.gate.format == 0
                && expert.up.format == 0
                && group_size != 0
                && group_size == expert.up.group_size as usize
                && group_size.is_multiple_of(8)
                && hidden_size.is_multiple_of(group_size)
                && matches!(expert.gate.scale_dtype, 0 | 1 | 2)
                && matches!(expert.up.scale_dtype, 0 | 1 | 2)
        });
        if route_count <= experts.len() && decode_fused_compatible && options().decode_moe_fused {
            return with_deferred_tensor_workspace(device_id, &[input_workspace_bytes, intermediate_elements * 2], |workspace| {
                launch_ct_decode_experts_bf16(
                    device_id,
                    input,
                    input_rows,
                    hidden_size,
                    intermediate_size,
                    route_ids,
                    route_weights_device,
                    route_count,
                    experts.len(),
                    &all_metas,
                    workspace.buffer(0),
                    workspace.buffer(1),
                    epilogue,
                    integrated_shared_index.map(|index| (index, integrated_shared.expect("shared index 必须有 residual").1)),
                    false,
                )
            });
        }
        if integrated_shared.is_some() {
            return Err("ROCm integrated shared 只支持 direct decode expert 路径".to_owned());
        }

        // 活跃 expert 数不可能超过 route_count；小批次直接计算，避免为稀疏 route 建立 grouped workspace。
        // token 与 weight 总容量严格等于 B*top_k，不按 expert 展开。
        let slot_count = route_count.min(experts.len());
        if slot_count == 0 {
            return Err("ROCm grouped decode expert 数量非法".to_owned());
        }
        // gate/up 的每条 route 独立写入，不存在跨 expert 归约；多行输入可以安全使用
        // F32 累加 WMMA。down 仍由 grouped_down_reduce 按固定 expert 顺序归约。
        let wmma_expert_count = usize::from(input_rows > 1 && options().grouped_wmma) * slot_count;
        let gate_elements = intermediate_elements;
        let meta_bytes = slot_count.checked_mul(std::mem::size_of::<CtGroupedExpertMeta>()).ok_or("grouped decode metadata 大小溢出")?;
        let functions = ct_quantized_functions(device_id)?;
        let runtime = RocmRuntime::open()?;
        let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
        return with_deferred_tensor_workspace(device_id, &[input_workspace_bytes, gate_elements * 4, intermediate_elements * 2, route_count * 4, route_count * 4, route_count * 4, (slot_count + 1) * 4, meta_bytes], |workspace| {
            let mut ids = route_ids.pointer;
            let mut route_weights = route_weights_device.pointer;
            let mut source_metas = all_metas.pointer;
            let mut tokens = workspace.buffer(3).pointer;
            let mut weights = workspace.buffer(4).pointer;
            let mut route_to_grouped = workspace.buffer(5).pointer;
            let mut offsets = workspace.buffer(6).pointer;
            let mut grouped_metas = workspace.buffer(7).pointer;
            let mut route_count_u32 = u32::try_from(route_count).map_err(|_| "grouped decode route_count 超过 u32")?;
            let mut top_k_u32 = u32::try_from(route_count / input_rows).map_err(|_| "grouped decode top_k 超过 u32")?;
            let mut expert_count_u32 = u32::try_from(experts.len()).map_err(|_| "grouped decode expert_count 超过 u32")?;
            let mut slot_count_u32 = u32::try_from(slot_count).map_err(|_| "grouped decode slot_count 超过 u32")?;
            let route_state_bytes = expert_count_u32.checked_mul(3 * 4).ok_or("grouped decode route state 超过 u32")?;
            let mut args = [
                (&mut ids as *mut *mut c_void).cast(),
                (&mut route_weights as *mut *mut c_void).cast(),
                (&mut source_metas as *mut *mut c_void).cast(),
                (&mut tokens as *mut *mut c_void).cast(),
                (&mut weights as *mut *mut c_void).cast(),
                (&mut route_to_grouped as *mut *mut c_void).cast(),
                (&mut offsets as *mut *mut c_void).cast(),
                (&mut grouped_metas as *mut *mut c_void).cast(),
                (&mut route_count_u32 as *mut u32).cast(),
                (&mut top_k_u32 as *mut u32).cast(),
                (&mut expert_count_u32 as *mut u32).cast(),
                (&mut slot_count_u32 as *mut u32).cast(),
            ];
            let status = {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result = unsafe { launch(functions.group_decode_routes as *mut c_void, 1, 1, 1, 256, 1, 1, route_state_bytes, crate::kernel::rocm::hip::active_compute_stream(), args.as_mut_ptr(), ptr::null_mut()) };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "grouped decode route compact"));
            }
            launch_ct_grouped_experts_bf16(
                device_id,
                input,
                input_rows,
                hidden_size,
                intermediate_size,
                input_elements,
                u32::try_from(input_rows).map_err(|_| "grouped decode max_rows 超过 u32")?,
                wmma_expert_count,
                slot_count,
                true,
                workspace.buffer(3),
                workspace.buffer(4),
                workspace.buffer(6),
                workspace.buffer(7),
                workspace.buffer(0),
                workspace.buffer(1),
                workspace.buffer(2),
                Some((route_ids, route_weights_device, workspace.buffer(5), &all_metas, route_count / input_rows)),
                epilogue,
            )
        });
    }

    let max_rows = route_offsets.windows(2).map(|pair| pair[1] - pair[0]).max().unwrap_or(0);
    let wmma_expert_count = if options().grouped_wmma { route_offsets.windows(2).take_while(|pair| pair[1] - pair[0] >= 4).count() } else { 0 };
    if max_rows == 0 {
        return Err("ROCm grouped expert 没有 route".to_owned());
    }
    let route_count = route_tokens.len();
    let intermediate_elements = route_count.checked_mul(intermediate_size).ok_or("grouped intermediate 大小溢出")?;
    let d_tokens = upload_pod(device_id, route_tokens)?;
    let d_weights = upload_pod(device_id, route_weights)?;
    let d_offsets = upload_pod(device_id, route_offsets)?;
    let d_metas = upload_pod(device_id, &metas)?;
    // expert-grouped route 只决定 gate/up 的连续布局；down 按 token、固定 expert
    // 顺序归约，避免 atomicAdd 的调度顺序改变 greedy token。
    let (reduce_ids, reduce_weights, route_to_grouped, top_k) = deterministic_grouped_routes(input_rows, route_tokens, route_weights, route_offsets, experts.len())?;
    let d_reduce_ids = upload_pod(device_id, &reduce_ids)?;
    let d_reduce_weights = upload_pod(device_id, &reduce_weights)?;
    let d_route_to_grouped = upload_pod(device_id, &route_to_grouped)?;
    let d_input_bf16 = DeviceBuffer::allocate(device_id, input_workspace_bytes)?;
    // 所有 expert 都走 fused WMMA 时 gate 从未落盘，只保留 scalar fallback 所需容量。
    let gate_elements = if wmma_expert_count == experts.len() { 1 } else { intermediate_elements };
    let d_gate = DeviceBuffer::allocate(device_id, gate_elements * 4)?;
    let d_activated = DeviceBuffer::allocate(device_id, intermediate_elements * 2)?;
    launch_ct_grouped_experts_bf16(
        device_id,
        input,
        input_rows,
        hidden_size,
        intermediate_size,
        input_elements,
        max_rows,
        wmma_expert_count,
        experts.len(),
        false,
        &d_tokens,
        &d_weights,
        &d_offsets,
        &d_metas,
        &d_input_bf16,
        &d_gate,
        &d_activated,
        Some((&d_reduce_ids, &d_reduce_weights, &d_route_to_grouped, &d_metas, top_k)),
        epilogue,
    )
}

#[cfg(test)]
mod grouped_route_tests {
    use super::deterministic_grouped_routes;

    #[test]
    fn deterministic_routes_are_token_major_and_expert_ordered() {
        let (ids, weights, grouped, top_k) = deterministic_grouped_routes(2, &[1, 0, 1, 0], &[0.4, 0.1, 0.3, 0.2], &[0, 1, 3, 4], 3).unwrap();
        assert_eq!(top_k, 2);
        assert_eq!(ids, [1, 2, 0, 1]);
        assert_eq!(weights, [0.1, 0.2, 0.4, 0.3]);
        assert_eq!(grouped, [1, 3, 0, 2]);
    }

    #[test]
    fn deterministic_routes_reject_missing_top_k_slot() {
        let error = deterministic_grouped_routes(2, &[0, 0], &[0.5, 0.5], &[0, 2, 2], 2).unwrap_err();
        assert!(error.contains("期望 top_k=1"));
    }
}
