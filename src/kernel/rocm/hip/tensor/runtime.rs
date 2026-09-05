use super::*;

#[derive(Default)]
pub(crate) struct CtQuantizedWorkspace {
    pub(crate) input_f32: Option<DeviceBuffer>,
    pub(crate) input_bf16: Option<DeviceBuffer>,
}

impl CtQuantizedWorkspace {
    pub(crate) fn ensure(&mut self, device_id: i32, input_f32: usize, input_bf16: usize, _output: usize) -> Result<(), String> {
        if input_f32 != 0 {
            Self::ensure_buffer(&mut self.input_f32, device_id, input_f32)?;
        }
        Self::ensure_buffer(&mut self.input_bf16, device_id, input_bf16)
    }

    pub(crate) fn ensure_buffer(buffer: &mut Option<DeviceBuffer>, device_id: i32, bytes: usize) -> Result<(), String> {
        if buffer.as_ref().is_some_and(|buffer| buffer.device_id == device_id && buffer.bytes >= bytes) {
            return Ok(());
        }
        *buffer = Some(DeviceBuffer::allocate(device_id, bytes)?);
        Ok(())
    }
}

thread_local! {
    pub(crate) static CT_QUANTIZED_WORKSPACES: RefCell<HashMap<(i32, usize), CtQuantizedWorkspace>> = RefCell::new(HashMap::new());
}

#[derive(Default)]
pub(crate) struct TensorWorkspace {
    pub(crate) buffers: Vec<Option<DeviceBuffer>>,
}

impl TensorWorkspace {
    pub(crate) fn ensure(&mut self, device_id: i32, sizes: &[usize]) -> Result<(), String> {
        self.buffers.resize_with(sizes.len(), || None);
        for (index, &bytes) in sizes.iter().enumerate() {
            CtQuantizedWorkspace::ensure_buffer(&mut self.buffers[index], device_id, bytes.max(1))?;
        }
        Ok(())
    }

    pub(crate) fn buffer(&self, index: usize) -> &DeviceBuffer {
        self.buffers[index].as_ref().unwrap()
    }
}

thread_local! {
    pub(crate) static TENSOR_WORKSPACES: RefCell<HashMap<(i32, usize), TensorWorkspace>> = RefCell::new(HashMap::new());
    pub(crate) static ATTENTION_WORKSPACES: RefCell<HashMap<(i32, usize), TensorWorkspace>> = RefCell::new(HashMap::new());
}

#[derive(Clone, Copy)]
pub(crate) struct TensorFunctions {
    pub(crate) add_scaled: usize,
    pub(crate) subtract: usize,
    pub(crate) relative_l1_delta_partial: usize,
    pub(crate) add_bf16_f32: usize,
    pub(crate) fill_zero: usize,
    pub(crate) fill_u32: usize,
    pub(crate) validate_finite: usize,
    pub(crate) rmsnorm: usize,
    pub(crate) rmsnorm_adaln_segment: usize,
    pub(crate) layernorm_bias: usize,
    pub(crate) rmsnorm_heads_unit: usize,
    pub(crate) rmsnorm_rope_pair_unit: usize,
    pub(crate) scaled_residual_columns: usize,
    pub(crate) unpatch_affine: usize,
    pub(crate) sigmoid_gate: usize,
    pub(crate) split_interleaved_columns: usize,
    pub(crate) gated_activation: usize,
    pub(crate) gated_activation_bf16: usize,
    pub(crate) split_gated_activation: usize,
    pub(crate) split_columns: usize,
    pub(crate) concat_columns: usize,
    pub(crate) compact_qkv_head_range: usize,
    pub(crate) rope: usize,
    pub(crate) rope_indirect: usize,
    #[cfg(test)]
    pub(crate) rope_segmented_pair: usize,
    pub(crate) select_rows: usize,
    pub(crate) argmax_rows_excluding: usize,
    pub(crate) argmax_rows_excluding_partial: usize,
    pub(crate) argmax_rows_merge_partial: usize,
    pub(crate) argmax_add_rows_partial: usize,
    pub(crate) sample_top_p_rows_excluding: usize,
    pub(crate) adaln_modulate: usize,
    pub(crate) adaln_modulate_segment: usize,
    pub(crate) gated_residual_segment: usize,
    pub(crate) pack_qkv_bf16: usize,
    pub(crate) pack_gqa_bf16: usize,
    pub(crate) pack_attention_heads_bf16: usize,
    pub(crate) unpack_attention_heads_f32: usize,
    pub(crate) prepare_attention_qkv_bf16: usize,
    pub(crate) pack_attention_kv_native_bf16: usize,
    pub(crate) full_attention: usize,
    pub(crate) block_attention: usize,
    pub(crate) block_attention_prefix_suffix: usize,
    pub(crate) full_attention_128: usize,
    pub(crate) full_attention_128_native_kv: usize,
    pub(crate) full_attention_batched: usize,
    pub(crate) full_attention_batched_128: usize,
    pub(crate) full_attention_batched_64: usize,
    pub(crate) gqa_attention_128: usize,
    pub(crate) gqa_cache_append: usize,
    pub(crate) gqa_decode_partial: usize,
    pub(crate) gqa_decode_merge: usize,
    pub(crate) concat_weight_rows_batched: usize,
    pub(crate) take_rows_batched: usize,
    pub(crate) silu: usize,
    pub(crate) add_row_bias: usize,
    pub(crate) modulation_chunk: usize,
    pub(crate) audio_unpack_affine: usize,
    pub(crate) audio_weight_norm: usize,
    pub(crate) audio_conv1d: usize,
    pub(crate) group_norm_time_isolated: usize,
    pub(crate) snake: usize,
    pub(crate) gelu_tanh: usize,
    pub(crate) channels_to_time: usize,
    pub(crate) causal_attention: usize,
    pub(crate) vision_rope: usize,
    pub(crate) vision_attention: usize,
    pub(crate) scatter_rows: usize,
    pub(crate) audio_conv_transpose1d: usize,
    pub(crate) alias_upsample2: usize,
    pub(crate) snake_beta: usize,
    pub(crate) alias_downsample2: usize,
    pub(crate) scale_tensor: usize,
    pub(crate) tanh_tensor: usize,
    pub(crate) timestep_embedding: usize,
    pub(crate) group_norm: usize,
    pub(crate) conv3d: usize,
    pub(crate) conv3d_single_frame_wmma: usize,
    pub(crate) pack_conv3d_single_frame_weight: usize,
    pub(crate) pixel_shuffle: usize,
}

fn tensor_source() -> String {
    [
        super::super::hiprtc::DEVICE_CONVERSIONS_PREAMBLE,
        super::elementwise::PREFIX,
        super::rope::SOURCE,
        super::elementwise::SELECTION,
        super::attention::SOURCE,
        super::elementwise::DIFFUSION,
        super::super::audio::TENSOR_SOURCE,
        super::elementwise::VAE,
        super::conv3d_vision::SOURCE,
    ]
    .concat()
}

fn tensor_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(&tensor_source(), "zllm_rocm_tensor.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

pub(crate) fn tensor_functions(device_id: i32) -> Result<TensorFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, TensorFunctions), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm tensor kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = tensor_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData tensor"));
        }
        let function = |name: &str| -> Result<usize, String> {
            let symbol = CString::new(name).unwrap();
            let mut function = ptr::null_mut();
            let status = unsafe { module_get_function(&mut function, module, symbol.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, &format!("hipModuleGetFunction tensor::{name}")));
            }
            Ok(function as usize)
        };
        Ok((
            module as usize,
            TensorFunctions {
                add_scaled: function("add_scaled_f32")?,
                subtract: function("subtract_f32")?,
                relative_l1_delta_partial: function("relative_l1_delta_partial_f32")?,
                add_bf16_f32: function("add_bf16_f32")?,
                fill_zero: function("fill_zero_f32")?,
                fill_u32: function("fill_u32")?,
                validate_finite: function("validate_finite_f32")?,
                rmsnorm: function("rmsnorm_to_f32")?,
                rmsnorm_adaln_segment: function("rmsnorm_adaln_segment_f32")?,
                layernorm_bias: function("layernorm_bias_f32")?,
                rmsnorm_heads_unit: function("rmsnorm_heads_unit_f32")?,
                rmsnorm_rope_pair_unit: function("rmsnorm_rope_pair_unit_f32")?,
                scaled_residual_columns: function("scaled_residual_columns_f32")?,
                unpatch_affine: function("unpatch_affine_f32")?,
                sigmoid_gate: function("sigmoid_gate_f32")?,
                split_interleaved_columns: function("split_interleaved_columns_f32")?,
                gated_activation: function("gated_activation_f32")?,
                gated_activation_bf16: function("gated_activation_bf16")?,
                split_gated_activation: function("split_gated_activation_bf16")?,
                split_columns: function("split_columns_f32")?,
                concat_columns: function("concat_columns_f32")?,
                compact_qkv_head_range: function("compact_qkv_head_range_f32")?,
                rope: function("rope_f32")?,
                rope_indirect: function("rope_indirect_f32")?,
                #[cfg(test)]
                rope_segmented_pair: function("rope_segmented_pair_f32")?,
                select_rows: function("select_rows_to_f32")?,
                argmax_rows_excluding: function("argmax_rows_excluding_f32")?,
                argmax_rows_excluding_partial: function("argmax_rows_excluding_partial_f32")?,
                argmax_rows_merge_partial: function("argmax_rows_merge_partial_f32")?,
                argmax_add_rows_partial: function("argmax_add_rows_partial_f32")?,
                sample_top_p_rows_excluding: function("sample_top_p_rows_excluding_f32")?,
                adaln_modulate: function("adaln_modulate_f32")?,
                adaln_modulate_segment: function("adaln_modulate_segment_f32")?,
                gated_residual_segment: function("gated_residual_segment_f32")?,
                pack_qkv_bf16: function("pack_qkv_bf16")?,
                pack_gqa_bf16: function("pack_gqa_bf16")?,
                pack_attention_heads_bf16: function("pack_attention_heads_bf16")?,
                unpack_attention_heads_f32: function("unpack_attention_heads_f32")?,
                prepare_attention_qkv_bf16: function("prepare_attention_qkv_bf16")?,
                pack_attention_kv_native_bf16: function("pack_attention_kv_native_bf16")?,
                full_attention: function("full_attention_f32")?,
                block_attention: function("block_attention_f32")?,
                block_attention_prefix_suffix: function("block_attention_prefix_suffix_f32")?,
                full_attention_128: function("full_attention_f32_128")?,
                full_attention_128_native_kv: function("full_attention_f32_128_native_kv")?,
                full_attention_batched: function("full_attention_batched_f32")?,
                full_attention_batched_128: function("full_attention_batched_f32_128")?,
                full_attention_batched_64: function("full_attention_batched_f32_64")?,
                gqa_attention_128: function("full_attention_gqa_causal_f32_128")?,
                gqa_cache_append: function("gqa_cache_append_f32")?,
                gqa_decode_partial: function("gqa_decode_partial_f32")?,
                gqa_decode_merge: function("gqa_decode_merge_f32")?,
                concat_weight_rows_batched: function("concat_weight_rows_batched_f32")?,
                take_rows_batched: function("take_rows_batched_f32")?,
                silu: function("silu_f32")?,
                add_row_bias: function("add_row_bias_f32")?,
                modulation_chunk: function("modulation_chunk_f32")?,
                audio_unpack_affine: function("audio_unpack_affine_f32")?,
                audio_weight_norm: function("audio_weight_norm_f32")?,
                audio_conv1d: function("audio_conv1d_f32")?,
                group_norm_time_isolated: function("group_norm_time_isolated_f32")?,
                snake: function("snake_f32")?,
                gelu_tanh: function("gelu_tanh_f32")?,
                channels_to_time: function("channels_to_time_f32")?,
                causal_attention: function("causal_attention_f32")?,
                vision_rope: function("vision_rope_f32")?,
                vision_attention: function("vision_attention_f32")?,
                scatter_rows: function("scatter_rows_f32")?,
                audio_conv_transpose1d: function("audio_conv_transpose1d_f32")?,
                alias_upsample2: function("alias_upsample2_f32")?,
                snake_beta: function("snake_beta_f32")?,
                alias_downsample2: function("alias_downsample2_f32")?,
                scale_tensor: function("scale_tensor_f32")?,
                tanh_tensor: function("tanh_tensor_f32")?,
                timestep_embedding: function("timestep_embedding_f32")?,
                group_norm: function("group_norm_f32")?,
                conv3d: function("conv3d_f32")?,
                conv3d_single_frame_wmma: function("conv3d_single_frame_wmma_f32")?,
                pack_conv3d_single_frame_weight: function("pack_conv3d_single_frame_weight_bf16")?,
                pixel_shuffle: function("pixel_shuffle_f32")?,
            },
        ))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

pub(crate) fn current_device() -> Result<i32, String> {
    let runtime = RocmRuntime::open()?;
    let get_device: Symbol<HipGetDevice> = runtime.symbol(&runtime.hip, b"hipGetDevice\0")?;
    let mut device_id = 0;
    let status = unsafe { get_device(&mut device_id) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipGetDevice"));
    }
    Ok(device_id)
}

const DEFERRED_WORKSPACE_SLOTS: usize = 8;
// 少量 in-flight 槽位足以覆盖 owner/peer；过深排队会让同一 NUMA 内两卡的
// allocator 与 P2P 提交集中成簇，反而拉长 stage completion。
const DEFERRED_COOPERATIVE_MOE_WORKSPACE_SLOTS: usize = 4;

struct DeferredTensorWorkspaceSlot {
    workspace: TensorWorkspace,
    event: usize,
}

#[derive(Default)]
struct DeferredTensorWorkspaces {
    available: Vec<DeferredTensorWorkspaceSlot>,
    pending: Vec<DeferredTensorWorkspaceSlot>,
}

thread_local! {
    static DEFERRED_TENSOR_WORKSPACES: RefCell<HashMap<i32, DeferredTensorWorkspaces>> =
        RefCell::new(HashMap::new());
    static DEFERRED_MOE_ROUTE_WORKSPACES: RefCell<HashMap<i32, DeferredTensorWorkspaces>> =
        RefCell::new(HashMap::new());
    static DEFERRED_COOPERATIVE_MOE_WORKSPACES: RefCell<HashMap<i32, DeferredTensorWorkspaces>> =
        RefCell::new(HashMap::new());
}

/// producer 提交后记录 event；未完成的 slot 留在 pending，不阻塞 CPU，也不允许后续 kernel 覆盖。
fn with_deferred_workspace<R>(
    pool: &'static std::thread::LocalKey<RefCell<HashMap<i32, DeferredTensorWorkspaces>>>,
    slot_limit: usize,
    device_id: i32,
    sizes: &[usize],
    run: impl FnOnce(&TensorWorkspace) -> Result<R, String>,
) -> Result<R, String> {
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0")?;
    let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0")?;
    let query: Symbol<HipEventQuery> = runtime.symbol(&runtime.hip, b"hipEventQuery\0")?;
    let synchronize: Symbol<HipEventSynchronize> = runtime.symbol(&runtime.hip, b"hipEventSynchronize\0")?;

    pool.with(|workspaces| {
        let mut workspaces = workspaces.borrow_mut();
        let workspaces = workspaces.entry(device_id).or_default();
        let mut index = 0;
        while index < workspaces.pending.len() {
            let status = unsafe { query(workspaces.pending[index].event as HipEvent) };
            if status == HIP_SUCCESS {
                let slot = workspaces.pending.swap_remove(index);
                workspaces.available.push(slot);
            } else if status == HIP_ERROR_NOT_READY {
                index += 1;
            } else {
                return Err(runtime.hip_error(status, "hipEventQuery deferred tensor workspace"));
            }
        }

        let mut slot = if let Some(slot) = workspaces.available.pop() {
            slot
        } else if workspaces.pending.len() >= slot_limit {
            // 槽位已满时等待最早 producer；这同时约束同一 stage 的提交深度。
            // 真机 trace 证明只做 stream wait 会把 allocator/P2P 提交集中成簇，
            // 拉长 owner completion 并把空泡传播到下游 stage。
            let slot = workspaces.pending.remove(0);
            let status = unsafe { synchronize(slot.event as HipEvent) };
            if status != HIP_SUCCESS {
                workspaces.pending.insert(0, slot);
                return Err(runtime.hip_error(status, "hipEventSynchronize deferred tensor workspace"));
            }
            slot
        } else {
            let mut event = ptr::null_mut();
            let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventCreateWithFlags deferred tensor workspace"));
            }
            DeferredTensorWorkspaceSlot { workspace: TensorWorkspace::default(), event: event as usize }
        };
        if let Err(error) = slot.workspace.ensure(device_id, sizes) {
            workspaces.available.push(slot);
            return Err(error);
        }

        let result = run(&slot.workspace);
        let status = unsafe { record(slot.event as HipEvent, crate::kernel::rocm::hip::active_compute_stream()) };
        if status != HIP_SUCCESS {
            let sync = synchronize_device(device_id, "deferred tensor workspace event fallback");
            workspaces.available.push(slot);
            return sync.and_then(|()| Err(runtime.hip_error(status, "hipEventRecord deferred tensor workspace")));
        }
        workspaces.pending.push(slot);
        result
    })
}

pub(crate) fn with_deferred_tensor_workspace<R>(device_id: i32, sizes: &[usize], run: impl FnOnce(&TensorWorkspace) -> Result<R, String>) -> Result<R, String> {
    with_deferred_workspace(&DEFERRED_TENSOR_WORKSPACES, DEFERRED_WORKSPACE_SLOTS, device_id, sizes, run)
}

pub(crate) fn with_deferred_moe_route_workspace<R>(device_id: i32, sizes: &[usize], run: impl FnOnce(&TensorWorkspace) -> Result<R, String>) -> Result<R, String> {
    with_deferred_workspace(&DEFERRED_MOE_ROUTE_WORKSPACES, DEFERRED_WORKSPACE_SLOTS, device_id, sizes, run)
}

/// cooperative MoE 的槽数与双卡组最大层数绑定，避免按 chunk 无限增长。
pub(crate) fn with_deferred_cooperative_moe_workspace<R>(device_id: i32, sizes: &[usize], run: impl FnOnce(&TensorWorkspace) -> Result<R, String>) -> Result<R, String> {
    with_deferred_workspace(&DEFERRED_COOPERATIVE_MOE_WORKSPACES, DEFERRED_COOPERATIVE_MOE_WORKSPACE_SLOTS, device_id, sizes, run)
}

pub(crate) fn release_deferred_tensor_workspace(device_id: i32) {
    let Ok(runtime) = RocmRuntime::open() else { return };
    let Ok(destroy) = runtime.symbol::<HipEventDestroy>(&runtime.hip, b"hipEventDestroy\0") else { return };
    for pool in [&DEFERRED_TENSOR_WORKSPACES, &DEFERRED_MOE_ROUTE_WORKSPACES, &DEFERRED_COOPERATIVE_MOE_WORKSPACES] {
        pool.with(|workspaces| {
            let Some(workspaces) = workspaces.borrow_mut().remove(&device_id) else {
                return;
            };
            for slot in workspaces.available.into_iter().chain(workspaces.pending) {
                let _ = unsafe { destroy(slot.event as HipEvent) };
            }
        });
    }
}

pub(crate) fn with_tensor_workspace<R>(device_id: i32, sizes: &[usize], run: impl FnOnce(&TensorWorkspace) -> Result<R, String>) -> Result<R, String> {
    TENSOR_WORKSPACES.with(|workspaces| {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        workspace.ensure(device_id, sizes)?;
        run(workspace)
    })
}

pub(crate) fn with_attention_workspace<R>(device_id: i32, sizes: &[usize], run: impl FnOnce(&TensorWorkspace) -> Result<R, String>) -> Result<R, String> {
    ATTENTION_WORKSPACES.with(|workspaces| {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        workspace.ensure(device_id, sizes)?;
        run(workspace)
    })
}

pub(crate) fn launch_tensor_kernel(function: usize, grid: u32, block: u32, arguments: &mut [*mut c_void], action: &str) -> Result<(), String> {
    let started = super::hip_api_stats::start();
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = unsafe { launch(function as *mut c_void, grid, 1, 1, block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, action));
    }
    if options().kernel_sync {
        synchronize_device(current_device()?, action)?;
    }
    Ok(())
}

pub(crate) fn validate_resident(buffer: &DeviceBuffer, device_id: i32, bytes: usize, what: &str) -> Result<(), String> {
    if buffer.device_id != device_id || buffer.bytes < bytes {
        return Err(format!("{what} resident buffer 不匹配: device={}/{} bytes={}/{}", buffer.device_id, device_id, buffer.bytes, bytes,));
    }
    Ok(())
}

fn try_validate_finite_resident(device_id: i32, input: &DeviceBuffer, offset: usize, elements: usize, element_bytes: usize, input_is_bf16: bool) -> Result<(), String> {
    if elements == 0 {
        return Err("finite scan 元素数为 0".to_owned());
    }
    let end = offset.checked_add(elements).ok_or("finite scan range 溢出")?;
    let bytes = end.checked_mul(element_bytes).ok_or("finite scan 大小溢出")?;
    validate_resident(input, device_id, bytes, "finite scan input")?;
    set_device(device_id)?;
    let invalid = DeviceBuffer::upload(device_id, &0u32.to_ne_bytes())?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_invalid = invalid.pointer;
    let mut offset = u64::try_from(offset).map_err(|_| "finite scan offset 超过 u64".to_owned())?;
    let mut elements = u32::try_from(elements).map_err(|_| "finite scan 元素数超过 u32".to_owned())?;
    let mut input_is_bf16 = u32::from(input_is_bf16);
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_invalid as *mut *mut c_void).cast(), (&mut offset as *mut u64).cast(), (&mut elements as *mut u32).cast(), (&mut input_is_bf16 as *mut u32).cast()];
    launch_tensor_kernel(functions.validate_finite, elements.div_ceil(256), 256, &mut arguments, "HIP finite scan")?;
    let mut host = [0u8; 4];
    invalid.copy_to_host(&mut host)?;
    if u32::from_ne_bytes(host) == 0 { Ok(()) } else { Err("HIP tensor 包含非有限值或异常幅值".to_owned()) }
}

pub fn try_validate_finite_resident_f32(device_id: i32, input: &DeviceBuffer, elements: usize) -> Result<(), String> {
    try_validate_finite_resident(device_id, input, 0, elements, 4, false)
}

pub fn try_validate_finite_resident_range_f32(device_id: i32, input: &DeviceBuffer, offset: usize, elements: usize) -> Result<(), String> {
    try_validate_finite_resident(device_id, input, offset, elements, 4, false)
}

pub fn try_validate_finite_resident_range_bf16(device_id: i32, input: &DeviceBuffer, offset: usize, elements: usize) -> Result<(), String> {
    try_validate_finite_resident(device_id, input, offset, elements, 2, true)
}
