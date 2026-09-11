//! 块内动态卷积与候选链选择；维度和权重编码均由调用方传入。
use super::{tensor::validate_resident, *};

fn functions(device: i32) -> Result<[usize; 3], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, [usize; 3]>>> = OnceLock::new();
    let mut cache = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new())).lock().map_err(|_| "speculative kernel cache poisoned")?;
    if let Some(functions) = cache.get(&device) {
        return Ok(*functions);
    }
    set_device(device)?;
    let code = CODE.get_or_init(|| compile_hip_source(include_str!("speculative/source.hip"), "zllm_speculative.hip")).as_ref().map_err(Clone::clone)?;
    let runtime = RocmRuntime::open()?;
    let load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
    let get: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
    let mut module = ptr::null_mut();
    let status = unsafe { load(&mut module, code.as_ptr().cast()) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "加载 speculative kernel"));
    }
    let mut result = [0; 3];
    for (index, name) in ["grouped_block_conv_f32", "candidate_topk_f32", "candidate_greedy_path_f32"].iter().enumerate() {
        let name = CString::new(*name).unwrap();
        let mut function = ptr::null_mut();
        let status = unsafe { get(&mut function, module, name.as_ptr()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "查找 speculative kernel"));
        }
        result[index] = function as usize;
    }
    cache.insert(device, result);
    Ok(result)
}

fn launch(function: usize, grid: u32, args: &mut [*mut c_void]) -> Result<(), String> {
    let status = unsafe { kernel_launch_trampoline(function as *mut c_void, grid, 1, 1, 256, 1, 1, 0, active_compute_stream(), args.as_mut_ptr(), ptr::null_mut()) };
    if status != HIP_SUCCESS {
        return Err(RocmRuntime::open()?.hip_error(status, "launch speculative kernel"));
    }
    Ok(())
}

fn dimension(value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("speculative kernel dimension={value} 超过 u32"))
}
fn bytes(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or_else(|| format!("speculative buffer shape={shape:?} 溢出"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn grouped_block_conv(
    device: i32,
    input: &DeviceBuffer,
    delta: &DeviceBuffer,
    base: &DeviceBuffer,
    base_bf16: bool,
    rows: usize,
    cols: usize,
    block_rows: usize,
    group_size: usize,
    taps: usize,
    side: usize,
) -> Result<DeviceBuffer, String> {
    if rows == 0 || cols == 0 || group_size == 0 || block_rows == 0 || taps == 0 || taps > block_rows || side > 1 || !rows.is_multiple_of(block_rows) || !cols.is_multiple_of(group_size) {
        return Err("grouped block conv 规格非法".into());
    }
    set_device(device)?;
    let size = bytes(&[rows, cols, 4])?;
    validate_resident(input, device, size, "block conv input")?;
    validate_resident(delta, device, bytes(&[rows, 2, taps, cols / group_size, 4])?, "block conv delta")?;
    validate_resident(base, device, bytes(&[2, taps, cols, if base_bf16 { 2 } else { 4 }])?, "block conv base")?;
    let output = DeviceBuffer::allocate_reusable(device, size)?;
    let mut pointers = [input.pointer, delta.pointer, base.pointer, output.pointer];
    let mut dims = [dimension(rows)?, dimension(cols)?, dimension(block_rows)?, dimension(cols / group_size)?, dimension(taps)?, dimension(side)?, u32::from(base_bf16)];
    let mut args: Vec<*mut c_void> = pointers.iter_mut().map(|p| (p as *mut *mut c_void).cast()).chain(dims.iter_mut().map(|d| (d as *mut u32).cast())).collect();
    launch(functions(device)?[0], dimension((size / 4).div_ceil(256))?, &mut args)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn candidate_greedy(
    device: i32,
    logits: &DeviceBuffer,
    gate: &DeviceBuffer,
    pred: &DeviceBuffer,
    succ: &DeviceBuffer,
    pred_bf16: bool,
    succ_bf16: bool,
    rows: usize,
    vocab: usize,
    rank: usize,
    top_k: usize,
    anchor: u32,
) -> Result<Vec<u32>, String> {
    if rows == 0 || rank == 0 || top_k == 0 || top_k > vocab || anchor as usize >= vocab {
        return Err("candidate greedy 规格非法".into());
    }
    set_device(device)?;
    validate_resident(logits, device, bytes(&[rows, vocab, 4])?, "candidate logits")?;
    validate_resident(gate, device, bytes(&[rows, rank, 4])?, "candidate gate")?;
    validate_resident(pred, device, bytes(&[vocab, rank, if pred_bf16 { 2 } else { 4 }])?, "candidate predecessor")?;
    validate_resident(succ, device, bytes(&[vocab, rank, if succ_bf16 { 2 } else { 4 }])?, "candidate successor")?;
    let ids = DeviceBuffer::allocate_reusable(device, bytes(&[rows, top_k, 4])?)?;
    let unary = DeviceBuffer::allocate_reusable(device, bytes(&[rows, top_k, 4])?)?;
    let output = DeviceBuffer::allocate_reusable(device, bytes(&[rows, 4])?)?;
    let kernels = functions(device)?;
    let mut pointers = [logits.pointer, ids.pointer, unary.pointer];
    let mut dims = [dimension(vocab)?, dimension(top_k)?];
    let mut args: Vec<*mut c_void> = pointers.iter_mut().map(|p| (p as *mut *mut c_void).cast()).chain(dims.iter_mut().map(|d| (d as *mut u32).cast())).collect();
    launch(kernels[1], dimension(rows)?, &mut args)?;
    let mut pointers = [ids.pointer, unary.pointer, gate.pointer, pred.pointer, succ.pointer, output.pointer];
    let mut dims = [dimension(rows)?, dimension(rank)?, dimension(top_k)?, anchor, u32::from(pred_bf16), u32::from(succ_bf16)];
    let mut args: Vec<*mut c_void> = pointers.iter_mut().map(|p| (p as *mut *mut c_void).cast()).chain(dims.iter_mut().map(|d| (d as *mut u32).cast())).collect();
    launch(kernels[2], 1, &mut args)?;
    // 下载只发生在草稿块完成处；copy_to_host 同步当前流后读取最终 IDs。
    let mut data = vec![0u8; rows * 4];
    output.copy_to_host(&mut data)?;
    let tokens: Vec<_> = data.chunks_exact(4).map(|b| u32::from_ne_bytes(b.try_into().unwrap())).collect();
    if tokens.iter().any(|&id| id as usize >= vocab) {
        return Err("candidate greedy 没有有限候选或输入存在非法数值".into());
    }
    Ok(tokens)
}
