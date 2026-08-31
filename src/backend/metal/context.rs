//! Metal device、queue、resident buffer 和 pipeline 生命周期。
//!
//! MetalContext 持有 device、command queue、编译好的 kernel library。
//! 各算子模块通过 `pipeline(name)` 拿 ComputePipelineState。

use super::api::{Buffer, CommandBuffer, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device, FunctionConstantValues, Library, MTLDataType, MTLResourceOptions};
use std::collections::HashMap;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::{ffi::c_void, mem};

use half::{bf16, f16};

/// decode 路径每个 command buffer 的算子上限(经验值,稳态下 GPU 利用率最高)。
pub(super) const DECODE_BATCH_MAX_OPERATIONS: u64 = 16;
/// prefill 路径每个 command buffer 的算子上限。过大会延长大 activation
/// 生命周期并导致 UMA 压力；真机完整 Gemma4 prefill 以 8 为当前最优点。
pub(super) const PREFILL_BATCH_MAX_OPERATIONS: u64 = 8;

#[link(name = "QuartzCore", kind = "framework")]
unsafe extern "C" {
    fn CACurrentMediaTime() -> f64;
}

fn metal_host_seconds() -> f64 {
    unsafe { CACurrentMediaTime() }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalTensorDType {
    F16,
    Bf16,
    F32,
}

/// 行优先 `[rows, cols]` GPU buffer；BF16/F16 都使用真实 16-bit 存储。
#[derive(Clone)]
pub struct MetalTensor {
    pub buffer: Buffer,
    pub rows: usize,
    pub cols: usize,
    pub dtype: MetalTensorDType,
}

impl MetalTensor {
    pub fn new(buffer: Buffer, rows: usize, cols: usize) -> Self {
        Self { buffer, rows, cols, dtype: MetalTensorDType::F16 }
    }

    pub fn new_f32(buffer: Buffer, rows: usize, cols: usize) -> Self {
        Self { buffer, rows, cols, dtype: MetalTensorDType::F32 }
    }

    pub fn new_bf16(buffer: Buffer, rows: usize, cols: usize) -> Self {
        Self { buffer, rows, cols, dtype: MetalTensorDType::Bf16 }
    }

    pub fn reshape(&self, rows: usize, cols: usize) -> Self {
        let elements = rows.checked_mul(cols).expect("MetalTensor reshape 元素数溢出");
        assert_eq!(self.len(), elements, "MetalTensor reshape 元素数不匹配");
        Self { buffer: self.buffer.clone(), rows, cols, dtype: self.dtype }
    }

    pub fn len(&self) -> usize {
        self.rows.checked_mul(self.cols).expect("MetalTensor 元素数溢出")
    }
}

/// 分配字节数显式溢出检查：乘积静默回绕会分配出过小的 buffer，随后 kernel 越界写。
fn checked_tensor_bytes(rows: usize, cols: usize, element_size: usize) -> usize {
    rows.checked_mul(cols).and_then(|elements| elements.checked_mul(element_size)).expect("MetalTensor 分配大小溢出")
}

/// Metal 执行上下文。一个 device + 一个 queue + kernel pipeline 缓存。
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    library: Library,
    pipelines: Mutex<HashMap<String, ComputePipelineState>>,
    defer_waits: AtomicBool,
    defer_layer_scope_sync: AtomicBool,
    decode_batch_max_operations: AtomicU64,
    deferred_batch_max_operations: AtomicU64,
    decode_batch: Mutex<Option<DecodeCommandBatch>>,
    /// 自上次 CB 提交以来分配的临时 buffer;flush 时移交 PendingMetalProfile,
    /// 在该 CB 完成前保持存活(防止 driver 在 CB 在飞时复用页面污染在飞读)。
    batch_keep_alive: Mutex<Vec<Buffer>>,
    pending_profiles: Mutex<Vec<PendingMetalProfile>>,
    resident_f16_weights: Mutex<HashMap<([u8; 32], usize), Buffer>>,
    resident_byte_weights: Mutex<HashMap<([u8; 32], usize), Buffer>>,
    decode_rope_table: Mutex<Option<DecodeRopeTable>>,
    routing_readback: Mutex<Option<RoutingReadbackBuffers>>,
    token_readback: Mutex<Option<Buffer>>,
    f16_casts: Mutex<HashMap<(usize, usize), (Buffer, MetalTensor)>>,
    /// 会话期稳定的权重 F32->F16 转换缓存；key 是源 buffer 地址，value 持有
    /// 源 buffer 引用防止地址复用，因此不会被 batch 边界清空。
    weight_f16_casts: Mutex<HashMap<usize, (Buffer, MetalTensor)>>,
    /// 按(用途,尺寸)池化的大 workspace 复用(GDN chunked/GEMM accumulator 等,
    /// kernel 全覆盖写,免清零)。
    scratch_pool: Mutex<HashMap<(&'static str, usize, usize), Buffer>>,
    // decode CPU 成本打点:tensor 分配与提交路径的累计纳秒(原子,~ns 级开销)
    decode_alloc_nanoseconds: AtomicU64,
    decode_commit_nanoseconds: AtomicU64,
    submit_wait_nanoseconds: AtomicU64,
    gpu_nanoseconds: AtomicU64,
    inter_command_gap_nanoseconds: AtomicU64,
    last_profile_gpu_end_bits: AtomicU64,
    completion_tail_nanoseconds: AtomicU64,
    command_buffers: AtomicU64,
    detailed_gpu_profiles: AtomicBool,
    gpu_profiles: Mutex<HashMap<(String, String), MetalGpuProfileCounter>>,
    metal4: bool,
    /// backend 级执行策略：Metal 默认开启静态命令重放。
    replay_enabled: bool,
}

/// 常驻 GPU 的 decode RoPE 全表 F16(cos/sin 各一份)。按 CPU 源表的
/// (指针,长度,half_dim) 身份缓存:RopeTable 会话期常驻,指针稳定即同一张表。
struct DecodeRopeTable {
    source: (*const f32, usize),
    half_dim: usize,
    cos: Buffer,
    sin: Buffer,
}

struct RoutingReadbackBuffers {
    ids: Buffer,
    weights: Buffer,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MetalGpuStats {
    pub submit_wait_seconds: f64,
    pub seconds: f64,
    pub inter_command_gap_seconds: f64,
    pub completion_tail_seconds: f64,
    pub command_buffers: u64,
}

/// 同一算子与 shape 的 command-buffer 级 GPU 聚合统计。
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetalGpuProfile {
    pub operator: String,
    pub shape: String,
    pub calls: u64,
    pub gpu_seconds: f64,
    pub estimated_read_bytes: u64,
    pub estimated_write_bytes: u64,
}

#[derive(Default)]
struct MetalGpuProfileCounter {
    calls: u64,
    gpu_nanoseconds: u64,
    estimated_read_bytes: u64,
    estimated_write_bytes: u64,
}

struct PendingMetalProfile {
    command: CommandBuffer,
    submitted_host_seconds: f64,
    operator: Option<String>,
    shape: Option<String>,
    estimated_read_bytes: u64,
    estimated_write_bytes: u64,
    /// 本 CB 期间分配的中间 buffer:持有到 CB 完成才释放,防止 driver 在
    /// CB 仍在飞时复用页面(CPU memset/新 kernel 写污染在飞读,见竞态定位记录)。
    _keep_alive: Vec<Buffer>,
}

struct DecodeCommandBatch {
    command: CommandBuffer,
    operations: u64,
    estimated_read_bytes: u64,
    estimated_write_bytes: u64,
    last_operator: String,
    last_shape: String,
}

impl MetalContext {
    pub fn new(kernel_source: &str) -> Result<Self, String> {
        Self::new_with_replay(kernel_source, true)
    }

    pub fn new_with_replay(kernel_source: &str, replay_enabled: bool) -> Result<Self, String> {
        let device = Device::system_default().ok_or_else(|| "Metal device not found".to_owned())?;
        // Metal 4 cooperative tensor 在新设备上显著加速量化 prefill；旧系统或
        // 旧 GPU 编译失败时仍用同一源码的 MSL 默认版本，保留传统 kernel。
        let (library, metal4_source) = match device.new_library_with_source(kernel_source, &CompileOptions::metal4()) {
            Ok(library) => (library, true),
            Err(_) => (device.new_library_with_source(kernel_source, &CompileOptions::new()).map_err(|e| format!("编译 Metal library: {e}"))?, false),
        };
        let mut pipelines = HashMap::new();
        // MSL 4 能编译不代表当前 GPU 支持 cooperative tensor。预建一次 PSO，
        // 只有 function 与 pipeline 都成功才开放快路径，否则继续用旧 fused kernel。
        let metal4 = if metal4_source {
            library.get_function("gguf_gemm_iq4nl_mpp_f16", None).ok().and_then(|function| device.new_compute_pipeline_state_icb(&function).ok()).is_some_and(|pipeline| {
                pipelines.insert("gguf_gemm_iq4nl_mpp_f16".to_owned(), pipeline);
                true
            })
        } else {
            false
        };
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            library,
            pipelines: Mutex::new(pipelines),
            defer_waits: AtomicBool::new(false),
            defer_layer_scope_sync: AtomicBool::new(false),
            decode_batch_max_operations: AtomicU64::new(DECODE_BATCH_MAX_OPERATIONS),
            deferred_batch_max_operations: AtomicU64::new(DECODE_BATCH_MAX_OPERATIONS),
            decode_batch: Mutex::new(None),
            batch_keep_alive: Mutex::new(Vec::new()),
            pending_profiles: Mutex::new(Vec::new()),
            resident_f16_weights: Mutex::new(HashMap::new()),
            resident_byte_weights: Mutex::new(HashMap::new()),
            decode_rope_table: Mutex::new(None),
            routing_readback: Mutex::new(None),
            token_readback: Mutex::new(None),
            f16_casts: Mutex::new(HashMap::new()),
            weight_f16_casts: Mutex::new(HashMap::new()),
            scratch_pool: Mutex::new(HashMap::new()),
            decode_alloc_nanoseconds: AtomicU64::new(0),
            decode_commit_nanoseconds: AtomicU64::new(0),
            submit_wait_nanoseconds: AtomicU64::new(0),
            gpu_nanoseconds: AtomicU64::new(0),
            inter_command_gap_nanoseconds: AtomicU64::new(0),
            last_profile_gpu_end_bits: AtomicU64::new(0),
            completion_tail_nanoseconds: AtomicU64::new(0),
            command_buffers: AtomicU64::new(0),
            detailed_gpu_profiles: AtomicBool::new(false),
            gpu_profiles: Mutex::new(HashMap::new()),
            metal4,
            replay_enabled,
        })
    }

    pub fn new_default() -> Result<Self, String> {
        Self::new_default_with_replay(true)
    }

    pub fn new_default_with_replay(replay_enabled: bool) -> Result<Self, String> {
        Self::new_with_replay(crate::kernel::metal::kernels_source(), replay_enabled)
    }

    pub fn replay_enabled(&self) -> bool {
        self.replay_enabled
    }

    pub(crate) fn metal4_available(&self) -> bool {
        self.metal4
    }

    /// 拿(按需缓存)kernel function 的 pipeline state。统一走 descriptor 路径并打开
    /// supportIndirectCommandBuffers:普通 encoder 与 ICB 命令都能引用同一份 PSO,
    /// 转录录制到的 PSO 无需二次编译。
    pub fn pipeline(&self, name: &str) -> Result<ComputePipelineState, String> {
        if let Some(p) = self.pipelines.lock().map_err(|_| "pipelines 锁中毒".to_owned())?.get(name) {
            return Ok(p.clone());
        }
        let function = self.library.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
        let pipeline = self.device.new_compute_pipeline_state_icb(&function).map_err(|e| format!("pipeline {name}: {e}"))?;
        self.pipelines.lock().map_err(|_| "pipelines 锁中毒".to_owned())?.insert(name.to_owned(), pipeline.clone());
        Ok(pipeline)
    }

    #[cfg(test)]
    pub(crate) fn cached_pipeline_name(&self, target: &ComputePipelineState) -> Option<String> {
        self.pipelines.lock().ok()?.iter().find_map(|(name, pipeline)| pipeline.same_handle(target).then(|| name.clone()))
    }

    /// 维度仍由 spec 提供，但作为 function constants 编译一次，避免动态循环阻止 Metal 展开。
    pub fn pipeline_u32_constants(&self, name: &str, values: &[u32]) -> Result<ComputePipelineState, String> {
        let key = format!("{name}<{values:?}>");
        if let Some(pipeline) = self.pipelines.lock().map_err(|_| "pipelines 锁中毒".to_owned())?.get(&key) {
            return Ok(pipeline.clone());
        }
        let constants = FunctionConstantValues::new();
        for (index, value) in values.iter().enumerate() {
            constants.set_constant_value_at_index(value as *const u32 as *const c_void, MTLDataType::UInt, index as u64);
        }
        let function = self.library.get_function(name, Some(constants)).map_err(|error| format!("kernel {key}: {error}"))?;
        let pipeline = self.device.new_compute_pipeline_state_icb(&function).map_err(|error| format!("pipeline {key}: {error}"))?;
        self.pipelines.lock().map_err(|_| "pipelines 锁中毒".to_owned())?.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// 创建 SharedMemory(MTLResourceOptions::StorageModeShared)buffer,CPU/GPU 都可访问。
    pub fn shared_buffer(&self, bytes: &[u8]) -> Buffer {
        self.try_shared_buffer(bytes).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_shared_buffer(&self, bytes: &[u8]) -> Result<Buffer, String> {
        self.device.try_new_buffer_with_data(bytes.as_ptr() as *const _, bytes.len() as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked)
    }

    /// 取(用途,槽位,尺寸)对应的池化 workspace;同尺寸多实例并存时用槽位区分,
    /// 例如 PLE 每步 42 个同宽切片必须同时存活,不能按尺寸折叠到同一块 buffer。
    pub fn cached_zero_buffer_slot(&self, tag: &'static str, slot: usize, len: usize) -> Buffer {
        let mut pool = self.scratch_pool.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let key = (tag, len, slot);
        if let Some(buffer) = pool.get(&key) {
            return buffer.clone();
        }
        let buffer = self.shared_buffer_zeros(len);
        pool.insert(key, buffer.clone());
        buffer
    }

    /// 取(用途,尺寸)对应的池化 workspace:命中直接复用,未命中分配后登记。
    pub fn cached_zero_buffer(&self, tag: &'static str, len: usize) -> Buffer {
        self.cached_zero_buffer_slot(tag, 0, len)
    }

    /// 创建空 SharedMemory buffer。
    pub fn shared_buffer_zeros(&self, len: usize) -> Buffer {
        self.try_shared_buffer_zeros(len).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_shared_buffer_zeros(&self, len: usize) -> Result<Buffer, String> {
        let buffer = self.device.try_new_buffer(len as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked)?;
        if len != 0 {
            unsafe { std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, len) };
        }
        self.batch_keep_alive.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(buffer.clone());
        Ok(buffer)
    }

    /// 创建由调用方完整写入后再读取的 SharedMemory buffer。
    pub fn shared_buffer_uninit(&self, len: usize) -> Buffer {
        self.try_shared_buffer_uninit(len).unwrap_or_else(|error| panic!("{error}"))
    }

    /// 诊断用:保活 buffer 防驱动层页面复用(仅调试竞态时使用,泄露进程生命周期)。
    pub fn retain_debug_buffers(&self, buffers: Vec<Buffer>) {
        static GRAVEYARD: std::sync::Mutex<Vec<Buffer>> = std::sync::Mutex::new(Vec::new());
        GRAVEYARD.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).extend(buffers);
    }

    pub fn try_shared_buffer_uninit(&self, len: usize) -> Result<Buffer, String> {
        let buffer = self.device.try_new_buffer(len as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked)?;
        self.batch_keep_alive.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(buffer.clone());
        Ok(buffer)
    }

    pub fn routing_readback_buffers(&self, len: usize) -> (Buffer, Buffer) {
        let bytes = len.saturating_mul(mem::size_of::<u32>());
        let mut cached = self.routing_readback.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if cached.as_ref().is_none_or(|buffers| buffers.ids.length() < bytes as u64) {
            *cached = Some(RoutingReadbackBuffers { ids: self.shared_buffer_uninit(bytes), weights: self.shared_buffer_uninit(bytes) });
        }
        let buffers = cached.as_ref().expect("routing readback buffers 已初始化");
        (buffers.ids.clone(), buffers.weights.clone())
    }

    pub fn token_readback_buffer(&self) -> Buffer {
        let mut cached = self.token_readback.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        cached.get_or_insert_with(|| self.shared_buffer_uninit(mem::size_of::<u32>())).clone()
    }

    pub fn shared_buffer_from_f32(&self, data: &[f32]) -> Buffer {
        self.try_shared_buffer_from_f32(data).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_shared_buffer_from_f32(&self, data: &[f32]) -> Result<Buffer, String> {
        let half: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
        let bytes = unsafe { std::slice::from_raw_parts(half.as_ptr() as *const u8, half.len() * mem::size_of::<f16>()) };
        self.device.try_new_buffer_with_data(bytes.as_ptr() as *const c_void, bytes.len() as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked)
    }

    /// 以内容身份缓存一次性 F16 上传；临时 Vec 即使复用宿主地址也不会命中旧权重。
    pub(crate) fn resident_f16_weight_buffer(&self, data: &[f32]) -> Buffer {
        self.try_resident_f16_weight_buffer(data).unwrap_or_else(|error| panic!("{error}"))
    }

    pub(crate) fn try_resident_f16_weight_buffer(&self, data: &[f32]) -> Result<Buffer, String> {
        if !self.defer_waits.load(Ordering::Acquire) || mem::size_of_val(data) <= 64 {
            return self.try_shared_buffer_from_f32(data);
        }
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), mem::size_of_val(data)) };
        let key = (*blake3::hash(bytes).as_bytes(), data.len());
        if let Some(buffer) = self.resident_f16_weights.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get(&key) {
            return Ok(buffer.clone());
        }
        let buffer = self.try_shared_buffer_from_f32(data)?;
        self.resident_f16_weights.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).insert(key, buffer.clone());
        Ok(buffer)
    }

    /// 以内容身份缓存大权重；小控制参数直接上传，避免为短值维护缓存。
    pub(crate) fn resident_byte_weight_buffer(&self, data: &[u8]) -> Buffer {
        self.try_resident_byte_weight_buffer(data).unwrap_or_else(|error| panic!("{error}"))
    }

    pub(crate) fn try_resident_byte_weight_buffer(&self, data: &[u8]) -> Result<Buffer, String> {
        if !self.defer_waits.load(Ordering::Acquire) || data.len() <= 64 {
            return self.try_shared_buffer(data);
        }
        let key = (*blake3::hash(data).as_bytes(), data.len());
        if let Some(buffer) = self.resident_byte_weights.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get(&key) {
            return Ok(buffer.clone());
        }
        let buffer = self.try_shared_buffer(data)?;
        self.resident_byte_weights.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).insert(key, buffer.clone());
        Ok(buffer)
    }

    pub fn read_f16_to_f32(&self, buffer: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buffer.contents() as *const f16;
        let data = unsafe { std::slice::from_raw_parts(ptr, len) };
        data.iter().map(|v| f16::to_f32(*v)).collect()
    }

    pub fn tensor_from_f32(&self, data: &[f32], rows: usize, cols: usize) -> Result<MetalTensor, String> {
        let elements = rows.checked_mul(cols).ok_or("MetalTensor shape 溢出")?;
        if data.len() != elements {
            return Err(format!("MetalTensor 上传长度不匹配: 实际={}, 期望={elements}", data.len()));
        }
        Ok(MetalTensor::new(self.try_shared_buffer_from_f32(data)?, rows, cols))
    }

    /// 由原始 F16 bit 直接构造 tensor(终点快照恢复路径)。
    pub fn tensor_from_f16_bits(&self, bytes: &[u8], rows: usize, cols: usize) -> Result<MetalTensor, String> {
        if bytes.len() != rows * cols * 2 {
            return Err(format!("F16 bits 长度 {}，期望 {}", bytes.len(), rows * cols * 2));
        }
        let buffer = self.try_shared_buffer(bytes)?;
        Ok(MetalTensor::new(buffer, rows, cols))
    }

    /// 由原始 BF16 bit 直接构造 tensor，不能借用 F16 构造器解释相同字节。
    pub fn tensor_from_bf16_bits(&self, bytes: &[u8], rows: usize, cols: usize) -> Result<MetalTensor, String> {
        let expected = rows.checked_mul(cols).and_then(|count| count.checked_mul(2)).ok_or("BF16 bits shape 溢出")?;
        if bytes.len() != expected {
            return Err(format!("BF16 bits 长度 {}，期望 {expected}", bytes.len()));
        }
        Ok(MetalTensor::new_bf16(self.try_shared_buffer(bytes)?, rows, cols))
    }

    pub fn tensor_from_f32_bf16(&self, data: &[f32], rows: usize, cols: usize) -> Result<MetalTensor, String> {
        let elements = rows.checked_mul(cols).ok_or("MetalTensor BF16 shape 溢出")?;
        if data.len() != elements {
            return Err(format!("MetalTensor BF16 上传长度不匹配: 实际={}, 期望={elements}", data.len()));
        }
        let values: Vec<bf16> = data.iter().copied().map(bf16::from_f32).collect();
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), mem::size_of_val(values.as_slice())) };
        Ok(MetalTensor::new_bf16(self.try_shared_buffer(bytes)?, rows, cols))
    }

    pub fn tensor_from_f32_preserve(&self, data: &[f32], rows: usize, cols: usize) -> Result<MetalTensor, String> {
        let elements = rows.checked_mul(cols).ok_or("MetalTensor F32 shape 溢出")?;
        if data.len() != elements {
            return Err(format!("MetalTensor F32 上传长度不匹配: 实际={}, 期望={elements}", data.len()));
        }
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        Ok(MetalTensor::new_f32(self.try_shared_buffer(bytes)?, rows, cols))
    }

    /// bf16 行 → F16 tensor,每调用新 buffer:embedding 行逐 token 内容不同,
    /// 无"常驻表+索引"结构;复用同一 buffer 的 CPU 覆写会与推测提交窗口内
    /// 在飞 kernel 的读重叠(同 rope 行 buffer 竞态),新 buffer 由 CB retain
    /// 保活到执行完,天然无竞态。
    pub fn tensor_from_bf16_row(&self, data: &[u8], cols: usize) -> Result<MetalTensor, String> {
        if data.len() != cols.checked_mul(mem::size_of::<bf16>()).ok_or("decode embedding 大小溢出")? {
            return Err(format!("decode BF16 embedding 长度 {}，期望 {}", data.len(), cols * mem::size_of::<bf16>()));
        }
        let packed: Vec<u16> = data.chunks_exact(2).map(|bytes| f16::from_f32(bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).to_bits()).collect();
        let bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) };
        Ok(MetalTensor::new(self.try_shared_buffer(bytes)?, 1, cols))
    }

    /// Decode 的 RoPE 全表 F16 常驻 buffer:CPU 源表(precompute 后不变的 RopeTable)
    /// 首次调用时一次性转换上传,之后每 token 只传 position(kernel 自行索引全表),
    /// CPU 不再逐 token 切片/覆写 GPU buffer——彻底消除推测提交窗口内 CPU 写与
    /// 在飞 rope kernel 读的重叠。按(指针,长度,half_dim)身份命中:RopeTable
    /// 会话期常驻,指针稳定即同一张表;表被替换(新指针)则重新上传。
    pub fn decode_rope_table_buffers(&self, cos: &[f32], sin: &[f32], half_dim: usize) -> Result<(Buffer, Buffer), String> {
        if cos.len() != sin.len() {
            return Err(format!("decode RoPE 全表 cos/sin 长度不一致: cos={} sin={}", cos.len(), sin.len()));
        }
        if !cos.len().is_multiple_of(half_dim) {
            return Err(format!("decode RoPE 全表长度 {} 不是 half_dim {half_dim} 的整数倍", cos.len()));
        }
        let identity = (cos.as_ptr(), cos.len());
        let mut cached = self.decode_rope_table.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if cached.as_ref().is_none_or(|table| table.source != identity || table.half_dim != half_dim) {
            let upload = |values: &[f32]| -> Result<Buffer, String> {
                let packed: Vec<u16> = values.iter().map(|&value| f16::from_f32(value).to_bits()).collect();
                let bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) };
                self.try_shared_buffer(bytes)
            };
            *cached = Some(DecodeRopeTable { source: identity, half_dim, cos: upload(cos)?, sin: upload(sin)? });
        }
        let table = cached.as_ref().expect("decode RoPE 全表已初始化");
        Ok((table.cos.clone(), table.sin.clone()))
    }

    pub fn tensor_zeros(&self, rows: usize, cols: usize) -> MetalTensor {
        let started = std::time::Instant::now();
        let tensor = MetalTensor::new(self.shared_buffer_zeros(checked_tensor_bytes(rows, cols, mem::size_of::<f16>())), rows, cols);
        self.decode_alloc_nanoseconds.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        tensor
    }

    pub fn tensor_uninit(&self, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new(self.shared_buffer_uninit(checked_tensor_bytes(rows, cols, mem::size_of::<f16>())), rows, cols)
    }

    /// Kernel 必须完成全部输出；单行 decode 保留清零以兼容累加算子，多行避免重复清零。
    pub(crate) fn tensor_kernel_output(&self, rows: usize, cols: usize) -> MetalTensor {
        if rows == 1 { self.tensor_zeros(rows, cols) } else { self.tensor_uninit(rows, cols) }
    }

    pub fn tensor_zeros_bf16(&self, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new_bf16(self.shared_buffer_zeros(checked_tensor_bytes(rows, cols, mem::size_of::<bf16>())), rows, cols)
    }

    pub fn tensor_uninit_bf16(&self, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new_bf16(self.shared_buffer_uninit(checked_tensor_bytes(rows, cols, mem::size_of::<bf16>())), rows, cols)
    }

    pub(crate) fn tensor_kernel_output_bf16(&self, rows: usize, cols: usize) -> MetalTensor {
        if rows == 1 { self.tensor_zeros_bf16(rows, cols) } else { self.tensor_uninit_bf16(rows, cols) }
    }

    pub fn tensor_zeros_f32(&self, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new_f32(self.shared_buffer_zeros(checked_tensor_bytes(rows, cols, mem::size_of::<f32>())), rows, cols)
    }

    pub fn tensor_uninit_f32(&self, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new_f32(self.shared_buffer_uninit(checked_tensor_bytes(rows, cols, mem::size_of::<f32>())), rows, cols)
    }

    /// 池化的 F16 kernel 输出:kernel 保证完整写出,decode 高频小切分按尺寸复用,
    /// 消掉每次调用的新 MetalBuffer 分配 + CPU 清零 churn。
    pub fn tensor_pooled(&self, tag: &'static str, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new(self.cached_zero_buffer(tag, checked_tensor_bytes(rows, cols, mem::size_of::<f16>())), rows, cols)
    }

    /// 池化的 BF16 kernel 输出,语义同 [`Self::tensor_pooled`]。
    pub fn tensor_pooled_bf16(&self, tag: &'static str, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new_bf16(self.cached_zero_buffer(tag, checked_tensor_bytes(rows, cols, mem::size_of::<bf16>())), rows, cols)
    }

    /// 池化的 F32 workspace；同一 queue 上按“写入→消费→下次写入”顺序复用。
    pub(crate) fn tensor_pooled_f32(&self, tag: &'static str, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new_f32(self.cached_zero_buffer(tag, checked_tensor_bytes(rows, cols, mem::size_of::<f32>())), rows, cols)
    }

    /// 池化且按槽位区分的 F16 kernel 输出:同尺寸的多个切片需要同时存活时使用。
    pub fn tensor_pooled_slot(&self, tag: &'static str, slot: usize, rows: usize, cols: usize) -> MetalTensor {
        MetalTensor::new(self.cached_zero_buffer_slot(tag, slot, checked_tensor_bytes(rows, cols, mem::size_of::<f16>())), rows, cols)
    }

    pub(crate) fn tensor_kernel_output_f32(&self, rows: usize, cols: usize) -> MetalTensor {
        if rows == 1 { self.tensor_zeros_f32(rows, cols) } else { self.tensor_uninit_f32(rows, cols) }
    }

    pub fn tensor_to_f32(&self, tensor: &MetalTensor) -> Vec<f32> {
        self.synchronize();
        match tensor.dtype {
            MetalTensorDType::F16 => self.read_f16_to_f32(&tensor.buffer, tensor.len()),
            MetalTensorDType::Bf16 => unsafe { std::slice::from_raw_parts(tensor.buffer.contents().cast::<bf16>(), tensor.len()) }.iter().map(|value| value.to_f32()).collect(),
            MetalTensorDType::F32 => unsafe { std::slice::from_raw_parts(tensor.buffer.contents().cast::<f32>(), tensor.len()) }.to_vec(),
        }
    }

    /// 只下载一行，避免 prefill 结束时把整段 hidden 往返 CPU。
    pub fn tensor_row_to_f32(&self, tensor: &MetalTensor, row: usize) -> Result<Vec<f32>, String> {
        if row >= tensor.rows {
            return Err(format!("MetalTensor 行 {row} 越界: rows={}", tensor.rows));
        }
        self.synchronize();
        let element_size = match tensor.dtype {
            MetalTensorDType::F16 => mem::size_of::<f16>(),
            MetalTensorDType::Bf16 => mem::size_of::<bf16>(),
            MetalTensorDType::F32 => mem::size_of::<f32>(),
        };
        let offset = row.checked_mul(tensor.cols).and_then(|value| value.checked_mul(element_size)).ok_or("MetalTensor 行 offset 溢出")?;
        let ptr = unsafe { tensor.buffer.contents().cast::<u8>().add(offset) };
        Ok(match tensor.dtype {
            MetalTensorDType::F16 => unsafe { std::slice::from_raw_parts(ptr.cast::<f16>(), tensor.cols) }.iter().map(|value| value.to_f32()).collect(),
            MetalTensorDType::Bf16 => unsafe { std::slice::from_raw_parts(ptr.cast::<bf16>(), tensor.cols) }.iter().map(|value| value.to_f32()).collect(),
            MetalTensorDType::F32 => unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), tensor.cols) }.to_vec(),
        })
    }

    pub(crate) fn cached_f16_cast(&self, input: &MetalTensor) -> Option<MetalTensor> {
        if !self.defer_waits.load(Ordering::Acquire) || input.dtype != MetalTensorDType::F32 {
            return None;
        }
        let source = input.buffer.contents() as usize;
        if source == 0 {
            return None;
        }
        self.f16_casts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get(&(source, input.len())).map(|(_, output)| output.clone())
    }

    pub(crate) fn retain_f16_cast(&self, input: &MetalTensor, output: &MetalTensor) {
        if !self.defer_waits.load(Ordering::Acquire) || input.dtype != MetalTensorDType::F32 {
            return;
        }
        let source = input.buffer.contents() as usize;
        if source == 0 {
            return;
        }
        self.f16_casts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).insert((source, input.len()), (input.buffer.clone(), output.clone()));
    }

    pub(crate) fn invalidate_f16_cast(&self, input: &MetalTensor) {
        let source = input.buffer.contents() as usize;
        if source != 0 {
            self.f16_casts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).remove(&(source, input.len()));
        }
    }

    /// 权重级 F32->F16 转换：与 activation cast 不同，权重在会话期不变，
    /// 缓存跨 decode batch 存活；value 持有源 buffer 防止地址被复用。
    pub(crate) fn weight_f16_view(&self, source: &Buffer, len: usize) -> Option<MetalTensor> {
        let key = source.contents() as usize;
        if key == 0 {
            return None;
        }
        self.weight_f16_casts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get(&key).filter(|(_, tensor)| tensor.len() == len).map(|(_, tensor)| tensor.clone())
    }

    pub(crate) fn retain_weight_f16_view(&self, source: &Buffer, output: &MetalTensor) {
        let key = source.contents() as usize;
        if key == 0 {
            return;
        }
        self.weight_f16_casts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).insert(key, (source.clone(), output.clone()));
    }

    pub(crate) fn clear_f16_casts(&self) {
        self.f16_casts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clear();
    }

    /// 提交并等待一个 command buffer，同时累计 Metal 报告的真实 GPU 执行时间。
    pub fn commit_and_wait(&self, command: &CommandBufferRef) {
        self.commit_and_wait_profiled(command, "unprofiled", "unknown", 0, 0);
    }

    /// Decode 模式复用当前 batch；普通模式返回独立 command buffer。
    pub fn command_buffer(&self) -> CommandBuffer {
        if !self.defer_waits.load(Ordering::Acquire) {
            return self.queue.new_command_buffer().to_owned();
        }
        let mut batch = self.decode_batch.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if batch.is_none() {
            *batch = Some(DecodeCommandBatch { command: self.queue.new_command_buffer().to_owned(), operations: 0, estimated_read_bytes: 0, estimated_write_bytes: 0, last_operator: String::new(), last_shape: String::new() });
        }
        batch.as_ref().expect("decode batch 已创建").command.to_owned()
    }

    /// Decode 的 GPU-only 算子按 queue 顺序提交，只在 router/最终 logits 的 CPU 读回边界等待。
    pub fn set_deferred_waits(&self, enabled: bool) {
        if !enabled && self.defer_waits.load(Ordering::Acquire) {
            self.synchronize();
        }
        self.defer_waits.store(enabled, Ordering::Release);
    }

    pub fn deferred_waits_enabled(&self) -> bool {
        self.defer_waits.load(Ordering::Acquire)
    }

    /// 诊断模式开关:reset_gpu_stats 打开、gpu_profile 取走结果后关闭。
    pub fn detailed_gpu_profiles_enabled(&self) -> bool {
        self.detailed_gpu_profiles.load(Ordering::Acquire)
    }

    pub fn layer_scope_requires_sync(&self) -> bool {
        !self.defer_layer_scope_sync.load(Ordering::Acquire)
    }

    pub fn set_deferred_layer_scope_sync(&self, enabled: bool) {
        self.defer_layer_scope_sync.store(enabled, Ordering::Release);
    }

    /// 临时改变延迟模式的 command-buffer 算子上限；返回旧值供调用方恢复。
    pub fn set_deferred_batch_max_operations(&self, operations: u64) -> u64 {
        self.deferred_batch_max_operations.swap(operations.max(1), Ordering::AcqRel)
    }

    /// 配置 decode 的 command-buffer 算子上限；模型工具可按设备实测覆盖默认值。
    pub fn set_decode_batch_max_operations(&self, operations: u64) {
        self.decode_batch_max_operations.store(operations.max(1), Ordering::Release);
    }

    pub(crate) fn decode_batch_max_operations(&self) -> u64 {
        self.decode_batch_max_operations.load(Ordering::Acquire)
    }

    fn record_decode_operation(&self, operator: &str, shape: &str, estimated_read_bytes: u64, estimated_write_bytes: u64) -> bool {
        let mut batch = self.decode_batch.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let batch = batch.as_mut().expect("decode 算子提交前必须先创建 command buffer");
        batch.operations += 1;
        batch.estimated_read_bytes = batch.estimated_read_bytes.saturating_add(estimated_read_bytes);
        batch.estimated_write_bytes = batch.estimated_write_bytes.saturating_add(estimated_write_bytes);
        if self.detailed_gpu_profiles.load(Ordering::Acquire) {
            batch.last_operator.clear();
            batch.last_operator.push_str(operator);
            batch.last_shape.clear();
            batch.last_shape.push_str(shape);
        }
        batch.operations >= self.deferred_batch_max_operations.load(Ordering::Acquire)
    }

    fn flush_decode_batch(&self) {
        let batch = self.decode_batch.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
        let Some(batch) = batch else {
            return;
        };
        if !self.detailed_gpu_profiles.load(Ordering::Acquire) {
            self.submit_profiled(&batch.command, "", "", batch.estimated_read_bytes, batch.estimated_write_bytes);
        } else if batch.operations == 1 {
            self.submit_profiled(&batch.command, &batch.last_operator, &batch.last_shape, batch.estimated_read_bytes, batch.estimated_write_bytes);
        } else {
            let shape = format!("ops={},boundary={} {}", batch.operations, batch.last_operator, batch.last_shape);
            self.submit_profiled(&batch.command, "decode_batch", &shape, batch.estimated_read_bytes, batch.estimated_write_bytes);
        }
    }

    /// 只提交当前延迟批次，不等待 GPU；后续同一 queue 的命令仍保持数据依赖顺序。
    pub fn submit_batch(&self) {
        if self.defer_waits.load(Ordering::Acquire) {
            self.flush_decode_batch();
        }
    }

    /// 提交当前延迟批次并返回(最后一个已提交 command buffer, 它在 pending 列表里的下标上界)。
    /// decode 流水线用它做精确等待点:queue 内 FIFO,该 CB 完成即此前全部工作完成,
    /// 之后提交的推测轮次不受影响继续跑。
    pub fn submit_batch_and_last_command(&self) -> Option<(CommandBuffer, usize)> {
        self.submit_batch();
        let pending = self.pending_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let through = pending.len();
        pending.last().map(|pending| (pending.command.to_owned(), through))
    }

    /// 记录并移除 pending 列表前 `through` 项;调用方保证这些 CB 已完成
    /// (配合 submit_batch_and_last_command 的等待点),避免长生成累积 CB 句柄。
    pub fn complete_profiles_through(&self, through: usize) {
        let mut pending = self.pending_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let count = through.min(pending.len());
        for profile in pending.drain(..count) {
            self.record_completed(profile);
        }
    }

    fn submit_profiled(&self, command: &CommandBufferRef, operator: &str, shape: &str, estimated_read_bytes: u64, estimated_write_bytes: u64) {
        // queue 按 FIFO 完成。提交新工作前无阻塞地回收队首已完成 CB，及时释放其
        // activation 保活引用；不能把这些 buffer 一直拖到整段 prefill 末尾。
        let completed = {
            let mut pending = self.pending_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let count = pending.iter().take_while(|profile| profile.command.is_completed()).count();
            pending.drain(..count).collect::<Vec<_>>()
        };
        for profile in completed {
            self.record_completed(profile);
        }
        let submitted_host_seconds = metal_host_seconds();
        command.commit();
        self.command_buffers.fetch_add(1, Ordering::Relaxed);
        let detailed = self.detailed_gpu_profiles.load(Ordering::Acquire);
        let keep_alive = std::mem::take(&mut *self.batch_keep_alive.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        self.pending_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(PendingMetalProfile {
            command: command.to_owned(),
            submitted_host_seconds,
            operator: detailed.then(|| operator.to_owned()),
            shape: detailed.then(|| shape.to_owned()),
            estimated_read_bytes,
            estimated_write_bytes,
            _keep_alive: keep_alive,
        });
    }

    /// 取出 pending 的 (host_submit, gpu_start, gpu_end, operator, shape) 时序,
    /// 不做完成记录(诊断用,同 CACurrentMediaTime 时钟基准)。
    pub fn take_pending_trace(&self) -> Vec<(f64, f64, f64, String, String)> {
        let pending = self.pending_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        pending.iter().map(|pending| (pending.submitted_host_seconds, pending.command.gpu_start_time(), pending.command.gpu_end_time(), pending.operator.clone().unwrap_or_default(), pending.shape.clone().unwrap_or_default())).collect()
    }

    fn record_completed(&self, pending: PendingMetalProfile) {
        let start = pending.command.gpu_start_time();
        let end = pending.command.gpu_end_time();
        let gpu_nanoseconds = if start.is_finite() && end.is_finite() && end >= start { ((end - start) * 1.0e9).round() as u64 } else { 0 };
        self.gpu_nanoseconds.fetch_add(gpu_nanoseconds, Ordering::Relaxed);
        if gpu_nanoseconds != 0 {
            let previous = f64::from_bits(self.last_profile_gpu_end_bits.swap(end.to_bits(), Ordering::Relaxed));
            if previous.is_finite() && previous > 0.0 && start >= previous {
                self.inter_command_gap_nanoseconds.fetch_add(((start - previous) * 1.0e9).round() as u64, Ordering::Relaxed);
            }
        }
        if let (Some(operator), Some(shape)) = (pending.operator, pending.shape) {
            let mut profiles = self.gpu_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let profile = profiles.entry((operator, shape)).or_default();
            profile.calls += 1;
            profile.gpu_nanoseconds += gpu_nanoseconds;
            profile.estimated_read_bytes = profile.estimated_read_bytes.saturating_add(pending.estimated_read_bytes);
            profile.estimated_write_bytes = profile.estimated_write_bytes.saturating_add(pending.estimated_write_bytes);
        }
    }

    pub fn synchronize(&self) {
        if self.defer_waits.load(Ordering::Acquire) {
            self.flush_decode_batch();
        }
        let last = self.pending_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).last().map(|pending| pending.command.to_owned());
        let Some(last) = last else {
            return;
        };
        last.wait_until_completed();
        let completed_host_seconds = metal_host_seconds();
        let pending = std::mem::take(&mut *self.pending_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        if let (Some(first), Some(last)) = (pending.first(), pending.last()) {
            let first_gpu_start = first.command.gpu_start_time();
            let last_gpu_end = last.command.gpu_end_time();
            if first_gpu_start.is_finite() && first_gpu_start >= first.submitted_host_seconds {
                self.submit_wait_nanoseconds.fetch_add(((first_gpu_start - first.submitted_host_seconds) * 1.0e9).round() as u64, Ordering::Relaxed);
            }
            if last_gpu_end.is_finite() && completed_host_seconds >= last_gpu_end {
                self.completion_tail_nanoseconds.fetch_add(((completed_host_seconds - last_gpu_end) * 1.0e9).round() as u64, Ordering::Relaxed);
            }
        }
        for profile in pending {
            self.record_completed(profile);
        }
    }

    /// 记录 command buffer 内的算子组。Decode 延迟模式下只提交，不在 GPU-only 数据依赖间等待。
    pub fn commit_and_wait_profiled(&self, command: &CommandBufferRef, operator: &str, shape: &str, estimated_read_bytes: u64, estimated_write_bytes: u64) {
        let started = std::time::Instant::now();
        if self.defer_waits.load(Ordering::Acquire) {
            if self.record_decode_operation(operator, shape, estimated_read_bytes, estimated_write_bytes) {
                self.flush_decode_batch();
            }
            self.decode_commit_nanoseconds.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return;
        }
        self.submit_profiled(command, operator, shape, estimated_read_bytes, estimated_write_bytes);
        self.synchronize();
        self.decode_commit_nanoseconds.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// CPU 即将读取 Shared buffer 时强制完成此前同一 queue 上的全部命令。
    pub fn commit_and_force_wait_profiled(&self, command: &CommandBufferRef, operator: &str, shape: &str, estimated_read_bytes: u64, estimated_write_bytes: u64) {
        let started = std::time::Instant::now();
        if self.defer_waits.load(Ordering::Acquire) {
            self.record_decode_operation(operator, shape, estimated_read_bytes, estimated_write_bytes);
            self.flush_decode_batch();
            self.synchronize();
            self.decode_commit_nanoseconds.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return;
        }
        self.submit_profiled(command, operator, shape, estimated_read_bytes, estimated_write_bytes);
        self.synchronize();
        self.decode_commit_nanoseconds.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// decode CPU 成本打点读取:返回(分配纳秒, 提交路径纳秒)。
    pub fn decode_cpu_breakdown(&self) -> (u64, u64) {
        (self.decode_alloc_nanoseconds.load(Ordering::Relaxed), self.decode_commit_nanoseconds.load(Ordering::Relaxed))
    }

    pub fn reset_decode_cpu_breakdown(&self) {
        self.decode_alloc_nanoseconds.store(0, Ordering::Relaxed);
        self.decode_commit_nanoseconds.store(0, Ordering::Relaxed);
    }

    pub fn reset_gpu_stats(&self) {
        self.synchronize();
        self.submit_wait_nanoseconds.store(0, Ordering::Relaxed);
        self.gpu_nanoseconds.store(0, Ordering::Relaxed);
        self.inter_command_gap_nanoseconds.store(0, Ordering::Relaxed);
        self.last_profile_gpu_end_bits.store(0, Ordering::Relaxed);
        self.completion_tail_nanoseconds.store(0, Ordering::Relaxed);
        self.command_buffers.store(0, Ordering::Relaxed);
        self.gpu_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clear();
        self.detailed_gpu_profiles.store(true, Ordering::Release);
    }

    pub fn gpu_stats(&self) -> MetalGpuStats {
        MetalGpuStats {
            submit_wait_seconds: self.submit_wait_nanoseconds.load(Ordering::Relaxed) as f64 * 1.0e-9,
            seconds: self.gpu_nanoseconds.load(Ordering::Relaxed) as f64 * 1.0e-9,
            inter_command_gap_seconds: self.inter_command_gap_nanoseconds.load(Ordering::Relaxed) as f64 * 1.0e-9,
            completion_tail_seconds: self.completion_tail_nanoseconds.load(Ordering::Relaxed) as f64 * 1.0e-9,
            command_buffers: self.command_buffers.load(Ordering::Relaxed),
        }
    }

    pub fn gpu_profile(&self) -> Vec<MetalGpuProfile> {
        self.detailed_gpu_profiles.store(false, Ordering::Release);
        let profiles = std::mem::take(&mut *self.gpu_profiles.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        let mut out: Vec<_> = profiles
            .into_iter()
            .map(|((operator, shape), profile)| MetalGpuProfile {
                operator,
                shape,
                calls: profile.calls,
                gpu_seconds: profile.gpu_nanoseconds as f64 * 1.0e-9,
                estimated_read_bytes: profile.estimated_read_bytes,
                estimated_write_bytes: profile.estimated_write_bytes,
            })
            .collect();
        out.sort_by(|a, b| b.gpu_seconds.total_cmp(&a.gpu_seconds).then_with(|| a.operator.cmp(&b.operator)).then_with(|| a.shape.cmp(&b.shape)));
        out
    }
}
