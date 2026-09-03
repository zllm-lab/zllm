//! ROCm 后端能力入口：保存设备资源并用 HIP kernel 实现 capability。

mod attention;
mod attn_res;
mod compressed_sparse;
mod context;
mod diffusion;
mod dsa;
mod expert;
mod gated_delta_net;
mod hyper_connection;
mod kda;
mod kv_cache;
mod tensor;
mod vae;
mod vision;

use std::{collections::HashMap, path::Path, sync::Arc};

use rayon::prelude::*;

use crate::attention::attn_res::AttnResBackend;
use crate::attention::gated_delta_net::GatedDeltaNetKernel;
use crate::attention::kda::KdaKernel;
use crate::backend::cpu::CpuKvCache;
use crate::backend::{
    Backend, BackendError, BackendResources, BlockAttentionBackend, DecodeBackend, DsaPrefillBackend, ExpertDecodeBackend, ExpertPrefillBackend, GqaPrefillBackend, LinearWeight, MlaPrefillBackend, MoePrefillBackend, MoePrefillRouting,
    checked_elements, compute_error,
};
use crate::kernel::cpu::ggml_quant;
use crate::kernel::rocm as ops;
use crate::moe::{Activation, routing::ExpertAssignments, topk_moe::TopkMoeSpec};
use crate::weight::{
    container::gguf::GgufMatrix,
    format::quantization::{QuantizedMatrixRef, ScaleDType},
};

thread_local! {
    /// CPU prefill 把同一层 Q/KV/RoPE 合并到一个 pinned D2H，避免三次
    /// pageable 传输各自同步 compute stream。
    static CPU_PREFILL_BF16_DOWNLOADS: std::cell::RefCell<HashMap<i32, ops::hip::AsyncHostDownload>> = std::cell::RefCell::new(HashMap::new());
    /// 每个 device worker 只保留一个 CPU attention 输出 staging；上一份 H2D
    /// 完成后立即复用，不随层数增长 pinned 内存。
    static CPU_PREFILL_BF16_UPLOADS: std::cell::RefCell<HashMap<i32, ops::hip::AsyncHostUpload>> = std::cell::RefCell::new(HashMap::new());
}
use crate::weight::{
    expert_source::{GgufExpertSource, Mxfp4ExpertSource, Mxfp4ExpertWeights},
    format::compressed_tensors_hybrid::CompressedTensorsSource,
    format::nvfp4::{Nvfp4ExpertWeights, NvidiaNvfp4Experts},
    format::official_fp8::OfficialExpertArchive,
};

pub use dsa::{DsaLayerSerde, RocmDsaSelection, RocmDsaState};
pub use expert::RocmPrefillExperts;
pub use gated_delta_net::RocmGatedDeltaNetStorage;
pub use kda::RocmKdaStorage;
pub use kv_cache::{MlaLayerSerde, RocmKvCache, RocmKvOwnership};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmTensorDType {
    F32,
    Bf16,
}

impl RocmTensorDType {
    const fn element_bytes(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::Bf16 => std::mem::size_of::<u16>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmTensorLayout {
    RowMajor,
}

/// 逻辑 shape、显式物理布局、迁移期 host shadow 与设备 resident buffer。
///
/// host shadow 不是 ROCm 的特质,而是"传输需要经过 host 内存"时的通用需要:
/// 当前 PCIe 机器的八卡链式迁移与双机 stage 流水(跨机必然走 host 侧网卡)
/// 都以 host 内存为中转,D2H 回退与 host 参考路径(DSA 选择/KDA 状态退化)
/// 也依赖它。若设备间有 NVLink/xGMI 直连,卡间迁移可设备直传,不需要 shadow;
/// 单机统一内存后端(Metal)无 host/device 之分,同样不需要常驻 shadow。
#[derive(Debug, Clone)]
pub struct RocmTensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub dtype: RocmTensorDType,
    pub layout: RocmTensorLayout,
    pub device: Option<Arc<ops::hip::DeviceBuffer>>,
}

pub struct RocmStageCompletion(ops::hip::DeviceCompletion);

/// 量化权重的设备 resident 形态。`linear()` 时 in-kernel dequant,不展开 F32。
#[derive(Debug, Clone)]
pub enum RocmQuantizedWeight {
    W4A16 {
        packed: Arc<ops::hip::DeviceBuffer>,
        scales: Arc<ops::hip::DeviceBuffer>,
        scale_dtype: ScaleDType,
        group_size: usize,
    },
    W8A16 {
        packed: Arc<ops::hip::DeviceBuffer>,
        scales: Arc<ops::hip::DeviceBuffer>,
        scale_dtype: ScaleDType,
        group_size: usize,
    },
    ConvRotInt8 {
        packed: Arc<ops::hip::DeviceBuffer>,
        scales: Arc<ops::hip::DeviceBuffer>,
        group_size: usize,
    },
    /// Block-scaled FP8(E4M3 codes + E8M0 block scales)。V4 Flash 唯一用到此变体的
    /// block 形态为 128×128。gfx11 无 FP8 计算单元,prefill GEMM 先解码成 BF16
    /// 走 WMMA;解码结果按权重缓存,只做一次,摊销到整个 prefill。
    BlockFp8 {
        codes: Arc<ops::hip::DeviceBuffer>,
        scales: Arc<ops::hip::DeviceBuffer>,
        block_rows: usize,
        block_cols: usize,
        bf16_cache: Arc<std::sync::OnceLock<Arc<ops::hip::DeviceBuffer>>>,
    },
    /// MXFP4(E2M1 + E8M0 32-group) 常驻形态;routed expert 专用,
    /// `linear()` 走 mxfp4_matmul in-kernel dequant,4bit 驻留为 BF16 的 1/4。
    Mxfp4 {
        packed: Arc<ops::hip::DeviceBuffer>,
        scales: Arc<ops::hip::DeviceBuffer>,
    },
    /// GGUF K-quant 原始 block 常驻；dense/expert kernel 直接从 block 解码 scale/min。
    GgufPacked {
        codes: Arc<ops::hip::DeviceBuffer>,
        tensor_type: u32,
    },
}

#[derive(Debug, Clone)]
enum RocmWeightInner {
    /// 量化 resident 形态。`RocmQuantizedWeight` 持有 codes/scales,linear 时 in-kernel dequant。
    Quantized(RocmQuantizedWeight),
    /// dense F32 内存 + 可选设备 resident buffer;`router_bf16` 是路由器专用惰性缓存。
    Dense { data: Vec<f32>, resident: Option<Arc<ops::hip::DeviceBuffer>>, resident_bf16: bool, router_bf16: Arc<std::sync::OnceLock<Arc<ops::hip::DeviceBuffer>>> },
    /// GGUF K-quant 矩阵。原始数据 + CPU decode 路径,无设备 buffer。
    #[allow(dead_code)] // GGUF ROCm 构造路径尚未接入，tensor 执行契约已就绪。
    Gguf(GgufMatrix),
}

/// ROCm resident 权重。CT 量化矩阵只保留 packed code 与 scale 的显存副本，不展开 F32。
#[derive(Debug, Clone)]
pub struct RocmWeight {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    inner: RocmWeightInner,
    /// expert decode 可额外保留原始 GGUF block；普通 linear 继续使用其
    /// prefill 友好的 resident 布局。
    expert_gguf: Option<(Arc<ops::hip::DeviceBuffer>, u32)>,
    /// CPU prefill 的 MLA 吸收路径只给 kv_b 保留 host F32 视图；普通权重不复制。
    cpu_mla_data: Option<Arc<Vec<f32>>>,
}

impl RocmWeight {
    pub fn data(&self) -> &[f32] {
        match &self.inner {
            RocmWeightInner::Dense { data, .. } => data,
            // 量化 / GGUF 路径无 host F32 shadow,返回空切片以保持调用方原
            // `weight.data.is_empty()` 判定语义。
            _ => &[],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn cpu_mla_data(&self) -> Option<&[f32]> {
        self.cpu_mla_data.as_deref().map(Vec::as_slice)
    }

    pub fn quantized(&self) -> Option<&RocmQuantizedWeight> {
        match &self.inner {
            RocmWeightInner::Quantized(weight) => Some(weight),
            _ => None,
        }
    }

    pub(crate) fn expert_gguf(&self) -> Option<(&Arc<ops::hip::DeviceBuffer>, u32)> {
        self.expert_gguf.as_ref().map(|(codes, tensor_type)| (codes, *tensor_type)).or_else(|| match &self.inner {
            RocmWeightInner::Quantized(RocmQuantizedWeight::GgufPacked { codes, tensor_type }) => Some((codes, *tensor_type)),
            _ => None,
        })
    }

    /// 为 decode 的输出行分片建立零复制 W8 view。packed/scales 都按行连续，
    /// 因此只需保留原 allocation，无需再占一份 q_b 显存。
    pub(crate) fn w8_row_view(&self, range: std::ops::Range<usize>) -> Result<Option<Self>, BackendError> {
        if range.start >= range.end || range.end > self.rows {
            return Err(compute_error(format!("ROCm W8 row view={range:?}/{} 非法", self.rows)));
        }
        let RocmWeightInner::Quantized(RocmQuantizedWeight::W8A16 { packed, scales, scale_dtype, group_size }) = &self.inner else {
            return Ok(None);
        };
        if *group_size == 0 || !self.cols.is_multiple_of(*group_size) {
            return Err(compute_error(format!("ROCm W8 row view columns={} group_size={} 非法", self.cols, group_size)));
        }
        let rows = range.len();
        let packed_row_bytes = self.cols;
        let scale_row_bytes = (self.cols / group_size).checked_mul(scale_dtype.bytes()).ok_or_else(|| compute_error("ROCm W8 row view scale 行大小溢出"))?;
        let packed_offset = range.start.checked_mul(packed_row_bytes).ok_or_else(|| compute_error("ROCm W8 row view packed offset 溢出"))?;
        let packed_bytes = rows.checked_mul(packed_row_bytes).ok_or_else(|| compute_error("ROCm W8 row view packed 大小溢出"))?;
        let scale_offset = range.start.checked_mul(scale_row_bytes).ok_or_else(|| compute_error("ROCm W8 row view scale offset 溢出"))?;
        let scale_bytes = rows.checked_mul(scale_row_bytes).ok_or_else(|| compute_error("ROCm W8 row view scale 大小溢出"))?;
        let packed = Arc::new(ops::hip::DeviceBuffer::view(packed.clone(), packed_offset, packed_bytes).map_err(compute_error)?);
        let scales = Arc::new(ops::hip::DeviceBuffer::view(scales.clone(), scale_offset, scale_bytes).map_err(compute_error)?);
        Ok(Some(Self { rows, cols: self.cols, inner: RocmWeightInner::Quantized(RocmQuantizedWeight::W8A16 { packed, scales, scale_dtype: *scale_dtype, group_size: *group_size }), expert_gguf: None, cpu_mla_data: None }))
    }

    pub fn gguf(&self) -> Option<&GgufMatrix> {
        match &self.inner {
            RocmWeightInner::Gguf(matrix) => Some(matrix),
            _ => None,
        }
    }

    /// dense 路径下的设备 resident buffer;量化 / GGUF 路径恒为 None。
    pub fn resident(&self) -> Option<&Arc<ops::hip::DeviceBuffer>> {
        match &self.inner {
            RocmWeightInner::Dense { resident, .. } => resident.as_ref(),
            _ => None,
        }
    }

    /// dense 路径下设备 buffer 是否 BF16 格式;其他路径恒为 false。
    pub fn resident_bf16(&self) -> bool {
        match &self.inner {
            RocmWeightInner::Dense { resident_bf16, .. } => *resident_bf16,
            _ => false,
        }
    }

    pub(crate) fn prepare_router(&self, device_id: i32) -> Result<(), BackendError> {
        self.router_resident(device_id, ops::hip::options().precise_router).map(|_| ())
    }

    fn router_resident(&self, device_id: i32, precise: bool) -> Result<&ops::hip::DeviceBuffer, BackendError> {
        let (data, resident, resident_bf16, router_bf16) = match &self.inner {
            RocmWeightInner::Dense { data, resident, resident_bf16, router_bf16 } => (data, resident, *resident_bf16, router_bf16),
            _ => return Err(compute_error("ROCm router 只支持 dense 权重")),
        };
        if precise {
            if resident_bf16 {
                return Err(compute_error("ROCm precise router 需要 F32 resident 权重"));
            }
            return resident.as_deref().ok_or_else(|| compute_error("ROCm precise router 缺少 resident buffer"));
        }
        if resident_bf16 {
            return resident.as_deref().ok_or_else(|| compute_error("ROCm BF16 router 缺少 resident buffer"));
        }
        if let Some(buffer) = router_bf16.get() {
            return Ok(buffer);
        }
        if data.len() != self.rows.checked_mul(self.cols).ok_or_else(|| compute_error("ROCm router shape 溢出"))? {
            return Err(compute_error("ROCm BF16 router 缺少 F32 host 权重"));
        }
        let values = data.iter().copied().map(half::bf16::from_f32).collect::<Vec<_>>();
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values.as_slice())) };
        let uploaded = Arc::new(ops::hip::DeviceBuffer::upload(device_id, bytes).map_err(compute_error)?);
        let _ = router_bf16.set(uploaded);
        router_bf16.get().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm BF16 router 初始化失败"))
    }
}
const ROCM_KV_BLOCK_SIZE: usize = 64;

fn committed_cache_rows(current: usize, required: usize, limit: usize) -> Result<usize, BackendError> {
    if required == 0 || required > limit {
        return Err(compute_error(format!("ROCm cache required={required} 超过逻辑上限 {limit}")));
    }
    if current >= required {
        return Ok(current);
    }
    let target = required.max(current.saturating_mul(2)).max(ROCM_KV_BLOCK_SIZE);
    let aligned = target.div_ceil(ROCM_KV_BLOCK_SIZE).checked_mul(ROCM_KV_BLOCK_SIZE).ok_or_else(|| compute_error("ROCm cache committed rows 溢出"))?;
    Ok(aligned.min(limit).max(required))
}

fn grow_cache_buffer(device_id: i32, current: &Arc<ops::hip::DeviceBuffer>, used_bytes: usize, capacity_bytes: usize) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
    if current.bytes() >= capacity_bytes {
        return Ok(current.clone());
    }
    let grown = Arc::new(ops::hip::DeviceBuffer::allocate(device_id, capacity_bytes).map_err(compute_error)?);
    grown.copy_from_device(0, current, 0, used_bytes).map_err(compute_error)?;
    Ok(grown)
}

/// SSD restore 直接按本轮已预留的行容量分配，再只上传有效前缀；避免首个 decode
/// append 先精确恢复、随后又为一行触发整层 cache 扩容和 D2D 复制。
fn upload_cache_buffer(device_id: i32, input: &[u8], capacity_bytes: usize) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
    if input.is_empty() || input.len() > capacity_bytes {
        return Err(compute_error(format!("ROCm cache restore bytes={}/{} 非法", input.len(), capacity_bytes)));
    }
    let buffer = Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, capacity_bytes).map_err(compute_error)?);
    buffer.copy_from_host(input).map_err(compute_error)?;
    Ok(buffer)
}

/// KV / DSA cache 共用的 block table：单调增长的 u32 block ID 数组，按需扩容并绑定单一 device。
struct RocmBlockTable {
    table: Option<Arc<ops::hip::DeviceBuffer>>,
    block_count: usize,
    device_id: Option<i32>,
}

impl RocmBlockTable {
    fn new() -> Self {
        Self { table: None, block_count: 0, device_id: None }
    }

    fn buffer(&self) -> Option<&ops::hip::DeviceBuffer> {
        self.table.as_deref()
    }

    /// `tag` 只用于错误信息区分调用方（KV / DSA）。
    ///
    /// 表内容固定是 identity（block i -> 物理块 i），与 required_rows 无关，
    /// 因此一旦需要（重）建就直接按 `capacity` 建全量：decode 单行 append
    /// 每 64 行就会跨一次块边界，若按 required 精确重建，每次都会在深队列
    /// 的流上做一次带 hipStreamSynchronize 的上传，把提交线程卡住。
    fn get(&mut self, tag: &str, capacity: usize, device_id: i32, required_rows: usize) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
        if self.device_id.is_some_and(|device| device != device_id) {
            return Err(compute_error(format!("ROCm {tag} cache 已绑定 device {:?}，不能用于 {device_id}", self.device_id)));
        }
        if required_rows == 0 || required_rows > capacity {
            return Err(compute_error(format!("ROCm {tag} block table rows={required_rows} 超过逻辑上限 {capacity}")));
        }
        self.device_id = Some(device_id);
        let blocks = required_rows.div_ceil(ROCM_KV_BLOCK_SIZE);
        if self.block_count >= blocks
            && let Some(table) = &self.table
        {
            return Ok(table.clone());
        }
        let blocks = capacity.div_ceil(ROCM_KV_BLOCK_SIZE).max(blocks);
        let ids = (0..blocks).map(|block| u32::try_from(block).map_err(|_| compute_error(format!("ROCm {tag} block ID 超过 u32")))).collect::<Result<Vec<_>, _>>()?;
        let bytes = unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), std::mem::size_of_val(ids.as_slice())) };
        let table = Arc::new(ops::hip::DeviceBuffer::upload(device_id, bytes).map_err(compute_error)?);
        self.table = Some(table.clone());
        self.block_count = blocks;
        Ok(table)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RocmContext {
    device_id: i32,
    allow_cpu_reference_fallback: bool,
    compute_stream: usize,
}

impl RocmContext {
    pub fn new(device_id: i32) -> Result<Self, String> {
        Self::configured(device_id, false)
    }

    pub fn configured(device_id: i32, allow_cpu_reference_fallback: bool) -> Result<Self, String> {
        if !available() {
            return Err("未检测到 ROCm runtime（libamdhip64）".to_owned());
        }
        ops::hip::set_device(device_id)?;
        Ok(Self { device_id, allow_cpu_reference_fallback, compute_stream: 0 })
    }

    pub fn for_device(&self, device_id: i32) -> Result<Self, String> {
        Self::configured(device_id, self.allow_cpu_reference_fallback)
    }

    pub(crate) fn with_independent_stream(&self) -> Result<Self, String> {
        Ok(Self { device_id: self.device_id, allow_cpu_reference_fallback: self.allow_cpu_reference_fallback, compute_stream: ops::hip::independent_compute_stream(self.device_id)? })
    }

    pub(crate) fn require_cpu_reference_fallback(&self, operation: &str) -> Result<(), BackendError> {
        if self.allow_cpu_reference_fallback { Ok(()) } else { Err(compute_error(format!("ROCm {operation} 尚无设备实现；如需使用慢速 CPU reference，请在 backend 中设置 allow_cpu_reference_fallback: true"))) }
    }

    pub(crate) fn upload_cpu_reference(&self, operation: &str, data: Vec<f32>, rows: usize, cols: usize) -> Result<RocmTensor, BackendError> {
        self.require_cpu_reference_fallback(operation)?;
        self.tensor_from_f32(data, rows, cols).map_err(compute_error)
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn activate(&self) -> Result<(), String> {
        ops::hip::activate_compute_stream(self.device_id, self.compute_stream)
    }

    pub(crate) fn synchronize_compute_stream(&self) -> Result<(), BackendError> {
        ops::hip::synchronize_compute_stream(self.device_id, "ROCm compute stream synchronize").map_err(compute_error)
    }

    pub fn enable_peer_access_from(&self, source_device_id: i32) -> Result<(), String> {
        ops::hip::enable_peer_access(self.device_id, source_device_id)
    }

    pub fn warmup_quantized(&self) -> Result<(), String> {
        ops::hip::warmup_ct_quantized(self.device_id)
    }

    pub fn tensor_from_f32(&self, data: Vec<f32>, rows: usize, cols: usize) -> Result<RocmTensor, String> {
        if data.len() != rows.checked_mul(cols).ok_or("ROCm tensor shape 溢出")? {
            return Err(format!("ROCm tensor 元素数 {}，期望 {}", data.len(), rows * cols));
        }
        let bytes = std::mem::size_of_val(data.as_slice());
        let device = ops::hip::DeviceBuffer::upload(self.device_id, unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), bytes) })?;
        Ok(RocmTensor { data, rows, cols, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: Some(Arc::new(device)) })
    }

    /// 接收/调度线程上传的输入已 ready，不与当前 compute stream 建立全流同步。
    pub fn tensor_from_f32_independent(&self, data: Vec<f32>, rows: usize, cols: usize) -> Result<RocmTensor, String> {
        if data.len() != rows.checked_mul(cols).ok_or("ROCm tensor shape 溢出")? {
            return Err(format!("ROCm tensor 元素数 {}，期望 {}", data.len(), rows * cols));
        }
        let bytes = std::mem::size_of_val(data.as_slice());
        let device = ops::hip::DeviceBuffer::upload_independent(self.device_id, unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), bytes) })?;
        Ok(RocmTensor { data, rows, cols, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: Some(Arc::new(device)) })
    }

    /// 从 fused QKV 中抽取连续 head 区间，输出仍按 `[Q|K|V]` 排列。
    pub(crate) fn compact_qkv_heads(&self, tensor: &RocmTensor, total_heads: usize, heads: std::ops::Range<usize>, head_dim: usize) -> Result<RocmTensor, BackendError> {
        if tensor.dtype != RocmTensorDType::F32 || heads.start >= heads.end || heads.end > total_heads {
            return Err(compute_error(format!("ROCm compact QKV dtype={:?} heads={heads:?}/{total_heads} 非法", tensor.dtype)));
        }
        let expected_cols = total_heads.checked_mul(head_dim).and_then(|value| value.checked_mul(3)).ok_or_else(|| compute_error("ROCm compact QKV cols 溢出"))?;
        if tensor.cols != expected_cols {
            return Err(compute_error(format!("ROCm compact QKV cols={}，期望 {expected_cols}", tensor.cols)));
        }
        let source = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm compact QKV 缺少 device buffer"))?;
        if source.device_id() != self.device_id {
            return Err(compute_error(format!("ROCm compact QKV device={}，当前 device={}", source.device_id(), self.device_id)));
        }
        let local_heads = heads.len();
        let output = ops::hip::try_compact_qkv_head_range_resident_f32(self.device_id, source, tensor.rows, total_heads, heads.start, local_heads, head_dim).map_err(compute_error)?;
        Ok(device_tensor_f32(output, tensor.rows, local_heads * head_dim * 3))
    }

    pub fn tensor_on_device(&self, tensor: RocmTensor) -> Result<RocmTensor, BackendError> {
        if tensor.device.as_deref().is_some_and(|buffer| buffer.device_id() == self.device_id) {
            return Ok(tensor);
        }
        let rows = tensor.rows;
        let cols = tensor.cols;
        if let Some(device) = tensor.device.as_deref() {
            let device = device.copy_to_device(self.device_id).map_err(compute_error)?;
            return Ok(device_tensor_with_dtype(device, rows, cols, tensor.dtype));
        }
        self.tensor_from_f32(tensor_data(&tensor)?, rows, cols).map_err(compute_error)
    }

    pub(crate) fn tensor_on_device_ordered(&self, tensor: RocmTensor) -> Result<RocmTensor, BackendError> {
        if let Some(buffer) = tensor.device.as_ref().filter(|buffer| buffer.device_id() == self.device_id) {
            buffer.enqueue_deferred_upload().map_err(compute_error)?;
            buffer.retain_for_active_stage();
            return Ok(tensor);
        }
        let rows = tensor.rows;
        let cols = tensor.cols;
        if let Some(device) = tensor.device.as_ref() {
            let device = device.copy_stable_to_device_ordered_async(self.device_id).map_err(compute_error)?;
            return Ok(device_tensor_with_dtype(device, rows, cols, tensor.dtype));
        }
        self.tensor_from_f32(tensor_data(&tensor)?, rows, cols).map_err(compute_error)
    }

    pub(crate) fn retire_ordered_p2p_sources(&self) {
        ops::hip::retire_pending_p2p_sources(self.device_id);
    }

    /// 阶段末尾异步复制到显式池，紧随其后的 completion event 负责完成判定。
    pub(crate) fn tensor_to_stable_deferred(&self, tensor: RocmTensor) -> Result<RocmTensor, BackendError> {
        let rows = tensor.rows;
        let cols = tensor.cols;
        let device = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm stage output 缺少 device buffer"))?;
        if device.device_id() != self.device_id {
            return Err(compute_error(format!("ROCm stage output device={}，期望 {}", device.device_id(), self.device_id)));
        }
        // 显式池输出可由 tensor Arc 直接持有到目标 completion；只有
        // hipMallocAsync allocation 才需要先复制到跨线程稳定存储。
        if !device.is_async_allocated() {
            return Ok(tensor);
        }
        let device = device.copy_to_stable_deferred().map_err(compute_error)?;
        Ok(device_tensor_with_dtype(device, rows, cols, tensor.dtype))
    }

    pub fn tensor_as_f32(&self, tensor: RocmTensor) -> Result<RocmTensor, BackendError> {
        let elements = checked_elements(tensor.rows, tensor.cols, "ROCm BF16 expand")?;
        let tensor = self.tensor_on_device(tensor)?;
        let device = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm tensor 缺少 device buffer"))?;
        if tensor.dtype == RocmTensorDType::F32 {
            return Ok(tensor);
        }
        validate_tensor_buffer(&tensor, elements)?;
        let output = ops::hip::try_cast_bf16_to_f32_resident(self.device_id, device, elements).map_err(compute_error)?;
        Ok(device_tensor_f32(output, tensor.rows, tensor.cols))
    }

    pub fn tensor_as_bf16(&self, tensor: RocmTensor) -> Result<RocmTensor, BackendError> {
        let elements = checked_elements(tensor.rows, tensor.cols, "ROCm BF16 compact")?;
        let tensor = self.tensor_on_device(tensor)?;
        let device = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm tensor 缺少 device buffer"))?;
        if tensor.dtype == RocmTensorDType::Bf16 {
            return Ok(tensor);
        }
        validate_tensor_buffer(&tensor, elements)?;
        let debug_finite = tensor.rows != 0 && ops::hip::options().debug_finite;
        let scan_rows = tensor.rows.min(64);
        let scan_offset = (tensor.rows - scan_rows) * tensor.cols;
        let scan_elements = scan_rows * tensor.cols;
        if debug_finite {
            ops::hip::try_validate_finite_resident_range_f32(self.device_id, device, scan_offset, scan_elements)
                .map_err(|error| compute_error(format!("ROCm BF16 compact 输入包含非有限值或异常幅值: rows={} cols={}: {error}", tensor.rows, tensor.cols,)))?;
        }
        let output = ops::hip::try_cast_f32_to_bf16_resident(self.device_id, device, elements).map_err(compute_error)?;
        if debug_finite {
            ops::hip::try_validate_finite_resident_range_bf16(self.device_id, &output, scan_offset, scan_elements)
                .map_err(|error| compute_error(format!("ROCm BF16 compact 输出包含非有限值或异常幅值: rows={} cols={}: {error}", tensor.rows, tensor.cols,)))?;
        }
        Ok(device_tensor_bf16(output, tensor.rows, tensor.cols))
    }

    pub fn tensor_to_bf16_bits(&self, tensor: &RocmTensor) -> Result<Vec<u16>, BackendError> {
        let elements = checked_elements(tensor.rows, tensor.cols, "ROCm BF16 download")?;
        let tensor = self.tensor_as_bf16(tensor.clone())?;
        tensor.device.as_deref().ok_or_else(|| compute_error("ROCm BF16 tensor 缺少 device buffer"))?.download_u16(elements).map_err(compute_error)
    }

    /// 把若干 tensor 按参数顺序下载到同一块 pinned host staging，并在 staging
    /// 有效期间消费 BF16 切片。所有 cast 和 D2H 都排在当前 compute stream，
    /// 只等待一个完成 event。
    pub(crate) fn with_tensors_bf16_bits<R>(&self, tensors: &[&RocmTensor], consume: impl FnOnce(&[&[u16]]) -> Result<R, BackendError>) -> Result<R, BackendError> {
        if tensors.is_empty() {
            return Err(compute_error("ROCm BF16 segmented download 不能为空"));
        }
        let tensors = tensors.iter().map(|tensor| self.tensor_as_bf16((*tensor).clone())).collect::<Result<Vec<_>, _>>()?;
        let mut lengths = Vec::with_capacity(tensors.len());
        let mut segments = Vec::with_capacity(tensors.len());
        let mut bytes = 0usize;
        for tensor in &tensors {
            let elements = checked_elements(tensor.rows, tensor.cols, "ROCm BF16 segmented download")?;
            let tensor_bytes = elements.checked_mul(std::mem::size_of::<u16>()).ok_or_else(|| compute_error("ROCm BF16 segmented download 大小溢出"))?;
            let device = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm BF16 segmented tensor 缺少 device buffer"))?;
            lengths.push(elements);
            segments.push((device, 0, tensor_bytes));
            bytes = bytes.checked_add(tensor_bytes).ok_or_else(|| compute_error("ROCm BF16 segmented download 总大小溢出"))?;
        }
        CPU_PREFILL_BF16_DOWNLOADS.with(|downloads| {
            let mut downloads = downloads.borrow_mut();
            if !downloads.contains_key(&self.device_id) {
                downloads.insert(self.device_id, ops::hip::AsyncHostDownload::new(self.device_id, bytes).map_err(compute_error)?);
            }
            let download = downloads.get_mut(&self.device_id).expect("CPU prefill D2H 已插入");
            download.enqueue_segments(&segments).map_err(compute_error)?;
            let packed = download.wait().map_err(compute_error)?;
            let mut offset = 0usize;
            let mut slices = Vec::with_capacity(lengths.len());
            for elements in lengths {
                let slice = unsafe { std::slice::from_raw_parts(packed.as_ptr().add(offset).cast::<u16>(), elements) };
                slices.push(slice);
                offset += elements * std::mem::size_of::<u16>();
            }
            debug_assert_eq!(offset, packed.len());
            consume(&slices)
        })
    }

    /// 融合 logits 选行、bias 相加与分片 argmax，避免两个词表宽度临时张量。
    pub fn argmax_add_rows(&self, logits: &RocmTensor, rows: &[u32], bias: &RocmTensor) -> Result<Vec<u32>, BackendError> {
        let logits = self.tensor_as_f32(logits.clone())?;
        let bias = self.tensor_as_f32(bias.clone())?;
        if rows.len() != bias.rows || bias.cols != logits.cols {
            return Err(compute_error(format!("ROCm add rows argmax 形状非法: logits=[{},{}] rows={} bias=[{},{}]", logits.rows, logits.cols, rows.len(), bias.rows, bias.cols)));
        }
        let logits_device = logits.device.as_deref().ok_or_else(|| compute_error("ROCm add rows argmax 缺少 logits device buffer"))?;
        let bias_device = bias.device.as_deref().ok_or_else(|| compute_error("ROCm add rows argmax 缺少 bias device buffer"))?;
        ops::hip::try_argmax_add_rows_resident_f32(self.device_id, logits_device, logits.rows, logits.cols, rows, bias_device).map_err(compute_error)
    }

    /// stage completion 已完成的 BF16 tensor 可绕开仍有后续批次的默认 stream。
    pub fn completed_tensor_to_bf16_bits(&self, tensor: &RocmTensor) -> Result<Vec<u16>, BackendError> {
        let elements = checked_elements(tensor.rows, tensor.cols, "ROCm completed BF16 download")?;
        if tensor.dtype != RocmTensorDType::Bf16 {
            return Err(compute_error(format!("ROCm completed BF16 tensor dtype={:?}，期望 Bf16", tensor.dtype)));
        }
        let device = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm completed BF16 tensor 缺少 device buffer"))?;
        let bytes = elements.checked_mul(std::mem::size_of::<u16>()).ok_or_else(|| compute_error("ROCm completed BF16 download 大小溢出"))?;
        if device.device_id() != self.device_id || device.bytes() != bytes {
            return Err(compute_error(format!("ROCm completed BF16 tensor device/bytes={}/{}, 期望 {self_device}/{bytes}", device.device_id(), device.bytes(), self_device = self.device_id)));
        }
        let mut output = vec![0_u16; elements];
        device.copy_completed_to_host(unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), bytes) }).map_err(compute_error)?;
        Ok(output)
    }

    pub fn tensor_from_bf16_bits(&self, values: Vec<u16>, rows: usize, cols: usize) -> Result<RocmTensor, BackendError> {
        let elements = checked_elements(rows, cols, "ROCm BF16 upload")?;
        if values.len() != elements {
            return Err(compute_error(format!("ROCm BF16 upload shape=[{rows},{cols}]，实际元素={}", values.len())));
        }
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values.as_slice())) };
        let device = ops::hip::DeviceBuffer::upload(self.device_id, bytes).map_err(compute_error)?;
        Ok(device_tensor_bf16(device, rows, cols))
    }

    pub(crate) fn tensor_from_bf16_bits_streamed(&self, values: Vec<u16>, rows: usize, cols: usize) -> Result<RocmTensor, BackendError> {
        let elements = checked_elements(rows, cols, "ROCm streamed BF16 upload")?;
        if values.len() != elements {
            return Err(compute_error(format!("ROCm streamed BF16 upload shape=[{rows},{cols}]，实际元素={}", values.len())));
        }
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values.as_slice())) };
        let device = CPU_PREFILL_BF16_UPLOADS.with(|uploads| {
            let mut uploads = uploads.borrow_mut();
            if !uploads.contains_key(&self.device_id) {
                uploads.insert(self.device_id, ops::hip::AsyncHostUpload::new(self.device_id, bytes.len()).map_err(compute_error)?);
            }
            uploads.get_mut(&self.device_id).expect("CPU prefill H2D 已插入").upload(bytes).map_err(compute_error)
        })?;
        Ok(device_tensor_bf16(device, rows, cols))
    }

    pub fn tensor_from_bf16_bits_independent(&self, values: Vec<u16>, rows: usize, cols: usize) -> Result<RocmTensor, BackendError> {
        let elements = checked_elements(rows, cols, "ROCm independent BF16 upload")?;
        if values.len() != elements {
            return Err(compute_error(format!("ROCm independent BF16 upload shape=[{rows},{cols}]，实际元素={}", values.len())));
        }
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values.as_slice())) };
        let device = ops::hip::DeviceBuffer::upload_independent(self.device_id, bytes).map_err(compute_error)?;
        Ok(device_tensor_bf16(device, rows, cols))
    }

    /// 双机边界专用上传；独立 H2D stream 只通过 event 交给该 context 的
    /// compute stream，设备 buffer 随 stage completion 进入显式池。
    pub fn tensor_from_bf16_bits_ordered(&self, values: Vec<u16>, rows: usize, cols: usize) -> Result<RocmTensor, BackendError> {
        let elements = checked_elements(rows, cols, "ROCm ordered BF16 upload")?;
        if values.len() != elements {
            return Err(compute_error(format!("ROCm ordered BF16 upload shape=[{rows},{cols}]，实际元素={}", values.len())));
        }
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values.as_slice())) };
        let device = ops::hip::DeviceBuffer::upload_ordered(self.device_id, bytes).map_err(compute_error)?;
        Ok(device_tensor_bf16(device, rows, cols))
    }

    pub fn tensor_to_f32(&self, tensor: &RocmTensor) -> Result<Vec<f32>, BackendError> {
        tensor_data(tensor)
    }
}

fn tensor_data(tensor: &RocmTensor) -> Result<Vec<f32>, BackendError> {
    let elements = checked_elements(tensor.rows, tensor.cols, "ROCm tensor")?;
    if tensor.data.len() == elements {
        return Ok(tensor.data.clone());
    }
    let device = tensor.device.as_ref().ok_or_else(|| compute_error(format!("ROCm tensor [{},{}] 同时缺少 host shadow 与 device buffer", tensor.rows, tensor.cols,)))?;
    validate_tensor_buffer(tensor, elements)?;
    match tensor.dtype {
        RocmTensorDType::Bf16 => {
            let expanded = ops::hip::try_cast_bf16_to_f32_resident(device.device_id(), device, elements).map_err(compute_error)?;
            expanded.download_f32(elements).map_err(compute_error)
        }
        RocmTensorDType::F32 => device.download_f32(elements).map_err(compute_error),
    }
}

fn host_tensor(tensor: &RocmTensor) -> Result<crate::kernel::cpu::CpuTensor, BackendError> {
    Ok(crate::kernel::cpu::CpuTensor { data: tensor_data(tensor)?, rows: tensor.rows, cols: tensor.cols })
}

fn validate_tensor_buffer(tensor: &RocmTensor, elements: usize) -> Result<(), BackendError> {
    let device = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm tensor 缺少 device buffer"))?;
    let expected = elements.checked_mul(tensor.dtype.element_bytes()).ok_or_else(|| compute_error("ROCm tensor 物理大小溢出"))?;
    if device.bytes() != expected {
        return Err(compute_error(format!("ROCm tensor dtype={:?} layout={:?} buffer={}，期望 {expected}", tensor.dtype, tensor.layout, device.bytes())));
    }
    Ok(())
}

fn device_tensor_with_dtype(buffer: ops::hip::DeviceBuffer, rows: usize, cols: usize, dtype: RocmTensorDType) -> RocmTensor {
    // 空张量允许 4 字节占位 buffer:DeviceBuffer 不接受 0 字节分配,
    // 空窗口路径(compressor 等)统一按 max(bytes, 4) 分配输出。
    debug_assert!(buffer.bytes() == rows.saturating_mul(cols).saturating_mul(dtype.element_bytes()) || (rows == 0 || cols == 0) && buffer.bytes() == 4);
    RocmTensor { data: Vec::new(), rows, cols, dtype, layout: RocmTensorLayout::RowMajor, device: Some(Arc::new(buffer)) }
}

fn device_tensor_with_arc(buffer: Arc<ops::hip::DeviceBuffer>, rows: usize, cols: usize, dtype: RocmTensorDType) -> RocmTensor {
    debug_assert_eq!(buffer.bytes(), rows.saturating_mul(cols).saturating_mul(dtype.element_bytes()));
    RocmTensor { data: Vec::new(), rows, cols, dtype, layout: RocmTensorLayout::RowMajor, device: Some(buffer) }
}

fn device_tensor_f32(buffer: ops::hip::DeviceBuffer, rows: usize, cols: usize) -> RocmTensor {
    device_tensor_with_dtype(buffer, rows, cols, RocmTensorDType::F32)
}

fn device_tensor_bf16(buffer: ops::hip::DeviceBuffer, rows: usize, cols: usize) -> RocmTensor {
    device_tensor_with_dtype(buffer, rows, cols, RocmTensorDType::Bf16)
}

fn f32_tensor(context: &RocmContext, tensor: &RocmTensor) -> Result<RocmTensor, BackendError> {
    context.tensor_as_f32(tensor.clone())
}

fn constant<'a>(weight: &'a RocmWeight, expected: usize, name: &str) -> Result<&'a ops::hip::DeviceBuffer, BackendError> {
    if weight.resident_bf16() || weight.data().len() != expected {
        return Err(compute_error(format!("ROCm {name} 需要 resident F32 常量，实际 len={} bf16={}", weight.data().len(), weight.resident_bf16())));
    }
    weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error(format!("ROCm {name} 缺少 resident buffer")))
}

fn resident_weight<'a>(weight: &'a RocmWeight, what: &str) -> Result<&'a ops::hip::DeviceBuffer, BackendError> {
    weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error(format!("ROCm {what} 需要 dense resident 权重")))
}

/// 确保 RocmWeight 以 F32 形态驻留 device,返回 device buffer 的 Arc clone。
/// BF16 resident 形态回退 host F32 上传;缺少 F32 数据(量化/GGUF)时直接报错,
/// 避免把 bf16 buffer 当 f32 读或上传空 buffer。
fn ensure_weight_device(device_id: i32, weight: &RocmWeight) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
    if !weight.resident_bf16()
        && let Some(resident) = weight.resident()
    {
        return Ok(resident.clone());
    }
    let data = weight.data();
    if data.is_empty() {
        return Err(compute_error("ROCm kernel 权重缺少 dense F32 host 数据"));
    }
    ops::hip::DeviceBuffer::upload_f32(device_id, data).map_err(compute_error).map(Arc::new)
}

fn append_gqa_cache(cache: &mut CpuKvCache, layer: usize, position: usize, key: &RocmTensor, value: &RocmTensor, spec: &crate::attention::gqa::GqaSpec, retain_full_cache: bool) -> Result<(), BackendError> {
    cache.append_gqa(layer, position, &host_tensor(key)?, &host_tensor(value)?, spec, retain_full_cache)
}

/// 判断当前进程是否能看到 ROCm runtime。
pub fn available() -> bool {
    ops::hip::is_hip_available()
}

pub fn device_name(context: &RocmContext) -> String {
    if available() { format!("ROCm backend (HIP device {})", context.device_id) } else { "ROCm backend (not ready)".to_owned() }
}

pub use compressed_sparse::{RocmCompressedKvSerde, RocmCompressedKvStorage, RocmGatedPoolSerde};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attention::{block::BlockAttentionSpec, dsa::DsaSpec, gqa::GqaGeometry, rope::RotaryLayout},
        backend::{Backend, BlockAttentionBackend, DecodeBackend, SegmentedTensorBackend},
    };

    #[test]
    fn completed_bf16_download_preserves_bits() {
        let Ok(context) = RocmContext::new(0) else { return };
        let expected = vec![0x3f80, 0x4000, 0xbf80, 0x0000];
        let tensor = context.tensor_from_bf16_bits(expected.clone(), 2, 2).expect("上传 BF16 测试 tensor");
        let actual = context.completed_tensor_to_bf16_bits(&tensor).expect("从独立 completed D2H stream 下载 BF16 tensor");
        assert_eq!(actual, expected);
    }

    #[test]
    fn independent_bf16_upload_preserves_bits() {
        let Ok(context) = RocmContext::new(0) else { return };
        let expected = vec![0x3f80, 0x4000, 0xbf80, 0x0000];
        let tensor = context.tensor_from_bf16_bits_independent(expected.clone(), 2, 2).expect("独立 H2D 上传 BF16 tensor");
        let actual = context.completed_tensor_to_bf16_bits(&tensor).expect("下载 independent H2D tensor");
        assert_eq!(actual, expected);
    }

    #[test]
    fn ordered_bf16_upload_hands_off_to_consumer_stream() {
        let Ok(context) = RocmContext::new(0) else { return };
        let expected = vec![0x3f80, 0x4000, 0xbf80, 0x0000];
        let tensor = context.tensor_from_bf16_bits_ordered(expected.clone(), 2, 2).expect("有序 H2D 上传 BF16 tensor");
        tensor.device.as_ref().expect("ordered H2D tensor 缺少 device buffer").enqueue_deferred_upload().expect("向 consumer stream 提交 H2D");
        context.synchronize_compute_stream().expect("等待 ordered H2D consumer stream");
        let actual = context.completed_tensor_to_bf16_bits(&tensor).expect("下载 ordered H2D tensor");
        assert_eq!(actual, expected);
    }

    #[test]
    fn reserved_token_concat_reuses_prefix_capacity() {
        let Ok(context) = RocmContext::new(0) else { return };
        let first_rows = context.tensor_from_f32(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let suffix = context.tensor_from_f32(vec![5.0, 6.0], 1, 2).unwrap();
        let first = context.concat_token_rows_reserved(&[&first_rows], 8).unwrap();
        let owner = first.device.as_deref().and_then(ops::hip::DeviceBuffer::prefix_capacity_owner).expect("reserved prefix 应持有 capacity owner");
        let actual = context.concat_token_rows_reserved(&[&first, &suffix], 8).unwrap();
        let actual_owner = actual.device.as_deref().and_then(ops::hip::DeviceBuffer::prefix_capacity_owner).expect("追加后应继续持有 capacity owner");
        assert_eq!(owner.device_pointer(), actual_owner.device_pointer());
        assert_eq!(context.tensor_to_f32(&actual).unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn split_block_attention_matches_contiguous_reference() {
        let Ok(context) = RocmContext::new(0) else { return };
        let query = context.tensor_from_f32(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let prefix_key = context.tensor_from_f32(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let prefix_value = context.tensor_from_f32(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let suffix_key = context.tensor_from_f32(vec![1.0, 1.0, -1.0, 1.0], 2, 2).unwrap();
        let suffix_value = context.tensor_from_f32(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        let key = context.concat_token_rows(&[&prefix_key, &suffix_key]).unwrap();
        let value = context.concat_token_rows(&[&prefix_value, &suffix_value]).unwrap();
        let spec = BlockAttentionSpec { geometry: GqaGeometry { num_heads: 1, num_kv_heads: 1, head_dim: 2 }, score_scale: 2.0_f32.sqrt().recip(), visible: vec![0..3, 0..4] };
        let expected = context.block_attention(&query, &key, &value, &spec).unwrap();
        let actual = context.block_attention_prefix_suffix(&query, &prefix_key, &prefix_value, &suffix_key, &suffix_value, &spec).unwrap();
        let expected = context.tensor_to_f32(&expected).unwrap();
        let actual = context.tensor_to_f32(&actual).unwrap();
        for (expected, actual) in expected.iter().zip(&actual) {
            assert!((expected - actual).abs() < 1.0e-5, "expected={expected} actual={actual}");
        }
    }

    #[test]
    fn split_block_attention_wave64_matches_shared_reduction_bits() {
        let Ok(context) = RocmContext::new(0) else { return };
        const QUERY_ROWS: usize = 3;
        const PREFIX_ROWS: usize = 37;
        const SUFFIX_ROWS: usize = 8;
        const HEADS: usize = 2;
        const KV_HEADS: usize = 1;
        const DIM: usize = 64;
        let values = |len: usize, multiplier: usize, offset: usize| {
            (0..len)
                .map(|index| {
                    let integer = (index.wrapping_mul(multiplier).wrapping_add(offset) % 257) as f32 - 128.0;
                    integer * (1.0 / 257.0)
                })
                .collect::<Vec<_>>()
        };
        let query = context.tensor_from_f32(values(QUERY_ROWS * HEADS * DIM, 29, 7), QUERY_ROWS, HEADS * DIM).unwrap();
        let prefix_key = context.tensor_from_f32(values(PREFIX_ROWS * KV_HEADS * DIM, 17, 11), PREFIX_ROWS, KV_HEADS * DIM).unwrap();
        let prefix_value = context.tensor_from_f32(values(PREFIX_ROWS * KV_HEADS * DIM, 23, 13), PREFIX_ROWS, KV_HEADS * DIM).unwrap();
        let suffix_key = context.tensor_from_f32(values(SUFFIX_ROWS * KV_HEADS * DIM, 31, 19), SUFFIX_ROWS, KV_HEADS * DIM).unwrap();
        let suffix_value = context.tensor_from_f32(values(SUFFIX_ROWS * KV_HEADS * DIM, 37, 23), SUFFIX_ROWS, KV_HEADS * DIM).unwrap();
        let key = context.concat_token_rows(&[&prefix_key, &suffix_key]).unwrap();
        let value = context.concat_token_rows(&[&prefix_value, &suffix_value]).unwrap();
        let spec = BlockAttentionSpec { geometry: GqaGeometry { num_heads: HEADS, num_kv_heads: KV_HEADS, head_dim: DIM }, score_scale: (DIM as f32).sqrt().recip(), visible: vec![0..38, 3..42, 7..45] };
        let expected = context.block_attention(&query, &key, &value, &spec).unwrap();
        let actual = context.block_attention_prefix_suffix(&query, &prefix_key, &prefix_value, &suffix_key, &suffix_value, &spec).unwrap();
        let expected = context.tensor_to_f32(&expected).unwrap();
        let actual = context.tensor_to_f32(&actual).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "wave64 split attention index={index} expected={expected} actual={actual}");
        }
    }

    #[test]
    fn dsa_fused_layernorm_rope_append_matches_raw_q8_cache_bits() {
        let Ok(context) = RocmContext::new(0) else { return };
        const ROWS: usize = 5;
        const HEAD_DIM: usize = 128;
        const ROPE_DIM: usize = 64;
        let weight_values = (0..HEAD_DIM).map(|column| 0.75 + (column % 17) as f32 * 0.03125).collect::<Vec<_>>();
        let bias_values = (0..HEAD_DIM).map(|column| ((column * 13 % 31) as f32 - 15.0) * 0.0078125).collect::<Vec<_>>();
        let norm_weight = context.prepare_f32(&weight_values, 1, HEAD_DIM).unwrap();
        let norm_bias = context.prepare_f32(&bias_values, 1, HEAD_DIM).unwrap();
        let cosine = (0..ROWS * (ROPE_DIM / 2)).map(|index| ((index * 7 % 101) as f32 * 0.013).cos()).collect::<Vec<_>>();
        let sine = (0..ROWS * (ROPE_DIM / 2)).map(|index| ((index * 7 % 101) as f32 * 0.013).sin()).collect::<Vec<_>>();
        let spec = DsaSpec { num_heads: 32, head_dim: HEAD_DIM, rope_dim: ROPE_DIM, top_k: 8, rotary_layout: RotaryLayout::SplitHalf, kpool: 0, always_select_tail: false };
        let mut reference = RocmDsaState::new(1, 16, HEAD_DIM, spec.top_k).unwrap();
        let mut fused = RocmDsaState::new(1, 16, HEAD_DIM, spec.top_k).unwrap();
        let row_values = |position: usize| {
            (0..HEAD_DIM)
                .map(|column| {
                    let integer = (position * 37 + column * 29 + column * column * 3) % 521;
                    (integer as f32 - 260.0) * (1.0 / 257.0)
                })
                .collect::<Vec<_>>()
        };
        for position in 0..ROWS {
            let key = context.tensor_from_f32(row_values(position), 1, HEAD_DIM).unwrap();
            let normalized = context.layernorm_bias(&key, &norm_weight, &norm_bias, 1.0e-6).unwrap();
            let rotated = context.rope_prefix(&normalized, 1, ROPE_DIM, RotaryLayout::SplitHalf, position, &cosine, &sine).unwrap();
            context.append_dsa_keys(&mut reference, 0, position, &rotated, &spec).unwrap();
            assert!(context.append_dsa_keys_layernorm_rope(&mut fused, 0, position, &key, &norm_weight, &norm_bias, 1.0e-6, &cosine, &sine, &spec).unwrap());
        }
        let reference = reference.download_layers().unwrap().remove(0).unwrap();
        let fused = fused.download_layers().unwrap().remove(0).unwrap();
        assert_eq!(fused.rows, reference.rows);
        assert_eq!(fused.key_group_size, reference.key_group_size);
        assert_eq!(fused.hadamard, reference.hadamard);
        assert_eq!(fused.keys, reference.keys);
        assert_eq!(fused.scales, reference.scales);

        let mut reference = RocmDsaState::new(1, 16, HEAD_DIM, spec.top_k).unwrap();
        let mut fused = RocmDsaState::new(1, 16, HEAD_DIM, spec.top_k).unwrap();
        let keys = context.tensor_from_f32((0..ROWS).flat_map(row_values).collect(), ROWS, HEAD_DIM).unwrap();
        assert!(context.supports_dsa_keys_layernorm_rope(&fused, &keys, &norm_weight, &norm_bias, &spec));
        let normalized = context.layernorm_bias(&keys, &norm_weight, &norm_bias, 1.0e-6).unwrap();
        let rotated = context.rope_prefix(&normalized, 1, ROPE_DIM, RotaryLayout::SplitHalf, 0, &cosine, &sine).unwrap();
        context.append_dsa_keys(&mut reference, 0, 0, &rotated, &spec).unwrap();
        assert!(context.append_dsa_keys_layernorm_rope(&mut fused, 0, 0, &keys, &norm_weight, &norm_bias, 1.0e-6, &cosine, &sine, &spec).unwrap());
        let reference = reference.download_layers().unwrap().remove(0).unwrap();
        let fused = fused.download_layers().unwrap().remove(0).unwrap();
        assert_eq!(fused.keys, reference.keys);
        assert_eq!(fused.scales, reference.scales);
    }
}
