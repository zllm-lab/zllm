//! ROCm 多卡节点引擎共用的设备链构造、层放置与 KV 容量上报。
//!
//! GLM-5.2（链头 + 下游 stage）与 Ornith（单机分层）都以 `devices` + `layer_ends`
//! 描述每张卡负责的层区间。这里集中三件事：校验并初始化设备上下文、按层找卡、
//! 用统一口径把每卡空闲显存折算成 KV token 容量。

use crate::backend::BackendError;
use crate::backend::StageExecutionBackend;
use crate::backend::rocm::RocmContext;
use crate::runtime::session::{KvCacheDeviceCapacity, NodeCapabilities};

pub type DynError = Box<dyn std::error::Error + Send + Sync>;

pub struct RocmDeviceChain {
    pub contexts: Vec<RocmContext>,
    /// 每卡负责的最后一层（含）；严格递增，最后一项必须等于 last_layer。
    pub layer_ends: Vec<usize>,
}

/// 请求终止后释放当前协调线程持有的 ROCm 临时 workspace。各 stage 工作线程
/// 由通用 scheduler 在 Close 时分别回收；这里覆盖输出头与 speculative runtime。
pub fn release_request_workspaces(devices: &[i32]) -> Result<(), String> {
    for &device in devices {
        crate::kernel::rocm::hip::release_tensor_workspace(device)?;
    }
    Ok(())
}

impl RocmDeviceChain {
    pub fn new(devices: &[i32], layer_ends: Vec<usize>, last_layer: usize, allow_cpu_reference_fallback: bool) -> Result<Self, DynError> {
        if devices.is_empty() || devices.len() != layer_ends.len() || *layer_ends.last().unwrap_or(&usize::MAX) != last_layer {
            return Err(format!("ROCm 设备链 layer_ends 必须非空、与 devices 一一对应且最后边界 = {last_layer}").into());
        }
        if layer_ends.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!("ROCm 设备链 layer_ends 必须严格递增: {layer_ends:?}").into());
        }
        if devices.len() > 1 {
            crate::kernel::rocm::hip::enable_device_buffer_reuse();
        }
        let contexts =
            devices.iter().map(|&device| RocmContext::configured(device, allow_cpu_reference_fallback).map_err(|error| -> DynError { format!("ROCm device {device} 初始化失败: {error}").into() })).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { contexts, layer_ends })
    }

    /// 层所在的设备下标。
    pub fn device_of_layer(&self, layer: usize) -> Result<usize, BackendError> {
        self.layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| crate::backend::BackendError::UnsupportedLayer { layer })
    }

    /// 设备负责的层区间（含两端）。
    pub fn layer_range(&self, device: usize) -> Result<(usize, usize), String> {
        let end = *self.layer_ends.get(device).ok_or_else(|| format!("ROCm 设备链没有 device {device}"))?;
        let start = if device == 0 { 0 } else { self.layer_ends[device - 1] + 1 };
        Ok((start, end))
    }
}

/// 用空闲显存折算单卡 KV token 容量。`kv_bytes_per_layer_token` 是调用方按模型
/// KV 几何算出的每层每 token 字节；`layers` 可被 `admission_layers` 抬高，
/// 避免极少层的卡给出虚高的 token 容量。
pub fn kv_capacity_from_free(label: String, free: usize, total: usize, layers: usize, kv_bytes_per_layer_token: usize, safety_bytes: usize, admission_layers: usize) -> Result<KvCacheDeviceCapacity, String> {
    let kv_budget = free.saturating_sub(safety_bytes).min(total / 2);
    let budget_layers = layers.max(admission_layers);
    let bytes_per_token = budget_layers.checked_mul(kv_bytes_per_layer_token).ok_or("KV token 显存估算溢出")?;
    Ok(crate::runtime::node::kv_device_capacity(label, kv_budget, bytes_per_token))
}

/// 对一张在本地进程内的卡查询空闲/总量后折算容量。
pub fn kv_device_capacity(context: &RocmContext, label: String, layers: usize, kv_bytes_per_layer_token: usize, safety_bytes: usize, admission_layers: usize) -> Result<KvCacheDeviceCapacity, String> {
    let free = context.stage_available_bytes().map_err(|error| format!("查询 ROCm device {} 可用显存: {error:?}", context.device_id()))?;
    let total = context.stage_total_bytes().map_err(|error| format!("查询 ROCm device {} 总显存: {error:?}", context.device_id()))?;
    kv_capacity_from_free(label, free, total, layers, kv_bytes_per_layer_token, safety_bytes, admission_layers)
}

/// 多卡 ROCm 节点的 NodeCapabilities 公共字段。CU 数按 W7900 60 CU/卡粗略上报，
/// scheduler 不依赖精确值；显存总量取首卡实测。
pub fn node_capabilities(contexts: &[RocmContext], max_seq_len: usize, kv_cache_format: &str, model_format: String, model_bytes: u64) -> NodeCapabilities {
    let accelerator_memory_bytes = contexts.first().and_then(|context| context.stage_total_bytes().ok()).map(|total| total as u64);
    crate::runtime::node::text_capabilities(
        crate::runtime::node::DeviceDescriptor {
            backend: "rocm",
            accelerator: format!("rocm-device-{}", contexts.first().map(RocmContext::device_id).unwrap_or_default()),
            compute_units: Some(contexts.len() * 60),
            compute_unit_kind: "cu",
            memory_kind: "dedicated",
            unified_memory: false,
            system_memory_bytes: None,
            accelerator_memory_bytes,
            recommended_working_set_bytes: accelerator_memory_bytes.map(|total| total / 6 * 5),
        },
        crate::runtime::node::SessionDescriptor { model_format: &model_format, model_bytes, max_seq_len, kv_cache_format, input_modalities: &["text"] },
    )
}
