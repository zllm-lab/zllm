//! 华为 NPU 后端。
//!
//! 真机路径不允许把未实现的算子静默交给 CPU：当前设备 tensor/算子桥还在
//! 下沉过程中，能力缺口必须显式报错，否则一次模型运行会同时混用 NPU 和
//! ARM，既破坏“层间零拷贝”目标，也会让 benchmark 失去意义。

use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetInputs, GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetWeightsRef},
        gqa::GqaSpec,
    },
    backend::{Backend, BackendError, BackendResources, GqaPrefillBackend, LinearWeight},
    moe::Activation,
};

#[cfg(not(target_env = "ohos"))]
use crate::kernel::cpu::CpuTensor;

use super::cpu::{CpuContext, CpuWeight};

#[cfg(not(target_env = "ohos"))]
use super::cpu::CpuKvCache;

#[cfg(not(target_env = "ohos"))]
use super::cpu::CpuGatedDeltaNetStorage;

#[cfg(target_env = "ohos")]
use crate::attention::gated_delta_net::GatedDeltaNetStorage;

#[cfg(target_env = "ohos")]
use crate::backend::compute_error as compute;

// ============================================================================
// Qwen3.5-4B dense FFN 的设备 decode 布局选择（实验路径）。
// 形状来自该模型规格；布局编号与设备侧重排 kernel 一一对应，
// 见 kernel/huawei/{q5,q6}_*_gemv.cpp 与 probe 工程的图生成脚本。
// ============================================================================
#[cfg(target_env = "ohos")]
const COMPACT_FFN_HIDDEN: usize = 2560;
#[cfg(target_env = "ohos")]
const COMPACT_FFN_INTERMEDIATE: usize = 9216;
/// Q5 compact gate（ffn_gate，[9216,2560] Q5_K）。
#[cfg(target_env = "ohos")]
const COMPACT_FFN_GATE_LAYOUT: u32 = 1;
/// Q5 dense up（ffn_up，[9216,2560] Q5_K）；三份 Const 合计需低于单图 64 MiB 访问窗口。
#[cfg(target_env = "ohos")]
const COMPACT_FFN_UP_LAYOUT: u32 = 2;
/// compact down（ffn_down，[2560,9216] Q6_K；Q5 down 的 compact kernel 已有独立验证）。
#[cfg(target_env = "ohos")]
const COMPACT_FFN_DOWN_LAYOUT: u32 = 3;
#[cfg(target_env = "ohos")]
const Q5_K_TYPE: u32 = 13;
#[cfg(target_env = "ohos")]
const Q6_K_TYPE: u32 = 14;

/// down 投影命中已验证的 compact FFN 形状时选择 decode 布局，其余保持 raw(0)。
#[cfg(target_env = "ohos")]
fn compact_ffn_down_layout(tensor_type: u32, rows: usize, cols: usize) -> u32 {
    match (tensor_type, rows, cols) {
        (Q6_K_TYPE, COMPACT_FFN_HIDDEN, COMPACT_FFN_INTERMEDIATE) => COMPACT_FFN_DOWN_LAYOUT,
        _ => 0,
    }
}

/// gate/up 成对命中 Qwen3.5 FFN 形状（两份 Q5_K [9216,2560]）时使用互补布局。
#[cfg(target_env = "ohos")]
fn compact_ffn_gate_up_pair(first: u32, second: u32, rows: usize, cols: usize) -> bool {
    first == Q5_K_TYPE && second == Q5_K_TYPE && rows == COMPACT_FFN_INTERMEDIATE && cols == COMPACT_FFN_HIDDEN
}

/// 单 token decode 的 FFN 形状：hidden 2560，gate/up [9216,2560]，down [2560,9216]。
#[cfg(target_env = "ohos")]
fn compact_ffn_decode_batch(input_rows: usize, input_cols: usize, gate_rows: usize, gate_cols: usize, up_rows: usize, up_cols: usize, down_rows: usize, down_cols: usize) -> bool {
    input_rows == 1
        && input_cols == COMPACT_FFN_HIDDEN
        && gate_rows == COMPACT_FFN_INTERMEDIATE
        && gate_cols == COMPACT_FFN_HIDDEN
        && up_rows == COMPACT_FFN_INTERMEDIATE
        && up_cols == COMPACT_FFN_HIDDEN
        && down_rows == COMPACT_FFN_HIDDEN
        && down_cols == COMPACT_FFN_INTERMEDIATE
}

/// Huawei 真机上的 resident tensor。句柄指向 C++ HIAI `AiTensor`，Rust 不拥有
/// 其 buffer，也不把每个线性层结果转成 `Vec<f32>`。
#[cfg(target_env = "ohos")]
#[derive(Debug)]
pub struct HuaweiTensor {
    handle: *mut std::ffi::c_void,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(target_env = "ohos")]
impl HuaweiTensor {
    pub fn from_f32(values: &[f32], rows: usize, cols: usize) -> Result<Self, BackendError> {
        if values.len() != rows.checked_mul(cols).ok_or_else(|| compute("Huawei tensor shape 溢出"))? {
            return Err(compute(format!("Huawei tensor upload 元素数 {}，期望 [{rows},{cols}]", values.len())));
        }
        let handle = unsafe { zllm_huawei_tensor_from_f32(values.as_ptr(), rows, cols) };
        if handle.is_null() {
            return Err(compute(format!("Huawei NPU tensor upload 失败 shape=[{rows},{cols}]")));
        }
        Ok(Self { handle, rows, cols })
    }

    pub fn to_f32(&self) -> Result<Vec<f32>, BackendError> {
        let elements = self.rows.checked_mul(self.cols).ok_or_else(|| compute("Huawei tensor download shape 溢出"))?;
        let mut values = vec![0.0; elements];
        let status = unsafe { zllm_huawei_tensor_to_f32(self.handle, values.as_mut_ptr(), elements) };
        if status != 0 {
            return Err(compute(format!("Huawei NPU tensor download 失败 status={status}")));
        }
        Ok(values)
    }

    /// 消费前置 Vector 算子生成的 task-major 16x16 K tile。输入和输出都是
    /// HIAI resident tensor；Gram 的两个操作数在 C++ 层引用同一份 K。
    #[allow(dead_code)]
    fn gdn_chunk_gram_npu(&self, tasks: usize) -> Result<Self, BackendError> {
        const TILE: usize = 16;
        if self.rows != tasks * TILE || self.cols != 128 {
            return Err(compute(format!("Huawei NPU GDN chunk Gram 输入 [{},{}]，期望 [{},128]", self.rows, self.cols, tasks * TILE,)));
        }
        let mut output = std::ptr::null_mut();
        let status = unsafe { zllm_huawei_gdn_chunk_gram_tensor(self.handle, tasks, self.cols, &mut output) };
        if status != 0 || output.is_null() {
            return Err(compute(format!("Huawei NPU GDN chunk Gram 执行失败 tasks={tasks} status={status}")));
        }
        Ok(Self { handle: output, rows: tasks * TILE, cols: TILE })
    }

    fn select_row_npu(&self, row: usize) -> Result<Self, BackendError> {
        if row >= self.rows {
            return Err(compute(format!("Huawei NPU select_row {row} 越界，rows={}", self.rows)));
        }
        // select_last_row kernel 按 128 元素 tile 寻址（columns / 128 向下取整），
        // 尾部不足 128 的列会被静默丢弃。
        if self.cols % 128 != 0 {
            return Err(compute(format!("Huawei NPU select_row 列数 {} 不是 128 的整数倍，select_last_row kernel 按 128 元素 tile 寻址", self.cols)));
        }
        let mut output = std::ptr::null_mut();
        let status = unsafe { zllm_huawei_select_row_tensor(self.handle, self.rows, self.cols, row, &mut output) };
        if status != 0 || output.is_null() {
            return Err(compute(format!("Huawei NPU select_row 执行失败 input=[{},{}] row={row} status={status}", self.rows, self.cols)));
        }
        Ok(Self { handle: output, rows: 1, cols: self.cols })
    }
}

#[cfg(target_env = "ohos")]
impl Drop for HuaweiTensor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { zllm_huawei_tensor_release(self.handle) };
        }
    }
}

#[cfg(not(target_env = "ohos"))]
pub type HuaweiTensor = CpuTensor;

#[cfg(target_env = "ohos")]
struct HuaweiGqaLayer {
    key: *mut std::ffi::c_void,
    value: *mut std::ffi::c_void,
    score_scratch: *mut std::ffi::c_void,
    rows: usize,
}

#[cfg(target_env = "ohos")]
impl Drop for HuaweiGqaLayer {
    fn drop(&mut self) {
        unsafe {
            zllm_huawei_tensor_release(self.key);
            zllm_huawei_tensor_release(self.value);
            zllm_huawei_tensor_release(self.score_scratch);
        }
    }
}

/// 每个 Full-Attention 层各自持有 NPU 常驻 K/V；只在首次进入该层时分配。
#[cfg(target_env = "ohos")]
pub struct HuaweiKvCache {
    layers: Vec<Option<HuaweiGqaLayer>>,
    max_seq_len: usize,
}

#[cfg(target_env = "ohos")]
impl HuaweiKvCache {
    pub fn new(layer_count: usize, max_seq_len: usize) -> Self {
        Self { layers: std::iter::repeat_with(|| None).take(layer_count).collect(), max_seq_len }
    }

    fn layer(&mut self, layer: usize, position: usize, batch: usize) -> Result<&mut HuaweiGqaLayer, BackendError> {
        let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if self.max_seq_len != 1040 || position.checked_add(batch).is_none_or(|end| end > self.max_seq_len) {
            return Err(compute(format!("Huawei NPU L{layer} GQA cache position={position} batch={batch} capacity={}", self.max_seq_len,)));
        }
        if slot.is_none() {
            let key = unsafe { zllm_huawei_tensor_zeros_2d(self.max_seq_len, 4096) };
            if key.is_null() {
                return Err(compute(format!("Huawei NPU L{layer} K cache 分配失败")));
            }
            let value = unsafe { zllm_huawei_tensor_zeros_2d(self.max_seq_len, 4096) };
            if value.is_null() {
                unsafe { zllm_huawei_tensor_release(key) };
                return Err(compute(format!("Huawei NPU L{layer} V cache 分配失败")));
            }
            let score_scratch = unsafe { zllm_huawei_tensor_zeros_2d(8, self.max_seq_len * 16) };
            if score_scratch.is_null() {
                unsafe {
                    zllm_huawei_tensor_release(key);
                    zllm_huawei_tensor_release(value);
                }
                return Err(compute(format!("Huawei NPU L{layer} attention score scratch 分配失败")));
            }
            *slot = Some(HuaweiGqaLayer { key, value, score_scratch, rows: 0 });
        }
        let cached = slot.as_mut().expect("刚完成 Huawei GQA cache 初始化");
        if cached.rows != position {
            return Err(compute(format!("Huawei NPU L{layer} GQA cache position={position}，当前 rows={}", cached.rows,)));
        }
        Ok(cached)
    }
}

#[cfg(target_env = "ohos")]
pub struct HuaweiGatedDeltaNetStorage {
    conv: *mut std::ffi::c_void,
    recurrent: *mut std::ffi::c_void,
    allocated_bytes: usize,
}

#[cfg(target_env = "ohos")]
unsafe impl Send for HuaweiGatedDeltaNetStorage {}

#[cfg(target_env = "ohos")]
impl Drop for HuaweiGatedDeltaNetStorage {
    fn drop(&mut self) {
        unsafe {
            zllm_huawei_tensor_release(self.conv);
            zllm_huawei_tensor_release(self.recurrent);
        }
    }
}

#[cfg(target_env = "ohos")]
impl GatedDeltaNetStorage for HuaweiGatedDeltaNetStorage {
    fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }
}

#[cfg(not(target_env = "ohos"))]
pub type HuaweiGatedDeltaNetStorage = CpuGatedDeltaNetStorage;

macro_rules! huawei_cpu_or_npu_error {
    ($backend:expr, $op:literal, $cpu:expr) => {{
        #[cfg(target_env = "ohos")]
        {
            $backend.npu_error($op)
        }
        #[cfg(not(target_env = "ohos"))]
        {
            $cpu
        }
    }};
}

#[derive(Debug)]
pub struct HuaweiWeight {
    /// GGUF 文件范围和 shape 元数据；量化 payload 不在这里展开为 CPU 数组。
    source: CpuWeight,
    /// 非量化权重在模型装配阶段转为 FP16 并常驻页对齐的 UMA mmap。算子
    /// 执行时只创建外部 AiTensor 视图，避免长期占用 HIAI 张量登记槽。
    #[cfg(target_env = "ohos")]
    fp16_device: Option<*mut std::ffi::c_void>,
    /// packed Q5/Q6/Q8 权重在模型装配阶段一次性上传；decode 只传句柄，
    /// 热路径没有文件描述符、offset 或 sidecar 生命周期。
    #[cfg(target_env = "ohos")]
    quant_device: Option<*mut std::ffi::c_void>,
}

#[cfg(target_env = "ohos")]
unsafe impl Send for HuaweiWeight {}

#[cfg(target_env = "ohos")]
unsafe impl Sync for HuaweiWeight {}

#[cfg(target_env = "ohos")]
impl Clone for HuaweiWeight {
    fn clone(&self) -> Self {
        let fp16_device = self.fp16_device.map(|handle| unsafe { zllm_huawei_fp16_weight_retain(handle) });
        let quant_device = self.quant_device.map(|handle| unsafe { zllm_huawei_quant_weight_retain(handle) });
        Self { source: self.source.clone(), fp16_device, quant_device }
    }
}

#[cfg(not(target_env = "ohos"))]
impl Clone for HuaweiWeight {
    fn clone(&self) -> Self {
        Self { source: self.source.clone() }
    }
}

#[cfg(target_env = "ohos")]
impl Drop for HuaweiWeight {
    fn drop(&mut self) {
        if let Some(handle) = self.fp16_device.take() {
            unsafe { zllm_huawei_fp16_weight_release(handle) };
        }
        if let Some(handle) = self.quant_device.take() {
            unsafe { zllm_huawei_quant_weight_release(handle) };
        }
    }
}

impl HuaweiWeight {
    pub fn rows(&self) -> usize {
        self.source.rows()
    }

    pub fn cols(&self) -> usize {
        self.source.cols()
    }

    #[cfg(target_env = "ohos")]
    fn device_view(&self) -> Result<HuaweiTensor, BackendError> {
        let Some(weight) = self.fp16_device else {
            return Err(compute(format!("Huawei NPU FP16 resident 权重缺失 shape=[{},{}]", self.rows(), self.cols(),)));
        };
        let handle = unsafe { zllm_huawei_fp16_weight_view(weight) };
        if handle.is_null() {
            return Err(compute(format!("Huawei NPU FP16 resident 权重视图创建失败 shape=[{},{}] status={}", self.rows(), self.cols(), unsafe { zllm_huawei_tensor_last_status() },)));
        }
        Ok(HuaweiTensor { handle, rows: self.rows(), cols: self.cols() })
    }
}

/// `require_npu` 保留在 API 中用于 host oracle 与真机配置兼容；OHOS 永远禁止
/// CPU fallback，不再由该开关改变设备路径语义。
#[derive(Debug, Clone, Copy, Default)]
pub struct HuaweiContext {
    #[allow(dead_code)]
    require_npu: bool,
}

impl HuaweiContext {
    pub fn new(require_npu: bool) -> Self {
        Self { require_npu }
    }

    pub fn quant_profile(&self) -> String {
        #[cfg(target_env = "ohos")]
        {
            let mut output = vec![0 as std::os::raw::c_char; 4096];
            let status = unsafe { zllm_huawei_quant_profile_dump(output.as_mut_ptr(), output.len()) };
            if status != 0 {
                return format!("quant_profile status={status}");
            }
            return unsafe { std::ffi::CStr::from_ptr(output.as_ptr()) }.to_string_lossy().into_owned();
        }
        #[cfg(not(target_env = "ohos"))]
        String::new()
    }

    /// prefill 已完成后，把 dense FFN 三个 raw packed 权重逐个转换为 decode
    /// 布局；每个转换完成立即释放对应 GGUF 映射，decode 热路径不做重排或 I/O。
    #[cfg(target_env = "ohos")]
    pub fn prepare_ffn_decode(&self, gate: &HuaweiWeight, up: &HuaweiWeight, down: &HuaweiWeight) -> Result<(), BackendError> {
        for (name, weight) in [("gate", gate), ("up", up), ("down", down)] {
            let handle = weight.quant_device.ok_or_else(|| compute(format!("Huawei NPU FFN {name} resident 权重缺失")))?;
            let status = unsafe { zllm_huawei_quant_weight_prepare_optimized(handle) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU FFN {name} decode layout 准备失败 status={status}")));
            }
        }
        Ok(())
    }

    #[cfg(target_env = "ohos")]
    pub fn quant_io_counters(&self) -> Result<(u64, u64), BackendError> {
        let mut calls = 0_u64;
        let mut bytes = 0_u64;
        let status = unsafe { zllm_huawei_quant_io_counters(&mut calls, &mut bytes) };
        if status != 0 {
            return Err(compute(format!("Huawei NPU quant I/O counter 读取失败 status={status}")));
        }
        Ok((calls, bytes))
    }

    /// 在 NPU 上完成 logits argmax 与下一 token 的 Q6_K embedding lookup。
    /// 返回的 token 只用于 host 控制流；下一轮 hidden 始终保持 AiTensor。
    #[cfg(target_env = "ohos")]
    pub fn q6_argmax_embedding(&self, logits: &HuaweiTensor, embedding: &HuaweiWeight, excluded: &[u32]) -> Result<(u32, HuaweiTensor), BackendError> {
        let Some(device) = embedding.quant_device else {
            return Err(compute("Huawei NPU Q6 embedding 权重未驻留"));
        };
        if logits.rows != 1 || logits.cols != 248_320 || embedding.rows() != 248_320 || embedding.cols() != 2_560 || excluded.len() != 2 {
            return Err(compute(format!("Huawei NPU argmax+embedding shape 不支持 logits=[{},{}] embedding=[{},{}] excluded={}", logits.rows, logits.cols, embedding.rows(), embedding.cols(), excluded.len(),)));
        }
        let mut token = 0_u32;
        let mut hidden = std::ptr::null_mut();
        let status = unsafe { zllm_huawei_q6_argmax_embedding_resident_tensor(logits.handle, device, excluded.as_ptr(), excluded.len(), &mut token, &mut hidden) };
        if status != 0 || hidden.is_null() {
            return Err(compute(format!("Huawei NPU argmax+Q6 embedding 执行失败 status={status}")));
        }
        Ok((token, HuaweiTensor { handle: hidden, rows: 1, cols: 2_560 }))
    }

    /// 在 prefill 前一次性上传 1024 行 prefill 与 16 行 decode RoPE。
    /// 后续 rope_prefix 只按 position 复用 resident AiTensor。
    #[cfg(target_env = "ohos")]
    pub fn prepare_rope_table(&self, cosine: &[f32], sine: &[f32], rows: usize) -> Result<(), BackendError> {
        if rows != 1_040 || cosine.len() != rows * 32 || sine.len() != rows * 32 {
            return Err(compute(format!("Huawei NPU RoPE resident table shape 非法 rows={rows} cos={} sin={}", cosine.len(), sine.len(),)));
        }
        let status = unsafe { zllm_huawei_rope_table_prepare(cosine.as_ptr(), sine.as_ptr(), rows) };
        if status != 0 {
            return Err(compute(format!("Huawei NPU RoPE resident table 准备失败 status={status}")));
        }
        Ok(())
    }

    /// 显式清理真机量化权重/模型缓存，只供冷启动探针使用。
    pub fn clear_quant_cache(&self) {
        #[cfg(target_env = "ohos")]
        unsafe {
            zllm_huawei_quant_cache_clear();
        }
    }

    fn cpu(&self) -> CpuContext {
        CpuContext
    }

    #[cfg(target_env = "ohos")]
    fn prepare_quantized_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize, optimized_layout: u32) -> Result<HuaweiWeight, BackendError> {
        let LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(matrix)) = weight else {
            return Err(compute("Huawei NPU optimized layout 只接受 GGUF 量化权重"));
        };
        let (file, offset) = matrix.file_range();
        let quant_device = unsafe { zllm_huawei_quant_weight_from_file_layout(matrix.tensor_type.0, file, offset, matrix.storage_len(), rows, cols, optimized_layout) };
        if quant_device.is_null() {
            return Err(compute(format!("Huawei NPU packed 权重驻留失败 type={} shape=[{rows},{cols}] bytes={} layout={optimized_layout}", matrix.tensor_type.0, matrix.storage_len())));
        }
        let source = self.cpu().prepare_weight(weight, rows, cols).map_err(|error| {
            unsafe { zllm_huawei_quant_weight_release(quant_device) };
            error
        })?;
        Ok(HuaweiWeight { source, fp16_device: None, quant_device: Some(quant_device) })
    }

    #[cfg(target_env = "ohos")]
    fn npu_error<T>(&self, op: &'static str) -> Result<T, BackendError> {
        Err(compute(format!("Huawei NPU 算子 {op} 尚未实现；设备路径禁止 CPU fallback")))
    }

    /// 量化线性层下沉 NPU。支持 Q5_K(13)、Q6_K(14)、Q8_0(8)。
    fn quantized_linear(&self, input: &HuaweiTensor, weight: &HuaweiWeight) -> Result<Option<HuaweiTensor>, BackendError> {
        let Some(matrix) = &weight.source.gguf else {
            #[cfg(target_env = "ohos")]
            return Err(compute(format!("Huawei NPU 量化线性层没有 GGUF resident 权重 shape=[{},{}] input=[{},{}]；禁止 CPU fallback", weight.rows(), weight.cols(), input.rows, input.cols)));
            #[cfg(not(target_env = "ohos"))]
            return Ok(None);
        };
        let tensor_type = matrix.tensor_type.0;
        if !matches!(tensor_type, 13 | 14 | 8) {
            #[cfg(target_env = "ohos")]
            return self.npu_error("未支持的量化格式");
            #[cfg(not(target_env = "ohos"))]
            return Ok(None);
        }
        #[cfg(target_env = "ohos")]
        {
            let Some(device) = weight.quant_device else {
                return Err(compute(format!("Huawei NPU packed 量化权重未驻留 shape=[{},{}]", matrix.rows, matrix.columns,)));
            };
            // 按 tensor_type 选择对应的块布局校验。
            match tensor_type {
                13 => crate::kernel::huawei::validate_q5_k_layout(matrix.storage_len(), matrix.rows, matrix.columns, input.rows, input.rows * input.cols),
                14 => crate::kernel::huawei::validate_q6_k_layout(matrix.storage_len(), matrix.rows, matrix.columns, input.rows, input.rows * input.cols),
                8 => crate::kernel::huawei::validate_q8_0_layout(matrix.storage_len(), matrix.rows, matrix.columns, input.rows, input.rows * input.cols),
                _ => unreachable!(),
            }
            .map_err(compute)?;
            if input.rows != 1 && input.rows != 4 && input.rows != 15 && input.rows != 1024 {
                return Err(compute(format!("Huawei NPU 量化图暂只支持 batch=1/4/15/1024，实际 batch={}", input.rows)));
            }
            let mut output = std::ptr::null_mut();
            let status = unsafe {
                if tensor_type == 14 && matrix.rows == 248320 && matrix.columns == 2560 && input.rows == 1 {
                    zllm_huawei_q6_vocab_matmul_resident_tensor(device, input.handle, matrix.rows, matrix.columns, input.rows, &mut output)
                } else {
                    zllm_huawei_quant_matmul_resident_tensor(tensor_type, device, input.handle, matrix.rows, matrix.columns, input.rows, &mut output)
                }
            };
            if status != 0 {
                return Err(compute(format!("Huawei NPU 量化执行失败: type={tensor_type} batch={} matrix=[{},{}] status={status}; 禁止 CPU fallback", input.rows, matrix.rows, matrix.columns)));
            }
            if output.is_null() {
                return Err(compute("Huawei NPU 量化图返回空 tensor"));
            }
            return Ok(Some(HuaweiTensor { handle: output, rows: input.rows, cols: matrix.rows }));
        }
        #[cfg(not(target_env = "ohos"))]
        {
            let _ = input;
            let _ = self.require_npu;
            Ok(None)
        }
    }

    #[cfg(target_env = "ohos")]
    fn f32_linear(&self, input: &HuaweiTensor, weight: &HuaweiWeight) -> Result<HuaweiTensor, BackendError> {
        let device = weight.device_view()?;
        if input.cols != weight.cols() || !matches!(input.rows, 1 | 4 | 15 | 1024) {
            return Err(compute(format!("Huawei NPU F32 linear shape 不支持 input=[{},{}] weight=[{},{}]", input.rows, input.cols, weight.rows(), weight.cols())));
        }
        let mut output = std::ptr::null_mut();
        let status = unsafe { zllm_huawei_f32_matmul_tensor(device.handle, input.handle, weight.rows(), weight.cols(), input.rows, &mut output) };
        if status != 0 {
            return Err(compute(format!("Huawei NPU F32 linear 执行失败 input=[{},{}] weight=[{},{}] status={status}；禁止 CPU fallback", input.rows, input.cols, weight.rows(), weight.cols())));
        }
        let Some(handle) = (!output.is_null()).then_some(output) else {
            return Err(compute("Huawei NPU F32 linear 返回空 tensor"));
        };
        Ok(HuaweiTensor { handle, rows: input.rows, cols: weight.rows() })
    }

    #[cfg(target_env = "ohos")]
    fn rmsnorm_tensor(&self, input: &HuaweiTensor, weight: &HuaweiWeight, eps: f32, op: &'static str) -> Result<HuaweiTensor, BackendError> {
        let gamma = weight.device_view()?;
        if input.cols != weight.cols() || !matches!(input.rows, 1 | 4 | 15 | 1024) {
            return Err(compute(format!("Huawei NPU {op} shape 不支持 input=[{},{}] gamma=[{},{}]；当前 OMC 仅支持 batch=1/4/15/1024", input.rows, input.cols, weight.rows(), weight.cols(),)));
        }
        let mut output = std::ptr::null_mut();
        let status = unsafe { zllm_huawei_rmsnorm_tensor(input.handle, gamma.handle, input.rows, input.cols, &mut output) };
        if status != 0 {
            return Err(compute(format!("Huawei NPU {op} 执行失败 input=[{},{}] eps={eps} status={status}；禁止 CPU fallback", input.rows, input.cols,)));
        }
        let Some(handle) = (!output.is_null()).then_some(output) else {
            return Err(compute(format!("Huawei NPU {op} 图返回空 tensor")));
        };
        Ok(HuaweiTensor { handle, rows: input.rows, cols: input.cols })
    }
}

#[cfg(target_env = "ohos")]
unsafe extern "C" {
    fn zllm_huawei_tensor_from_f32(values: *const f32, rows: usize, columns: usize) -> *mut std::ffi::c_void;
    fn zllm_huawei_fp16_weight_from_f32(values: *const f32, rows: usize, columns: usize, add_one: u32) -> *mut std::ffi::c_void;
    fn zllm_huawei_fp16_weight_retain(handle: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn zllm_huawei_fp16_weight_release(handle: *mut std::ffi::c_void);
    fn zllm_huawei_fp16_weight_view(handle: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn zllm_huawei_tensor_last_status() -> i32;
    fn zllm_huawei_tensor_zeros(elements: usize) -> *mut std::ffi::c_void;
    fn zllm_huawei_tensor_zeros_2d(rows: usize, columns: usize) -> *mut std::ffi::c_void;
    fn zllm_huawei_tensor_retain(handle: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn zllm_huawei_tensor_release(handle: *mut std::ffi::c_void);
    fn zllm_huawei_quant_weight_from_file(tensor_type: u32, file: i32, offset: u64, weight_bytes: usize, rows: usize, columns: usize) -> *mut std::ffi::c_void;
    fn zllm_huawei_quant_weight_from_file_layout(tensor_type: u32, file: i32, offset: u64, weight_bytes: usize, rows: usize, columns: usize, optimized_layout: u32) -> *mut std::ffi::c_void;
    fn zllm_huawei_quant_weight_prepare_optimized(handle: *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_quant_weight_retain(handle: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn zllm_huawei_quant_weight_release(handle: *mut std::ffi::c_void);
    fn zllm_huawei_select_row_tensor(input: *const std::ffi::c_void, rows: usize, columns: usize, row: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_quant_matmul_resident_tensor(tensor_type: u32, weight: *const std::ffi::c_void, input: *mut std::ffi::c_void, rows: usize, columns: usize, batch: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_ffn_mixed_resident_tensor(gate: *const std::ffi::c_void, up: *const std::ffi::c_void, down: *const std::ffi::c_void, input: *const std::ffi::c_void, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_q6_vocab_matmul_resident_tensor(weight: *const std::ffi::c_void, input: *mut std::ffi::c_void, rows: usize, columns: usize, batch: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_q6_argmax_embedding_resident_tensor(logits: *const std::ffi::c_void, embedding: *const std::ffi::c_void, excluded: *const u32, excluded_count: usize, token: *mut u32, hidden: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_quant_cache_clear();
    fn zllm_huawei_quant_io_counters(read_calls: *mut u64, read_bytes: *mut u64) -> i32;
    fn zllm_huawei_quant_profile_dump(output: *mut std::os::raw::c_char, capacity: usize) -> i32;
    fn zllm_huawei_argmax_tensor(input: *const std::ffi::c_void, excluded: *const u32, excluded_count: usize, token: *mut u32) -> i32;
    fn zllm_huawei_f32_matmul_tensor(weight: *const std::ffi::c_void, input: *const std::ffi::c_void, rows: usize, columns: usize, batch: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_gdn_chunk_gram_tensor(keys: *const std::ffi::c_void, tasks: usize, columns: usize, output: *mut *mut std::ffi::c_void) -> i32;
    #[allow(dead_code)]
    fn zllm_huawei_tensor_to_f32(handle: *const std::ffi::c_void, output: *mut f32, elements: usize) -> i32;
    fn zllm_huawei_rmsnorm_tensor(input: *const std::ffi::c_void, gamma: *const std::ffi::c_void, batch: usize, columns: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_add_tensor(left: *const std::ffi::c_void, right: *const std::ffi::c_void, batch: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_split_interleaved_tensor(input: *const std::ffi::c_void, batch: usize, columns: usize, block_columns: usize, left: *mut *mut std::ffi::c_void, right: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_gemma_rmsnorm_heads_tensor(input: *const std::ffi::c_void, gamma: *const std::ffi::c_void, batch: usize, columns: usize, head_count: usize, head_dim: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_rope_table_prepare(cosine: *const f32, sine: *const f32, rows: usize) -> i32;
    fn zllm_huawei_rope_prefix_tensor(
        input: *const std::ffi::c_void,
        batch: usize,
        columns: usize,
        head_count: usize,
        rotary_dim: usize,
        position: usize,
        cosine: *const f32,
        sine: *const f32,
        table_rows: usize,
        output: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn zllm_huawei_gqa_attention_cached_tensor(
        query: *const std::ffi::c_void,
        key: *const std::ffi::c_void,
        value: *const std::ffi::c_void,
        key_cache: *const std::ffi::c_void,
        value_cache: *const std::ffi::c_void,
        score_scratch: *const std::ffi::c_void,
        position: usize,
        batch: usize,
        output: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn zllm_huawei_sigmoid_gate_tensor(input: *const std::ffi::c_void, gate: *const std::ffi::c_void, batch: usize, columns: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_gated_activation_tensor(gate: *const std::ffi::c_void, up: *const std::ffi::c_void, batch: usize, columns: usize, output: *mut *mut std::ffi::c_void) -> i32;
    fn zllm_huawei_gated_delta_net_tensor(
        qkv: *const std::ffi::c_void,
        z: *const std::ffi::c_void,
        alpha: *const std::ffi::c_void,
        beta: *const std::ffi::c_void,
        conv_weight: *const std::ffi::c_void,
        a_log: *const std::ffi::c_void,
        dt_bias: *const std::ffi::c_void,
        norm: *const std::ffi::c_void,
        conv_state: *const std::ffi::c_void,
        recurrent: *const std::ffi::c_void,
        batch: usize,
        output: *mut *mut std::ffi::c_void,
    ) -> i32;
}

impl BackendResources for HuaweiContext {
    type Tensor = HuaweiTensor;
    type Weight = HuaweiWeight;
    #[cfg(target_env = "ohos")]
    type Cache = HuaweiKvCache;
    #[cfg(not(target_env = "ohos"))]
    type Cache = CpuKvCache;
    type LayerScope<'a>
        = ()
    where
        Self: 'a;

    fn layer_scope(&self) -> Self::LayerScope<'_> {}

    fn token_rows(&self, tensor: &Self::Tensor) -> usize {
        tensor.rows
    }

    fn token_cols(&self, tensor: &Self::Tensor) -> usize {
        tensor.cols
    }

    fn tensor_allocated_bytes(&self, tensor: &Self::Tensor) -> u64 {
        #[cfg(target_env = "ohos")]
        {
            tensor.rows.saturating_mul(tensor.cols).saturating_mul(std::mem::size_of::<f32>()) as u64
        }
        #[cfg(not(target_env = "ohos"))]
        {
            tensor.data.capacity().saturating_mul(std::mem::size_of::<f32>()) as u64
        }
    }

    fn begin_batch(&self) {}

    fn finish_batch(&self) {}

    fn prepare_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            let supported = match weight {
                LinearWeight::F32(_) => true,
                LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::W8A16(_)) => true,
                LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(matrix)) => {
                    matches!(matrix.tensor_type.0, 13 | 14 | 8)
                }
                _ => false,
            };
            if !supported {
                return Err(compute(format!("Huawei NPU 权重格式未实现 rows={rows} cols={cols}；禁止 CPU weight fallback")));
            }
        }
        #[cfg(target_env = "ohos")]
        if let LinearWeight::F32(values) = weight {
            let expected = rows.checked_mul(cols).ok_or_else(|| compute("Huawei NPU F32 weight shape 溢出"))?;
            if values.len() != expected {
                return Err(compute(format!("Huawei NPU F32 weight 元素数 {}，期望 [{rows},{cols}]", values.len())));
            }
            let fp16_device = unsafe { zllm_huawei_fp16_weight_from_f32(values.as_ptr(), rows, cols, 0) };
            if fp16_device.is_null() {
                return Err(compute(format!("Huawei NPU F32 weight UMA resident 失败 shape=[{rows},{cols}] status={}", unsafe { zllm_huawei_tensor_last_status() })));
            }
            let source = self.cpu().prepare_weight(weight, rows, cols).map_err(|error| {
                unsafe { zllm_huawei_fp16_weight_release(fp16_device) };
                error
            })?;
            return Ok(HuaweiWeight { source, fp16_device: Some(fp16_device), quant_device: None });
        }
        #[cfg(target_env = "ohos")]
        if let LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::W8A16(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute(format!("Huawei W8A16 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            // HIAI 暂无标准 W8A16 图，统一 LM-head 开关在真机上先显式解码并上传
            // FP16 resident；不允许执行期偷偷落回 CPU。
            let values = matrix.decode().map_err(compute)?;
            return self.prepare_weight(LinearWeight::F32(&values), rows, cols);
        }
        #[cfg(target_env = "ohos")]
        if let LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(matrix)) = weight {
            // 已验证形状的 dense FFN down 同时保留 raw prefill 布局与 compact decode
            // 布局；二者都只在模型装配阶段建立，decode 不再访问 GGUF。
            return self.prepare_quantized_weight(weight, rows, cols, compact_ffn_down_layout(matrix.tensor_type.0, rows, cols));
        }
        self.cpu().prepare_weight(weight, rows, cols).map(|source| HuaweiWeight {
            source,
            #[cfg(target_env = "ohos")]
            fp16_device: None,
            #[cfg(target_env = "ohos")]
            quant_device: None,
        })
    }

    fn prepare_weight_pair(&self, first: LinearWeight<'_>, second: LinearWeight<'_>, rows: usize, cols: usize) -> Result<(Self::Weight, Self::Weight), BackendError> {
        #[cfg(target_env = "ohos")]
        if let (LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(first_matrix)), LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(second_matrix))) = (first, second)
            && compact_ffn_gate_up_pair(first_matrix.tensor_type.0, second_matrix.tensor_type.0, rows, cols)
        {
            return Ok((self.prepare_quantized_weight(first, rows, cols, COMPACT_FFN_GATE_LAYOUT)?, self.prepare_quantized_weight(second, rows, cols, COMPACT_FFN_UP_LAYOUT)?));
        }
        Ok((self.prepare_weight(first, rows, cols)?, self.prepare_weight(second, rows, cols)?))
    }

    fn prepare_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            let expected = rows.checked_mul(cols).ok_or_else(|| compute("Huawei NPU F32 weight shape 溢出"))?;
            if values.len() != expected {
                return Err(compute(format!("Huawei NPU F32 weight 元素数 {}，期望 [{rows},{cols}]", values.len())));
            }
            let fp16_device = unsafe { zllm_huawei_fp16_weight_from_f32(values.as_ptr(), rows, cols, 0) };
            if fp16_device.is_null() {
                return Err(compute(format!("Huawei NPU F32 weight UMA resident 失败 shape=[{rows},{cols}] status={}", unsafe { zllm_huawei_tensor_last_status() })));
            }
            let source = self.cpu().prepare_f32(values, rows, cols).map_err(|error| {
                unsafe { zllm_huawei_fp16_weight_release(fp16_device) };
                error
            })?;
            return Ok(HuaweiWeight { source, fp16_device: Some(fp16_device), quant_device: None });
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().prepare_f32(values, rows, cols).map(|source| HuaweiWeight { source })
    }

    fn prepare_gemma_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            let expected = rows.checked_mul(cols).ok_or_else(|| compute("Huawei NPU GemmaRMSNorm weight shape 溢出"))?;
            if values.len() != expected {
                return Err(compute(format!("Huawei NPU GemmaRMSNorm weight 元素数 {}，期望 [{rows},{cols}]", values.len())));
            }
            // GGUF 已在 runtime 层还原为零中心 gamma；HIAI RmsNorm 需要实际缩放值，
            // 因此只在 device copy 上加一，source 仍保留语义值供 host oracle 使用。
            let fp16_device = unsafe { zllm_huawei_fp16_weight_from_f32(values.as_ptr(), rows, cols, 1) };
            if fp16_device.is_null() {
                return Err(compute(format!("Huawei NPU GemmaRMSNorm weight UMA resident 失败 shape=[{rows},{cols}] status={}", unsafe { zllm_huawei_tensor_last_status() })));
            }
            let source = self.cpu().prepare_f32(values, rows, cols).map_err(|error| {
                unsafe { zllm_huawei_fp16_weight_release(fp16_device) };
                error
            })?;
            return Ok(HuaweiWeight { source, fp16_device: Some(fp16_device), quant_device: None });
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().prepare_gemma_f32(values, rows, cols).map(|source| HuaweiWeight { source })
    }
}

#[allow(unused_variables)]
impl Backend for HuaweiContext {
    fn linear(&self, input: &Self::Tensor, weight: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        if weight.source.gguf.is_none() {
            return self.f32_linear(input, weight);
        }
        if input.rows > 0
            && let Some(output) = self.quantized_linear(input, weight)?
        {
            return Ok(output);
        }
        huawei_cpu_or_npu_error!(self, "linear", self.cpu().linear(input, &weight.source))
    }

    fn gated_mlp(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Weight, down: &Self::Weight, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        if compact_ffn_decode_batch(input.rows, input.cols, gate.rows(), gate.cols(), up.rows(), up.cols(), down.rows(), down.cols()) && matches!(activation, Activation::Silu) {
            let gate_device = gate.quant_device.ok_or_else(|| compute("Huawei NPU FFN gate resident 权重缺失"))?;
            let up_device = up.quant_device.ok_or_else(|| compute("Huawei NPU FFN up resident 权重缺失"))?;
            let down_device = down.quant_device.ok_or_else(|| compute("Huawei NPU FFN down resident 权重缺失"))?;
            let mut output = std::ptr::null_mut();
            let status = unsafe { zllm_huawei_ffn_mixed_resident_tensor(gate_device, up_device, down_device, input.handle, &mut output) };
            if status != 0 || output.is_null() {
                return Err(compute(format!("Huawei NPU resident mixed FFN 执行失败 status={status}")));
            }
            return Ok(HuaweiTensor { handle: output, rows: 1, cols: 2560 });
        }
        let activated = self.gated_linear(input, gate, up, activation)?;
        self.linear(&activated, down)
    }

    fn rmsnorm(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            return self.rmsnorm_tensor(input, weight, eps, "rmsnorm");
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().rmsnorm(input, &weight.source, eps)
    }

    fn gemma_rmsnorm(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            return self.rmsnorm_tensor(input, weight, eps, "gemma_rmsnorm");
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().gemma_rmsnorm(input, &weight.source, eps)
    }

    fn layernorm_bias(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "layernorm_bias", self.cpu().layernorm_bias(input, &weight.source, &bias.source, eps))
    }

    fn split_columns(&self, input: &Self::Tensor, left_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        huawei_cpu_or_npu_error!(self, "split_columns", self.cpu().split_columns(input, left_columns))
    }

    fn split_interleaved_columns(&self, input: &Self::Tensor, block_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        #[cfg(target_env = "ohos")]
        {
            if !matches!(input.rows, 1 | 4 | 15 | 1024) || input.cols != 8192 || block_columns != 256 {
                return Err(compute(format!("Huawei NPU split_interleaved_columns 当前 OMC 只支持 batch=1/4/15/1024、input=8192、block=256，实际 input=[{},{}] block={block_columns}", input.rows, input.cols)));
            }
            let mut left = std::ptr::null_mut();
            let mut right = std::ptr::null_mut();
            let status = unsafe { zllm_huawei_split_interleaved_tensor(input.handle, input.rows, input.cols, block_columns, &mut left, &mut right) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU split_interleaved_columns 执行失败 input=[{},{}] block={block_columns} status={status}；禁止 CPU fallback", input.rows, input.cols)));
            }
            if left.is_null() || right.is_null() {
                if !left.is_null() {
                    unsafe { zllm_huawei_tensor_release(left) };
                }
                if !right.is_null() {
                    unsafe { zllm_huawei_tensor_release(right) };
                }
                return Err(compute("Huawei NPU split_interleaved_columns 返回空 tensor"));
            }
            let columns = input.cols / 2;
            Ok((HuaweiTensor { handle: left, rows: input.rows, cols: columns }, HuaweiTensor { handle: right, rows: input.rows, cols: columns }))
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().split_interleaved_columns(input, block_columns)
    }

    fn concat_columns(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "concat_columns", self.cpu().concat_columns(left, right))
    }

    fn rope(&self, input: &Self::Tensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "rope", self.cpu().rope(input, head_count, rotary_dim, layout, position, cos, sin))
    }

    fn rope_prefix(&self, input: &Self::Tensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            if layout != crate::attention::rope::RotaryLayout::SplitHalf || !matches!(input.rows, 1 | 4 | 15 | 1024) || !matches!(input.cols, 1024 | 4096) || head_count.checked_mul(256) != Some(input.cols) || rotary_dim != 64 {
                return Err(compute(format!("Huawei NPU rope_prefix shape/参数不支持 input=[{},{}] heads={} rotary={} layout={layout:?}", input.rows, input.cols, head_count, rotary_dim,)));
            }
            let half = rotary_dim / 2;
            let start = position.checked_mul(half).ok_or_else(|| compute("Huawei NPU RoPE table offset 溢出"))?;
            let end = position.checked_add(input.rows).and_then(|value| value.checked_mul(half)).ok_or_else(|| compute("Huawei NPU RoPE table 范围溢出"))?;
            if cos.len() != sin.len() || end > cos.len() {
                return Err(compute(format!("Huawei NPU RoPE table 长度不足 cos={} sin={} range={start}..{end}", cos.len(), sin.len(),)));
            }
            let mut output = std::ptr::null_mut();
            let status = unsafe { zllm_huawei_rope_prefix_tensor(input.handle, input.rows, input.cols, head_count, rotary_dim, position, cos.as_ptr(), sin.as_ptr(), cos.len() / half, &mut output) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU rope_prefix 执行失败 input=[{},{}] position={} status={status}；禁止 CPU fallback", input.rows, input.cols, position,)));
            }
            let Some(handle) = (!output.is_null()).then_some(output) else {
                return Err(compute("Huawei NPU rope_prefix 返回空 tensor"));
            };
            Ok(HuaweiTensor { handle, rows: input.rows, cols: input.cols })
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().rope_prefix(input, head_count, rotary_dim, layout, position, cos, sin)
    }

    fn add(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            if left.rows != right.rows || left.cols != right.cols {
                return Err(compute(format!("Huawei NPU add shape 不一致 left=[{},{}] right=[{},{}]", left.rows, left.cols, right.rows, right.cols)));
            }
            if !matches!(left.rows, 1 | 4 | 15 | 1024) || left.cols != 2560 {
                return Err(compute(format!("Huawei NPU add 当前 OMC 只支持 batch=1/4/15/1024、hidden=2560，实际 [{},{}]", left.rows, left.cols)));
            }
            let mut output = std::ptr::null_mut();
            let status = unsafe { zllm_huawei_add_tensor(left.handle, right.handle, left.rows, &mut output) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU add 执行失败 shape=[{},{}] status={status}；禁止 CPU fallback", left.rows, left.cols)));
            }
            let Some(handle) = (!output.is_null()).then_some(output) else {
                return Err(compute("Huawei NPU add 返回空 tensor"));
            };
            Ok(HuaweiTensor { handle, rows: left.rows, cols: left.cols })
        }
        #[cfg(not(target_env = "ohos"))]
        {
            self.cpu().add(left, right)
        }
    }

    fn add_scaled(&self, left: &Self::Tensor, right: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "add_scaled", self.cpu().add_scaled(left, right, scale))
    }

    fn sigmoid_gate(&self, input: &Self::Tensor, gate: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            if input.rows != gate.rows || input.cols != gate.cols {
                return Err(compute(format!("Huawei NPU sigmoid_gate shape 不一致 input=[{},{}] gate=[{},{}]", input.rows, input.cols, gate.rows, gate.cols)));
            }
            if !matches!(input.rows, 1 | 4 | 15 | 1024) || input.cols != 4096 {
                return Err(compute(format!("Huawei NPU sigmoid_gate 当前 OMC 只支持 batch=1/4/15/1024、columns=4096，实际 [{},{}]", input.rows, input.cols)));
            }
            // kernel 把 elements 均分到各 AIV core 后按 128 元素 tile 处理；
            // elements 至少要是 128 的整数倍（cores 因子由固定 shape 的 OMC 保证）。
            let elements = input.rows * input.cols;
            if elements % 128 != 0 {
                return Err(compute(format!("Huawei NPU sigmoid_gate 元素数 {elements} 不是 128 的整数倍，kernel 按 128 元素分片 shape=[{},{}]", input.rows, input.cols)));
            }
            let mut output = std::ptr::null_mut();
            let status = unsafe { zllm_huawei_sigmoid_gate_tensor(input.handle, gate.handle, input.rows, input.cols, &mut output) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU sigmoid_gate 执行失败 shape=[{},{}] status={status}；禁止 CPU fallback", input.rows, input.cols)));
            }
            let Some(handle) = (!output.is_null()).then_some(output) else {
                return Err(compute("Huawei NPU sigmoid_gate 返回空 tensor"));
            };
            Ok(HuaweiTensor { handle, rows: input.rows, cols: input.cols })
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().sigmoid_gate(input, gate)
    }

    fn select_row(&self, input: &Self::Tensor, row: usize) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            return input.select_row_npu(row);
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().select_row(input, row)
    }

    fn select_rows(&self, input: &Self::Tensor, rows: &[u32]) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "select_rows", self.cpu().select_rows(input, rows))
    }

    fn argmax(&self, input: &Self::Tensor) -> Result<u32, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            // NPU FFI 只有"排除 2 个 token"一种形态：无排除请求时传两个 u32::MAX 哑值占位，内核视其为永不命中的哨兵。
            return self.argmax_excluding(input, &[u32::MAX, u32::MAX]);
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().argmax(input)
    }

    fn argmax_excluding(&self, input: &Self::Tensor, excluded: &[u32]) -> Result<u32, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            if input.rows != 1 || input.cols != 248_320 || excluded.len() != 2 || excluded.iter().any(|&token| token != u32::MAX && token as usize >= input.cols) {
                return Err(compute(format!("Huawei NPU argmax shape/排除项不支持 input=[{},{}] excluded={excluded:?}；FFI 只接受恰好 2 个排除槽，无排除时用 u32::MAX 占位", input.rows, input.cols)));
            }
            let mut token = 0_u32;
            let status = unsafe { zllm_huawei_argmax_tensor(input.handle, excluded.as_ptr(), excluded.len(), &mut token) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU argmax 执行失败 input=[{},{}] excluded={excluded:?} status={status}；禁止 CPU fallback", input.rows, input.cols)));
            }
            Ok(token)
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().argmax_excluding(input, excluded)
    }

    fn sample_top_p(&self, input: &Self::Tensor, temperature: f32, top_p: f32, random: f32) -> Result<u32, BackendError> {
        huawei_cpu_or_npu_error!(self, "sample_top_p", self.cpu().sample_top_p(input, temperature, top_p, random))
    }

    fn gated_activation(&self, gate: &Self::Tensor, up: &Self::Tensor, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            if !matches!(activation, Activation::Silu) {
                return Err(compute(format!("Huawei NPU gated_activation 当前只支持 SiLU，实际 {activation:?}")));
            }
            if gate.rows != up.rows || gate.cols != up.cols {
                return Err(compute(format!("Huawei NPU gated_activation shape 不一致 gate=[{},{}] up=[{},{}]", gate.rows, gate.cols, up.rows, up.cols)));
            }
            if !matches!(gate.rows, 1 | 4 | 15 | 1024) || gate.cols != 9216 {
                return Err(compute(format!("Huawei NPU gated_activation 当前 OMC 只支持 batch=1/4/15/1024、columns=9216，实际 [{},{}]", gate.rows, gate.cols)));
            }
            // kernel 把 elements 均分到各 AIV core 后按 128 元素 tile 处理；
            // elements 至少要是 128 的整数倍（cores 因子由固定 shape 的 OMC 保证）。
            let elements = gate.rows * gate.cols;
            if elements % 128 != 0 {
                return Err(compute(format!("Huawei NPU gated_activation 元素数 {elements} 不是 128 的整数倍，kernel 按 128 元素分片 shape=[{},{}]", gate.rows, gate.cols)));
            }
            let mut output = std::ptr::null_mut();
            let status = unsafe { zllm_huawei_gated_activation_tensor(gate.handle, up.handle, gate.rows, gate.cols, &mut output) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU gated_activation 执行失败 shape=[{},{}] status={status}；禁止 CPU fallback", gate.rows, gate.cols)));
            }
            let Some(handle) = (!output.is_null()).then_some(output) else {
                return Err(compute("Huawei NPU gated_activation 返回空 tensor"));
            };
            Ok(HuaweiTensor { handle, rows: gate.rows, cols: gate.cols })
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().gated_activation(gate, up, activation)
    }
}

#[allow(unused_variables)]
impl GqaPrefillBackend for HuaweiContext {
    fn gemma_rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            let gamma = weight.device_view()?;
            if !matches!(input.rows, 1 | 4 | 15 | 1024) || !matches!(input.cols, 1024 | 4096) || head_dim != 256 || head_count.checked_mul(head_dim) != Some(input.cols) || weight.cols() != head_dim || (eps - 1.0e-6).abs() > f32::EPSILON {
                return Err(compute(format!("Huawei NPU gemma_rmsnorm_heads shape/参数不支持 input=[{},{}] gamma=[{},{}] heads={}x{} eps={eps}", input.rows, input.cols, weight.rows(), weight.cols(), head_count, head_dim,)));
            }
            let mut output = std::ptr::null_mut();
            let status = unsafe { zllm_huawei_gemma_rmsnorm_heads_tensor(input.handle, gamma.handle, input.rows, input.cols, head_count, head_dim, &mut output) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU gemma_rmsnorm_heads 执行失败 input=[{},{}] heads={}x{} status={status}；禁止 CPU fallback", input.rows, input.cols, head_count, head_dim,)));
            }
            let Some(handle) = (!output.is_null()).then_some(output) else {
                return Err(compute("Huawei NPU gemma_rmsnorm_heads 返回空 tensor"));
            };
            Ok(HuaweiTensor { handle, rows: input.rows, cols: input.cols })
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().gemma_rmsnorm_heads(input, &weight.source, head_count, head_dim, eps)
    }

    fn gqa_prefill_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, spec: &GqaSpec) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "gqa_prefill_attention", self.cpu().gqa_prefill_attention(query, key, value, spec))
    }

    fn gqa_prefill_attention_cached(
        &self,
        cache: &mut Self::Cache,
        layer: usize,
        position: usize,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        spec: &GqaSpec,
        retain_full_cache: bool,
    ) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            if query.rows != key.rows
                || query.rows != value.rows
                || !matches!(query.rows, 1 | 4 | 15 | 1024)
                || query.cols != 4096
                || key.cols != 1024
                || value.cols != 1024
                || spec.num_heads != 16
                || spec.num_kv_heads != 4
                || spec.head_dim != 256
                || !matches!(spec.window, crate::attention::gqa::CausalWindow::Full)
                || retain_full_cache
            {
                return Err(compute(format!(
                    "Huawei NPU cached GQA 参数不支持 L{layer} position={position} Q=[{},{}] K=[{},{}] V=[{},{}]；期望 batch=1/4/15/1024、Q=[batch,4096]、K/V=[batch,1024]、16 query 头/4 KV 头/head_dim=256、全量 causal（Qwen3.5-4B 硬编码形状）",
                    query.rows, query.cols, key.rows, key.cols, value.rows, value.cols,
                )));
            }
            let cached = cache.layer(layer, position, query.rows)?;
            let mut output = std::ptr::null_mut();
            let status = unsafe { zllm_huawei_gqa_attention_cached_tensor(query.handle, key.handle, value.handle, cached.key, cached.value, cached.score_scratch, position, query.rows, &mut output) };
            if status != 0 {
                return Err(compute(format!("Huawei NPU cached GQA 执行失败 L{layer} position={position} batch={} status={status}；禁止 CPU fallback", query.rows,)));
            }
            cached.rows = position + query.rows;
            let Some(handle) = (!output.is_null()).then_some(output) else {
                return Err(compute("Huawei NPU cached GQA 返回空 tensor"));
            };
            Ok(HuaweiTensor { handle, rows: query.rows, cols: 4096 })
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().gqa_prefill_attention_cached(cache, layer, position, query, key, value, spec, retain_full_cache)
    }

    fn gqa_prefill_attention_cached_visible(
        &self,
        cache: &mut Self::Cache,
        layer: usize,
        position: usize,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        spec: &GqaSpec,
        visible_ends: &[u32],
        retain_full_cache: bool,
    ) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "gqa_prefill_attention_cached_visible", self.cpu().gqa_prefill_attention_cached_visible(cache, layer, position, query, key, value, spec, visible_ends, retain_full_cache))
    }

    fn gqa_prefill_attention_cached_from(&self, cache: &Self::Cache, source_layer: usize, position: usize, query: &Self::Tensor, spec: &GqaSpec) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "gqa_prefill_attention_cached_from", self.cpu().gqa_prefill_attention_cached_from(cache, source_layer, position, query, spec))
    }

    fn gqa_prefill_attention_cached_from_visible(&self, cache: &Self::Cache, source_layer: usize, position: usize, query: &Self::Tensor, spec: &GqaSpec, visible_ends: &[u32]) -> Result<Self::Tensor, BackendError> {
        huawei_cpu_or_npu_error!(self, "gqa_prefill_attention_cached_from_visible", self.cpu().gqa_prefill_attention_cached_from_visible(cache, source_layer, position, query, spec, visible_ends))
    }
}

#[allow(unused_variables)]
impl GatedDeltaNetKernel for HuaweiContext {
    type GatedDeltaNetStorage = HuaweiGatedDeltaNetStorage;

    fn allocate_gated_delta_net_storage(&self, spec: &GatedDeltaNetSpec) -> Result<Self::GatedDeltaNetStorage, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            let conv_elements = spec.conv_state_elements();
            let recurrent_elements = spec.recurrent_elements();
            let conv = unsafe { zllm_huawei_tensor_zeros(conv_elements) };
            if conv.is_null() {
                return Err(compute(format!("Huawei NPU DeltaNet conv state 分配失败 elements={conv_elements}")));
            }
            let recurrent = unsafe { zllm_huawei_tensor_zeros(recurrent_elements) };
            if recurrent.is_null() {
                unsafe { zllm_huawei_tensor_release(conv) };
                return Err(compute(format!("Huawei NPU DeltaNet recurrent state 分配失败 elements={recurrent_elements}")));
            }
            return Ok(HuaweiGatedDeltaNetStorage { conv, recurrent, allocated_bytes: (conv_elements + recurrent_elements) * std::mem::size_of::<half::f16>() });
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().allocate_gated_delta_net_storage(spec)
    }

    fn gated_delta_net_fused(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, Self::Tensor>,
        weights: GatedDeltaNetWeightsRef<'_, Self::Weight>,
        spec: &GatedDeltaNetSpec,
    ) -> Result<Self::Tensor, BackendError> {
        #[cfg(target_env = "ohos")]
        {
            let batch = inputs.qkv.rows;
            if !matches!(batch, 1 | 4 | 15 | 1024)
                || inputs.qkv.cols != spec.conv_dim()
                || inputs.z.rows != batch
                || inputs.z.cols != spec.value_dim()
                || inputs.alpha.rows != batch
                || inputs.alpha.cols != spec.value_heads
                || inputs.beta.rows != batch
                || inputs.beta.cols != spec.value_heads
                || spec.key_heads != 16
                || spec.value_heads != 32
                || spec.key_head_dim != 128
                || spec.value_head_dim != 128
                || spec.conv_kernel != 4
                || spec.rms_eps != 1.0e-6
            {
                return Err(compute(format!(
                    "Huawei NPU gated_delta_net_fused shape/spec 不支持 qkv=[{},{}] z=[{},{}] alpha=[{},{}] beta=[{},{}] spec={spec:?}；当前 OMC 仅支持 batch=1/4/15/1024 Qwen3.5-4B",
                    inputs.qkv.rows, inputs.qkv.cols, inputs.z.rows, inputs.z.cols, inputs.alpha.rows, inputs.alpha.cols, inputs.beta.rows, inputs.beta.cols,
                )));
            }
            if (weights.conv.rows(), weights.conv.cols()) != (spec.conv_dim(), spec.conv_kernel)
                || (weights.a_log.rows(), weights.a_log.cols()) != (1, spec.value_heads)
                || (weights.dt_bias.rows(), weights.dt_bias.cols()) != (1, spec.value_heads)
                || (weights.norm.rows(), weights.norm.cols()) != (1, spec.value_head_dim)
            {
                return Err(compute("Huawei NPU gated_delta_net_fused weight shape 与 spec 不一致"));
            }
            let conv_weight = weights.conv.device_view()?;
            let a_log = weights.a_log.device_view()?;
            let dt_bias = weights.dt_bias.device_view()?;
            let norm = weights.norm.device_view()?;
            let mut output = std::ptr::null_mut();
            let status = unsafe {
                zllm_huawei_gated_delta_net_tensor(
                    inputs.qkv.handle,
                    inputs.z.handle,
                    inputs.alpha.handle,
                    inputs.beta.handle,
                    conv_weight.handle,
                    a_log.handle,
                    dt_bias.handle,
                    norm.handle,
                    storage.conv,
                    storage.recurrent,
                    batch,
                    &mut output,
                )
            };
            if status != 0 {
                return Err(compute(format!("Huawei NPU gated_delta_net_fused 执行失败 status={status}；禁止 CPU fallback")));
            }
            let Some(handle) = (!output.is_null()).then_some(output) else {
                return Err(compute("Huawei NPU gated_delta_net_fused 返回空 tensor"));
            };
            Ok(HuaweiTensor { handle, rows: batch, cols: spec.value_dim() })
        }
        #[cfg(not(target_env = "ohos"))]
        self.cpu().gated_delta_net_fused(storage, inputs, GatedDeltaNetWeightsRef { conv: &weights.conv.source, a_log: &weights.a_log.source, dt_bias: &weights.dt_bias.source, norm: &weights.norm.source }, spec)
    }
}

#[cfg(test)]
mod tests {

    /// 字段对齐 cann_probe.cpp 的 decoded 权重缓存：字节预算 + coldest(最小
    /// stamp) 淘汰 + 仅 decode 插入。C++ 侧依赖 hiai 无法在主机编译，这里用
    /// 同一份记账逻辑做 oracle，保证预算不变式与循环访问不抖动。
    #[derive(Default)]
    struct DecodeWindowCache {
        map: std::collections::HashMap<u64, (u64, u64)>, // key -> (stamp, bytes)
        bytes: u64,
        stamp: u64,
        budget: u64,
    }

    impl DecodeWindowCache {
        fn with_budget(budget: u64) -> Self {
            Self { budget, ..Self::default() }
        }

        /// 返回 true 表示命中（跳过 scalar decode）。
        fn access(&mut self, key: u64, bytes: u64, fused_pack: bool) -> bool {
            if !fused_pack && let Some(entry) = self.map.get_mut(&key) {
                self.stamp += 1;
                entry.0 = self.stamp;
                return true;
            }
            if !fused_pack {
                while self.bytes + bytes > self.budget && !self.map.is_empty() {
                    let victim = *self.map.iter().min_by_key(|(_, v)| v.0).map(|(k, _)| k).unwrap();
                    self.bytes -= self.map.remove(&victim).unwrap().1;
                }
                if self.bytes + bytes <= self.budget {
                    self.stamp += 1;
                    self.map.insert(key, (self.stamp, bytes));
                    self.bytes += bytes;
                }
            }
            false
        }
    }

    #[test]
    fn decode_cache_respects_byte_budget() {
        let mut cache = DecodeWindowCache::with_budget(100);
        for key in 0..10 {
            cache.access(key, 20, false);
            assert!(cache.bytes <= 100, "总字节 {} 超出预算", cache.bytes);
        }
    }

    #[test]
    fn decode_cache_cyclic_access_does_not_thrash_when_budget_fits() {
        // 32 层每层 7 个矩阵 = 224 个 key，预算刚好装下 → 预热后命中率 100%。
        let keys = 224_u64;
        let mut cache = DecodeWindowCache::with_budget(keys * 16);
        for token in 0..4 {
            let mut hits = 0;
            for key in 0..keys {
                if cache.access(key, 16, false) {
                    hits += 1;
                }
            }
            if token == 0 {
                assert_eq!(hits, 0, "首个 token 全部 miss（冷启动）");
            } else {
                assert_eq!(hits, keys, "token {token} 预热后应全部命中，实际命中 {hits}");
            }
        }
    }

    #[test]
    fn decode_cache_evicts_coldest_stamp_under_pressure() {
        // 预算只装 4 项；访问 5 个 key 应淘汰最早访问的 key 0。
        let mut cache = DecodeWindowCache::with_budget(4 * 16);
        for key in 0..5 {
            cache.access(key, 16, false);
        }
        assert!(!cache.map.contains_key(&0), "应淘汰最小 stamp 的 key 0");
        for key in 1..5 {
            assert!(cache.map.contains_key(&key), "key {key} 应仍在缓存");
        }
    }

    #[test]
    fn prefill_never_populates_decode_cache() {
        let mut cache = DecodeWindowCache::with_budget(1 << 20);
        for key in 0..8 {
            assert!(!cache.access(key, 16, true), "prefill(fused_pack) 不应命中");
        }
        assert!(cache.map.is_empty(), "prefill 不应写入 decode 缓存");
        assert_eq!(cache.bytes, 0);
    }
}
