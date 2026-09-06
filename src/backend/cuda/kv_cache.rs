//! GQA 的 CUDA device-resident KV cache。

use cudarc::driver::safe::CudaSlice;
use half::f16;

use crate::{
    attention::gqa::GqaSpec,
    backend::cuda::{CudaContext, CudaTensor, CudaWeight},
    backend::{BackendError, GqaPrefillBackend},
    kernel::cuda as ops,
};

/// Q8G64(symmetric INT8 + per-64 组 scale)存储,布局镜像 Metal 的 Ornith Q8G64。
pub struct CudaKvLayerQ8 {
    pub key_codes: CudaSlice<i8>,
    pub key_scales: CudaSlice<f32>,
    pub value_codes: CudaSlice<i8>,
    pub value_scales: CudaSlice<f32>,
}

pub struct CudaKvLayer {
    pub key: CudaSlice<f16>,
    pub value: CudaSlice<f16>,
    pub q8: Option<CudaKvLayerQ8>,
    pub rows: usize,
    pub columns: usize,
    /// 该层存储容量(行数):sliding-window 层 = 窗口大小,按 ring 寻址。
    capacity: usize,
}

pub struct CudaKvCache {
    layers: Vec<Option<CudaKvLayer>>,
    max_seq_len: usize,
    columns: usize,
    /// Q8G64(symmetric INT8 + per-64 组 scale)存储,布局镜像 Metal 的 Ornith Q8G64。
    q8g64: bool,
    /// 逐层存储容量;hybrid 模型由引擎按 window.cache_capacity 提供,
    /// 空 slice 或逐层等于 max_seq_len 即 dense 语义(兼容既有调用方)。
    capacities: Vec<usize>,
}

impl CudaKvCache {
    pub fn new(layer_count: usize, max_seq_len: usize, columns: usize) -> Self {
        Self { layers: (0..layer_count).map(|_| None).collect(), max_seq_len, columns, q8g64: false, capacities: Vec::new() }
    }

    /// 按逐层容量构造(ring KV)。容量小于 max_seq_len 的层即 sliding-window 层;
    /// Q8G64 与 ring 组合当前不支持,构造期拒绝。
    pub fn new_with_capacities(layer_count: usize, max_seq_len: usize, columns: usize, capacities: Vec<usize>) -> Result<Self, String> {
        if capacities.len() != layer_count {
            return Err(format!("CUDA KV capacities 长度 {} 不等于层数 {layer_count}", capacities.len()));
        }
        for (layer, &capacity) in capacities.iter().enumerate() {
            if capacity == 0 || capacity > max_seq_len {
                return Err(format!("CUDA KV layer={layer} capacity={capacity} 非法(max={max_seq_len})"));
            }
        }
        Ok(Self { layers: (0..layer_count).map(|_| None).collect(), max_seq_len, columns, q8g64: false, capacities })
    }

    pub fn new_q8g64(layer_count: usize, max_seq_len: usize, columns: usize) -> Result<Self, String> {
        if columns == 0 || columns % 64 != 0 {
            return Err(format!("Q8G64 KV columns={columns} 必须按 64 对齐"));
        }
        Ok(Self { layers: (0..layer_count).map(|_| None).collect(), max_seq_len, columns, q8g64: true, capacities: Vec::new() })
    }

    /// QSA 掩码注意力等直接消费方需要 KV 层句柄(码/scale 或 F16)。
    pub fn layer(&self, layer: usize) -> Result<&CudaKvLayer, crate::backend::BackendError> {
        self.layers.get(layer).ok_or(crate::backend::BackendError::UnsupportedLayer { layer })?.as_ref().ok_or(crate::backend::BackendError::Compute { msg: format!("CUDA KV layer {layer} 尚未初始化") })
    }

    pub fn format(&self) -> &'static str {
        if self.q8g64 { "q8g64" } else { "f16" }
    }

    /// 游标回退到 `rows` 行(推测解码 verify 拒绝部分 draft 时)。
    /// ring 语义下槽位不清理:被拒绝位置的槽会在重新 append 时覆写。
    pub fn truncate(&mut self, rows: usize) {
        for layer in self.layers.iter_mut().flatten() {
            layer.rows = layer.rows.min(rows);
        }
    }

    pub fn allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .zip(self.capacity_of_layer_iter())
            .filter_map(|(layer, capacity)| {
                layer.as_ref().map(|layer| if self.q8g64 { 2 * (capacity * layer.columns + capacity * (layer.columns / 64) * std::mem::size_of::<f32>()) } else { capacity * layer.columns * std::mem::size_of::<f16>() * 2 })
            })
            .sum()
    }

    /// 第 layer 层的存储容量;未提供逐层容量时回退 max_seq_len(dense)。
    fn layer_capacity(&self, layer: usize) -> usize {
        self.capacities.get(layer).copied().unwrap_or(self.max_seq_len)
    }

    fn capacity_of_layer_iter(&self) -> impl Iterator<Item = usize> {
        (0..self.layers.len()).map(move |layer| self.layer_capacity(layer))
    }

    pub fn append<'a>(&'a mut self, ctx: &CudaContext, layer: usize, position: usize, key: &CudaTensor, value: &CudaTensor) -> Result<&'a CudaKvLayer, BackendError> {
        if key.rows != value.rows || key.cols != value.cols || (self.columns != 0 && key.cols != self.columns) {
            return Err(BackendError::Compute { msg: format!("CUDA GQA KV shape 异常: key=[{},{}], value=[{},{}], columns={}", key.rows, key.cols, value.rows, value.cols, self.columns) });
        }
        if position + key.rows > self.max_seq_len {
            return Err(BackendError::Compute { msg: format!("CUDA GQA KV 超出 max_seq_len: position={position}, rows={}, max={}", key.rows, self.max_seq_len) });
        }
        // 先取容量(不可变借用),再进入 layers 的可变借用。
        let capacity = self.layer_capacity(layer);
        let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if slot.is_none() {
            let columns = key.cols;
            if self.q8g64 && capacity < self.max_seq_len {
                return Err(BackendError::Compute { msg: format!("CUDA Q8G64 KV layer={layer} 容量 {capacity} < max {}:Q8G64 尚不支持 ring 寻址", self.max_seq_len) });
            }
            let elements = capacity * columns;
            if self.q8g64 {
                if !columns.is_multiple_of(64) {
                    return Err(BackendError::Compute { msg: format!("CUDA Q8G64 KV columns={columns} 必须按 64 对齐") });
                }
                let scale_elements = capacity * (columns / 64);
                *slot = Some(CudaKvLayer {
                    key: ctx.buffer_uninit::<f16>(1).map_err(|e| BackendError::Compute { msg: format!("CUDA KV q8 占位分配失败: {e:?}") })?,
                    value: ctx.buffer_uninit::<f16>(1).map_err(|e| BackendError::Compute { msg: format!("CUDA KV q8 占位分配失败: {e:?}") })?,
                    q8: Some(CudaKvLayerQ8 {
                        key_codes: ctx.buffer_uninit::<i8>(elements).map_err(|e| BackendError::Compute { msg: format!("CUDA KV q8 key codes 分配失败: {e:?}") })?,
                        key_scales: ctx.buffer_uninit::<f32>(scale_elements).map_err(|e| BackendError::Compute { msg: format!("CUDA KV q8 key scales 分配失败: {e:?}") })?,
                        value_codes: ctx.buffer_uninit::<i8>(elements).map_err(|e| BackendError::Compute { msg: format!("CUDA KV q8 value codes 分配失败: {e:?}") })?,
                        value_scales: ctx.buffer_uninit::<f32>(scale_elements).map_err(|e| BackendError::Compute { msg: format!("CUDA KV q8 value scales 分配失败: {e:?}") })?,
                    }),
                    rows: 0,
                    columns,
                    capacity,
                });
            } else {
                // 分配失败时附带元素规模与设备空闲显存,区分"预算真不够"与"尺寸异常"。
                let describe = |error: String, name: &str| {
                    let (free, total) = ctx.mem_info().unwrap_or((0, 0));
                    BackendError::Compute { msg: format!("CUDA KV {name} 分配失败: {error}(elements={elements}, max_seq_len={}, columns={columns}, free={free}/total={total})", self.max_seq_len) }
                };
                // 清零分配:ring 容量内未写行的内容会进 softmax/加权重读,
                // 未初始化位模式含 NaN 时 0×NaN=NaN 污染整层输出(实测 L47)。
                let key = ctx.stream().alloc_zeros::<f16>(elements).map_err(|error| describe(format!("{error:?}"), "key"))?;
                let value = ctx.stream().alloc_zeros::<f16>(elements).map_err(|error| describe(format!("{error:?}"), "value"))?;
                *slot = Some(CudaKvLayer { key, value, q8: None, rows: 0, columns, capacity });
            }
        }
        let cache = slot.as_mut().expect("CUDA KV layer 已初始化");
        if cache.columns != key.cols {
            return Err(BackendError::Compute { msg: format!("CUDA GQA KV layer={layer} columns={}，输入={}", cache.columns, key.cols) });
        }
        if cache.rows != position {
            return Err(BackendError::Compute { msg: format!("CUDA GQA KV position 不连续: layer={layer}, cache={}, input={position}", cache.rows) });
        }
        if let Some(q8) = cache.q8.as_mut() {
            ops::attention::q8g64_quantize_rows(ctx, key, &mut q8.key_codes, &mut q8.key_scales, position).map_err(|msg| BackendError::Compute { msg })?;
            ops::attention::q8g64_quantize_rows(ctx, value, &mut q8.value_codes, &mut q8.value_scales, position).map_err(|msg| BackendError::Compute { msg })?;
        } else {
            ops::attention::append_rows_f16(ctx, key, &cache.key, position, cache.capacity).map_err(|msg| BackendError::Compute { msg })?;
            ops::attention::append_rows_f16(ctx, value, &cache.value, position, cache.capacity).map_err(|msg| BackendError::Compute { msg })?;
        }
        cache.rows += key.rows;
        Ok(cache)
    }
}

impl GqaPrefillBackend for CudaContext {
    fn gemma_rmsnorm_heads(&self, input: &CudaTensor, weight: &CudaWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, BackendError> {
        ops::attention::gemma_rmsnorm_heads_f16(self, input, &weight.data, head_count, head_dim, eps).map_err(|msg| BackendError::Compute { msg })
    }

    fn rmsnorm_heads(&self, input: &CudaTensor, weight: &CudaWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, BackendError> {
        ops::diffusion::rmsnorm_heads_f16(self, input, &weight.data, head_count, head_dim, eps).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention(&self, query: &CudaTensor, key: &CudaTensor, value: &CudaTensor, spec: &GqaSpec) -> Result<CudaTensor, BackendError> {
        // 非缓存路径持有恰好 kv_rows 行的临时 KV,容量即行数(dense)。
        ops::attention::gqa_attention_f16(self, query, &key.slice, &value.slice, key.rows, 0, spec, key.rows).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention_cached(&self, cache: &mut CudaKvCache, layer: usize, position: usize, query: &CudaTensor, key: &CudaTensor, value: &CudaTensor, spec: &GqaSpec, _retain_full_cache: bool) -> Result<CudaTensor, BackendError> {
        let kv_layer = cache.append(self, layer, position, key, value)?;
        if std::env::var_os("ZLLM_CUDA_NAN_AUDIT").is_some() && std::env::var("ZLLM_CUDA_AUDIT_LAYER").map(|value| value.parse::<usize>() == Ok(layer)).unwrap_or(false) {
            // 调试插桩:dump cache 首行/末行 K/V,区分 cache 写坏与 kernel 输出坏。
            let dump = |slice: &CudaSlice<f16>, row: usize| -> String {
                match self.stream().clone_dtoh(slice) {
                    Ok(data) => {
                        let start = row * kv_layer.columns;
                        data[start..start + 8].iter().map(|v| v.to_f32()).map(|v| if v.is_nan() { "NaN".to_owned() } else { format!("{v:.2}") }).collect::<Vec<_>>().join(",")
                    }
                    Err(error) => format!("err {error:?}"),
                }
            };
            eprintln!("[laguna-kv-audit] L{layer} rows={} K0=[{}] Klast=[{}] V0=[{}]", kv_layer.rows, dump(&kv_layer.key, 0), dump(&kv_layer.key, kv_layer.rows.saturating_sub(1)), dump(&kv_layer.value, 0));
        }
        if let Some(q8) = kv_layer.q8.as_ref() {
            // Q8G64:prefill 走 flash 变体(tile 加载时反量化进 smem),约束不满足回退。
            if query.rows > 1 && spec.head_dim.is_multiple_of(32) && spec.head_dim <= 256 && spec.num_heads.is_multiple_of(spec.num_kv_heads) {
                return ops::attention::gqa_attention_q8g64_flash(self, query, &q8.key_codes, &q8.key_scales, &q8.value_codes, &q8.value_scales, kv_layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg });
            }
            return ops::attention::gqa_attention_q8g64(self, query, &q8.key_codes, &q8.key_scales, &q8.value_codes, &q8.value_scales, kv_layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg });
        }
        // prefill(多行)走 flash 式 kernel:block=8 行×1 head 共享 smem K/V tile,
        // KV 全局重复读缩小 8 倍(548 token 实测 8.3 → ~1.5 ms/layer);约束不满足时回退。
        // ZLLM_CUDA_NO_FLASH:消元开关,强制 plain kernel。
        if query.rows > 1 && spec.head_dim.is_multiple_of(32) && spec.head_dim <= 256 && spec.num_heads.is_multiple_of(spec.num_kv_heads) && std::env::var_os("ZLLM_CUDA_NO_FLASH").is_none() {
            return ops::attention::gqa_attention_f16_flash(self, query, &kv_layer.key, &kv_layer.value, kv_layer.rows, position, spec, kv_layer.capacity).map_err(|msg| BackendError::Compute { msg });
        }
        ops::attention::gqa_attention_f16(self, query, &kv_layer.key, &kv_layer.value, kv_layer.rows, position, spec, kv_layer.capacity).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention_cached_from(&self, cache: &CudaKvCache, source_layer: usize, position: usize, query: &CudaTensor, spec: &GqaSpec) -> Result<CudaTensor, BackendError> {
        let layer =
            cache.layers.get(source_layer).ok_or(BackendError::UnsupportedLayer { layer: source_layer })?.as_ref().ok_or_else(|| BackendError::Compute { msg: format!("CUDA GQA shared KV source layer {source_layer} 尚未初始化") })?;
        if let Some(q8) = layer.q8.as_ref() {
            return ops::attention::gqa_attention_q8g64(self, query, &q8.key_codes, &q8.key_scales, &q8.value_codes, &q8.value_scales, layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg });
        }
        ops::attention::gqa_attention_f16(self, query, &layer.key, &layer.value, layer.rows, position, spec, layer.capacity).map_err(|msg| BackendError::Compute { msg })
    }
}
