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
struct CudaKvLayerQ8 {
    key_codes: CudaSlice<i8>,
    key_scales: CudaSlice<f32>,
    value_codes: CudaSlice<i8>,
    value_scales: CudaSlice<f32>,
}

struct CudaKvLayer {
    key: CudaSlice<f16>,
    value: CudaSlice<f16>,
    q8: Option<CudaKvLayerQ8>,
    rows: usize,
    columns: usize,
}

pub struct CudaKvCache {
    layers: Vec<Option<CudaKvLayer>>,
    max_seq_len: usize,
    columns: usize,
    /// Q8G64 模式:append 时量化,attention 走 q8g64 kernel;显存 ~49% 于 f16。
    q8g64: bool,
}

impl CudaKvCache {
    pub fn new(layer_count: usize, max_seq_len: usize, columns: usize) -> Self {
        Self { layers: (0..layer_count).map(|_| None).collect(), max_seq_len, columns, q8g64: false }
    }

    pub fn new_q8g64(layer_count: usize, max_seq_len: usize, columns: usize) -> Result<Self, String> {
        if columns == 0 || columns % 64 != 0 {
            return Err(format!("Q8G64 KV columns={columns} 必须按 64 对齐"));
        }
        Ok(Self { layers: (0..layer_count).map(|_| None).collect(), max_seq_len, columns, q8g64: true })
    }

    pub fn format(&self) -> &'static str {
        if self.q8g64 { "q8g64" } else { "f16" }
    }

    pub fn allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .filter_map(Option::as_ref)
            .map(|layer| if self.q8g64 { 2 * (self.max_seq_len * layer.columns + self.max_seq_len * (layer.columns / 64) * std::mem::size_of::<f32>()) } else { self.max_seq_len * layer.columns * std::mem::size_of::<f16>() * 2 })
            .sum()
    }

    fn append<'a>(&'a mut self, ctx: &CudaContext, layer: usize, position: usize, key: &CudaTensor, value: &CudaTensor) -> Result<&'a CudaKvLayer, BackendError> {
        if key.rows != value.rows || key.cols != value.cols || (self.columns != 0 && key.cols != self.columns) {
            return Err(BackendError::Compute { msg: format!("CUDA GQA KV shape 异常: key=[{},{}], value=[{},{}], columns={}", key.rows, key.cols, value.rows, value.cols, self.columns) });
        }
        if position + key.rows > self.max_seq_len {
            return Err(BackendError::Compute { msg: format!("CUDA GQA KV 超出 max_seq_len: position={position}, rows={}, max={}", key.rows, self.max_seq_len) });
        }
        let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if slot.is_none() {
            let columns = key.cols;
            let elements = self.max_seq_len * columns;
            if self.q8g64 {
                if !columns.is_multiple_of(64) {
                    return Err(BackendError::Compute { msg: format!("CUDA Q8G64 KV columns={columns} 必须按 64 对齐") });
                }
                let scale_elements = self.max_seq_len * (columns / 64);
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
                });
            } else {
                let key = ctx.buffer_uninit::<f16>(elements).map_err(|error| BackendError::Compute { msg: format!("CUDA KV key 分配失败: {error:?}") })?;
                let value = ctx.buffer_uninit::<f16>(elements).map_err(|error| BackendError::Compute { msg: format!("CUDA KV value 分配失败: {error:?}") })?;
                *slot = Some(CudaKvLayer { key, value, q8: None, rows: 0, columns });
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
            ops::attention::append_rows_f16(ctx, key, &cache.key, position).map_err(|msg| BackendError::Compute { msg })?;
            ops::attention::append_rows_f16(ctx, value, &cache.value, position).map_err(|msg| BackendError::Compute { msg })?;
        }
        cache.rows += key.rows;
        Ok(cache)
    }
}

impl GqaPrefillBackend for CudaContext {
    fn gemma_rmsnorm_heads(&self, input: &CudaTensor, weight: &CudaWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, BackendError> {
        ops::attention::gemma_rmsnorm_heads_f16(self, input, &weight.data, head_count, head_dim, eps).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention(&self, query: &CudaTensor, key: &CudaTensor, value: &CudaTensor, spec: &GqaSpec) -> Result<CudaTensor, BackendError> {
        ops::attention::gqa_attention_f16(self, query, &key.slice, &value.slice, key.rows, 0, spec).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention_cached(&self, cache: &mut CudaKvCache, layer: usize, position: usize, query: &CudaTensor, key: &CudaTensor, value: &CudaTensor, spec: &GqaSpec, _retain_full_cache: bool) -> Result<CudaTensor, BackendError> {
        let layer = cache.append(self, layer, position, key, value)?;
        if let Some(q8) = layer.q8.as_ref() {
            // Q8G64:prefill 走 flash 变体(tile 加载时反量化进 smem),约束不满足回退。
            if query.rows > 1 && spec.head_dim.is_multiple_of(32) && spec.head_dim <= 256 && spec.num_heads.is_multiple_of(spec.num_kv_heads) {
                return ops::attention::gqa_attention_q8g64_flash(self, query, &q8.key_codes, &q8.key_scales, &q8.value_codes, &q8.value_scales, layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg });
            }
            return ops::attention::gqa_attention_q8g64(self, query, &q8.key_codes, &q8.key_scales, &q8.value_codes, &q8.value_scales, layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg });
        }
        // prefill(多行)走 flash 式 kernel:block=8 行×1 head 共享 smem K/V tile,
        // KV 全局重复读缩小 8 倍(548 token 实测 8.3 → ~1.5 ms/layer);约束不满足时回退。
        if query.rows > 1 && spec.head_dim.is_multiple_of(32) && spec.head_dim <= 256 && spec.num_heads.is_multiple_of(spec.num_kv_heads) {
            return ops::attention::gqa_attention_f16_flash(self, query, &layer.key, &layer.value, layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg });
        }
        ops::attention::gqa_attention_f16(self, query, &layer.key, &layer.value, layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention_cached_from(&self, cache: &CudaKvCache, source_layer: usize, position: usize, query: &CudaTensor, spec: &GqaSpec) -> Result<CudaTensor, BackendError> {
        let layer =
            cache.layers.get(source_layer).ok_or(BackendError::UnsupportedLayer { layer: source_layer })?.as_ref().ok_or_else(|| BackendError::Compute { msg: format!("CUDA GQA shared KV source layer {source_layer} 尚未初始化") })?;
        if let Some(q8) = layer.q8.as_ref() {
            return ops::attention::gqa_attention_q8g64(self, query, &q8.key_codes, &q8.key_scales, &q8.value_codes, &q8.value_scales, layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg });
        }
        ops::attention::gqa_attention_f16(self, query, &layer.key, &layer.value, layer.rows, position, spec).map_err(|msg| BackendError::Compute { msg })
    }
}
