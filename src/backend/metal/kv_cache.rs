//! GPU resident KV cache 的 Metal 后端资源。spec 层在 [`crate::kv_cache`]，
//! 本模块只负责"怎么做":Metal buffer 分配 + INT8 量化 GPU kernel + 生命周期。
//!
//! 单块 `StorageModeShared` Metal buffer,layer-major 布局。每层三段连续区域:
//! `[codes | scales | rope]`,各自 `[capacity × per_token]`。
//!
//! 量化/反量化用 GPU kernel(`mla_kv_quantize_f16` / `mla_kv_dequantize_f16`),
//! append 时上传 f16 staging → 在线量化写入;attention 前 GPU 反量化重建。

use super::context::{MetalContext, MetalTensor};
use crate::attention::gqa::HybridGqaSpec;
use crate::backend::metal::api::{Buffer, MTLResourceOptions, MTLSize};
use crate::kv_cache::{DEFAULT_GROUP_SIZE, HybridGqaCacheLayout, HybridGqaCacheState, KvCacheFormat, KvCacheLayerMap, KvCacheLayout, KvCacheSpec, KvCacheState, MlaRecordLayout, QUANT_BITS};
use half::f16;
use std::{
    fs::File,
    io::{Read, Write},
    mem,
    path::Path,
};

const KV_LAYER_MAGIC: [u8; 8] = *b"ZLLMKV1\0";
const KV_LAYER_VERSION: u32 = 2;
const KV_LAYER_HEADER_BYTES: u64 = 52;

pub type MetalKvCacheFormat = KvCacheFormat;

pub(crate) struct MetalGqaCacheView {
    pub buffer: Buffer,
    pub key_offset: u64,
    pub key_scale_offset: Option<u64>,
    pub value_offset: u64,
    pub value_scale_offset: Option<u64>,
    pub rows: usize,
    pub start: usize,
    pub capacity: usize,
    pub format: MetalKvCacheFormat,
    pub group_size: usize,
}

/// GPU resident KV cache。一块 Metal buffer 承载所有层。
///
/// append 语义:每层维护已写入 token 数(`lengths[layer]`),新 token 紧接已写入区域。
/// MLA 路径:`append_layer` 接收 f32 latent + f32 rope,内部上传 f16 staging → GPU 量化 kernel
/// 写入 codes/scales/rope 三段。attention 前用 [`Self::dequant_layer`] 反量化重建 latent。
pub struct MetalKvCache {
    buffer: Buffer,
    state: KvCacheState,
    hybrid_gqa: Option<HybridGqaCacheState>,
}

impl MetalKvCache {
    /// 分配 cache。buffer = `layer_count × capacity × bytes_per_token` 字节,Shared 内存。
    pub fn new(ctx: &MetalContext, spec: KvCacheSpec, layer_count: usize, capacity: usize) -> Result<Self, String> {
        Self::with_group_size(ctx, spec, layer_count, capacity, DEFAULT_GROUP_SIZE)
    }

    pub fn new_mapped(ctx: &MetalContext, spec: KvCacheSpec, layer_map: KvCacheLayerMap, capacity: usize) -> Result<Self, String> {
        let layout = KvCacheLayout::new_mapped(spec, MetalKvCacheFormat::Int8, layer_map, capacity, DEFAULT_GROUP_SIZE)?;
        let buffer = ctx.device.try_new_buffer(layout.total_bytes() as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked)?;
        Ok(Self { buffer, state: KvCacheState::new(layout), hybrid_gqa: None })
    }

    /// 未量化 MLA cache，作为 decode 正确性和性能基线。
    pub fn new_f16(ctx: &MetalContext, spec: KvCacheSpec, layer_count: usize, capacity: usize) -> Result<Self, String> {
        Self::with_format(ctx, spec, layer_count, capacity, DEFAULT_GROUP_SIZE, MetalKvCacheFormat::F16)
    }

    /// 只为映射中声明的逻辑层分配物理槽；访问 API 仍使用模型逻辑层号。
    pub fn new_f16_mapped(ctx: &MetalContext, spec: KvCacheSpec, layer_map: KvCacheLayerMap, capacity: usize) -> Result<Self, String> {
        let layout = KvCacheLayout::new_mapped(spec, MetalKvCacheFormat::F16, layer_map, capacity, DEFAULT_GROUP_SIZE)?;
        let buffer = ctx.device.try_new_buffer(layout.total_bytes() as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked)?;
        Ok(Self { buffer, state: KvCacheState::new(layout), hybrid_gqa: None })
    }

    /// 自定义 group_size 的构造器(测试或调参用)。
    pub fn with_group_size(ctx: &MetalContext, spec: KvCacheSpec, layer_count: usize, capacity: usize, group_size: usize) -> Result<Self, String> {
        Self::with_format(ctx, spec, layer_count, capacity, group_size, MetalKvCacheFormat::Int8)
    }

    fn with_format(ctx: &MetalContext, spec: KvCacheSpec, layer_count: usize, capacity: usize, group_size: usize, format: MetalKvCacheFormat) -> Result<Self, String> {
        let layout = KvCacheLayout::new(spec, format, layer_count, capacity, group_size)?;
        let buffer = ctx.device.try_new_buffer(layout.total_bytes() as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked)?;
        Ok(Self { buffer, state: KvCacheState::new(layout), hybrid_gqa: None })
    }

    pub fn new_hybrid_gqa(ctx: &MetalContext, spec: HybridGqaSpec, max_sequence_len: usize) -> Result<Self, String> {
        let layout = HybridGqaCacheLayout::new(&spec, max_sequence_len)?;
        let bytes = layout.total_elements().checked_mul(2).ok_or("hybrid GQA KV cache 大小溢出")?;
        let buffer = ctx.device.try_new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked)?;
        let placeholder_layout = KvCacheLayout::new(KvCacheSpec::Gqa { num_kv_heads: 1, head_dim: 1 }, KvCacheFormat::F16, layout.layer_count(), 1, DEFAULT_GROUP_SIZE)?;
        Ok(Self { buffer, state: KvCacheState::new(placeholder_layout), hybrid_gqa: Some(HybridGqaCacheState::new(layout)) })
    }

    pub fn is_hybrid_gqa(&self) -> bool {
        self.hybrid_gqa.is_some()
    }

    pub fn hybrid_gqa_state(&self) -> Option<&crate::kv_cache::HybridGqaCacheState> {
        self.hybrid_gqa.as_ref()
    }

    pub fn hybrid_gqa_state_mut(&mut self) -> Option<&mut crate::kv_cache::HybridGqaCacheState> {
        self.hybrid_gqa.as_mut()
    }

    pub fn spec(&self) -> &KvCacheSpec {
        self.state.layout().spec()
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn format(&self) -> MetalKvCacheFormat {
        self.state.layout().format()
    }

    pub fn layer_count(&self) -> usize {
        self.state.layout().layer_count()
    }

    pub fn cache_slot_count(&self) -> usize {
        self.state.layout().cache_slot_count()
    }

    pub fn capacity(&self) -> usize {
        self.state.layout().capacity()
    }

    pub fn group_size(&self) -> usize {
        self.state.layout().group_size()
    }

    pub fn mla_layout(&self) -> Option<&MlaRecordLayout> {
        self.state.layout().mla_layout()
    }

    pub fn bytes_per_token(&self) -> usize {
        self.state.layout().bytes_per_token()
    }

    pub fn layer_len(&self, layer: usize) -> usize {
        self.state.layer_len(layer)
    }

    /// 快照恢复：写回单层有效长度；调用方需先恢复该层数据段。
    pub fn set_layer_len(&mut self, layer: usize, len: usize) -> Result<(), String> {
        self.state.set_layer_len(layer, len)
    }

    pub fn gqa_columns(&self) -> Result<usize, String> {
        self.state.layout().gqa_columns()
    }

    pub fn layer_gqa_key_offset(&self, layer: usize) -> Result<usize, String> {
        self.state.layout().layer_gqa_key_offset(layer)
    }

    pub fn layer_gqa_key_scale_offset(&self, layer: usize) -> Result<usize, String> {
        self.state.layout().layer_gqa_key_scale_offset(layer)
    }

    pub fn layer_gqa_value_offset(&self, layer: usize) -> Result<usize, String> {
        self.state.layout().layer_gqa_value_offset(layer)
    }

    pub fn layer_gqa_value_scale_offset(&self, layer: usize) -> Result<usize, String> {
        self.state.layout().layer_gqa_value_scale_offset(layer)
    }

    pub fn gqa_groups_per_token(&self) -> Result<usize, String> {
        self.state.layout().gqa_groups_per_token()
    }

    /// 已分配字节数(全模型)。
    pub fn allocated_bytes(&self) -> usize {
        self.buffer.length() as usize
    }

    fn check_layer(&self, layer: usize) -> Result<usize, String> {
        self.state.layout().layer_base(layer)
    }

    /// 该层 latent 段的字节偏移。F16 和 INT8 cache 都从 layer base 开始。
    pub fn layer_latent_offset(&self, layer: usize) -> Result<u64, String> {
        Ok(self.state.layout().layer_latent_offset(layer)? as u64)
    }

    /// 该层 codes 段的字节偏移(append/反量化用)。
    pub fn layer_codes_offset(&self, layer: usize) -> Result<u64, String> {
        Ok(self.state.layout().layer_codes_offset(layer)? as u64)
    }

    /// 该层 scales 段的字节偏移。
    pub fn layer_scales_offset(&self, layer: usize) -> Result<u64, String> {
        Ok(self.state.layout().layer_scales_offset(layer)? as u64)
    }

    /// 该层 rope 段的字节偏移。
    pub fn layer_rope_offset(&self, layer: usize) -> Result<u64, String> {
        Ok(self.state.layout().layer_rope_offset(layer)? as u64)
    }

    /// 把 `n_new_tokens` 行 latent(f32)+ rope(f32)追加到 `layer`。
    ///
    /// 测试与 CPU 路径用:内部上传 f16 staging → GPU 量化 kernel。生产路径
    /// 应优先用 [`Self::append_layer_mla_tensor`](GPU buffer 直入,无 CPU 往返)。
    pub fn append_layer_mla(&mut self, ctx: &MetalContext, layer: usize, latent_f32: &[f32], rope_f32: &[f32], n_new_tokens: usize) -> Result<(), String> {
        let layout = *self.mla_layout().ok_or("append_layer_mla 仅支持 MLA cache")?;
        if latent_f32.len() != n_new_tokens * layout.kv_lora_rank {
            return Err(format!("latent_f32 长度 {} 期望 {}(n_new={n_new_tokens} × kv_lora_rank={})", latent_f32.len(), n_new_tokens * layout.kv_lora_rank, layout.kv_lora_rank));
        }
        if rope_f32.len() != n_new_tokens * layout.qk_rope_head_dim {
            return Err(format!("rope_f32 长度 {} 期望 {}(n_new={n_new_tokens} × qk_rope_head_dim={})", rope_f32.len(), n_new_tokens * layout.qk_rope_head_dim, layout.qk_rope_head_dim));
        }
        let latent_buf = ctx.shared_buffer_from_f32(latent_f32);
        let rope_buf = ctx.shared_buffer_from_f32(rope_f32);
        self.quantize_into_cache(ctx, layer, &latent_buf, &rope_buf, n_new_tokens)
    }

    /// device-resident 路径:两个 GPU MetalTensor(latent norm 后 + rope RoPE 后)直入 cache。
    /// latent 形状 `[n_new, kv_lora_rank]`、rope 形状 `[n_new, qk_rope_head_dim]`。
    pub fn append_layer_mla_tensor(&mut self, ctx: &MetalContext, layer: usize, latent: &MetalTensor, rope: &MetalTensor) -> Result<(), String> {
        let layout = *self.mla_layout().ok_or("append_layer_mla_tensor 仅支持 MLA cache")?;
        if latent.cols != layout.kv_lora_rank || rope.cols != layout.qk_rope_head_dim || latent.rows != rope.rows {
            return Err(format!("append tensor 形状不符: latent=[{},{}] rope=[{},{}] (期望 latent=[*,{}], rope=[*,{}])", latent.rows, latent.cols, rope.rows, rope.cols, layout.kv_lora_rank, layout.qk_rope_head_dim));
        }
        self.quantize_into_cache(ctx, layer, &latent.buffer, &rope.buffer, latent.rows)
    }

    /// 当前 K/V 直接写入常驻显存；SSD 换入换出由更上层资源策略决定。
    /// 只推进长度状态不做 GPU 写入:配合量化融合 attention kernel(新行由 kernel 回写)。
    pub fn reserve_layer_gqa_row(&mut self, layer: usize, position: usize) -> Result<(), String> {
        let current = self.layer_len(layer);
        if current != position {
            return Err(format!("L{layer} GQA cache position 不连续: cached={current} position={position}"));
        }
        let end = self.state.append_end(layer, 1)?;
        self.state.set_layer_len(layer, end)
    }

    pub fn append_layer_gqa_tensor(&mut self, ctx: &MetalContext, layer: usize, position: usize, key: &MetalTensor, value: &MetalTensor) -> Result<(), String> {
        if key.dtype != value.dtype || !matches!(key.dtype, super::MetalTensorDType::F16 | super::MetalTensorDType::Bf16) {
            return Err(format!("GQA cache 需要相同的 F16/BF16 K/V，实际 {:?}/{:?}", key.dtype, value.dtype));
        }
        if let Some(state) = &mut self.hybrid_gqa {
            let plan = state.plan_append(layer, position, key.rows)?;
            let layout = state.layout();
            let layer_layout = layout.layer(layer)?;
            let columns = layer_layout.columns;
            if key.rows != value.rows || key.cols != columns || value.cols != columns {
                return Err(format!("L{layer} hybrid GQA cache shape 不符: K=[{},{}] V=[{},{}]，期望 [*,{columns}]", key.rows, key.cols, value.rows, value.cols));
            }
            if key.rows == 0 {
                return Ok(());
            }
            let row_bytes = columns.checked_mul(2).ok_or("hybrid GQA KV row bytes 溢出")?;
            let source_offset = plan.source_start.checked_mul(row_bytes).ok_or("hybrid GQA KV source offset 溢出")?;
            let key_base = layer_layout.key_offset.checked_mul(2).ok_or("hybrid GQA K offset 溢出")?;
            let value_base = layer_layout.value_offset.checked_mul(2).ok_or("hybrid GQA V offset 溢出")?;
            let first_bytes = plan.first_count.checked_mul(row_bytes).ok_or("hybrid GQA first bytes 溢出")?;
            let second_bytes = plan.second_count.checked_mul(row_bytes).ok_or("hybrid GQA second bytes 溢出")?;
            let command = ctx.command_buffer();
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&key.buffer, source_offset as u64, &self.buffer, (key_base + plan.first_slot * row_bytes) as u64, first_bytes as u64);
            blit.copy_from_buffer(&value.buffer, source_offset as u64, &self.buffer, (value_base + plan.first_slot * row_bytes) as u64, first_bytes as u64);
            if second_bytes != 0 {
                let second_source = source_offset + first_bytes;
                blit.copy_from_buffer(&key.buffer, second_source as u64, &self.buffer, key_base as u64, second_bytes as u64);
                blit.copy_from_buffer(&value.buffer, second_source as u64, &self.buffer, value_base as u64, second_bytes as u64);
            }
            blit.end_encoding();
            let shape = format!("layer={layer},position={position},rows={},cols={columns}", key.rows);
            let operator = if key.dtype == super::MetalTensorDType::Bf16 { "hybrid_gqa_kv_append_bf16" } else { "hybrid_gqa_kv_append_f16" };
            ctx.commit_and_wait_profiled(&command, operator, &shape, ((first_bytes + second_bytes) * 2) as u64, ((first_bytes + second_bytes) * 2) as u64);
            state.commit_append(&plan)?;
            return Ok(());
        }
        let columns = self.state.layout().gqa_columns()?;
        if key.rows != value.rows || key.cols != columns || value.cols != columns {
            return Err(format!("L{layer} GQA cache shape 不符: K=[{},{}] V=[{},{}]，期望 [*,{columns}]", key.rows, key.cols, value.rows, value.cols));
        }
        let current = self.layer_len(layer);
        if current != position {
            return Err(format!("L{layer} GQA cache position 不连续: cached={current} position={position}"));
        }
        let end = self.state.append_end(layer, key.rows)?;
        if key.rows == 0 {
            return Ok(());
        }
        if self.format() == MetalKvCacheFormat::Int8 {
            self.quantize_gqa_into_cache(ctx, layer, current, key, value)?;
            self.state.set_layer_len(layer, end)?;
            return Ok(());
        }
        let row_bytes = columns.checked_mul(2).ok_or("GQA cache row bytes 溢出")?;
        let copy_bytes = key.rows.checked_mul(row_bytes).ok_or("GQA cache copy bytes 溢出")? as u64;
        let current_bytes = current.checked_mul(row_bytes).ok_or("GQA cache append offset 溢出")? as u64;
        let key_offset = self.state.layout().layer_gqa_key_offset(layer)? as u64 + current_bytes;
        let value_offset = self.state.layout().layer_gqa_value_offset(layer)? as u64 + current_bytes;
        let command = ctx.command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(&key.buffer, 0, &self.buffer, key_offset, copy_bytes);
        blit.copy_from_buffer(&value.buffer, 0, &self.buffer, value_offset, copy_bytes);
        blit.end_encoding();
        let shape = format!("layer={layer},position={position},rows={},cols={columns}", key.rows);
        ctx.commit_and_wait_profiled(&command, "gqa_kv_append_f16", &shape, copy_bytes * 2, copy_bytes * 2);
        self.state.set_layer_len(layer, end)?;
        Ok(())
    }

    fn quantize_gqa_into_cache(&self, ctx: &MetalContext, layer: usize, current: usize, key: &MetalTensor, value: &MetalTensor) -> Result<(), String> {
        let KvCacheSpec::Gqa { num_kv_heads, head_dim } = self.spec() else {
            return Err("GQA quantize 需要 GQA cache spec".to_owned());
        };
        let groups_per_head = head_dim / self.group_size();
        let groups_per_token = self.state.layout().gqa_groups_per_token()?;
        let key_codes_offset = self.state.layout().layer_gqa_key_offset(layer)? as u64 + (current * key.cols) as u64;
        let key_scales_offset = self.state.layout().layer_gqa_key_scale_offset(layer)? as u64 + (current * groups_per_token * 2) as u64;
        let value_codes_offset = self.state.layout().layer_gqa_value_offset(layer)? as u64 + (current * value.cols) as u64;
        let value_scales_offset = self.state.layout().layer_gqa_value_scale_offset(layer)? as u64 + (current * groups_per_token * 2) as u64;
        let rows = u32::try_from(key.rows).map_err(|_| "GQA Q8 rows 超过 u32".to_owned())?;
        let kv_heads = u32::try_from(*num_kv_heads).map_err(|_| "GQA Q8 kv_heads 超过 u32".to_owned())?;
        let head_dim = u32::try_from(*head_dim).map_err(|_| "GQA Q8 head_dim 超过 u32".to_owned())?;
        let group_size = u32::try_from(self.group_size()).map_err(|_| "GQA Q8 group_size 超过 u32".to_owned())?;
        let groups_per_head_u32 = u32::try_from(groups_per_head).map_err(|_| "GQA Q8 groups_per_head 超过 u32".to_owned())?;
        let bf16 = u32::from(key.dtype == super::MetalTensorDType::Bf16);
        let total_groups = key.rows.checked_mul(groups_per_token).and_then(|count| count.checked_mul(2)).ok_or("GQA Q8 group 数溢出")?;
        let pipeline = ctx.pipeline("gqa_kv_quantize_q8")?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&key.buffer), 0);
        encoder.set_buffer(1, Some(&value.buffer), 0);
        encoder.set_buffer(2, Some(&self.buffer), key_codes_offset);
        encoder.set_buffer(3, Some(&self.buffer), key_scales_offset);
        encoder.set_buffer(4, Some(&self.buffer), value_codes_offset);
        encoder.set_buffer(5, Some(&self.buffer), value_scales_offset);
        crate::kernel::metal::set_bytes(&encoder, 6, &rows);
        crate::kernel::metal::set_bytes(&encoder, 7, &kv_heads);
        crate::kernel::metal::set_bytes(&encoder, 8, &head_dim);
        crate::kernel::metal::set_bytes(&encoder, 9, &group_size);
        crate::kernel::metal::set_bytes(&encoder, 10, &groups_per_head_u32);
        crate::kernel::metal::set_bytes(&encoder, 11, &bf16);
        let width = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(MTLSize::new(total_groups as u64, 1, 1), MTLSize::new(width, 1, 1));
        encoder.end_encoding();
        let output_bytes = key.rows * (key.cols * 2 + groups_per_token * 4);
        let shape = format!("rows={},kv_heads={},head_dim={},group={}", key.rows, num_kv_heads, head_dim, self.group_size());
        ctx.commit_and_wait_profiled(&command, "gqa_kv_quantize_q8", &shape, key.buffer.length() + value.buffer.length(), output_bytes as u64);
        Ok(())
    }

    pub(crate) fn gqa_layer_view(&self, layer: usize) -> Result<MetalGqaCacheView, String> {
        if let Some(state) = &self.hybrid_gqa {
            let layout = state.layout();
            let retained = state.retained_range(layer)?;
            let layer_layout = layout.layer(layer)?;
            return Ok(MetalGqaCacheView {
                buffer: self.buffer.clone(),
                key_offset: (layer_layout.key_offset * 2) as u64,
                key_scale_offset: None,
                value_offset: (layer_layout.value_offset * 2) as u64,
                value_scale_offset: None,
                rows: retained.end,
                start: retained.start,
                capacity: layer_layout.capacity,
                format: MetalKvCacheFormat::F16,
                group_size: self.group_size(),
            });
        }
        self.state.layout().gqa_columns()?;
        let int8 = self.format() == MetalKvCacheFormat::Int8;
        Ok(MetalGqaCacheView {
            buffer: self.buffer.clone(),
            key_offset: self.state.layout().layer_gqa_key_offset(layer)? as u64,
            key_scale_offset: int8.then(|| self.state.layout().layer_gqa_key_scale_offset(layer)).transpose()?.map(|offset| offset as u64),
            value_offset: self.state.layout().layer_gqa_value_offset(layer)? as u64,
            value_scale_offset: int8.then(|| self.state.layout().layer_gqa_value_scale_offset(layer)).transpose()?.map(|offset| offset as u64),
            rows: self.layer_len(layer),
            start: 0,
            // F16 position-replay 仍使用普通连续 cache；把真实容量交给
            // position kernel 后，position<capacity 时 ring 寻址与连续布局一致。
            capacity: if int8 { 0 } else { self.capacity() },
            format: self.format(),
            group_size: self.group_size(),
        })
    }

    /// 量化写入的共用内核。latent_buf/rope_buf 是 f16 GPU buffer,长度由调用方保证。
    fn quantize_into_cache(&mut self, ctx: &MetalContext, layer: usize, latent_buf: &Buffer, rope_buf: &Buffer, n_new_tokens: usize) -> Result<(), String> {
        let layout = *self.mla_layout().ok_or("quantize_into_cache 仅支持 MLA cache")?;
        let current = self.layer_len(layer);
        let end = self.state.append_end(layer, n_new_tokens)?;
        if n_new_tokens == 0 {
            return Ok(());
        }

        if self.format() == MetalKvCacheFormat::F16 {
            let latent_bytes = (n_new_tokens * layout.kv_lora_rank * 2) as u64;
            let rope_bytes = (n_new_tokens * layout.rope_bytes_per_token()) as u64;
            let latent_offset = self.layer_latent_offset(layer)? + (current * layout.kv_lora_rank * 2) as u64;
            let rope_offset = self.layer_rope_offset(layer)? + (current * layout.rope_bytes_per_token()) as u64;
            let command = ctx.command_buffer();
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(latent_buf, 0, &self.buffer, latent_offset, latent_bytes);
            blit.copy_from_buffer(rope_buf, 0, &self.buffer, rope_offset, rope_bytes);
            blit.end_encoding();
            let shape = format!("n_new={n_new_tokens},kv_lora={},rope={}", layout.kv_lora_rank, layout.qk_rope_head_dim);
            ctx.commit_and_wait_profiled(&command, "mla_kv_append_f16", &shape, latent_bytes + rope_bytes, latent_bytes + rope_bytes);
            self.state.set_layer_len(layer, end)?;
            return Ok(());
        }

        let codes_offset = self.layer_codes_offset(layer)? + (current * layout.codes_bytes_per_token()) as u64;
        let scales_offset = self.layer_scales_offset(layer)? + (current * layout.scale_bytes_per_token()) as u64;
        let rope_offset = self.layer_rope_offset(layer)? + (current * layout.rope_bytes_per_token()) as u64;

        let rows_u32 = u32::try_from(n_new_tokens).map_err(|_| "append rows 超 u32".to_owned())?;
        let columns_u32 = u32::try_from(layout.kv_lora_rank).map_err(|_| "kv_lora_rank 超 u32".to_owned())?;
        let group_size_u32 = u32::try_from(layout.group_size).map_err(|_| "group_size 超 u32".to_owned())?;
        let quant_bits = QUANT_BITS;
        let rope_columns_u32 = u32::try_from(layout.qk_rope_head_dim).map_err(|_| "qk_rope_head_dim 超 u32".to_owned())?;

        let pipeline = ctx.pipeline("mla_kv_quantize_f16")?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(latent_buf), 0);
        encoder.set_buffer(1, Some(&self.buffer), codes_offset);
        encoder.set_buffer(2, Some(&self.buffer), scales_offset);
        crate::kernel::metal::set_bytes(&encoder, 3, &rows_u32);
        crate::kernel::metal::set_bytes(&encoder, 4, &columns_u32);
        crate::kernel::metal::set_bytes(&encoder, 5, &group_size_u32);
        crate::kernel::metal::set_bytes(&encoder, 6, &quant_bits);
        encoder.set_buffer(7, Some(rope_buf), 0);
        encoder.set_buffer(8, Some(&self.buffer), rope_offset);
        crate::kernel::metal::set_bytes(&encoder, 9, &rope_columns_u32);

        let total_groups = n_new_tokens * layout.groups_per_row();
        let width = pipeline.max_total_threads_per_threadgroup().min(256);
        encoder.dispatch_thread_groups(MTLSize::new((total_groups as u64).div_ceil(width.max(1)), 1, 1), MTLSize::new(width, 1, 1));
        encoder.end_encoding();
        let shape = format!("n_new={n_new_tokens},kv_lora={},group={}", layout.kv_lora_rank, layout.group_size);
        ctx.commit_and_wait_profiled(&command, "mla_kv_quantize_f16", &shape, latent_buf.length() + rope_buf.length(), (n_new_tokens * layout.bytes_per_token()) as u64);

        self.state.set_layer_len(layer, end)?;
        Ok(())
    }

    /// 反量化该层 `[0, layer_len)` 的 latent → f32(attention 重建前用)。
    /// rope 段直接读出(已是 f16)。返回 (latent_f32, rope_f32)。
    /// 反量化该层 `[0, layer_len)` 的 latent → f32,rope 直读 f16→f32。
    /// 测试与 CPU 路径用。生产路径应优先用 [`Self::dequant_layer_mla_tensor`]。
    pub fn dequant_layer_mla(&self, ctx: &MetalContext, layer: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let layout = *self.mla_layout().ok_or("dequant_layer_mla 仅支持 MLA cache")?;
        let len = self.layer_len(layer);
        if len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if self.format() == MetalKvCacheFormat::F16 {
            let ptr = self.buffer.contents() as *const u8;
            let latent_elements = len * layout.kv_lora_rank;
            let latent_ptr = unsafe { ptr.add(self.layer_latent_offset(layer)? as usize) as *const f16 };
            let rope_elements = len * layout.qk_rope_head_dim;
            let rope_ptr = unsafe { ptr.add(self.layer_rope_offset(layer)? as usize) as *const f16 };
            let latent = unsafe { std::slice::from_raw_parts(latent_ptr, latent_elements) }.iter().map(|value| value.to_f32()).collect();
            let rope = unsafe { std::slice::from_raw_parts(rope_ptr, rope_elements) }.iter().map(|value| value.to_f32()).collect();
            return Ok((latent, rope));
        }
        let latent_elements = len * layout.kv_lora_rank;
        let latent_out = ctx.shared_buffer_zeros(latent_elements * 2);
        self.dequant_latent_into_buffer(ctx, layer, len, &latent_out)?;
        let latent_f32 = ctx.read_f16_to_f32(&latent_out, latent_elements);

        // rope 直接从 cache buffer 读 f16(Shared,CPU 可见)。
        let rope_offset = self.layer_rope_offset(layer)?;
        let rope_elements = len * layout.qk_rope_head_dim;
        let rope_f32 = {
            let ptr = self.buffer.contents() as *const u8;
            let rope_ptr = unsafe { (ptr.add(rope_offset as usize)) as *const f16 };
            let slice = unsafe { std::slice::from_raw_parts(rope_ptr, rope_elements) };
            slice.iter().map(|v| v.to_f32()).collect()
        };
        Ok((latent_f32, rope_f32))
    }

    /// device-resident 路径:反量化 latent → MetalTensor `[layer_len, kv_lora_rank]`,
    /// rope 段用 blit copy 到独立 MetalTensor `[layer_len, qk_rope_head_dim]`。全程 GPU。
    pub fn dequant_layer_mla_tensor(&self, ctx: &MetalContext, layer: usize) -> Result<(MetalTensor, MetalTensor), String> {
        let layout = *self.mla_layout().ok_or("dequant_layer_mla_tensor 仅支持 MLA cache")?;
        let len = self.layer_len(layer);
        if len == 0 {
            return Ok((ctx.tensor_zeros(0, layout.kv_lora_rank), ctx.tensor_zeros(0, layout.qk_rope_head_dim)));
        }
        if self.format() == MetalKvCacheFormat::F16 {
            let latent = ctx.tensor_kernel_output(len, layout.kv_lora_rank);
            let rope = ctx.tensor_kernel_output(len, layout.qk_rope_head_dim);
            let latent_bytes = (len * layout.kv_lora_rank * 2) as u64;
            let rope_bytes = (len * layout.rope_bytes_per_token()) as u64;
            let command = ctx.command_buffer();
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&self.buffer, self.layer_latent_offset(layer)?, &latent.buffer, 0, latent_bytes);
            blit.copy_from_buffer(&self.buffer, self.layer_rope_offset(layer)?, &rope.buffer, 0, rope_bytes);
            blit.end_encoding();
            let shape = format!("rows={len},kv_lora={},rope={}", layout.kv_lora_rank, layout.qk_rope_head_dim);
            ctx.commit_and_wait_profiled(&command, "mla_kv_read_f16", &shape, latent_bytes + rope_bytes, latent_bytes + rope_bytes);
            return Ok((latent, rope));
        }
        let latent = ctx.tensor_kernel_output(len, layout.kv_lora_rank);
        self.dequant_latent_into_buffer(ctx, layer, len, &latent.buffer)?;

        // rope 用 blit copy 从 cache buffer 拷到独立 MetalTensor(同 command buffer)。
        let rope_offset = self.layer_rope_offset(layer)?;
        let rope_bytes = (len * layout.rope_bytes_per_token()) as u64;
        let rope = ctx.tensor_kernel_output(len, layout.qk_rope_head_dim);
        let command = ctx.command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(&self.buffer, rope_offset, &rope.buffer, 0, rope_bytes);
        blit.end_encoding();
        let shape = format!("rope_copy=[{len},{}]", layout.qk_rope_head_dim);
        ctx.commit_and_wait_profiled(&command, "mla_kv_rope_copy", &shape, rope_bytes, rope_bytes);
        Ok((latent, rope))
    }

    /// GPU 反量化 latent 的共用内核:codes + scales → latent_out(f16 buffer)。
    fn dequant_latent_into_buffer(&self, ctx: &MetalContext, layer: usize, len: usize, latent_out: &Buffer) -> Result<(), String> {
        let layout = *self.mla_layout().ok_or("dequant 仅支持 MLA cache")?;
        let codes_offset = self.layer_codes_offset(layer)?;
        let scales_offset = self.layer_scales_offset(layer)?;
        let latent_elements = len * layout.kv_lora_rank;

        let rows_u32 = u32::try_from(len).map_err(|_| "dequant rows 超 u32".to_owned())?;
        let columns_u32 = u32::try_from(layout.kv_lora_rank).map_err(|_| "kv_lora_rank 超 u32".to_owned())?;
        let group_size_u32 = u32::try_from(layout.group_size).map_err(|_| "group_size 超 u32".to_owned())?;
        let quant_bits = QUANT_BITS;

        let pipeline = ctx.pipeline("mla_kv_dequantize_f16")?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&self.buffer), codes_offset);
        encoder.set_buffer(1, Some(&self.buffer), scales_offset);
        encoder.set_buffer(2, Some(latent_out), 0);
        crate::kernel::metal::set_bytes(&encoder, 3, &rows_u32);
        crate::kernel::metal::set_bytes(&encoder, 4, &columns_u32);
        crate::kernel::metal::set_bytes(&encoder, 5, &group_size_u32);
        crate::kernel::metal::set_bytes(&encoder, 6, &quant_bits);
        let width = pipeline.max_total_threads_per_threadgroup().min(256);
        encoder.dispatch_thread_groups(MTLSize::new((latent_elements as u64).div_ceil(width.max(1)), 1, 1), MTLSize::new(width, 1, 1));
        encoder.end_encoding();
        let shape = format!("rows={len},kv_lora={}", layout.kv_lora_rank);
        ctx.commit_and_wait_profiled(&command, "mla_kv_dequantize_f16", &shape, (len * layout.codes_bytes_per_token()) as u64, latent_out.length());
        Ok(())
    }

    /// 把一层实际使用的 cache 紧凑写入文件，不保存 capacity 对应的空洞。
    ///
    /// 文件内容为固定头部和 `[codes | scales | rope]`，可加载到更大的 capacity。
    pub fn dump_layer(&self, path: impl AsRef<Path>, layer: usize) -> Result<usize, String> {
        let path = path.as_ref();
        let layout = *self.mla_layout().ok_or("dump_layer 当前只支持 MLA cache")?;
        self.check_layer(layer)?;
        let len = self.layer_len(layer);
        let payload_bytes = len.checked_mul(self.bytes_per_token()).ok_or_else(|| format!("L{layer} dump payload 长度溢出"))?;
        let layer_u32 = u32::try_from(layer).map_err(|_| format!("L{layer} 超出 u32"))?;
        let len_u64 = u64::try_from(len).map_err(|_| format!("L{layer} token 数超出 u64"))?;
        let payload_u64 = u64::try_from(payload_bytes).map_err(|_| format!("L{layer} payload 超出 u64"))?;
        let tmp_path = path.with_extension("kv.tmp");
        let mut file = File::create(&tmp_path).map_err(|error| format!("创建 {}: {error}", tmp_path.display()))?;
        file.write_all(&KV_LAYER_MAGIC).map_err(|error| format!("写 {} magic: {error}", tmp_path.display()))?;
        write_u32(&mut file, KV_LAYER_VERSION, &tmp_path)?;
        write_u32(&mut file, layer_u32, &tmp_path)?;
        write_u64(&mut file, len_u64, &tmp_path)?;
        write_u32(&mut file, layout.kv_lora_rank as u32, &tmp_path)?;
        write_u32(&mut file, layout.qk_rope_head_dim as u32, &tmp_path)?;
        write_u32(&mut file, layout.group_size as u32, &tmp_path)?;
        write_u32(&mut file, self.format().code(), &tmp_path)?;
        write_u32(&mut file, self.bytes_per_token() as u32, &tmp_path)?;
        write_u64(&mut file, payload_u64, &tmp_path)?;

        let rope_bytes = len * layout.rope_bytes_per_token();
        match self.format() {
            MetalKvCacheFormat::F16 => {
                let latent_bytes = len * layout.kv_lora_rank * 2;
                write_buffer_region(&mut file, &self.buffer, self.layer_latent_offset(layer)? as usize, latent_bytes, &tmp_path)?;
            }
            MetalKvCacheFormat::Int8 => {
                let codes_bytes = len * layout.codes_bytes_per_token();
                let scales_bytes = len * layout.scale_bytes_per_token();
                write_buffer_region(&mut file, &self.buffer, self.layer_codes_offset(layer)? as usize, codes_bytes, &tmp_path)?;
                write_buffer_region(&mut file, &self.buffer, self.layer_scales_offset(layer)? as usize, scales_bytes, &tmp_path)?;
            }
        }
        write_buffer_region(&mut file, &self.buffer, self.layer_rope_offset(layer)? as usize, rope_bytes, &tmp_path)?;
        file.sync_all().map_err(|error| format!("同步 {}: {error}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, path).map_err(|error| format!("提交 {} -> {}: {error}", tmp_path.display(), path.display()))?;
        Ok(payload_bytes + KV_LAYER_HEADER_BYTES as usize)
    }

    /// 运行时保持 F16，checkpoint 按 INT8 per-group + F16 scale 紧凑写盘。
    pub fn dump_layer_int8(&self, path: impl AsRef<Path>, layer: usize) -> Result<usize, String> {
        if self.format() == MetalKvCacheFormat::Int8 {
            return self.dump_layer(path, layer);
        }
        let path = path.as_ref();
        let layout = *self.mla_layout().ok_or("dump_layer_int8 当前只支持 MLA cache")?;
        self.check_layer(layer)?;
        let stored = MlaRecordLayout::new(layout.kv_lora_rank, layout.qk_rope_head_dim, self.group_size())?;
        let len = self.layer_len(layer);
        let payload_bytes = len.checked_mul(stored.bytes_per_token()).ok_or_else(|| format!("L{layer} INT8 dump payload 长度溢出"))?;
        let tmp_path = path.with_extension("kv.tmp");
        let mut file = File::create(&tmp_path).map_err(|error| format!("创建 {}: {error}", tmp_path.display()))?;
        file.write_all(&KV_LAYER_MAGIC).map_err(|error| format!("写 {} magic: {error}", tmp_path.display()))?;
        write_u32(&mut file, KV_LAYER_VERSION, &tmp_path)?;
        write_u32(&mut file, u32::try_from(layer).map_err(|_| format!("L{layer} 超出 u32"))?, &tmp_path)?;
        write_u64(&mut file, u64::try_from(len).map_err(|_| format!("L{layer} token 数超出 u64"))?, &tmp_path)?;
        write_u32(&mut file, stored.kv_lora_rank as u32, &tmp_path)?;
        write_u32(&mut file, stored.qk_rope_head_dim as u32, &tmp_path)?;
        write_u32(&mut file, stored.group_size as u32, &tmp_path)?;
        write_u32(&mut file, MetalKvCacheFormat::Int8.code(), &tmp_path)?;
        write_u32(&mut file, stored.bytes_per_token() as u32, &tmp_path)?;
        write_u64(&mut file, payload_bytes as u64, &tmp_path)?;

        let latent_elements = len * stored.kv_lora_rank;
        let latent = unsafe {
            let ptr = self.buffer.contents().cast::<u8>().add(self.layer_latent_offset(layer)? as usize).cast::<f16>();
            std::slice::from_raw_parts(ptr, latent_elements)
        };
        let mut codes = Vec::with_capacity(latent_elements);
        let mut scales = Vec::with_capacity(len * stored.scale_bytes_per_token());
        for row in 0..len {
            let row_base = row * stored.kv_lora_rank;
            for group in 0..stored.groups_per_row() {
                let begin = row_base + group * stored.group_size;
                let values = &latent[begin..begin + stored.group_size];
                let maximum = values.iter().fold(0.0f32, |maximum, value| maximum.max(value.to_f32().abs()));
                let scale = if maximum > 0.0 { maximum / 127.0 } else { 1.0 };
                scales.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
                let inverse_scale = 1.0 / scale;
                codes.extend(values.iter().map(|value| (value.to_f32() * inverse_scale).round_ties_even().clamp(-127.0, 127.0) as i8 as u8));
            }
        }
        file.write_all(&codes).map_err(|error| format!("写 {} codes: {error}", tmp_path.display()))?;
        file.write_all(&scales).map_err(|error| format!("写 {} scales: {error}", tmp_path.display()))?;
        write_buffer_region(&mut file, &self.buffer, self.layer_rope_offset(layer)? as usize, len * stored.rope_bytes_per_token(), &tmp_path)?;
        file.sync_all().map_err(|error| format!("同步 {}: {error}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, path).map_err(|error| format!("提交 {} -> {}: {error}", tmp_path.display(), path.display()))?;
        Ok(payload_bytes + KV_LAYER_HEADER_BYTES as usize)
    }

    /// 从紧凑层文件恢复 cache，目标 capacity 可以大于 dump 时的实际 token 数。
    pub fn load_layer(&mut self, path: impl AsRef<Path>, layer: usize) -> Result<usize, String> {
        let path = path.as_ref();
        let layout = *self.mla_layout().ok_or("load_layer 当前只支持 MLA cache")?;
        self.check_layer(layer)?;
        let mut file = File::open(path).map_err(|error| format!("打开 {}: {error}", path.display()))?;
        let file_bytes = file.metadata().map_err(|error| format!("读取 {} metadata: {error}", path.display()))?.len();
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic).map_err(|error| format!("读取 {} magic: {error}", path.display()))?;
        if magic != KV_LAYER_MAGIC {
            return Err(format!("{} 不是 zLLM KV layer 文件", path.display()));
        }
        let version = read_u32(&mut file, path)?;
        let stored_layer = read_u32(&mut file, path)? as usize;
        let len = usize::try_from(read_u64(&mut file, path)?).map_err(|_| format!("{} token 数超出 usize", path.display()))?;
        let kv_lora_rank = read_u32(&mut file, path)? as usize;
        let rope_dim = read_u32(&mut file, path)? as usize;
        let group_size = read_u32(&mut file, path)? as usize;
        let format_code = read_u32(&mut file, path)?;
        let stored_format = MetalKvCacheFormat::from_code(format_code).ok_or_else(|| format!("{} cache format {format_code} 未知", path.display()))?;
        let bytes_per_token = read_u32(&mut file, path)? as usize;
        let payload_bytes = usize::try_from(read_u64(&mut file, path)?).map_err(|_| format!("{} payload 超出 usize", path.display()))?;
        if version != KV_LAYER_VERSION {
            return Err(format!("{} KV version {version}，当前只支持 {KV_LAYER_VERSION}", path.display()));
        }
        if stored_layer != layer {
            return Err(format!("{} 保存的是 L{stored_layer}，不能加载到 L{layer}", path.display()));
        }
        let stored_layout = MlaRecordLayout::new(kv_lora_rank, rope_dim, group_size).map_err(|error| format!("{} cache layout: {error}", path.display()))?;
        let expected_bytes_per_token = match stored_format {
            MetalKvCacheFormat::F16 => (kv_lora_rank + rope_dim) * mem::size_of::<f16>(),
            MetalKvCacheFormat::Int8 => stored_layout.bytes_per_token(),
        };
        if kv_lora_rank != layout.kv_lora_rank || rope_dim != layout.qk_rope_head_dim || bytes_per_token != expected_bytes_per_token {
            return Err(format!("{} cache spec 不匹配: kv={kv_lora_rank},rope={rope_dim},group={group_size},format={stored_format:?},bytes/token={bytes_per_token}", path.display()));
        }
        if stored_format == MetalKvCacheFormat::Int8 && group_size != self.group_size() {
            return Err(format!("{} INT8 group_size {group_size}，当前要求 {}", path.display(), self.group_size()));
        }
        if len > self.capacity() {
            return Err(format!("{} 含 {len} token，超过当前 capacity {}", path.display(), self.capacity()));
        }
        let expected_payload = len.checked_mul(bytes_per_token).ok_or_else(|| format!("{} payload 长度溢出", path.display()))?;
        if payload_bytes != expected_payload || file_bytes != KV_LAYER_HEADER_BYTES + payload_bytes as u64 {
            return Err(format!("{} 长度不符: header payload={payload_bytes}，期望={expected_payload}，文件={file_bytes}", path.display()));
        }

        let rope_bytes = len * layout.rope_bytes_per_token();
        match (self.format(), stored_format) {
            (MetalKvCacheFormat::F16, MetalKvCacheFormat::F16) => {
                let latent_bytes = len * layout.kv_lora_rank * 2;
                read_buffer_region(&mut file, &self.buffer, self.layer_latent_offset(layer)? as usize, latent_bytes, path)?;
            }
            (MetalKvCacheFormat::Int8, MetalKvCacheFormat::Int8) => {
                let codes_bytes = len * stored_layout.codes_bytes_per_token();
                let scales_bytes = len * stored_layout.scale_bytes_per_token();
                read_buffer_region(&mut file, &self.buffer, self.layer_codes_offset(layer)? as usize, codes_bytes, path)?;
                read_buffer_region(&mut file, &self.buffer, self.layer_scales_offset(layer)? as usize, scales_bytes, path)?;
            }
            (MetalKvCacheFormat::F16, MetalKvCacheFormat::Int8) => {
                let mut codes = vec![0u8; len * stored_layout.codes_bytes_per_token()];
                let mut scale_bytes = vec![0u8; len * stored_layout.scale_bytes_per_token()];
                file.read_exact(&mut codes).map_err(|error| format!("读 {} codes: {error}", path.display()))?;
                file.read_exact(&mut scale_bytes).map_err(|error| format!("读 {} scales: {error}", path.display()))?;
                let output = unsafe {
                    let ptr = self.buffer.contents().cast::<u8>().add(self.layer_latent_offset(layer)? as usize).cast::<f16>();
                    std::slice::from_raw_parts_mut(ptr, len * layout.kv_lora_rank)
                };
                for row in 0..len {
                    for column in 0..layout.kv_lora_rank {
                        let scale_index = row * stored_layout.groups_per_row() + column / stored_layout.group_size;
                        let scale_offset = scale_index * 2;
                        let scale = f16::from_bits(u16::from_le_bytes([scale_bytes[scale_offset], scale_bytes[scale_offset + 1]])).to_f32();
                        output[row * layout.kv_lora_rank + column] = f16::from_f32((codes[row * layout.kv_lora_rank + column] as i8 as f32) * scale);
                    }
                }
            }
            (MetalKvCacheFormat::Int8, MetalKvCacheFormat::F16) => {
                return Err(format!("{} F16 checkpoint 暂不直接加载到 INT8 runtime cache", path.display()));
            }
        }
        let rope_offset = self.layer_rope_offset(layer)? as usize;
        read_buffer_region(&mut file, &self.buffer, rope_offset, rope_bytes, path)?;
        self.state.set_layer_len(layer, len)?;
        Ok(payload_bytes + KV_LAYER_HEADER_BYTES as usize)
    }

    /// 重置所有层长度为 0(buffer 内容不主动清零,语义上视为未初始化前缀)。
    pub fn clear(&mut self) {
        self.state.clear();
    }

    /// 把所有层回滚到最多 `len` 个 token。
    pub fn truncate(&mut self, len: usize) {
        self.state.truncate(len);
    }
}

fn write_u32(file: &mut File, value: u32, path: &Path) -> Result<(), String> {
    file.write_all(&value.to_le_bytes()).map_err(|error| format!("写 {}: {error}", path.display()))
}

fn write_u64(file: &mut File, value: u64, path: &Path) -> Result<(), String> {
    file.write_all(&value.to_le_bytes()).map_err(|error| format!("写 {}: {error}", path.display()))
}

fn read_u32(file: &mut File, path: &Path) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes).map_err(|error| format!("读取 {}: {error}", path.display()))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File, path: &Path) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes).map_err(|error| format!("读取 {}: {error}", path.display()))?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_buffer_region(file: &mut File, buffer: &Buffer, offset: usize, len: usize, path: &Path) -> Result<(), String> {
    if offset.checked_add(len).is_none_or(|end| end > buffer.length() as usize) {
        return Err(format!("写 {} 时 Metal buffer 区间越界: offset={offset},len={len}", path.display()));
    }
    let bytes = unsafe { std::slice::from_raw_parts((buffer.contents() as *const u8).add(offset), len) };
    file.write_all(bytes).map_err(|error| format!("写 {} payload: {error}", path.display()))
}

fn read_buffer_region(file: &mut File, buffer: &Buffer, offset: usize, len: usize, path: &Path) -> Result<(), String> {
    if offset.checked_add(len).is_none_or(|end| end > buffer.length() as usize) {
        return Err(format!("读 {} 时 Metal buffer 区间越界: offset={offset},len={len}", path.display()));
    }
    let bytes = unsafe { std::slice::from_raw_parts_mut((buffer.contents() as *mut u8).add(offset), len) };
    file.read_exact(bytes).map_err(|error| format!("读取 {} payload: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_mla_spec() -> KvCacheSpec {
        KvCacheSpec::Mla { kv_lora_rank: 8, qk_rope_head_dim: 4 }
    }

    fn small_gqa_spec() -> KvCacheSpec {
        KvCacheSpec::Gqa { num_kv_heads: 2, head_dim: 4 }
    }

    fn make_ctx() -> Option<MetalContext> {
        match MetalContext::new_default() {
            Ok(ctx) => Some(ctx),
            Err(error) => {
                eprintln!("skip Metal test: {error}");
                None
            }
        }
    }

    #[test]
    fn allocated_bytes_formula() {
        let Some(ctx) = make_ctx() else { return };
        // 3 层 × 5 token × 20 bytes(kv_lora=8,rope=4,group=4)
        let cache = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 3, 5, 4).unwrap();
        assert_eq!(cache.allocated_bytes(), 3 * 5 * 20);
    }

    #[test]
    fn append_and_dequant_mla_roundtrip() {
        // 端到端:append f32 latent+rope → GPU 量化 → GPU 反量化 → 验证近似相等(INT8 误差)。
        let Some(ctx) = make_ctx() else { return };
        let mut cache = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 2, 4, 4).unwrap();
        // 2 token × kv_lora 8,值在 [-1,1] 内(INT8 量化友好)。
        let latent: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) / 8.0).collect();
        let rope: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) / 8.0).collect();
        cache.append_layer_mla(&ctx, 0, &latent, &rope, 2).unwrap();
        assert_eq!(cache.layer_len(0), 2);

        let (latent_back, rope_back) = cache.dequant_layer_mla(&ctx, 0).unwrap();
        // INT8 per-group 量化误差应 < 1%(这里宽松用 5%,因为数据小 scale 接近 0)。
        for (orig, got) in latent.iter().zip(latent_back.iter()) {
            let rel = (orig - got).abs() / orig.abs().max(1e-6);
            assert!(rel < 0.05, "latent 量化误差过大: orig={orig}, got={got}, rel={rel}");
        }
        // rope 不量化,应逐元素相等。
        for (orig, got) in rope.iter().zip(rope_back.iter()) {
            assert!((orig - got).abs() < 1e-3, "rope 不应变化: orig={orig}, got={got}");
        }
    }

    #[test]
    fn append_beyond_capacity_errors() {
        let Some(ctx) = make_ctx() else { return };
        let mut cache = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 1, 2, 4).unwrap();
        let latent: Vec<f32> = vec![0.0; 2 * 8];
        let rope: Vec<f32> = vec![0.0; 2 * 4];
        cache.append_layer_mla(&ctx, 0, &latent, &rope, 2).unwrap();
        let one_latent: Vec<f32> = vec![0.0; 8];
        let one_rope: Vec<f32> = vec![0.0; 4];
        let result = cache.append_layer_mla(&ctx, 0, &one_latent, &one_rope, 1);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("capacity"));
    }

    #[test]
    fn append_wrong_length_errors() {
        let Some(ctx) = make_ctx() else { return };
        let mut cache = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 1, 4, 4).unwrap();
        let bad_latent: Vec<f32> = vec![0.0; 5]; // 应是 8
        let rope: Vec<f32> = vec![0.0; 4];
        let result = cache.append_layer_mla(&ctx, 0, &bad_latent, &rope, 1);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("期望"));
    }

    #[test]
    fn append_layer_out_of_range_errors() {
        let Some(ctx) = make_ctx() else { return };
        let mut cache = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 2, 4, 4).unwrap();
        let latent: Vec<f32> = vec![0.0; 8];
        let rope: Vec<f32> = vec![0.0; 4];
        let result = cache.append_layer_mla(&ctx, 5, &latent, &rope, 1);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("越界"));
    }

    #[test]
    fn clear_and_truncate() {
        let Some(ctx) = make_ctx() else { return };
        let mut cache = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 2, 4, 4).unwrap();
        let latent: Vec<f32> = vec![0.1; 3 * 8];
        let rope: Vec<f32> = vec![0.1; 3 * 4];
        cache.append_layer_mla(&ctx, 0, &latent, &rope, 3).unwrap();
        cache.append_layer_mla(&ctx, 1, &latent, &rope, 3).unwrap();
        assert_eq!(cache.layer_len(0), 3);

        cache.truncate(1);
        assert_eq!(cache.layer_len(0), 1);
        assert_eq!(cache.layer_len(1), 1);

        cache.clear();
        assert_eq!(cache.layer_len(0), 0);
    }

    #[test]
    fn gqa_buffer_allocates() {
        // GQA 形态:验证 buffer 能正常分配(attention 消费留阶段 4)。
        let Some(ctx) = make_ctx() else { return };
        let cache = MetalKvCache::with_group_size(&ctx, small_gqa_spec(), 2, 3, 4).unwrap();
        assert_eq!(cache.bytes_per_token(), 2 * 4 * 2 + 2 * 2 * 2);
        assert_eq!(cache.allocated_bytes(), 2 * 3 * cache.bytes_per_token());
    }

    #[test]
    fn zero_layer_count_errors() {
        let Some(ctx) = make_ctx() else { return };
        let result = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 0, 4, 4);
        assert!(result.is_err(), "0 层应报错");
        assert!(result.err().unwrap().contains("层数"));
    }

    #[test]
    fn mla_layout_group_size_must_divide() {
        let Some(ctx) = make_ctx() else { return };
        // kv_lora=8, group=3 不能整除 → 应报错。
        let result = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 1, 4, 3);
        assert!(result.is_err());
    }

    #[test]
    fn append_dequant_tensor_roundtrip() {
        // device-resident 路径:MetalTensor 直入 → GPU 量化 → GPU 反量化 → MetalTensor。
        // 验证与 f32 版往返一致的精度。
        let Some(ctx) = make_ctx() else { return };
        let mut cache = MetalKvCache::with_group_size(&ctx, small_mla_spec(), 1, 4, 4).unwrap();
        let latent: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) / 8.0).collect();
        let rope: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) / 8.0).collect();
        let latent_t = ctx.tensor_from_f32(&latent, 2, 8).unwrap();
        let rope_t = ctx.tensor_from_f32(&rope, 2, 4).unwrap();
        cache.append_layer_mla_tensor(&ctx, 0, &latent_t, &rope_t).unwrap();
        assert_eq!(cache.layer_len(0), 2);

        let (latent_back_t, rope_back_t) = cache.dequant_layer_mla_tensor(&ctx, 0).unwrap();
        assert_eq!(latent_back_t.rows, 2);
        assert_eq!(latent_back_t.cols, 8);
        assert_eq!(rope_back_t.rows, 2);
        assert_eq!(rope_back_t.cols, 4);
        let latent_back = ctx.tensor_to_f32(&latent_back_t);
        let rope_back = ctx.tensor_to_f32(&rope_back_t);

        // INT8 量化误差宽松阈值(见 append_and_dequant_mla_roundtrip)。
        for (orig, got) in latent.iter().zip(latent_back.iter()) {
            let rel = (orig - got).abs() / orig.abs().max(1e-6);
            assert!(rel < 0.05, "tensor latent 量化误差过大: orig={orig}, got={got}");
        }
        for (orig, got) in rope.iter().zip(rope_back.iter()) {
            assert!((orig - got).abs() < 1e-3, "tensor rope 不应变化: orig={orig}, got={got}");
        }
    }

    #[test]
    fn attention_with_cache_matches_direct() {
        // 核心回归:cache 路径(append→INT8→dequant→kv_b_proj→attention)与直接路径
        // (compressed_kv→kv_b_proj→attention)输出在 INT8 量化误差内一致。
        let Some(ctx) = make_ctx() else { return };

        // 小 MLA:heads=2, q_head_dim=4(nope 2 + rope 2),kv_head_dim=6(nope 2 + value 4)。
        // kernel 约束 value_dim == q_head_dim → kv_head_dim = qk_nope + q_head_dim = 2+4 = 6。
        // q_proj = 2×4 = 8,kv_proj = 2×6 = 12,kv_lora=4, rope=2。group_size=4。
        use crate::kernel::metal as metal_ops;
        use crate::weight::Fp8Matrix;
        let n = 3; // 3 个 token

        // compressed_kv [3,4](norm 后的 latent,值小以利 INT8)。
        let compressed_kv: Vec<f32> = (0..n * 4).map(|i| ((i as f32 % 8.0) - 4.0) / 8.0).collect();
        // k_rope [3,2](RoPE 后)。
        let k_rope: Vec<f32> = (0..n * 2).map(|i| (i as f32 - 3.0) / 8.0).collect();
        // q [3,8]。
        let q: Vec<f32> = (0..n * 8).map(|i| ((i as f32 % 8.0) - 4.0) / 16.0).collect();
        // kv_b_proj [kv_proj=12, kv_lora=4] = [out=12, in=4]。
        let mut kv_b = vec![0.0_f32; 12 * 4];
        for r in 0..4 {
            kv_b[r * 4 + r] = 0.5; // 对角线,让 latent 部分通过
        }
        // scale_inv:每 128×128 block 一个 f32。rows=12/128=1, cols=4/128=1 → 1 个 scale。
        let kv_b_fp8 = Fp8Matrix::new(kv_b.iter().map(|&v| (v.clamp(-1.0, 1.0) * 127.0) as i8 as u8).collect(), 1.0f32.to_le_bytes().to_vec(), 12, 4).expect("Fp8Matrix 构造");

        // 直接路径:compressed_kv → kv_b_proj → expanded_kv → attention。
        let q_t = ctx.tensor_from_f32(&q, n, 8).unwrap();
        let compressed_kv_t = ctx.tensor_from_f32(&compressed_kv, n, 4).unwrap();
        let k_rope_t = ctx.tensor_from_f32(&k_rope, n, 2).unwrap();
        let expanded_kv_direct = metal_ops::fp8::fp8_matmul_tensor(&ctx, &compressed_kv_t, &kv_b_fp8).unwrap();
        let direct = metal_ops::attention::mla_attention_tensor(&ctx, &q_t, &expanded_kv_direct, &k_rope_t, 2, 2).unwrap();

        // cache 路径:append compressed_kv + k_rope → with_cache attention。
        let spec = KvCacheSpec::Mla { kv_lora_rank: 4, qk_rope_head_dim: 2 };
        let mut cache = MetalKvCache::with_group_size(&ctx, spec, 1, n, 4).unwrap();
        cache.append_layer_mla_tensor(&ctx, 0, &compressed_kv_t, &k_rope_t).unwrap();
        let cached = metal_ops::mla::mla_attention_with_cache_tensor(&ctx, &q_t, &cache, metal_ops::mla::KvBWeight::Fp8(&kv_b_fp8), 2, 2, 0).unwrap();

        assert_eq!(cached.rows, n);
        assert_eq!(cached.cols, 8);
        let direct_f = ctx.tensor_to_f32(&direct);
        let cached_f = ctx.tensor_to_f32(&cached);
        // INT8 量化 + causal attention 的复合误差,宽松用 10% 相对误差。
        for (d, c) in direct_f.iter().zip(cached_f.iter()) {
            assert!(d.is_finite() && c.is_finite(), "出现 NaN/Inf: direct={d}, cached={c}");
            let rel = (d - c).abs() / d.abs().max(1e-3);
            assert!(rel < 0.10, "cache vs direct 误差过大: direct={d}, cached={c}, rel={rel}");
        }
    }

    #[test]
    fn decode_attention_absorption_matches_reconstruct() {
        // 阶段 A 核心验收:decode absorption 路径(GPU)vs 标准重建 attention(CPU 基准)。
        // 小 MLA:heads=2, qk_nope=2, rope=2, kv_lora=4, value_dim=2, kv_head_dim=qk_nope+value=4。
        // q_proj = 2×4=8, kv_proj = 2×4=8, kv_b_proj [8,4]。group_size=4。
        // N=2 历史 token,position=2,query=1 token。
        let Some(ctx) = make_ctx() else { return };
        use crate::kernel::metal as metal_ops;
        use crate::weight::Fp8Matrix;

        const HEADS: usize = 2;
        const QK_NOPE: usize = 2;
        const ROPE: usize = 2;
        const KV_LORA: usize = 4;
        const VALUE: usize = 2;
        const KV_HEAD: usize = QK_NOPE + VALUE; // 4
        const Q_HEAD: usize = QK_NOPE + ROPE; // 4
        const Q_PROJ: usize = HEADS * Q_HEAD; // 8
        const KV_PROJ: usize = HEADS * KV_HEAD; // 8
        const N: usize = 2; // 历史 token 数

        // latent [N, kv_lora=4](值小利 INT8)。
        let latent: Vec<f32> = (0..N * KV_LORA).map(|i| (i as f32 + 1.0) * 0.1 - 0.2).collect();
        // k_rope [N, ROPE=2]。
        let k_rope: Vec<f32> = (0..N * ROPE).map(|i| (i as f32 + 1.0) * 0.05 - 0.05).collect();
        // q [1, Q_PROJ=8]。
        let q: Vec<f32> = (0..Q_PROJ).map(|i| (i as f32 + 1.0) * 0.03 - 0.1).collect();
        // kv_b_proj [KV_PROJ=8, KV_LORA=4] = [out=8, in=4]。构造可解的对角 + 小随机。
        let mut kv_b = vec![0.0_f32; KV_PROJ * KV_LORA];
        for r in 0..KV_PROJ {
            for c in 0..KV_LORA {
                // 小值,让 FP8 编码误差小。
                kv_b[r * KV_LORA + c] = if r % KV_LORA == c { 0.5 } else { 0.02 * ((r + c) as f32) };
            }
        }
        // FP8 scale_inv:每 128×128 block 一个 f32。rows=8/128=1, cols=4/128=1 → 1 个 scale。
        let kv_b_fp8 = Fp8Matrix::new(kv_b.iter().map(|&v| (v.clamp(-1.0, 1.0) * 127.0) as i8 as u8).collect(), 1.0f32.to_le_bytes().to_vec(), KV_PROJ, KV_LORA).expect("Fp8Matrix 构造");

        // ---- CPU 基准:标准重建 attention(query=1,看 N 个历史) ----
        // expanded_kv[t] = latent[t] × kv_b_proj^T → [KV_PROJ]。
        // kv_b_proj 用 FP8 解码还原(和 GPU 同源,排除 FP8 编码差异)。
        let kv_b_decoded = kv_b_fp8.decode();
        let mut expanded_kv = vec![0.0_f32; N * KV_PROJ];
        for t in 0..N {
            for out in 0..KV_PROJ {
                let mut s = 0.0;
                for c in 0..KV_LORA {
                    s += latent[t * KV_LORA + c] * kv_b_decoded[out * KV_LORA + c];
                }
                expanded_kv[t * KV_PROJ + out] = s;
            }
        }
        let scale = 1.0f32 / (Q_HEAD as f32).sqrt();
        let mut cpu_out = vec![0.0_f32; HEADS * VALUE];
        for h in 0..HEADS {
            // 每 head 的 q 切片:[q_nope, q_rope],expanded_kv 切片:[k_nope, v]。
            let q_base = h * Q_HEAD;
            let kv_base_head = h * KV_HEAD;
            // scores per token。
            let mut scores = vec![0.0_f32; N];
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..N {
                let mut s = 0.0;
                // nope 点积。
                for j in 0..QK_NOPE {
                    s += q[q_base + j] * expanded_kv[t * KV_PROJ + kv_base_head + j];
                }
                // rope 点积(q 的 rope 段 vs cache k_rope)。
                for j in 0..ROPE {
                    s += q[q_base + QK_NOPE + j] * k_rope[t * ROPE + j];
                }
                s *= scale;
                scores[t] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            // softmax。
            let mut sum = 0.0;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            // value 加权:expanded_kv 的 nope 之后是 value。
            for v in 0..VALUE {
                let mut acc = 0.0;
                for t in 0..N {
                    acc += scores[t] * expanded_kv[t * KV_PROJ + kv_base_head + QK_NOPE + v];
                }
                cpu_out[h * VALUE + v] = acc;
            }
        }

        // ---- GPU absorption 路径 ----
        let q_t = ctx.tensor_from_f32(&q, 1, Q_PROJ).unwrap();
        let latent_t = ctx.tensor_from_f32(&latent, N, KV_LORA).unwrap();
        let k_rope_t = ctx.tensor_from_f32(&k_rope, N, ROPE).unwrap();
        let spec = KvCacheSpec::Mla { kv_lora_rank: KV_LORA, qk_rope_head_dim: ROPE };
        let caches = [MetalKvCache::with_group_size(&ctx, spec.clone(), 1, N, KV_LORA).unwrap(), MetalKvCache::new_f16(&ctx, spec, 1, N).unwrap()];
        let selection = ctx.shared_buffer(&[0, 0, 0, 0, 1, 0, 0, 0]);
        for mut cache in caches {
            cache.append_layer_mla_tensor(&ctx, 0, &latent_t, &k_rope_t).unwrap();
            let full = metal_ops::mla::mla_decode_attention(&ctx, &q_t, &cache, metal_ops::mla::KvBWeight::Fp8(&kv_b_fp8), 0, N, HEADS, QK_NOPE, ROPE, VALUE).unwrap();
            let selected = metal_ops::mla::mla_decode_attention_selected(&ctx, &q_t, &cache, metal_ops::mla::KvBWeight::Fp8(&kv_b_fp8), 0, N, HEADS, QK_NOPE, ROPE, VALUE, &selection, N).unwrap();
            for (path, gpu) in [("full", full), ("selected", selected)] {
                assert_eq!(gpu.rows, 1);
                assert_eq!(gpu.cols, HEADS * VALUE);
                let gpu_f = ctx.tensor_to_f32(&gpu);
                let tolerance = if cache.format() == MetalKvCacheFormat::F16 { 0.05 } else { 0.15 };
                for (i, (c, g)) in cpu_out.iter().zip(gpu_f.iter()).enumerate() {
                    assert!(c.is_finite() && g.is_finite(), "NaN/Inf: cpu={c}, gpu={g}");
                    let rel = (c - g).abs() / c.abs().max(1e-3);
                    assert!(rel < tolerance, "{:?} {path} decode absorption vs 重建误差过大 [i={i}]: cpu={c}, gpu={g}, rel={rel}", cache.format());
                }
            }
        }
    }
}
