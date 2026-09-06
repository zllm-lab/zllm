//! Metal MoE decode 的专家驻留、预测预取与异步 I/O 能力。

use crate::backend::metal::api as metal;

use std::{
    collections::{BTreeMap, VecDeque},
    mem,
    sync::Arc,
    thread,
    time::Instant,
};

use crate::{
    backend::{Backend, BackendError, BackendResources, ExpertDecodeBackend, ExpertPrefillBackend, LinearWeight, MoePrefillBackend, MoePrefillRouting},
    kernel::metal as ops,
    moe::{
        routing::{ExpertAssignments, execute_routed_experts},
        topk_moe::{ScoringFunc, TopkMoeSpec},
    },
    weight::{
        container::gguf::GgufMatrix,
        expert_source::{GgufExpertSource, Mxfp4ExpertSource, Nvfp4ExpertSource, W4A16ExpertSource},
        format::nvfp4::{Nvfp4ArchiveLayout, Nvfp4ExpertBufferMut, Nvfp4MatrixBufferMut, NvidiaNvfp4Experts, nvfp4_storage_lengths},
        format::official_fp8::OfficialExpertArchive,
    },
};

use super::{MetalContext, MetalTensor, MetalWeight};

pub enum MetalPrefillExperts {
    Fp8(OfficialExpertArchive),
    Mxfp4 { source: Arc<dyn Mxfp4ExpertSource>, activation: MetalMxfp4ActivationMode },
    W4A16(Arc<dyn W4A16ExpertSource>),
    Nvfp4(NvidiaNvfp4Experts),
    Gguf { source: Arc<dyn GgufExpertSource>, decode_state: Option<MetalMoeDecodeState> },
}

impl MetalPrefillExperts {
    pub fn fp8(root: &std::path::Path, intermediate: usize, hidden: usize, expert_count: usize) -> Result<Self, String> {
        Ok(Self::Fp8(OfficialExpertArchive::open(root, intermediate, hidden, expert_count)?))
    }

    pub fn nvfp4(source: NvidiaNvfp4Experts) -> Self {
        Self::Nvfp4(source)
    }

    pub fn mxfp4(source: Arc<dyn Mxfp4ExpertSource>) -> Self {
        Self::Mxfp4 { source, activation: MetalMxfp4ActivationMode::F16 }
    }

    pub fn mxfp4_mxfp8(source: Arc<dyn Mxfp4ExpertSource>) -> Self {
        Self::Mxfp4 { source, activation: MetalMxfp4ActivationMode::Mxfp8 }
    }

    pub fn w4a16(source: Arc<dyn W4A16ExpertSource>) -> Self {
        Self::W4A16(source)
    }

    pub fn gguf(source: Arc<dyn GgufExpertSource>) -> Self {
        Self::Gguf { source, decode_state: None }
    }

    /// prefill 首次触达 expert 时直接填充 decode cache，避免随后重复上传。
    pub fn gguf_resident(source: Arc<dyn GgufExpertSource>, decode_state: MetalMoeDecodeState) -> Self {
        Self::Gguf { source, decode_state: Some(decode_state) }
    }

    /// prefill 结束后把已填充的资源所有权交给 decode。
    pub fn take_gguf_decode_state(&mut self) -> Option<MetalMoeDecodeState> {
        match self {
            Self::Gguf { decode_state, .. } => decode_state.take(),
            _ => None,
        }
    }
}

fn metal_moe_route(ctx: &MetalContext, input: &MetalTensor, router_weight: &MetalWeight, router_bias: &MetalWeight, spec: &TopkMoeSpec) -> Result<ops::moe::MetalRouting, BackendError> {
    match (spec.scoring_func, router_weight, router_bias) {
        (ScoringFunc::Softmax, MetalWeight::F16(weight), _) => ops::moe::moe_router_softmax_tensor_resident(ctx, input, weight, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected),
        (ScoringFunc::Softmax, MetalWeight::F32 { buffer: weight, len: weight_len }, _) => {
            ops::moe::moe_router_softmax_tensor_resident_f32(ctx, input, weight, *weight_len, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected)
        }
        (ScoringFunc::SigmoidBias, MetalWeight::F16(weight), MetalWeight::F16(bias)) => ops::moe::moe_router_tensor_resident_f16_bias(ctx, input, weight, bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor),
        (ScoringFunc::SigmoidBias, MetalWeight::F32 { buffer: weight, len: weight_len }, MetalWeight::F32 { buffer: bias, len: bias_len }) => {
            ops::moe::moe_router_tensor_resident_f32(ctx, input, weight, *weight_len, bias, *bias_len, spec.num_experts, spec.top_k, spec.routed_scaling_factor, 1)
        }
        (ScoringFunc::SqrtSoftplusBias, MetalWeight::F32 { buffer: weight, len: weight_len }, MetalWeight::F32 { buffer: bias, len: bias_len }) => {
            ops::moe::moe_router_tensor_resident_f32(ctx, input, weight, *weight_len, bias, *bias_len, spec.num_experts, spec.top_k, spec.routed_scaling_factor, 2)
        }
        _ => return Err(BackendError::Compute { msg: "MoE router resident weight 格式不受支持".to_owned() }),
    }
    .map_err(|msg| BackendError::Compute { msg })
}

impl MoePrefillBackend for MetalContext {
    type MoeAccumulator = ops::moe::MetalF32Accumulator;

    fn moe_route(&self, input: &MetalTensor, router_weight: &MetalWeight, router_bias: &MetalWeight, spec: &TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        let routing = metal_moe_route(self, input, router_weight, router_bias, spec)?;
        Ok(MoePrefillRouting { expert_ids: routing.expert_ids, weights: routing.weights, rows: routing.rows, top_k: routing.top_k })
    }

    fn moe_route_selected(&self, input: &MetalTensor, router_weight: &MetalWeight, selected_experts: &[u32], spec: &TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        let MetalWeight::F32 { buffer, len } = router_weight else {
            return Err(BackendError::Compute { msg: "Metal fixed-expert router 需要 F32 权重".to_owned() });
        };
        if spec.scoring_func != ScoringFunc::SqrtSoftplusBias {
            return Err(BackendError::Compute { msg: format!("Metal fixed-expert router 不支持 {:?}", spec.scoring_func) });
        }
        let routing = ops::moe::moe_router_sqrt_softplus_selected_f32(self, input, buffer, *len, selected_experts, spec.num_experts, spec.top_k, spec.routed_scaling_factor).map_err(|msg| BackendError::Compute { msg })?;
        Ok(MoePrefillRouting { expert_ids: routing.expert_ids, weights: routing.weights, rows: routing.rows, top_k: routing.top_k })
    }

    fn moe_zeros(&self, rows: usize, cols: usize) -> Result<Self::MoeAccumulator, BackendError> {
        ops::moe::moe_accumulator_zeros(self, rows, cols).map_err(|msg| BackendError::Compute { msg })
    }

    fn moe_gather_rows(&self, input: &MetalTensor, rows: &[u32]) -> Result<MetalTensor, BackendError> {
        let input = ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
        ops::moe::gather_rows_tensor_f16(self, &input, rows).map_err(|msg| BackendError::Compute { msg })
    }

    fn moe_gather_rows_batch(&self, input: &MetalTensor, rows: &[Vec<u32>]) -> Result<Vec<MetalTensor>, BackendError> {
        let input = ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
        ops::moe::gather_rows_batch_tensor_f16(self, &input, rows).map_err(|msg| BackendError::Compute { msg })
    }

    fn moe_scatter_add_rows(&self, output: &mut Self::MoeAccumulator, input: &MetalTensor, rows: &[u32], weights: &[f32]) -> Result<(), BackendError> {
        ops::moe::scatter_add_rows_weighted_f32(self, output, input, rows, weights).map_err(|msg| BackendError::Compute { msg })
    }

    fn moe_scatter_add_rows_batch(&self, output: &mut Self::MoeAccumulator, inputs: &[MetalTensor], rows: &[Vec<u32>], weights: &[Vec<f32>]) -> Result<(), BackendError> {
        ops::moe::scatter_add_rows_batch_weighted_f32(self, output, inputs, rows, weights).map_err(|msg| BackendError::Compute { msg })
    }

    fn moe_finish(&self, output: Self::MoeAccumulator) -> Result<MetalTensor, BackendError> {
        ops::moe::finish_moe_accumulator(self, output, None).map_err(|msg| BackendError::Compute { msg })
    }
}

impl ExpertPrefillBackend for MetalContext {
    type PrefillExperts = MetalPrefillExperts;

    fn prefill_routed_experts(&self, spec: &TopkMoeSpec, layer: usize, experts: &mut MetalPrefillExperts, input: &MetalTensor, assignments: &ExpertAssignments) -> Result<Option<MetalTensor>, BackendError> {
        let routed = match experts {
            MetalPrefillExperts::Fp8(archive) => fp8_routed(self, spec, layer, archive, input, assignments)?,
            MetalPrefillExperts::Mxfp4 { source, activation } => mxfp4_routed(self, spec, layer, source.as_ref(), input, assignments, *activation)?,
            MetalPrefillExperts::W4A16(source) => w4a16_routed(self, spec, layer, source.as_ref(), input, assignments)?.0,
            MetalPrefillExperts::Nvfp4(source) => nvfp4_prefill_routed(self, spec, layer, source, input, assignments)?,
            MetalPrefillExperts::Gguf { source, decode_state } => {
                let cache = decode_state.as_mut().map(|state| &mut state.gguf_cache);
                gguf_routed(self, spec, layer, source.as_ref(), input, assignments, cache)?
            }
        };
        Ok(Some(routed))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MetalMoeDecodeStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub predicted: usize,
    pub prefetch_requested: usize,
    pub prefetch_hits: usize,
    pub prefetch_elapsed_seconds: f64,
    pub prefetch_wait_seconds: f64,
    pub sync_expert_read_seconds: f64,
    pub sync_expert_wait_seconds: f64,
    pub expert_upload_seconds: f64,
    pub resident_experts: usize,
    pub resident_bytes: usize,
}

pub struct MetalMoeDecodeState {
    gguf_cache: MetalGgufDecodeCache,
    nvfp4_prefetch: Option<MetalNvfp4Prefetch>,
    nvfp4_recycle: Option<MetalNvfp4Layer>,
    stats: MetalMoeDecodeStats,
    mxfp4_activation: MetalMxfp4ActivationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalMxfp4ActivationMode {
    F16,
    Mxfp8,
}

pub struct MetalDecodeRouting {
    expert_ids: metal::Buffer,
    weights: metal::Buffer,
    top_k: usize,
}

struct MetalNvfp4Prefetch {
    layer: usize,
    experts: Vec<usize>,
    layer_experts: MetalNvfp4Layer,
}

impl MetalMoeDecodeState {
    pub fn new() -> Result<Self, String> {
        Self::with_gguf_cache_gib(12)
    }

    pub fn with_gguf_cache_gib(gib: usize) -> Result<Self, String> {
        if gib == 0 {
            return Err("GGUF expert cache 至少需要 1 GiB".to_owned());
        }
        Ok(Self { gguf_cache: MetalGgufDecodeCache::with_max_bytes(gib.saturating_mul(1024 * 1024 * 1024)), nvfp4_prefetch: None, nvfp4_recycle: None, stats: MetalMoeDecodeStats::default(), mxfp4_activation: MetalMxfp4ActivationMode::F16 })
    }

    pub fn with_mxfp8_activations(mut self) -> Self {
        self.mxfp4_activation = MetalMxfp4ActivationMode::Mxfp8;
        self
    }

    /// 只检查 GGUF expert 的元数据、shape 与 Metal 解码能力，不读取权重正文。
    /// Ornith 在模型加载期调用它，避免长 prompt prefill 完成后才发现某层 expert
    /// 使用了没有 Metal 算子的量化类型。
    pub fn validate_gguf_expert_formats(source: &dyn GgufExpertSource, layer_count: usize, expert_count: usize) -> Result<(), BackendError> {
        Self::validate_gguf_expert_range(source, 0, layer_count, expert_count)
    }

    pub fn validate_gguf_expert_range(source: &dyn GgufExpertSource, first_layer: usize, layer_count: usize, expert_count: usize) -> Result<(), BackendError> {
        let hidden = source.hidden();
        let intermediate = source.intermediate();
        for layer in first_layer..layer_count {
            for expert in 0..expert_count {
                let weights = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
                for (projection, matrix, rows, cols) in [("gate", &weights.gate, intermediate, hidden), ("up", &weights.up, intermediate, hidden), ("down", &weights.down, hidden, intermediate)] {
                    if matrix.rows != rows || matrix.columns != cols {
                        return Err(BackendError::ExpertLoad(format!("GGUF L{layer} E{expert} {projection} shape=[{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
                    }
                    if !crate::weight::codec::ggml::supports_decode(matrix.tensor_type.0) {
                        return Err(BackendError::ExpertLoad(format!("GGUF L{layer} E{expert} {projection} type={} 没有 Metal expert 算子", matrix.tensor_type.name())));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn preload_gguf_experts(&mut self, ctx: &MetalContext, source: &dyn GgufExpertSource, layer_count: usize, expert_count: usize) -> Result<(usize, usize), BackendError> {
        self.preload_gguf_expert_range(ctx, source, 0, layer_count, expert_count)
    }

    pub fn preload_gguf_expert_range(&mut self, ctx: &MetalContext, source: &dyn GgufExpertSource, first_layer: usize, layer_count: usize, expert_count: usize) -> Result<(usize, usize), BackendError> {
        if first_layer >= layer_count {
            return Err(BackendError::ExpertLoad(format!("GGUF expert layer range {first_layer}..{layer_count} 为空")));
        }
        for layer in first_layer..layer_count {
            for expert in 0..expert_count {
                self.gguf_cache.ensure(ctx, source, layer, expert)?;
            }
        }
        for layer in first_layer..layer_count {
            self.gguf_cache.pack_layer(ctx, layer, expert_count)?;
        }
        ops::moe::prewarm_resident_decode_router(ctx).map_err(|msg| BackendError::Compute { msg })?;
        for weights in self.gguf_cache.layers.iter().flatten() {
            ops::gguf::prewarm_indexed_experts(ctx, &weights.gate, &weights.down).map_err(|msg| BackendError::Compute { msg })?;
            ops::gguf::prefetch_indexed_expert_buffers(ctx, &weights.gate, &weights.up, &weights.down).map_err(|msg| BackendError::Compute { msg })?;
        }
        // CPU packing 会让 GPU 量化路径降频；丢弃一次零输入结果，只预热 resident 资源，不触碰模型状态。
        let warmup_top_k = expert_count.min(8);
        if warmup_top_k != 0 {
            let warmup_input = ctx.tensor_zeros(1, source.hidden());
            let warmup_ids = (0..warmup_top_k).map(|slot| (slot * expert_count / warmup_top_k) as u32).collect::<Vec<_>>();
            let warmup_weights = vec![0.0f32; warmup_top_k];
            let id_bytes = unsafe { std::slice::from_raw_parts(warmup_ids.as_ptr().cast::<u8>(), std::mem::size_of_val(warmup_ids.as_slice())) };
            let weight_bytes = unsafe { std::slice::from_raw_parts(warmup_weights.as_ptr().cast::<u8>(), std::mem::size_of_val(warmup_weights.as_slice())) };
            let id_buffer = ctx.shared_buffer(id_bytes);
            let weight_buffer = ctx.shared_buffer(weight_bytes);
            let mut warmup_outputs = Vec::with_capacity(layer_count - first_layer);
            for weights in self.gguf_cache.layers[first_layer..layer_count].iter().flatten() {
                warmup_outputs.push(
                    ops::gguf::gguf_indexed_experts_tensor_resident(ctx, &warmup_input, &weights.gate, &weights.up, &weights.down, &id_buffer, &weight_buffer, warmup_top_k, &crate::backend::Activation::Silu)
                        .map_err(|msg| BackendError::Compute { msg })?,
                );
            }
            ctx.synchronize();
            drop(warmup_outputs);
        }
        self.gguf_cache.fully_resident = self.gguf_cache.len() == (layer_count - first_layer).saturating_mul(expert_count);
        if self.gguf_cache.fully_resident {
            self.gguf_cache.lru.clear();
        }
        self.stats.resident_experts = self.gguf_cache.len();
        self.stats.resident_bytes = self.gguf_cache.bytes;
        Ok((self.gguf_cache.len(), self.gguf_cache.bytes))
    }

    pub fn take_stats(&mut self) -> MetalMoeDecodeStats {
        std::mem::take(&mut self.stats)
    }
}

struct MetalGgufExpertWeights {
    gate: MetalWeight,
    up: MetalWeight,
    down: MetalWeight,
    bytes: usize,
}

impl MetalGgufExpertWeights {
    fn load(ctx: &MetalContext, source: &dyn GgufExpertSource, layer: usize, expert: usize) -> Result<Self, BackendError> {
        let matrices = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
        Self::upload(ctx, &matrices.gate, &matrices.up, &matrices.down)
    }

    fn upload(ctx: &MetalContext, gate: &GgufMatrix, up: &GgufMatrix, down: &GgufMatrix) -> Result<Self, BackendError> {
        let gate_weight = MetalWeight::allocate_gguf(ctx, gate, gate.rows, gate.columns).map_err(BackendError::ExpertLoad)?;
        let up_weight = MetalWeight::allocate_gguf(ctx, up, up.rows, up.columns).map_err(BackendError::ExpertLoad)?;
        let down_weight = MetalWeight::allocate_gguf(ctx, down, down.rows, down.columns).map_err(BackendError::ExpertLoad)?;
        std::thread::scope(|scope| {
            let gate_read = scope.spawn(|| gate_weight.fill_gguf(gate));
            let up_read = scope.spawn(|| up_weight.fill_gguf(up));
            let down_result = down_weight.fill_gguf(down);
            let gate_result = gate_read.join().map_err(|_| "Metal GGUF gate 读取线程 panic".to_owned())?;
            let up_result = up_read.join().map_err(|_| "Metal GGUF up 读取线程 panic".to_owned())?;
            down_result?;
            gate_result?;
            up_result?;
            Ok::<(), String>(())
        })
        .map_err(BackendError::ExpertLoad)?;
        let (gate, up, down) = (gate_weight, up_weight, down_weight);
        let bytes = [&gate, &up, &down]
            .into_iter()
            .map(|weight| match weight {
                MetalWeight::Gguf { blob, .. } => blob.length() as usize,
                _ => 0,
            })
            .sum();
        Ok(Self { gate, up, down, bytes })
    }
}

fn pack_gguf_projection(ctx: &MetalContext, weights: Vec<&MetalWeight>) -> Result<MetalWeight, BackendError> {
    let Some(MetalWeight::Gguf { tensor_type, row_bytes, rows, cols, .. }) = weights.first().copied() else {
        return Err(BackendError::ExpertLoad("GGUF layer pack 收到非 GGUF 权重".to_owned()));
    };
    let expert_bytes = row_bytes.checked_mul(*rows).ok_or_else(|| BackendError::ExpertLoad("GGUF layer expert 大小溢出".to_owned()))?;
    let total_bytes = expert_bytes.checked_mul(weights.len()).ok_or_else(|| BackendError::ExpertLoad("GGUF layer pack 大小溢出".to_owned()))?;
    let packed = ctx.shared_buffer_uninit(total_bytes);
    for (expert, weight) in weights.iter().copied().enumerate() {
        let MetalWeight::Gguf { blob, tensor_type: actual_type, row_bytes: actual_row_bytes, rows: actual_rows, cols: actual_cols } = weight else {
            return Err(BackendError::ExpertLoad("GGUF layer projection 格式不一致".to_owned()));
        };
        if actual_type != tensor_type || actual_row_bytes != row_bytes || actual_rows != rows || actual_cols != cols || blob.length() as usize != expert_bytes {
            return Err(BackendError::ExpertLoad(format!(
                "GGUF layer projection E{expert} 布局不一致: type={actual_type}/{tensor_type}, row_bytes={actual_row_bytes}/{row_bytes}, shape=[{actual_rows},{actual_cols}]/[{rows},{cols}], bytes={}/{expert_bytes}",
                blob.length(),
            )));
        }
    }
    let worker_count = weights.len().min(4);
    let chunk_size = weights.len().div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in weights.chunks(chunk_size).enumerate() {
            let packed = &packed;
            workers.push(scope.spawn(move || {
                for (offset, weight) in chunk.iter().copied().enumerate() {
                    let MetalWeight::Gguf { blob, .. } = weight else { unreachable!("GGUF layer projection 已校验") };
                    let expert = chunk_index * chunk_size + offset;
                    unsafe {
                        std::ptr::copy_nonoverlapping(blob.contents().cast::<u8>(), packed.contents().cast::<u8>().add(expert * expert_bytes), expert_bytes);
                    }
                }
            }));
        }
        for worker in workers {
            worker.join().map_err(|_| BackendError::ExpertLoad("Metal GGUF layer pack 线程 panic".to_owned()))?;
        }
        Ok::<(), BackendError>(())
    })?;
    Ok(MetalWeight::Gguf { blob: packed, tensor_type: *tensor_type, row_bytes: *row_bytes, rows: *rows, cols: *cols })
}

struct MetalGgufCacheEntry {
    weights: MetalGgufExpertWeights,
    generation: u64,
}

struct MetalGgufDecodeCache {
    entries: Vec<Vec<Option<MetalGgufCacheEntry>>>,
    layers: Vec<Option<MetalGgufExpertWeights>>,
    entry_count: usize,
    packed_experts: usize,
    lru: VecDeque<((usize, usize), u64)>,
    bytes: usize,
    max_bytes: usize,
    generation: u64,
    fully_resident: bool,
}

impl MetalGgufDecodeCache {
    fn with_max_bytes(max_bytes: usize) -> Self {
        Self { entries: Vec::new(), layers: Vec::new(), entry_count: 0, packed_experts: 0, lru: VecDeque::new(), bytes: 0, max_bytes, generation: 0, fully_resident: false }
    }

    fn len(&self) -> usize {
        self.entry_count + self.packed_experts
    }

    fn layer(&self, layer: usize) -> Option<&MetalGgufExpertWeights> {
        self.layers.get(layer)?.as_ref()
    }

    fn pack_layer(&mut self, ctx: &MetalContext, layer: usize, expert_count: usize) -> Result<bool, BackendError> {
        if self.layer(layer).is_some() {
            return Ok(true);
        }
        let experts = (0..expert_count).map(|expert| self.get(layer, expert)).collect::<Result<Vec<_>, _>>()?;
        let supported = experts.iter().all(|weights| {
            matches!(
                (&weights.gate, &weights.up, &weights.down),
                (
                    MetalWeight::Gguf { tensor_type: gate_type, .. },
                    MetalWeight::Gguf { tensor_type: up_type, .. },
                    MetalWeight::Gguf { tensor_type: down_type, .. },
                ) if gate_type == up_type
                    && crate::weight::codec::ggml::supports_decode(*gate_type)
                    && crate::weight::codec::ggml::supports_decode(*down_type)
            )
        });
        if !supported {
            return Ok(false);
        }
        let gate = pack_gguf_projection(ctx, experts.iter().map(|weights| &weights.gate).collect())?;
        let up = pack_gguf_projection(ctx, experts.iter().map(|weights| &weights.up).collect())?;
        let down = pack_gguf_projection(ctx, experts.iter().map(|weights| &weights.down).collect())?;
        drop(experts);

        let bytes = [&gate, &up, &down]
            .into_iter()
            .map(|weight| match weight {
                MetalWeight::Gguf { blob, .. } => blob.length() as usize,
                _ => 0,
            })
            .sum();
        let mut removed_bytes = 0usize;
        for expert in 0..expert_count {
            if let Some(entry) = self.remove(layer, expert) {
                removed_bytes = removed_bytes.saturating_add(entry.weights.bytes);
            }
        }
        self.bytes = self.bytes.saturating_sub(removed_bytes).saturating_add(bytes);
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
        }
        self.layers[layer] = Some(MetalGgufExpertWeights { gate, up, down, bytes });
        self.packed_experts += expert_count;
        Ok(true)
    }

    fn entry(&self, layer: usize, expert: usize) -> Option<&MetalGgufCacheEntry> {
        self.entries.get(layer)?.get(expert)?.as_ref()
    }

    fn entry_mut(&mut self, layer: usize, expert: usize) -> Option<&mut MetalGgufCacheEntry> {
        self.entries.get_mut(layer)?.get_mut(expert)?.as_mut()
    }

    fn insert(&mut self, layer: usize, expert: usize, entry: MetalGgufCacheEntry) {
        if self.entries.len() <= layer {
            self.entries.resize_with(layer + 1, Vec::new);
        }
        if self.entries[layer].len() <= expert {
            self.entries[layer].resize_with(expert + 1, || None);
        }
        debug_assert!(self.entries[layer][expert].is_none());
        self.entries[layer][expert] = Some(entry);
        self.entry_count += 1;
    }

    fn remove(&mut self, layer: usize, expert: usize) -> Option<MetalGgufCacheEntry> {
        let removed = self.entries.get_mut(layer)?.get_mut(expert)?.take();
        if removed.is_some() {
            self.entry_count -= 1;
        }
        removed
    }

    fn touch(&mut self, layer: usize, expert: usize) -> bool {
        let key = (layer, expert);
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        if let Some(entry) = self.entry_mut(layer, expert) {
            entry.generation = generation;
            self.lru.push_back((key, generation));
            return true;
        }
        false
    }

    fn insert_loaded(&mut self, layer: usize, expert: usize, weights: MetalGgufExpertWeights) -> Result<(), BackendError> {
        let bytes = weights.bytes;
        // 单条超过上限时跳过缓存而不是报错或清空 LRU：保留已有条目，调用方随后的 get 会以"cache 缺少"报出具体 L/E。
        if bytes > self.max_bytes {
            return Ok(());
        }
        while self.bytes.saturating_add(bytes) > self.max_bytes {
            let Some((old_key, old_generation)) = self.lru.pop_front() else {
                break;
            };
            let current = self.entry(old_key.0, old_key.1).map(|entry| entry.generation);
            if current != Some(old_generation) {
                continue;
            }
            if let Some(removed) = self.remove(old_key.0, old_key.1) {
                self.bytes = self.bytes.saturating_sub(removed.weights.bytes);
            }
        }
        // LRU 清空后仍放不下（如 packed 层字节不可驱逐）时跳过插入，避免缓存静默突破 max_bytes。
        if self.bytes.saturating_add(bytes) > self.max_bytes {
            return Ok(());
        }
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.bytes = self.bytes.saturating_add(bytes);
        self.insert(layer, expert, MetalGgufCacheEntry { weights, generation });
        self.lru.push_back(((layer, expert), generation));
        Ok(())
    }

    fn ensure(&mut self, ctx: &MetalContext, source: &dyn GgufExpertSource, layer: usize, expert: usize) -> Result<(bool, f64, f64), BackendError> {
        if self.touch(layer, expert) {
            return Ok((true, 0.0, 0.0));
        }
        let started = Instant::now();
        let matrices = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
        let weights = MetalGgufExpertWeights::upload(ctx, &matrices.gate, &matrices.up, &matrices.down)?;
        let elapsed = started.elapsed().as_secs_f64();
        self.insert_loaded(layer, expert, weights)?;
        Ok((false, elapsed, 0.0))
    }

    fn get(&self, layer: usize, expert: usize) -> Result<&MetalGgufExpertWeights, BackendError> {
        self.entry(layer, expert).map(|entry| &entry.weights).ok_or_else(|| BackendError::ExpertLoad(format!("GGUF expert cache 缺少 L{layer} E{expert}")))
    }
}

impl ExpertDecodeBackend for MetalContext {
    type MoeState = MetalMoeDecodeState;
    type DecodeRouting = MetalDecodeRouting;

    fn decode_resident_routed_experts(
        &self,
        spec: &TopkMoeSpec,
        weights: crate::moe::topk_moe::RoutedMoeWeightsRef<'_, MetalWeight>,
        layer: usize,
        source: crate::weight::expert_source::ExpertSource<'_>,
        state: &mut Self::MoeState,
        inputs: crate::moe::topk_moe::RoutedMoeInputs<'_, MetalTensor>,
    ) -> Result<Option<(MetalTensor, Option<Vec<u16>>)>, BackendError> {
        let router_weight = weights.router;
        let router_bias = weights.bias;
        let route_input = inputs.route;
        let expert_input = inputs.expert;
        if !matches!(source, crate::weight::expert_source::ExpertSource::Gguf(_)) || route_input.rows != 1 || expert_input.rows != 1 {
            return Ok(None);
        }
        let MetalWeight::F32 { buffer: router_weight, len: router_weight_len } = router_weight else {
            return Ok(None);
        };
        let Some(expert_weights) = state.gguf_cache.layer(layer) else {
            return Ok(None);
        };
        let (expert_ids, route_weights) = match (spec.scoring_func, router_bias) {
            (ScoringFunc::Softmax, _) => ops::moe::moe_router_softmax_decode_resident_f32(self, route_input, router_weight, *router_weight_len, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected),
            (ScoringFunc::SigmoidBias, MetalWeight::F32 { buffer: bias, len: bias_len }) => {
                ops::moe::moe_router_sigmoid_decode_resident_f32(self, route_input, router_weight, *router_weight_len, bias, *bias_len, spec.num_experts, spec.top_k, spec.routed_scaling_factor)
            }
            _ => return Ok(None),
        }
        .map_err(|msg| BackendError::Compute { msg })?;
        let output = ops::gguf::gguf_indexed_experts_tensor_resident(self, expert_input, &expert_weights.gate, &expert_weights.up, &expert_weights.down, &expert_ids, &route_weights, spec.top_k, &spec.activation)
            .map_err(|msg| BackendError::Compute { msg })?;
        state.stats.cache_hits += spec.top_k;
        state.stats.resident_experts = state.gguf_cache.len();
        state.stats.resident_bytes = state.gguf_cache.bytes;
        Ok(Some((output, None)))
    }

    fn decode_route(&self, input: &MetalTensor, router_weight: &MetalWeight, router_bias: &MetalWeight, spec: &TopkMoeSpec) -> Result<(MoePrefillRouting, Self::DecodeRouting), BackendError> {
        let mut routing = metal_moe_route(self, input, router_weight, router_bias, spec)?;
        for row in 0..routing.rows {
            let begin = row * routing.top_k;
            let end = begin + routing.top_k;
            let mut selected = routing.expert_ids[begin..end].iter().copied().zip(routing.weights[begin..end].iter().copied()).collect::<Vec<_>>();
            selected.sort_unstable_by_key(|&(expert, _)| expert);
            for (slot, (expert, weight)) in selected.into_iter().enumerate() {
                routing.expert_ids[begin + slot] = expert;
                routing.weights[begin + slot] = weight;
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(routing.expert_ids.as_ptr(), routing.expert_ids_buffer.contents().cast::<u32>(), routing.expert_ids.len());
            std::ptr::copy_nonoverlapping(routing.weights.as_ptr(), routing.weights_buffer.contents().cast::<f32>(), routing.weights.len());
        }
        let backend_routing = MetalDecodeRouting { expert_ids: routing.expert_ids_buffer, weights: routing.weights_buffer, top_k: routing.top_k };
        Ok((MoePrefillRouting { expert_ids: routing.expert_ids, weights: routing.weights, rows: routing.rows, top_k: routing.top_k }, backend_routing))
    }

    fn prefetch_experts(&self, spec: &TopkMoeSpec, state: &mut Self::MoeState, request: crate::backend::ExpertPrefetchRequest<'_>) -> Result<usize, BackendError> {
        let crate::backend::ExpertPrefetchRequest { layer, source, experts } = request;
        if experts.is_empty() {
            return Ok(0);
        }
        let crate::weight::expert_source::ExpertSource::Nvfp4(source) = source else {
            return Ok(0);
        };
        if state.nvfp4_prefetch.as_ref().is_some_and(|prefetch| prefetch.layer == layer && prefetch.experts.get(..experts.len()) == Some(experts.as_slice()) && prefetch.experts[experts.len()..].iter().all(|expert| *expert == usize::MAX)) {
            return Ok(0);
        }
        let started = Instant::now();
        let reuse = state.nvfp4_recycle.take();
        let layer_experts = MetalNvfp4Layer::load_selected_reusing(self, source, layer, spec.num_experts, &experts, spec.top_k, reuse)?;
        let elapsed = started.elapsed().as_secs_f64();
        let bytes = layer_experts.bytes();
        let requested = experts.len();
        let mut slot_experts = experts;
        slot_experts.resize(spec.top_k, usize::MAX);
        state.nvfp4_prefetch = Some(MetalNvfp4Prefetch { layer, experts: slot_experts, layer_experts });
        state.stats.predicted += requested;
        state.stats.prefetch_requested += requested;
        state.stats.prefetch_elapsed_seconds += elapsed;
        state.stats.resident_experts = requested;
        state.stats.resident_bytes = bytes;
        Ok(requested)
    }

    fn decode_routed_experts<'a, F>(
        &self,
        spec: &TopkMoeSpec,
        layer: usize,
        source: crate::weight::expert_source::ExpertSource<'_>,
        state: &mut Self::MoeState,
        input: &MetalTensor,
        assignments: &ExpertAssignments,
        routing: &Self::DecodeRouting,
        on_ready: F,
    ) -> Result<MetalTensor, BackendError>
    where
        F: FnOnce(&mut Self::MoeState) -> Result<Option<crate::backend::ExpertPrefetchRequest<'a>>, BackendError>,
    {
        let active = assignments.iter().filter(|rows| !rows.is_empty()).count();
        match source {
            crate::weight::expert_source::ExpertSource::Fp8(source) => {
                state.stats.cache_misses += active;
                if let Some(request) = on_ready(state)? {
                    self.prefetch_experts(spec, state, request)?;
                }
                fp8_routed(self, spec, layer, source, input, assignments)
            }
            crate::weight::expert_source::ExpertSource::Nvfp4(source) => {
                let active_experts: Vec<usize> = assignments.iter().enumerate().filter_map(|(expert, rows)| (!rows.is_empty()).then_some(expert)).collect();
                let prefetched = state.nvfp4_prefetch.take();
                let current_recycle = prefetched.is_none().then(|| state.nvfp4_recycle.take()).flatten();
                let request = on_ready(state)?;
                let mut next_prefetch = match request {
                    Some(request) => match request.source {
                        crate::weight::expert_source::ExpertSource::Nvfp4(next_source) => {
                            let next_reuse = state.nvfp4_recycle.take();
                            let arena = MetalNvfp4Layer::prepare_selected_reusing(self, next_source, request.layer, spec.num_experts, spec.top_k, next_reuse)?;
                            Some((request.layer, next_source, request.experts, arena))
                        }
                        _ => {
                            self.prefetch_experts(spec, state, request)?;
                            None
                        }
                    },
                    None => None,
                };
                let current_read_concurrency = NVFP4_SSD_READ_CONCURRENCY;
                let (output, layer_experts, cache_hits, sync_read_seconds, prefetch_elapsed_seconds, prefetch_wait_seconds) = thread::scope(|scope| {
                    let next_read = next_prefetch.as_ref().map(|(next_layer, next_source, experts, arena)| {
                        let next_layer = *next_layer;
                        let next_source = *next_source;
                        scope.spawn(move || {
                            let started = Instant::now();
                            arena.fill_selected(next_source, next_layer, spec.num_experts, experts, NVFP4_SSD_READ_CONCURRENCY)?;
                            Ok::<f64, BackendError>(started.elapsed().as_secs_f64())
                        })
                    });
                    let started = Instant::now();
                    let (layer_experts, slot_experts, cache_hits) = match prefetched {
                        Some(mut prefetch) if prefetch.layer == layer => match prefetch.layer_experts.align_selected(source, layer, spec.num_experts, &mut prefetch.experts, &active_experts, current_read_concurrency)? {
                            Some(hits) => (prefetch.layer_experts, prefetch.experts, hits),
                            None => {
                                let loaded = MetalNvfp4Layer::load_selected_reusing(self, source, layer, spec.num_experts, &active_experts, active_experts.len(), Some(prefetch.layer_experts))?;
                                (loaded, active_experts.clone(), 0)
                            }
                        },
                        prefetched => {
                            let reuse = prefetched.map(|prefetch| prefetch.layer_experts).or(current_recycle);
                            let loaded = MetalNvfp4Layer::load_selected_reusing(self, source, layer, spec.num_experts, &active_experts, active_experts.len(), reuse)?;
                            (loaded, active_experts.clone(), 0)
                        }
                    };
                    let cache_misses = active.saturating_sub(cache_hits);
                    let sync_read_seconds = if cache_misses != 0 { started.elapsed().as_secs_f64() } else { 0.0 };
                    let (output, layer_experts) = nvfp4_decode_routed(self, spec, layer, source, input, assignments, Some((layer_experts, slot_experts)))?;
                    let wait_started = Instant::now();
                    let prefetch_elapsed_seconds = match next_read {
                        Some(handle) => handle.join().map_err(|_| BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch thread panic")))??,
                        None => 0.0,
                    };
                    let prefetch_wait_seconds = wait_started.elapsed().as_secs_f64();
                    Ok::<_, BackendError>((output, layer_experts, cache_hits, sync_read_seconds, prefetch_elapsed_seconds, prefetch_wait_seconds))
                })?;
                let cache_misses = active.saturating_sub(cache_hits);
                state.stats.cache_hits += cache_hits;
                state.stats.cache_misses += cache_misses;
                state.stats.prefetch_hits += cache_hits;
                state.stats.sync_expert_read_seconds += sync_read_seconds;
                state.stats.prefetch_elapsed_seconds += prefetch_elapsed_seconds;
                state.stats.prefetch_wait_seconds += prefetch_wait_seconds;
                if let Some((next_layer, _, mut experts, arena)) = next_prefetch.take() {
                    let requested = experts.len();
                    experts.resize(spec.top_k, usize::MAX);
                    state.stats.predicted += requested;
                    state.stats.prefetch_requested += requested;
                    state.stats.resident_experts = requested;
                    state.stats.resident_bytes = arena.bytes();
                    state.nvfp4_prefetch = Some(MetalNvfp4Prefetch { layer: next_layer, experts, layer_experts: arena });
                }
                state.nvfp4_recycle = Some(layer_experts);
                Ok(output)
            }
            crate::weight::expert_source::ExpertSource::Gguf(source) => {
                let mut cache_hits = if state.gguf_cache.fully_resident { active } else { 0 };
                let mut read_seconds = 0.0;
                if !state.gguf_cache.fully_resident {
                    for (expert, rows) in assignments.iter().enumerate() {
                        if rows.is_empty() {
                            continue;
                        }
                        let (hit, read, _) = state.gguf_cache.ensure(self, source, layer, expert)?;
                        cache_hits += usize::from(hit);
                        read_seconds += read;
                    }
                }
                state.stats.cache_hits += cache_hits;
                state.stats.cache_misses += active - cache_hits;
                state.stats.sync_expert_read_seconds += read_seconds;
                state.stats.resident_experts = state.gguf_cache.len();
                state.stats.resident_bytes = state.gguf_cache.bytes;
                if let Some(request) = on_ready(state)? {
                    self.prefetch_experts(spec, state, request)?;
                }
                if input.rows == 1
                    && let Some(weights) = state.gguf_cache.layer(layer)
                {
                    return ops::gguf::gguf_indexed_experts_tensor_resident(self, input, &weights.gate, &weights.up, &weights.down, &routing.expert_ids, &routing.weights, routing.top_k, &spec.activation)
                        .map_err(|msg| BackendError::Compute { msg });
                }
                gguf_routed(self, spec, layer, source, input, assignments, Some(&mut state.gguf_cache))
            }
            crate::weight::expert_source::ExpertSource::Mxfp8(_) => Err(BackendError::ExpertLoad("Metal 通用 decode 尚未接入 MXFP8 expert arena".to_owned())),
            crate::weight::expert_source::ExpertSource::Mxfp4(source) => {
                state.stats.cache_misses += active;
                if let Some(request) = on_ready(state)? {
                    self.prefetch_experts(spec, state, request)?;
                }
                mxfp4_routed(self, spec, layer, source, input, assignments, state.mxfp4_activation)
            }
            crate::weight::expert_source::ExpertSource::W4A16(source) => {
                state.stats.cache_misses += active;
                if let Some(request) = on_ready(state)? {
                    self.prefetch_experts(spec, state, request)?;
                }
                let (output, read_seconds, wait_seconds, upload_seconds) = w4a16_routed(self, spec, layer, source, input, assignments)?;
                state.stats.sync_expert_read_seconds += read_seconds;
                state.stats.sync_expert_wait_seconds += wait_seconds;
                state.stats.expert_upload_seconds += upload_seconds;
                Ok(output)
            }
        }
    }
}

fn mxfp4_routed(
    ctx: &MetalContext,
    spec: &TopkMoeSpec,
    layer: usize,
    source: &dyn crate::weight::expert_source::Mxfp4ExpertSource,
    input: &MetalTensor,
    assignments: &ExpertAssignments,
    activation: MetalMxfp4ActivationMode,
) -> Result<MetalTensor, BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("MXFP4 expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }
    execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let weights = source.load_expert_mxfp4(layer, expert).map_err(BackendError::ExpertLoad)?;
        match activation {
            MetalMxfp4ActivationMode::F16 => {
                let activated = ops::low_bit::mxfp4_gated_matmul_tensor(ctx, expert_input, &weights.gate, &weights.up, &spec.activation).map_err(|msg| BackendError::Compute { msg })?;
                ops::low_bit::mxfp4_matmul_tensor(ctx, &activated, &weights.down).map_err(|msg| BackendError::Compute { msg })
            }
            MetalMxfp4ActivationMode::Mxfp8 => {
                let activated = ops::low_bit::mxfp4_mxfp8_gated_matmul_tensor(ctx, expert_input, &weights.gate, &weights.up, &spec.activation).map_err(|msg| BackendError::Compute { msg })?;
                ops::low_bit::mxfp4_mxfp8_matmul_tensor(ctx, &activated, &weights.down).map_err(|msg| BackendError::Compute { msg })
            }
        }
    })
}

fn w4a16_routed(ctx: &MetalContext, spec: &TopkMoeSpec, layer: usize, source: &dyn W4A16ExpertSource, input: &MetalTensor, assignments: &ExpertAssignments) -> Result<(MetalTensor, f64, f64, f64), BackendError> {
    const SSD_READ_CONCURRENCY: usize = 4;
    let active_experts = assignments.iter().enumerate().filter_map(|(expert, rows)| (!rows.is_empty()).then_some(expert)).collect::<Vec<_>>();
    let indexed_experts = active_experts.into_iter().enumerate().collect::<Vec<_>>();
    let next_job = std::sync::atomic::AtomicUsize::new(0);
    thread::scope(|scope| {
        // 允许读取 worker 在 Metal 编码当前专家时继续下一次 I/O，容量保持为并发窗口以限制内存峰值。
        let (sender, receiver) = std::sync::mpsc::sync_channel(SSD_READ_CONCURRENCY);
        for _ in 0..SSD_READ_CONCURRENCY {
            let sender = sender.clone();
            let jobs = &indexed_experts;
            let next_job = &next_job;
            scope.spawn(move || {
                while let Some(&(sequence, expert)) = jobs.get(next_job.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                    let started = Instant::now();
                    let loaded = source.load_expert_w4a16(layer, expert);
                    if sender.send((sequence, expert, loaded, started.elapsed().as_secs_f64())).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        let mut read_seconds = 0.0;
        let mut wait_seconds = 0.0;
        let mut upload_seconds = 0.0;
        let mut sequence = 0usize;
        let mut ready = BTreeMap::new();
        let output = execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
            let wait_started = Instant::now();
            let (loaded_expert, weights, read_elapsed) = loop {
                if let Some(loaded) = ready.remove(&sequence) {
                    break loaded;
                }
                let (loaded_sequence, loaded_expert, weights, read_elapsed) = receiver.recv().map_err(|_| BackendError::ExpertLoad(format!("W4A16 L{layer} E{expert} 加载线程提前结束")))?;
                ready.insert(loaded_sequence, (loaded_expert, weights, read_elapsed));
            };
            sequence += 1;
            wait_seconds += wait_started.elapsed().as_secs_f64();
            read_seconds += read_elapsed;
            if loaded_expert != expert {
                return Err(BackendError::ExpertLoad(format!("W4A16 L{layer} expert 顺序不一致: 加载 E{loaded_expert}，执行 E{expert}",)));
            }
            let weights = weights.map_err(BackendError::ExpertLoad)?;
            if weights.gate.rows != spec.intermediate_size
                || weights.gate.cols != input.cols
                || weights.up.rows != spec.intermediate_size
                || weights.up.cols != input.cols
                || weights.down.rows != input.cols
                || weights.down.cols != spec.intermediate_size
            {
                return Err(BackendError::ExpertLoad(format!(
                    "W4A16 L{layer} E{expert} shape gate=[{},{}] up=[{},{}] down=[{},{}]，期望 intermediate={} hidden={}",
                    weights.gate.rows, weights.gate.cols, weights.up.rows, weights.up.cols, weights.down.rows, weights.down.cols, spec.intermediate_size, input.cols,
                )));
            }
            let upload_started = Instant::now();
            let gate = ctx.prepare_weight(LinearWeight::w4a16(&weights.gate), spec.intermediate_size, input.cols)?;
            let up = ctx.prepare_weight(LinearWeight::w4a16(&weights.up), spec.intermediate_size, input.cols)?;
            let down = ctx.prepare_weight(LinearWeight::w4a16(&weights.down), input.cols, spec.intermediate_size)?;
            upload_seconds += upload_started.elapsed().as_secs_f64();
            let activated = ctx.gated_linear(expert_input, &gate, &up, &spec.activation)?;
            ctx.linear(&activated, &down)
        })?;
        Ok((output, read_seconds, wait_seconds, upload_seconds))
    })
}

fn fp8_routed(ctx: &MetalContext, spec: &TopkMoeSpec, layer: usize, source: &dyn crate::weight::expert_source::Fp8ExpertSource, input: &MetalTensor, assignments: &ExpertAssignments) -> Result<MetalTensor, BackendError> {
    let host_input = ctx.tensor_to_f32(input);
    let mut host_output = vec![0.0; input.rows * input.cols];
    for (expert, rows) in assignments.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let mut gathered = Vec::with_capacity(rows.len() * input.cols);
        for &(token, _) in rows {
            let begin = token as usize * input.cols;
            gathered.extend_from_slice(&host_input[begin..begin + input.cols]);
        }
        let weights = source.load_expert_fp8(layer, expert).map_err(BackendError::ExpertLoad)?;
        let gate = ops::dense::fp8_matmul(ctx, &gathered, &weights.gate, rows.len()).map_err(|msg| BackendError::Compute { msg })?;
        let up = ops::dense::fp8_matmul(ctx, &gathered, &weights.up, rows.len()).map_err(|msg| BackendError::Compute { msg })?;
        let activated = ops::dense::gated_activation(ctx, &gate, &up, &spec.activation).map_err(|msg| BackendError::Compute { msg })?;
        let down = ops::dense::fp8_matmul(ctx, &activated, &weights.down, rows.len()).map_err(|msg| BackendError::Compute { msg })?;
        for (row, &(token, route_weight)) in rows.iter().enumerate() {
            let target = token as usize * input.cols;
            let source = row * input.cols;
            for column in 0..input.cols {
                host_output[target + column] += route_weight * down[source + column];
            }
        }
    }
    ctx.tensor_from_f32(&host_output, input.rows, input.cols).map_err(|msg| BackendError::Compute { msg })
}

fn nvfp4_prefill_routed(ctx: &MetalContext, spec: &TopkMoeSpec, layer: usize, source: &dyn Nvfp4ExpertSource, input: &MetalTensor, assignments: &[Vec<(u32, f32)>]) -> Result<MetalTensor, BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("NVFP4 expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }
    // NVFP4 expert 很大，只让当前计算批次进入 arena；路由和累加顺序保持不变。
    const EXPERTS_PER_COMPUTE_BATCH: usize = 8;
    const OPERATIONS_PER_COMMAND_BUFFER: u64 = 24;

    let active_experts = assignments.iter().enumerate().filter_map(|(expert, rows)| (!rows.is_empty()).then_some(expert)).collect::<Vec<_>>();
    let was_deferred = ctx.deferred_waits_enabled();
    let previous_batch_limit = ctx.set_deferred_batch_max_operations(OPERATIONS_PER_COMMAND_BUFFER);
    ctx.set_deferred_waits(true);
    let result = {
        let mut batch_start = 0;
        let mut batch_end = 0;
        let mut next_expert = 0;
        let mut batch = None;
        let output = execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
            if next_expert == batch_end {
                batch_start = next_expert;
                batch_end = (batch_start + EXPERTS_PER_COMPUTE_BATCH).min(active_experts.len());
                if batch.is_some() {
                    ctx.submit_batch();
                }
                let next_batch = MetalNvfp4Layer::load_selected(ctx, source, layer, spec.num_experts, &active_experts[batch_start..batch_end], batch_end - batch_start)?;
                if batch.is_some() {
                    ctx.synchronize();
                }
                batch = Some(next_batch);
            }
            if active_experts.get(next_expert) != Some(&expert) {
                return Err(BackendError::Compute { msg: format!("L{layer} NVFP4 prefill expert 顺序不一致: expected={:?}, actual={expert}", active_experts.get(next_expert)) });
            }
            let (gate_weight, up_weight, down_weight) = batch.as_ref().expect("NVFP4 prefill batch 已加载").weights(next_expert - batch_start)?;
            let (gate, up) = ctx.dual_linear(expert_input, &gate_weight, &up_weight)?;
            let activated = ctx.gated_activation(&gate, &up, &spec.activation)?;
            let down = ctx.linear(&activated, &down_weight)?;
            next_expert += 1;
            Ok(down)
        });
        // 最后一批完成前保持 arena 存活，随后才允许 closure 释放 buffer。
        ctx.synchronize();
        output
    };

    if !was_deferred {
        ctx.set_deferred_waits(false);
    }
    ctx.set_deferred_batch_max_operations(previous_batch_limit);
    result
}

fn gguf_resident_decode_fused(ctx: &MetalContext, spec: &TopkMoeSpec, layer: usize, input: &MetalTensor, assignments: &[Vec<(u32, f32)>], cache: &MetalGgufDecodeCache) -> Result<MetalTensor, BackendError> {
    let input = ops::to_f16_tensor(ctx, input).map_err(|msg| BackendError::Compute { msg })?;

    let output = ops::moe::moe_accumulator_zeros(ctx, 1, input.cols).map_err(|msg| BackendError::Compute { msg })?;
    let scratch = ctx.tensor_zeros(spec.top_k, spec.intermediate_size);
    let grid = ops::gguf::gguf_iq2s_grid_buffer(ctx);
    let fence = ctx.device.new_fence();
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    let mut active_slot = 0;
    let mut read_bytes = input.buffer.length() + grid.length();
    for (expert, assigned) in assignments.iter().enumerate() {
        if assigned.is_empty() {
            continue;
        }
        let expert_weights = cache.get(layer, expert)?;
        let (
            MetalWeight::Gguf { blob: gate_blob, tensor_type: gate_type, row_bytes: gate_row_bytes, rows: gate_rows, cols: gate_cols },
            MetalWeight::Gguf { blob: up_blob, tensor_type: up_type, row_bytes: up_row_bytes, rows: up_rows, cols: up_cols },
            MetalWeight::Gguf { blob: down_blob, tensor_type: down_type, row_bytes: down_row_bytes, rows: down_rows, cols: down_cols },
        ) = (&expert_weights.gate, &expert_weights.up, &expert_weights.down)
        else {
            unreachable!("GGUF fused decode 只接收已预检的 GGUF expert");
        };
        ops::gguf::encode_gguf_expert_accumulate_f32(
            ctx,
            &encoder,
            &input,
            &scratch,
            active_slot,
            &grid,
            gate_blob,
            *gate_type,
            *gate_row_bytes,
            *gate_rows,
            *gate_cols,
            up_blob,
            *up_type,
            *up_row_bytes,
            *up_rows,
            *up_cols,
            down_blob,
            *down_type,
            *down_row_bytes,
            *down_rows,
            *down_cols,
            &spec.activation,
            &output,
            assigned[0].1,
        )
        .map_err(|msg| BackendError::Compute { msg })?;
        active_slot += 1;
        read_bytes = read_bytes.saturating_add(gate_blob.length() + up_blob.length() + down_blob.length());
    }
    encoder.update_fence(&fence);
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, "gguf_experts_accumulate_f32", "resident experts", read_bytes, scratch.buffer.length() + (input.cols * std::mem::size_of::<f32>()) as u64);
    ops::moe::finish_moe_accumulator(ctx, output, Some(fence.as_ref())).map_err(|msg| BackendError::Compute { msg })
}

fn gguf_resident_decode_routed(ctx: &MetalContext, spec: &TopkMoeSpec, layer: usize, input: &MetalTensor, assignments: &[Vec<(u32, f32)>], cache: &MetalGgufDecodeCache) -> Result<MetalTensor, BackendError> {
    let mut fused = true;
    for (expert, assigned) in assignments.iter().enumerate() {
        if assigned.is_empty() {
            continue;
        }
        if assigned.len() != 1 || assigned[0].0 != 0 {
            return Err(BackendError::Compute { msg: format!("GGUF resident decode expert {expert} assignments 非单行: {assigned:?}") });
        }
        let weights = cache.get(layer, expert)?;
        fused &= matches!((&weights.gate, &weights.up, &weights.down), (MetalWeight::Gguf { .. }, MetalWeight::Gguf { .. }, MetalWeight::Gguf { .. }));
    }
    if fused {
        return gguf_resident_decode_fused(ctx, spec, layer, input, assignments, cache);
    }
    let previous_batch_limit = ctx.set_deferred_batch_max_operations(1);
    // 先收集 result 再恢复 batch 上限：循环内任何 `?` 提前返回都不能把全局上限永久留成 1。
    let result = (|| -> Result<MetalTensor, BackendError> {
        let output = ops::moe::moe_accumulator_zeros(ctx, 1, input.cols).map_err(|msg| BackendError::Compute { msg })?;
        let scratch = ctx.tensor_zeros(spec.top_k, spec.intermediate_size);
        let grid = ops::gguf::gguf_iq2s_grid_buffer(ctx);
        let fence = ctx.device.new_fence();
        let mut fence_ready = false;
        let mut active_slot = 0;
        for (expert, assigned) in assignments.iter().enumerate() {
            if assigned.is_empty() {
                continue;
            }
            let scratch_row = active_slot;
            active_slot += 1;
            if assigned.len() != 1 || assigned[0].0 != 0 {
                return Err(BackendError::Compute { msg: format!("GGUF resident decode expert {expert} assignments 非单行: {assigned:?}") });
            }
            let expert_weights = cache.get(layer, expert)?;
            match (&expert_weights.gate, &expert_weights.up, &expert_weights.down) {
                (
                    MetalWeight::Gguf { blob: gate_blob, tensor_type: gate_type, row_bytes: gate_row_bytes, rows: gate_rows, cols: gate_cols },
                    MetalWeight::Gguf { blob: up_blob, tensor_type: up_type, row_bytes: up_row_bytes, rows: up_rows, cols: up_cols },
                    MetalWeight::Gguf { blob: down_blob, tensor_type: down_type, row_bytes: down_row_bytes, rows: down_rows, cols: down_cols },
                ) if matches!(*down_type, 11 | 22) => {
                    ops::gguf::gguf_expert_accumulate_f32(
                        ctx,
                        input,
                        &scratch,
                        scratch_row,
                        &grid,
                        &fence,
                        fence_ready,
                        gate_blob,
                        *gate_type,
                        *gate_row_bytes,
                        *gate_rows,
                        *gate_cols,
                        up_blob,
                        *up_type,
                        *up_row_bytes,
                        *up_rows,
                        *up_cols,
                        down_blob,
                        *down_type,
                        *down_row_bytes,
                        *down_rows,
                        *down_cols,
                        &spec.activation,
                        &output,
                        assigned[0].1,
                    )
                    .map_err(|msg| BackendError::Compute { msg })?;
                    fence_ready = true;
                    continue;
                }
                _ => {}
            }
            if fence_ready {
                ctx.synchronize();
                fence_ready = false;
            }
            let activated = ctx.gated_linear(input, &expert_weights.gate, &expert_weights.up, &spec.activation)?;
            match &expert_weights.down {
                MetalWeight::Gguf { blob, tensor_type: 11, row_bytes, rows, cols } => {
                    ops::gguf::gguf_q3k_gemv_accumulate_f32(ctx, &activated, blob, *row_bytes, *rows, *cols, &output, assigned[0].1).map_err(|msg| BackendError::Compute { msg })?;
                }
                MetalWeight::Gguf { blob, tensor_type: 22, row_bytes, rows, cols } => {
                    ops::gguf::gguf_iq2s_gemv_accumulate_f32(ctx, &activated, blob, *row_bytes, *rows, *cols, &output, assigned[0].1).map_err(|msg| BackendError::Compute { msg })?;
                }
                down_weight => {
                    let down = ctx.linear(&activated, down_weight)?;
                    ops::moe::scatter_add_rows_weighted_f32(ctx, &output, &down, &[0], &[assigned[0].1]).map_err(|msg| BackendError::Compute { msg })?;
                }
            }
            ctx.synchronize();
        }
        ops::moe::finish_moe_accumulator(ctx, output, fence_ready.then_some(fence.as_ref())).map_err(|msg| BackendError::Compute { msg })
    })();
    ctx.set_deferred_batch_max_operations(previous_batch_limit);
    result
}
fn gguf_routed(ctx: &MetalContext, spec: &TopkMoeSpec, layer: usize, source: &dyn GgufExpertSource, input: &MetalTensor, assignments: &[Vec<(u32, f32)>], mut cache: Option<&mut MetalGgufDecodeCache>) -> Result<MetalTensor, BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("GGUF expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }
    let was_deferred = ctx.deferred_waits_enabled();
    let resident_decode = input.rows == 1 && cache.is_some() && was_deferred;
    if resident_decode {
        let cache = cache.as_deref_mut().expect("resident decode 应持有 GGUF cache");
        // prefill 只会预热当时命中的专家；下一 token 的路由可变化，执行前按需补齐。
        for (expert, rows) in assignments.iter().enumerate() {
            if !rows.is_empty() && !cache.fully_resident {
                cache.ensure(ctx, source, layer, expert)?;
            }
        }
        return gguf_resident_decode_routed(ctx, spec, layer, input, assignments, cache);
    }
    // 常驻 decode 的 expert 输出仍由 GPU 消费，保持在外层 batch 中直到下一次 router 读回。
    // Prefill 合并若干 expert 流水线，避免量化矩阵解码产生过碎的提交边界。
    let previous_batch_limit = if resident_decode {
        None
    } else {
        let batch_operations = (spec.top_k as u64).max(1).saturating_mul(6);
        Some(ctx.set_deferred_batch_max_operations(batch_operations))
    };
    ctx.set_deferred_waits(true);
    let mut resident_weights = Vec::new();
    let result = execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let mut loaded_weights = None;
        let expert_weights = if let Some(cache) = cache.as_deref_mut() {
            if !cache.fully_resident {
                cache.ensure(ctx, source, layer, expert)?;
            }
            cache.get(layer, expert)?
        } else {
            loaded_weights = Some(MetalGgufExpertWeights::load(ctx, source, layer, expert)?);
            loaded_weights.as_ref().expect("临时 GGUF expert 权重应已加载")
        };
        let activated = ctx.gated_linear(expert_input, &expert_weights.gate, &expert_weights.up, &spec.activation)?;
        let down = ctx.linear(&activated, &expert_weights.down)?;
        if let Some(weights) = loaded_weights {
            resident_weights.push(weights);
        }
        Ok(down)
    });

    // 临时权重必须等 GPU 用完；常驻 decode 由下一次 CPU 读回边界统一等待。
    // defer 模式下由外层统一 sync，不重复等待。
    if !resident_decode && !was_deferred {
        ctx.synchronize();
    }
    drop(resident_weights);
    if !was_deferred {
        ctx.set_deferred_waits(false);
    }
    if let Some(previous_batch_limit) = previous_batch_limit {
        ctx.set_deferred_batch_max_operations(previous_batch_limit);
    }
    result
}

struct MetalNvfp4Layer {
    codes: Option<[metal::Buffer; 3]>,
    scales: Option<[metal::Buffer; 3]>,
    global_scales: Option<[metal::Buffer; 3]>,
    archive_chunks: Option<[metal::Buffer; 2]>,
    expert_count: usize,
    code_stride: usize,
    scale_stride: usize,
    hidden: usize,
    intermediate: usize,
}

const NVFP4_SSD_READ_CONCURRENCY: usize = 2;

/// 并发把 `slots` 指定的 expert 段从 archive 读进 chunk buffer。
/// `slots` 为 (目标 slot, 源 expert) 对;`label` 用于错误信息区分调用路径。
fn read_archive_slots(source: &dyn Nvfp4ExpertSource, layer: usize, layout: &Nvfp4ArchiveLayout, chunks: &[metal::Buffer; 2], experts_per_chunk: usize, slots: &[(usize, usize)], concurrency: usize, label: &str) -> Result<(), BackendError> {
    let slots_per_worker = slots.len().div_ceil(concurrency);
    thread::scope(|scope| {
        let handles = slots
            .chunks(slots_per_worker)
            .map(|work| {
                scope.spawn(move || {
                    for &(slot, expert) in work {
                        let chunk = slot / experts_per_chunk;
                        let local_expert = slot % experts_per_chunk;
                        let source_offset = layout.expert_offset(expert).map_err(BackendError::ExpertLoad)?;
                        let destination = unsafe { std::slice::from_raw_parts_mut(chunks[chunk].contents().cast::<u8>().add(local_expert * layout.expert_bytes()), layout.expert_bytes()) };
                        source.read_layer_archive_range(layer, source_offset, destination).map_err(BackendError::ExpertLoad)?;
                    }
                    Ok::<(), BackendError>(())
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().map_err(|_| BackendError::ExpertLoad(format!("L{layer} NVFP4 {label} reader panic")))??;
        }
        Ok::<(), BackendError>(())
    })
}

impl MetalNvfp4Layer {
    fn bytes(&self) -> usize {
        self.codes
            .iter()
            .flat_map(|buffers| buffers.iter())
            .chain(self.scales.iter().flat_map(|buffers| buffers.iter()))
            .chain(self.global_scales.iter().flat_map(|buffers| buffers.iter()))
            .chain(self.archive_chunks.iter().flat_map(|buffers| buffers.iter()))
            .map(|buffer| buffer.length() as usize)
            .sum()
    }

    fn prepare_selected_reusing(ctx: &MetalContext, source: &dyn Nvfp4ExpertSource, layer: usize, source_expert_count: usize, slot_count: usize, reuse: Option<Self>) -> Result<Self, BackendError> {
        if slot_count == 0 {
            return Err(BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch slot 数不能为 0")));
        }
        let layout = source.layer_archive_layout(layer, source_expert_count).map_err(BackendError::ExpertLoad)?;
        if let Some(mut arena) = reuse {
            let reusable = match (&arena.archive_chunks, layout) {
                (Some(chunks), Some(layout)) => {
                    let experts_per_chunk = slot_count.div_ceil(NVFP4_SSD_READ_CONCURRENCY);
                    let chunk_bytes = experts_per_chunk.checked_mul(layout.expert_bytes()).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch chunk 大小溢出")))?;
                    if chunks.iter().all(|buffer| buffer.length() >= chunk_bytes as u64) {
                        arena.code_stride = layout.code_bytes();
                        arena.scale_stride = layout.scale_bytes();
                        true
                    } else {
                        false
                    }
                }
                (None, None) => {
                    let (code_stride, scale_stride) = nvfp4_storage_lengths(source.intermediate(), source.hidden()).map_err(BackendError::ExpertLoad)?;
                    let code_bytes = code_stride.checked_mul(slot_count).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch codes 大小溢出")))?;
                    let scale_bytes = scale_stride.checked_mul(slot_count).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch scales 大小溢出")))?;
                    let global_bytes = mem::size_of::<f32>().checked_mul(slot_count).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch global scales 大小溢出")))?;
                    let fits = arena.codes.as_ref().is_some_and(|buffers| buffers.iter().all(|buffer| buffer.length() >= code_bytes as u64))
                        && arena.scales.as_ref().is_some_and(|buffers| buffers.iter().all(|buffer| buffer.length() >= scale_bytes as u64))
                        && arena.global_scales.as_ref().is_some_and(|buffers| buffers.iter().all(|buffer| buffer.length() >= global_bytes as u64));
                    if fits {
                        arena.code_stride = code_stride;
                        arena.scale_stride = scale_stride;
                    }
                    fits
                }
                _ => false,
            };
            if reusable {
                arena.expert_count = slot_count;
                arena.hidden = source.hidden();
                arena.intermediate = source.intermediate();
                return Ok(arena);
            }
        }
        if let Some(layout) = layout {
            let experts_per_chunk = slot_count.div_ceil(NVFP4_SSD_READ_CONCURRENCY);
            let chunk_bytes = experts_per_chunk.checked_mul(layout.expert_bytes()).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch chunk 大小溢出")))?;
            return Ok(Self {
                codes: None,
                scales: None,
                global_scales: None,
                archive_chunks: Some(std::array::from_fn(|_| ctx.shared_buffer_uninit(chunk_bytes))),
                expert_count: slot_count,
                code_stride: layout.code_bytes(),
                scale_stride: layout.scale_bytes(),
                hidden: source.hidden(),
                intermediate: source.intermediate(),
            });
        }
        Self::load_empty(ctx, source, layer, slot_count)
    }

    fn fill_selected(&self, source: &dyn Nvfp4ExpertSource, layer: usize, source_expert_count: usize, experts: &[usize], read_concurrency: usize) -> Result<(), BackendError> {
        if experts.is_empty() || experts.len() > self.expert_count {
            return Err(BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch expert 数无效: experts={}, slots={}", experts.len(), self.expert_count,)));
        }
        let read_concurrency = read_concurrency.clamp(1, NVFP4_SSD_READ_CONCURRENCY);
        if let Some(chunks) = &self.archive_chunks {
            let layout = source.layer_archive_layout(layer, source_expert_count).map_err(BackendError::ExpertLoad)?.ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch archive layout 缺失")))?;
            let experts_per_chunk = self.expert_count.div_ceil(NVFP4_SSD_READ_CONCURRENCY);
            let slots: Vec<(usize, usize)> = experts.iter().copied().enumerate().collect();
            return read_archive_slots(source, layer, &layout, chunks, experts_per_chunk, &slots, read_concurrency, "prefetch");
        }
        if self.codes.is_some() {
            let slots: Vec<(usize, usize)> = experts.iter().copied().enumerate().collect();
            return load_nvfp4_experts_into_with_concurrency(source, layer, self, &slots, read_concurrency);
        }
        Err(BackendError::ExpertLoad(format!("L{layer} NVFP4 prefetch arena 未初始化")))
    }

    fn load_selected(ctx: &MetalContext, source: &dyn Nvfp4ExpertSource, layer: usize, source_expert_count: usize, experts: &[usize], slot_count: usize) -> Result<Self, BackendError> {
        if experts.is_empty() || experts.len() > slot_count {
            return Err(BackendError::ExpertLoad(format!("L{layer} NVFP4 decode slot 数无效: experts={}, slots={slot_count}", experts.len(),)));
        }
        if let Some(layout) = source.layer_archive_layout(layer, source_expert_count).map_err(BackendError::ExpertLoad)? {
            return Self::load_selected_archive(ctx, source, layer, layout, experts, slot_count);
        }
        let arena = Self::load_empty(ctx, source, layer, slot_count)?;
        let slots: Vec<(usize, usize)> = experts.iter().copied().enumerate().collect();
        load_nvfp4_experts_into(source, layer, &arena, &slots)?;
        Ok(arena)
    }

    fn load_selected_reusing(ctx: &MetalContext, source: &dyn Nvfp4ExpertSource, layer: usize, source_expert_count: usize, experts: &[usize], slot_count: usize, reuse: Option<Self>) -> Result<Self, BackendError> {
        if let Some(mut arena) = reuse
            && arena.refill_selected_archive(source, layer, source_expert_count, experts, slot_count)?
        {
            return Ok(arena);
        }
        Self::load_selected(ctx, source, layer, source_expert_count, experts, slot_count)
    }

    fn refill_selected_archive(&mut self, source: &dyn Nvfp4ExpertSource, layer: usize, source_expert_count: usize, experts: &[usize], slot_count: usize) -> Result<bool, BackendError> {
        if experts.is_empty() || experts.len() > slot_count {
            return Err(BackendError::ExpertLoad(format!("L{layer} NVFP4 decode slot 数无效: experts={}, slots={slot_count}", experts.len(),)));
        }
        let Some(layout) = source.layer_archive_layout(layer, source_expert_count).map_err(BackendError::ExpertLoad)? else {
            return Ok(false);
        };
        let Some(chunks) = &self.archive_chunks else {
            return Ok(false);
        };
        let experts_per_chunk = slot_count.div_ceil(NVFP4_SSD_READ_CONCURRENCY);
        let chunk_bytes = experts_per_chunk.checked_mul(layout.expert_bytes()).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 decode chunk 大小溢出")))?;
        if chunks.iter().any(|buffer| buffer.length() < chunk_bytes as u64) {
            return Ok(false);
        }
        let slots: Vec<(usize, usize)> = experts.iter().copied().enumerate().collect();
        read_archive_slots(source, layer, &layout, chunks, experts_per_chunk, &slots, NVFP4_SSD_READ_CONCURRENCY, "decode archive")?;
        self.expert_count = slot_count;
        self.code_stride = layout.code_bytes();
        self.scale_stride = layout.scale_bytes();
        self.hidden = source.hidden();
        self.intermediate = source.intermediate();
        Ok(true)
    }

    fn align_selected(&mut self, source: &dyn Nvfp4ExpertSource, layer: usize, source_expert_count: usize, loaded_experts: &mut [usize], active_experts: &[usize], read_concurrency: usize) -> Result<Option<usize>, BackendError> {
        if loaded_experts.len() != self.expert_count || active_experts.len() != self.expert_count {
            return Ok(None);
        }
        let free_slots: Vec<usize> = loaded_experts.iter().enumerate().filter_map(|(slot, expert)| (!active_experts.contains(expert)).then_some(slot)).collect();
        let missing: Vec<usize> = active_experts.iter().copied().filter(|expert| !loaded_experts.contains(expert)).collect();
        if free_slots.len() != missing.len() {
            return Ok(None);
        }
        let hits = active_experts.len() - missing.len();
        let replacements: Vec<(usize, usize)> = free_slots.into_iter().zip(missing).collect();
        if !replacements.is_empty() {
            let read_concurrency = read_concurrency.clamp(1, NVFP4_SSD_READ_CONCURRENCY);
            if let Some(chunks) = &self.archive_chunks {
                let Some(layout) = source.layer_archive_layout(layer, source_expert_count).map_err(BackendError::ExpertLoad)? else {
                    return Ok(None);
                };
                if self.code_stride != layout.code_bytes() || self.scale_stride != layout.scale_bytes() {
                    return Ok(None);
                }
                let experts_per_chunk = self.expert_count.div_ceil(NVFP4_SSD_READ_CONCURRENCY);
                read_archive_slots(source, layer, &layout, chunks, experts_per_chunk, &replacements, read_concurrency, "decode archive")?;
            } else if self.codes.is_some() {
                load_nvfp4_experts_into_with_concurrency(source, layer, self, &replacements, read_concurrency)?;
            } else {
                return Ok(None);
            }
            for &(slot, expert) in &replacements {
                loaded_experts[slot] = expert;
            }
        }
        Ok(Some(hits))
    }

    fn load_selected_archive(ctx: &MetalContext, source: &dyn Nvfp4ExpertSource, layer: usize, layout: Nvfp4ArchiveLayout, experts: &[usize], slot_count: usize) -> Result<Self, BackendError> {
        if experts.is_empty() {
            return Err(BackendError::ExpertLoad(format!("L{layer} NVFP4 decode 没有 active expert")));
        }
        let experts_per_chunk = slot_count.div_ceil(NVFP4_SSD_READ_CONCURRENCY);
        let chunk_bytes = experts_per_chunk.checked_mul(layout.expert_bytes()).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 decode chunk 大小溢出")))?;
        // 保持 archive 的 expert-major 原始布局，避免把一个连续 expert 拆成 9 次读取再重排。
        let chunks = std::array::from_fn(|_| ctx.shared_buffer_uninit(chunk_bytes));
        let slots: Vec<(usize, usize)> = experts.iter().copied().enumerate().collect();
        read_archive_slots(source, layer, &layout, &chunks, experts_per_chunk, &slots, NVFP4_SSD_READ_CONCURRENCY, "decode archive")?;
        Ok(Self {
            codes: None,
            scales: None,
            global_scales: None,
            archive_chunks: Some(chunks),
            expert_count: slot_count,
            code_stride: layout.code_bytes(),
            scale_stride: layout.scale_bytes(),
            hidden: source.hidden(),
            intermediate: source.intermediate(),
        })
    }

    fn load_empty(ctx: &MetalContext, source: &dyn Nvfp4ExpertSource, layer: usize, expert_count: usize) -> Result<Self, BackendError> {
        let (code_stride, scale_stride) = nvfp4_storage_lengths(source.intermediate(), source.hidden()).map_err(BackendError::ExpertLoad)?;
        let code_bytes = code_stride.checked_mul(expert_count).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 codes arena 大小溢出")))?;
        let scale_bytes = scale_stride.checked_mul(expert_count).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 scales arena 大小溢出")))?;
        let global_scale_bytes = mem::size_of::<f32>().checked_mul(expert_count).ok_or_else(|| BackendError::ExpertLoad(format!("L{layer} NVFP4 global scale arena 大小溢出")))?;
        Ok(Self {
            codes: Some(std::array::from_fn(|_| ctx.shared_buffer_uninit(code_bytes))),
            scales: Some(std::array::from_fn(|_| ctx.shared_buffer_uninit(scale_bytes))),
            global_scales: Some(std::array::from_fn(|_| ctx.shared_buffer_uninit(global_scale_bytes))),
            archive_chunks: None,
            expert_count,
            code_stride,
            scale_stride,
            hidden: source.hidden(),
            intermediate: source.intermediate(),
        })
    }

    fn expert_buffer(&self, expert: usize, slot: usize) -> Result<Nvfp4ExpertBufferMut<'_>, BackendError> {
        if slot >= self.expert_count {
            return Err(BackendError::ExpertLoad(format!("NVFP4 slot 越界: {slot} >= {}", self.expert_count)));
        }
        let code_offset = slot * self.code_stride;
        let scale_offset = slot * self.scale_stride;
        let global_scale_offset = slot * mem::size_of::<f32>();
        let codes = self.codes.as_ref().ok_or_else(|| BackendError::ExpertLoad("NVFP4 archive arena 不支持 slot 写入".to_owned()))?;
        let scales = self.scales.as_ref().expect("matrix-major codes/scales 同时存在");
        let global_scales = self.global_scales.as_ref().expect("matrix-major codes/global scales 同时存在");
        let matrix = |index: usize, rows: usize, cols: usize| unsafe {
            Nvfp4MatrixBufferMut::new(
                std::slice::from_raw_parts_mut(codes[index].contents().cast::<u8>().add(code_offset), self.code_stride),
                std::slice::from_raw_parts_mut(scales[index].contents().cast::<u8>().add(scale_offset), self.scale_stride),
                std::slice::from_raw_parts_mut(global_scales[index].contents().cast::<u8>().add(global_scale_offset), mem::size_of::<f32>()),
                rows,
                cols,
            )
        };
        let gate = matrix(0, self.intermediate, self.hidden).map_err(BackendError::ExpertLoad)?;
        let up = matrix(1, self.intermediate, self.hidden).map_err(BackendError::ExpertLoad)?;
        let down = matrix(2, self.hidden, self.intermediate).map_err(BackendError::ExpertLoad)?;
        Ok(Nvfp4ExpertBufferMut { expert, gate, up, down })
    }

    fn weights(&self, expert: usize) -> Result<(MetalWeight, MetalWeight, MetalWeight), BackendError> {
        if expert >= self.expert_count {
            return Err(BackendError::ExpertLoad(format!("NVFP4 layer arena expert 越界: {expert} >= {}", self.expert_count)));
        }
        Ok((self.weight(expert, 0, self.intermediate, self.hidden), self.weight(expert, 1, self.intermediate, self.hidden), self.weight(expert, 2, self.hidden, self.intermediate)))
    }

    fn weight(&self, expert: usize, matrix: usize, rows: usize, cols: usize) -> MetalWeight {
        if let Some(chunks) = &self.archive_chunks {
            let experts_per_chunk = self.expert_count.div_ceil(NVFP4_SSD_READ_CONCURRENCY);
            let chunk = expert / experts_per_chunk;
            let local_expert = expert % experts_per_chunk;
            let matrix_stride = self.code_stride + self.scale_stride + mem::size_of::<f32>();
            let base = local_expert * matrix_stride * 3 + matrix * matrix_stride;
            return MetalWeight::Nvfp4 {
                codes: chunks[chunk].clone(),
                codes_offset: base,
                scales: chunks[chunk].clone(),
                scales_offset: base + self.code_stride,
                global_scale: chunks[chunk].clone(),
                global_scale_offset: base + self.code_stride + self.scale_stride,
                rows,
                cols,
            };
        }
        let codes = self.codes.as_ref().expect("matrix-major arena codes 已初始化");
        let scales = self.scales.as_ref().expect("matrix-major arena scales 已初始化");
        let global_scales = self.global_scales.as_ref().expect("matrix-major arena global scales 已初始化");
        MetalWeight::Nvfp4 {
            codes: codes[matrix].clone(),
            codes_offset: expert * self.code_stride,
            scales: scales[matrix].clone(),
            scales_offset: expert * self.scale_stride,
            global_scale: global_scales[matrix].clone(),
            global_scale_offset: expert * mem::size_of::<f32>(),
            rows,
            cols,
        }
    }
}

fn load_nvfp4_experts_into(source: &dyn Nvfp4ExpertSource, layer: usize, arena: &MetalNvfp4Layer, experts: &[(usize, usize)]) -> Result<(), BackendError> {
    load_nvfp4_experts_into_with_concurrency(source, layer, arena, experts, NVFP4_SSD_READ_CONCURRENCY)
}

fn load_nvfp4_experts_into_with_concurrency(source: &dyn Nvfp4ExpertSource, layer: usize, arena: &MetalNvfp4Layer, experts: &[(usize, usize)], read_concurrency: usize) -> Result<(), BackendError> {
    if experts.is_empty() {
        return Ok(());
    }
    let experts_per_worker = experts.len().div_ceil(read_concurrency.clamp(1, NVFP4_SSD_READ_CONCURRENCY));
    thread::scope(|scope| {
        let handles: Vec<_> = experts
            .chunks(experts_per_worker)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut targets = Vec::with_capacity(chunk.len());
                    for &(slot, expert) in chunk {
                        targets.push(arena.expert_buffer(expert, slot)?);
                    }
                    source.load_experts_nvfp4_into(layer, targets).map_err(BackendError::ExpertLoad)
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().map_err(|_| BackendError::ExpertLoad(format!("L{layer} NVFP4 SSD 加载线程 panic")))?).collect::<Result<Vec<_>, _>>()?;
        Ok(())
    })
}

fn nvfp4_decode_routed(
    ctx: &MetalContext,
    spec: &TopkMoeSpec,
    layer: usize,
    source: &dyn Nvfp4ExpertSource,
    input: &MetalTensor,
    assignments: &[Vec<(u32, f32)>],
    layer_experts: Option<(MetalNvfp4Layer, Vec<usize>)>,
) -> Result<(MetalTensor, MetalNvfp4Layer), BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("NVFP4 expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }

    // Decode 每层只有一个 token，只读取路由实际命中的专家；整层驻留只属于 prefill。
    let routed_experts: Vec<usize> = assignments.iter().enumerate().filter_map(|(expert, rows)| (!rows.is_empty()).then_some(expert)).collect();
    let (layer_experts, active) = match layer_experts {
        Some((layer_experts, active)) => (layer_experts, active),
        None => {
            let layer_experts = MetalNvfp4Layer::load_selected(ctx, source, layer, spec.num_experts, &routed_experts, routed_experts.len())?;
            (layer_experts, routed_experts)
        }
    };
    let route_weights = active
        .iter()
        .map(|&expert| {
            let assigned = &assignments[expert];
            if assigned.len() != 1 || assigned[0].0 != 0 {
                return Err(BackendError::Compute { msg: format!("L{layer} NVFP4 decode expert {expert} assignment 非单 token") });
            }
            Ok(assigned[0].1)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(chunks) = &layer_experts.archive_chunks {
        let output = ops::low_bit::nvfp4_grouped_experts_decode(
            ctx,
            input,
            &chunks[0],
            &chunks[1],
            active.len(),
            layer_experts.expert_count.div_ceil(NVFP4_SSD_READ_CONCURRENCY),
            layer_experts.code_stride,
            layer_experts.scale_stride,
            layer_experts.hidden,
            layer_experts.intermediate,
            &route_weights,
            &spec.activation,
        )
        .map_err(|msg| BackendError::Compute { msg })?;
        return Ok((output, layer_experts));
    }
    if let (Some(codes), Some(scales), Some(global_scales)) = (&layer_experts.codes, &layer_experts.scales, &layer_experts.global_scales) {
        let output = ops::low_bit::nvfp4_grouped_matrix_experts_decode(
            ctx,
            input,
            codes,
            scales,
            global_scales,
            active.len(),
            layer_experts.code_stride,
            layer_experts.scale_stride,
            layer_experts.hidden,
            layer_experts.intermediate,
            &route_weights,
            &spec.activation,
        )
        .map_err(|msg| BackendError::Compute { msg })?;
        return Ok((output, layer_experts));
    }
    let was_deferred = ctx.deferred_waits_enabled();
    let previous_batch_limit = ctx.set_deferred_batch_max_operations(6);
    ctx.set_deferred_waits(true);
    let result = (|| {
        let mut output = ctx.moe_zeros(input.rows, input.cols)?;
        for (slot, expert) in active.iter().copied().enumerate() {
            let assigned = &assignments[expert];
            let rows: Vec<u32> = assigned.iter().map(|&(row, _)| row).collect();
            let route_weights: Vec<f32> = assigned.iter().map(|&(_, weight)| weight).collect();
            let gathered = if input.rows == 1 && rows.len() == 1 && rows[0] == 0 { None } else { Some(ctx.moe_gather_rows(input, &rows)?) };
            let expert_input = gathered.as_ref().unwrap_or(input);
            let (gate_weight, up_weight, down_weight) = layer_experts.weights(slot)?;
            let (gate, up) = ctx.dual_linear(expert_input, &gate_weight, &up_weight)?;
            let activated = ctx.gated_activation(&gate, &up, &spec.activation)?;
            let down = ctx.linear(&activated, &down_weight)?;
            ctx.moe_scatter_add_rows(&mut output, &down, &rows, &route_weights)?;
        }
        ctx.moe_finish(output)
    })();

    ctx.synchronize();
    if !was_deferred {
        ctx.set_deferred_waits(false);
    }
    ctx.set_deferred_batch_max_operations(previous_batch_limit);
    Ok((result?, layer_experts))
}
