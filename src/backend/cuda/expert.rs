//! CUDA 专家权重驻留、量化 kernel 与 routed batch capability。

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use cudarc::driver::safe::{CudaSlice, DevicePtr};
use half::f16;

use crate::{
    backend::cuda::{CudaContext, CudaTensor, CudaWeight},
    backend::{Backend, BackendError, ExpertDecodeBackend, ExpertPrefetchRequest, ExpertPrefillBackend, ExpertPrefillBatch, MoePrefillBackend, MoePrefillRouting},
    kernel::cuda as ops,
    moe::{
        Activation,
        routing::ExpertAssignments,
        topk_moe::{ScoringFunc, TopkMoeSpec},
    },
    weight::expert_source::{ExpertSource, GgufExpertSource, GgufExpertWeights},
};

pub struct CudaDecodeRouting {
    expert_ids: Vec<u32>,
    route_weights: CudaSlice<f32>,
}

pub struct CudaMoeAccumulator {
    data: CudaSlice<f32>,
    rows: usize,
    cols: usize,
}

pub struct CudaPrefillExperts {
    source: Arc<dyn GgufExpertSource>,
    state: CudaMoeState,
}

impl CudaPrefillExperts {
    pub fn gguf(source: Arc<dyn GgufExpertSource>, max_bytes: usize) -> Self {
        Self { source, state: CudaMoeState::new(max_bytes) }
    }

    pub(crate) fn reserve_arena(&mut self, ctx: &CudaContext) -> Result<(), String> {
        self.state.reserve_arena(ctx)
    }

    pub(crate) fn set_transfer_group(&mut self, group: usize) {
        self.state.transfer_group = group;
    }

    pub fn set_frequency_cache(&mut self, enabled: bool) {
        self.state.frequency_cache = enabled;
    }

    pub fn swap_decode_state(&mut self, state: &mut CudaMoeState) {
        std::mem::swap(&mut self.state, state);
    }

    pub fn into_decode_state(self) -> CudaMoeState {
        self.state
    }
}

// 固定预算只对应一份设备分配;空闲段由最后一个权重引用归还。
struct CudaExpertArena {
    data: CudaSlice<u8>,
    free: Mutex<BTreeMap<usize, usize>>,
}

impl CudaExpertArena {
    fn allocate(self: &Arc<Self>, bytes: usize) -> Result<Option<Arc<CudaExpertRegion>>, BackendError> {
        let mut free = self.free.lock().map_err(|_| BackendError::ExpertLoad("CUDA expert arena 空闲段锁已中毒".into()))?;
        // 优先消费最小的足够区间,避免小专家切碎可供较大编码使用的空闲段。
        let Some((&offset, &available)) = free.iter().filter(|(_, len)| **len >= bytes).min_by_key(|(_, len)| **len) else { return Ok(None) };
        free.remove(&offset);
        if available > bytes {
            free.insert(offset + bytes, available - bytes);
        }
        Ok(Some(Arc::new(CudaExpertRegion { arena: self.clone(), offset, bytes })))
    }
}

struct CudaExpertRegion {
    arena: Arc<CudaExpertArena>,
    offset: usize,
    bytes: usize,
}

impl Drop for CudaExpertRegion {
    fn drop(&mut self) {
        let mut free = self.arena.free.lock().unwrap_or_else(|error| error.into_inner());
        let (mut offset, mut bytes) = (self.offset, self.bytes);
        if let Some((&previous, &length)) = free.range(..offset).next_back() {
            if previous + length == offset {
                free.remove(&previous);
                offset = previous;
                bytes += length;
            }
        }
        if let Some((&next, &length)) = free.range(offset..).next() {
            if offset + bytes == next {
                free.remove(&next);
                bytes += length;
            }
        }
        free.insert(offset, bytes);
    }
}

// 该别名只借用区域里的地址,绝不对内部指针调用 cuMemFree;区域所有者负责整块生命周期。
enum CudaExpertBuffer {
    Owned(CudaSlice<u8>),
    Arena { view: ManuallyDrop<CudaSlice<u8>>, _region: Arc<CudaExpertRegion> },
}
impl Deref for CudaExpertBuffer {
    type Target = CudaSlice<u8>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(data) => data,
            Self::Arena { view, .. } => view,
        }
    }
}
impl DerefMut for CudaExpertBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(data) => data,
            Self::Arena { view, .. } => view,
        }
    }
}
impl Drop for CudaExpertBuffer {
    fn drop(&mut self) {
        if let Self::Arena { view, .. } = self {
            // leak 释放 cudarc 的 stream/event 引用,只跳过内部地址的设备 free。
            unsafe { ManuallyDrop::take(view) }.leak();
        }
    }
}

enum CudaGateUp {
    F16 { gate: CudaWeight, up: CudaWeight },
    Q4K { gate: CudaExpertBuffer, up: CudaExpertBuffer, rows: usize, cols: usize },
    Q5K { gate: CudaExpertBuffer, up: CudaExpertBuffer, rows: usize },
}

impl CudaGateUp {
    fn activated(&self, ctx: &CudaContext, input: &CudaTensor, activation: &Activation) -> Result<CudaTensor, BackendError> {
        match self {
            Self::F16 { gate, up } => ctx.gated_linear(input, gate, up, activation),
            Self::Q5K { gate, up, rows } => {
                let gate = ops::linear::gguf_kq_matmul_f16(ctx, input, gate, *rows, 13).map_err(|msg| BackendError::Compute { msg })?;
                let up = ops::linear::gguf_kq_matmul_f16(ctx, input, up, *rows, 13).map_err(|msg| BackendError::Compute { msg })?;
                ctx.gated_activation(&gate, &up, activation)
            }
            Self::Q4K { gate, up, rows, cols } => {
                if !matches!(activation, Activation::Silu) {
                    return Err(BackendError::Compute { msg: "CUDA Q4_K gate/up 当前只支持 SiLU".to_owned() });
                }
                if input.cols != *cols {
                    return Err(BackendError::Compute { msg: format!("CUDA Q4_K gate/up input cols={}，weight cols={cols}", input.cols) });
                }
                ops::linear::gated_linear_q4_k_silu_f16(ctx, input, gate, up, *rows).map_err(|msg| BackendError::Compute { msg })
            }
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Self::F16 { gate, up } => (gate.rows * gate.cols + up.rows * up.cols) * std::mem::size_of::<f16>(),
            Self::Q4K { gate, up, .. } | Self::Q5K { gate, up, .. } => gate.len() + up.len(),
        }
    }
}

enum CudaDown {
    F16(CudaWeight),
    Packed { data: CudaExpertBuffer, rows: usize, tensor_type: u32 },
    Q5K { data: CudaExpertBuffer, rows: usize, cols: usize },
    Q6K { data: CudaExpertBuffer, rows: usize, cols: usize },
}

impl CudaDown {
    fn linear(&self, ctx: &CudaContext, input: &CudaTensor) -> Result<CudaTensor, BackendError> {
        match self {
            Self::F16(weight) => ctx.linear(input, weight),
            Self::Packed { data, rows, tensor_type } => ops::linear::gguf_kq_matmul_f16(ctx, input, data, *rows, *tensor_type).map_err(|msg| BackendError::Compute { msg }),
            Self::Q5K { data, rows, cols } => {
                if input.cols != *cols {
                    return Err(BackendError::Compute { msg: format!("CUDA Q5_K down input cols={}，weight cols={cols}", input.cols) });
                }
                ops::linear::linear_q5_k_f16(ctx, input, data, *rows).map_err(|msg| BackendError::Compute { msg })
            }
            Self::Q6K { data, rows, cols } => {
                if input.cols != *cols {
                    return Err(BackendError::Compute { msg: format!("CUDA Q6_K down input cols={}，weight cols={cols}", input.cols) });
                }
                ops::linear::linear_q6_k_f16(ctx, input, data, *rows).map_err(|msg| BackendError::Compute { msg })
            }
        }
    }

    fn accumulate(&self, ctx: &CudaContext, input: &CudaTensor, output: &CudaMoeAccumulator, route_weights: &CudaSlice<f32>, route: usize) -> Result<(), BackendError> {
        match self {
            Self::Packed { .. } => {
                let down = self.linear(ctx, input)?;
                ops::routing::scatter_add_route_f32(ctx, &output.data, &down, route_weights, route).map_err(|msg| BackendError::Compute { msg })
            }
            Self::F16(weight) => {
                let down = ctx.linear(input, weight)?;
                ops::routing::scatter_add_route_f32(ctx, &output.data, &down, route_weights, route).map_err(|msg| BackendError::Compute { msg })
            }
            Self::Q5K { data, rows, cols } => {
                if input.cols != *cols {
                    return Err(BackendError::Compute { msg: format!("CUDA Q5_K down input cols={}，weight cols={cols}", input.cols) });
                }
                ops::linear::linear_q5_k_accumulate_f32(ctx, input, data, &output.data, *rows, route_weights, route).map_err(|msg| BackendError::Compute { msg })
            }
            Self::Q6K { data, rows, cols } => {
                if input.cols != *cols {
                    return Err(BackendError::Compute { msg: format!("CUDA Q6_K down input cols={}，weight cols={cols}", input.cols) });
                }
                ops::linear::linear_q6_k_accumulate_f32(ctx, input, data, &output.data, *rows, route_weights, route).map_err(|msg| BackendError::Compute { msg })
            }
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Self::F16(weight) => weight.rows * weight.cols * std::mem::size_of::<f16>(),
            Self::Packed { data, .. } | Self::Q5K { data, .. } | Self::Q6K { data, .. } => data.len(),
        }
    }
}

#[derive(Clone)]
struct CudaExpertWeights {
    // 活跃 batch 延长缓存权重生命周期,不能 clone CudaSlice 触发整份 D2D 复制。
    gate_up: Arc<CudaGateUp>,
    down: Arc<CudaDown>,
    bytes: usize,
    used_at: u64,
    frequency: u32,
}

impl CudaExpertWeights {
    /// 只覆写没有活跃 batch 引用的同布局权重;同 stream 的上传在旧计算之后执行。
    fn reload_packed(&mut self, ctx: &CudaContext, matrices: &GgufExpertWeights) -> Result<bool, BackendError> {
        // 旧 copy-stream 路线的生命周期另有约束,这里仅复用单计算流的缓冲。
        if std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() {
            return Ok(false);
        }
        let (Some(gate_up), Some(down)) = (Arc::get_mut(&mut self.gate_up), Arc::get_mut(&mut self.down)) else {
            return Ok(false);
        };
        let (CudaGateUp::Q4K { gate, up, rows, cols }, CudaDown::Packed { data, rows: down_rows, tensor_type }) = (gate_up, down) else {
            return Ok(false);
        };
        if matrices.gate.tensor_type.0 != 12
            || matrices.up.tensor_type.0 != 12
            || matrices.down.tensor_type.0 != *tensor_type
            || matrices.gate.rows != *rows
            || matrices.up.rows != *rows
            || matrices.gate.columns != *cols
            || matrices.up.columns != *cols
            || matrices.down.rows != *down_rows
            || matrices.down.columns != *rows
            || gate.len() != matrices.gate.storage_len()
            || up.len() != matrices.up.storage_len()
            || data.len() != matrices.down.storage_len()
        {
            return Ok(false);
        }
        ctx.upload_u8_pinned(matrices.down.bytes().map_err(BackendError::ExpertLoad)?, data).map_err(BackendError::ExpertLoad)?;
        ctx.upload_u8_pinned(matrices.gate.bytes().map_err(BackendError::ExpertLoad)?, gate).map_err(BackendError::ExpertLoad)?;
        ctx.upload_u8_pinned(matrices.up.bytes().map_err(BackendError::ExpertLoad)?, up).map_err(BackendError::ExpertLoad)?;
        Ok(true)
    }
}

pub struct CudaMoeState {
    cache: HashMap<(usize, usize), CudaExpertWeights>,
    lru: VecDeque<(u64, (usize, usize))>,
    bytes: usize,
    max_bytes: usize,
    arena: Option<Arc<CudaExpertArena>>,
    transfer_group: usize,
    clock: u64,
    accumulator: Option<CudaSlice<f32>>,
    /// 累计 miss(load)次数,用于命中率观测。
    pub loaded: usize,
    /// 累计上传的实际编码字节,包含预取 miss。
    pub uploaded_bytes: u64,
    /// 显式开启的分段同步诊断,默认关闭。
    pub profile: bool,
    pub upload_wall: f64,
    pub compute_wall: f64,
    /// 可选频次优先缓存;同频时仍按 LRU,频次定期衰减以适应上下文变化。
    pub frequency_cache: bool,
    frequencies: HashMap<(usize, usize), u32>,
}

impl CudaMoeState {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            cache: HashMap::new(),
            lru: VecDeque::new(),
            bytes: 0,
            max_bytes,
            arena: None,
            transfer_group: 0,
            clock: 0,
            accumulator: None,
            loaded: 0,
            uploaded_bytes: 0,
            profile: false,
            upload_wall: 0.0,
            compute_wall: 0.0,
            frequency_cache: false,
            frequencies: HashMap::new(),
        }
    }

    fn record_touch(&mut self, key: (usize, usize)) {
        self.lru.push_back((self.clock, key));
        // 重复命中留下的旧记录可丢弃,把队列空间限制在缓存规模的常数倍。
        if self.lru.len() > 4 * self.cache.len().max(1) {
            self.lru.retain(|(stamp, key)| self.cache.get(key).is_some_and(|weight| weight.used_at == *stamp));
        }
    }

    fn oldest_key(&mut self) -> Result<(usize, usize), BackendError> {
        if self.frequency_cache {
            return self.cache.iter().min_by_key(|(_, weight)| (weight.frequency, weight.used_at)).map(|(&key, _)| key).ok_or_else(|| BackendError::ExpertLoad("CUDA expert cache 无法释放足够空间".to_owned()));
        }
        while let Some((stamp, key)) = self.lru.pop_front() {
            if self.cache.get(&key).is_some_and(|weight| weight.used_at == stamp) {
                return Ok(key);
            }
        }
        Err(BackendError::ExpertLoad("CUDA expert LRU 队列缺少有效缓存记录".to_owned()))
    }

    pub(crate) fn reserve_arena(&mut self, ctx: &CudaContext) -> Result<(), String> {
        if !self.cache.is_empty() || self.arena.is_some() || std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() {
            return Err("CUDA expert arena 必须在空缓存、单计算流上初始化".into());
        }
        let data = ctx.buffer_uninit(self.max_bytes)?;
        ctx.stream().synchronize().map_err(|e| format!("CUDA expert arena 分配同步: {e:?}"))?;
        self.arena = Some(Arc::new(CudaExpertArena { data, free: Mutex::new(BTreeMap::from([(0, self.max_bytes)])) }));
        Ok(())
    }

    /// 保持每行专家累加顺序,在当前组计算期间上传下一组。
    fn packed_groups(&mut self, ctx: &CudaContext, source: &dyn GgufExpertSource, layer: usize, input: &CudaTensor, ids: &[u32], routes: &CudaSlice<f32>, top_k: usize, first: &GgufExpertWeights) -> Result<CudaTensor, BackendError> {
        ctx.stream().synchronize().map_err(|e| crate::backend::compute_error(format!("CUDA overlap 旧专家读者同步: {e:?}")))?;
        let order: Vec<Vec<_>> = ids
            .chunks_exact(top_k)
            .enumerate()
            .map(|(row, ids)| {
                let mut order: Vec<_> = ids.iter().enumerate().map(|(route, &id)| (id, row * top_k + route)).collect();
                order.sort_unstable_by_key(|&(id, _)| id);
                order
            })
            .collect();
        // 所有已经提交计算的区域都保留到整层结束,缓存驱逐不会覆写在途读者。
        let mut active = HashMap::new();
        let mut output = None;
        for start in (0..top_k).step_by(self.transfer_group) {
            let end = (start + self.transfer_group).min(top_k);
            for row in &order {
                for &(id, _) in &row[start..end] {
                    if !active.contains_key(&id) {
                        self.ensure_with_upload(ctx, source, layer, id as usize, true, true)?;
                        active.insert(id, self.cache[&(layer, id as usize)].clone());
                    }
                }
            }
            let mut references = Vec::with_capacity(input.rows * (end - start));
            for row in &order {
                for &(id, route) in &row[start..end] {
                    let expert = &active[&id];
                    match (expert.gate_up.as_ref(), expert.down.as_ref()) {
                        (CudaGateUp::Q4K { gate, up, rows, cols }, CudaDown::Packed { data, rows: out, tensor_type })
                            if *rows == first.gate.rows && *cols == input.cols && *out == first.down.rows && *tensor_type == first.down.tensor_type.0 =>
                        {
                            references.push((&**gate, &**up, &**data, route))
                        }
                        _ => return Err(crate::backend::compute_error(format!("CUDA overlap L{layer} E{id} 编码或形状不一致"))),
                    }
                }
            }
            output = Some(ops::linear::moe_q4k_block32_accumulate(ctx, input, &references, first.gate.rows, first.down.rows, first.down.tensor_type.0, routes, output.as_ref()).map_err(crate::backend::compute_error)?);
        }
        output.ok_or_else(|| crate::backend::compute_error("CUDA overlap 缺少专家"))
    }

    pub fn resident_bytes(&self) -> usize {
        self.bytes
    }

    fn required_bytes(matrices: &crate::weight::expert_source::GgufExpertWeights, allow_q4_k: bool) -> Result<usize, BackendError> {
        let f16_bytes = |rows: usize, columns: usize| rows.checked_mul(columns).and_then(|elements| elements.checked_mul(std::mem::size_of::<f16>())).ok_or_else(|| BackendError::ExpertLoad("CUDA expert 驻留大小溢出".to_owned()));
        let down = if allow_q4_k && matches!(matrices.down.tensor_type.0, 7 | 8 | 13 | 14) { matrices.down.storage_len() } else { f16_bytes(matrices.down.rows, matrices.down.columns)? };
        let gate_up = if allow_q4_k && matches!(matrices.gate.tensor_type.0, 12 | 13) && matrices.up.tensor_type == matrices.gate.tensor_type && matrices.gate.rows == matrices.up.rows && matrices.gate.columns == matrices.up.columns {
            matrices.gate.storage_len().checked_add(matrices.up.storage_len())
        } else {
            f16_bytes(matrices.gate.rows, matrices.gate.columns)?.checked_add(f16_bytes(matrices.up.rows, matrices.up.columns)?)
        }
        .ok_or_else(|| BackendError::ExpertLoad("CUDA expert gate/up 驻留大小溢出".to_owned()))?;
        gate_up.checked_add(down).ok_or_else(|| BackendError::ExpertLoad("CUDA expert 总驻留大小溢出".to_owned()))
    }

    fn load(ctx: &CudaContext, matrices: GgufExpertWeights, layer: usize, expert: usize, allow_q4_k: bool, region: Option<Arc<CudaExpertRegion>>, overlap: bool) -> Result<CudaExpertWeights, BackendError> {
        // packed 字节走 pinned 中转:pageable clone_htod 在 PCIe3 上带宽仅 ~1/3
        // 且有 SyncOnDrop stall,expert 流式路径对 H2D 带宽敏感。
        let mut cursor = 0usize;
        let mut upload_u8 = |bytes: &[u8]| -> Result<CudaExpertBuffer, BackendError> {
            // copy-on:DMA 目标块与 DMA 同在 copy 流 mallocAsync——块的 alloc/DMA/free
            // 依赖闭合在 copy 流 FIFO 内,池复用不会撞上在途 DMA。此前块在计算流分配、
            // copy 流裸 DMA 写入:池对跨流写入不可见,块驱逐后复用给 expert 计算输出,
            // 被迟到 DMA 写坏(实测 copy-on 首 decode 层 activated 686/1024 nonfinite
            // 而全部 expert 权重字节逐位完好)。
            let stream: &Arc<cudarc::driver::safe::CudaStream> = if std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() { ctx.copy_stream() } else { ctx.stream() };
            let mut slice = if let Some(region) = &region {
                if cursor + bytes.len() > region.bytes {
                    return Err(BackendError::ExpertLoad("CUDA expert 超过 arena 区域".into()));
                }
                let (pointer, _read) = region.arena.data.device_ptr(ctx.stream());
                // 区域持续持有唯一底层分配;每个子区间不重叠且按 256 bytes 对齐。
                let view = unsafe { ctx.stream().upgrade_device_ptr(pointer + (region.offset + cursor) as u64, bytes.len()) };
                cursor += bytes.len().next_multiple_of(256);
                CudaExpertBuffer::Arena { view: ManuallyDrop::new(view), _region: region.clone() }
            } else {
                let data = unsafe { stream.alloc(bytes.len()) }.map_err(|error| BackendError::ExpertLoad(format!("分配 expert L{layer} E{expert} buffer: {error:?}; {}", ctx.memory_diagnostics())))?;
                CudaExpertBuffer::Owned(data)
            };
            if overlap {
                // 整层入口已同步旧读者,active 引用保证当前层区域不能被回收。
                unsafe { ctx.upload_registered_expert_overlap(bytes, &mut slice) }.map_err(BackendError::ExpertLoad)?;
            } else {
                ctx.upload_u8_pinned(bytes, &mut slice).map_err(BackendError::ExpertLoad)?;
            }
            Ok(slice)
        };
        let decode = |matrix: crate::weight::container::gguf::GgufMatrix| {
            let rows = matrix.rows;
            let cols = matrix.columns;
            let data = matrix.decode().map_err(BackendError::ExpertLoad)?;
            let data = data.into_iter().map(f16::from_f32).collect::<Vec<_>>();
            // 其它格式的 F16 回退同样走 pinned 上传:pageable clone_htod 在
            // 与 copy 流 pinned DMA 并发时存在驱动 staging 干扰(实测 L47 全 NaN),
            // 且本身在重型流上有 SyncOnDrop stall。
            let mut slice = ctx.buffer_uninit::<f16>(data.len()).map_err(|error| BackendError::ExpertLoad(format!("分配 F16 expert buffer: {error:?}")))?;
            ctx.upload_f16_pinned(&data, &mut slice).map_err(BackendError::ExpertLoad)?;
            Ok(CudaWeight::new(slice, rows, cols))
        };
        let down = if allow_q4_k && matches!(matrices.down.tensor_type.0, 7 | 8) {
            let data = upload_u8(matrices.down.bytes().map_err(BackendError::ExpertLoad)?)?;
            CudaDown::Packed { data, rows: matrices.down.rows, tensor_type: matrices.down.tensor_type.0 }
        } else if allow_q4_k && matrices.down.tensor_type.0 == 13 {
            let data = upload_u8(matrices.down.bytes().map_err(BackendError::ExpertLoad)?)?;
            CudaDown::Q5K { data, rows: matrices.down.rows, cols: matrices.down.columns }
        } else if allow_q4_k && matrices.down.tensor_type.0 == 14 {
            let data = upload_u8(matrices.down.bytes().map_err(BackendError::ExpertLoad)?)?;
            CudaDown::Q6K { data, rows: matrices.down.rows, cols: matrices.down.columns }
        } else {
            CudaDown::F16(decode(matrices.down)?)
        };
        let gate_up = if allow_q4_k && matches!(matrices.gate.tensor_type.0, 12 | 13) && matrices.up.tensor_type == matrices.gate.tensor_type && matrices.gate.rows == matrices.up.rows && matrices.gate.columns == matrices.up.columns {
            let gate = upload_u8(matrices.gate.bytes().map_err(BackendError::ExpertLoad)?)?;
            let up = upload_u8(matrices.up.bytes().map_err(BackendError::ExpertLoad)?)?;
            if matrices.gate.tensor_type.0 == 12 { CudaGateUp::Q4K { gate, up, rows: matrices.gate.rows, cols: matrices.gate.columns } } else { CudaGateUp::Q5K { gate, up, rows: matrices.gate.rows } }
        } else {
            CudaGateUp::F16 { gate: decode(matrices.gate)?, up: decode(matrices.up)? }
        };
        let bytes = gate_up.bytes() + down.bytes();
        Ok(CudaExpertWeights { gate_up: Arc::new(gate_up), down: Arc::new(down), bytes, used_at: 0, frequency: 0 })
    }

    fn ensure_on_stream(&mut self, ctx: &CudaContext, source: &dyn GgufExpertSource, layer: usize, expert: usize, allow_q4_k: bool) -> Result<(), BackendError> {
        self.ensure_with_upload(ctx, source, layer, expert, allow_q4_k, false)
    }

    fn ensure_with_upload(&mut self, ctx: &CudaContext, source: &dyn GgufExpertSource, layer: usize, expert: usize, allow_q4_k: bool, overlap: bool) -> Result<(), BackendError> {
        if overlap && self.arena.is_none() {
            return Err(BackendError::ExpertLoad("CUDA overlap 需要固定专家 arena".into()));
        }
        let key = (layer, expert);
        let frequency = if self.frequency_cache {
            if self.clock != 0 && self.clock.is_multiple_of(16_384) {
                for value in self.frequencies.values_mut() {
                    *value = (*value / 2).max(1);
                }
                for cached in self.cache.values_mut() {
                    cached.frequency = (cached.frequency / 2).max(1);
                }
            }
            let value = self.frequencies.entry(key).or_default();
            *value = value.saturating_add(1);
            *value
        } else {
            0
        };
        if let Some(cached) = self.cache.get_mut(&key) {
            cached.frequency = frequency;
            self.clock = self.clock.wrapping_add(1);
            cached.used_at = self.clock;
            self.record_touch(key);
            return Ok(());
        }
        let matrices = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
        if self.arena.is_some()
            && (!allow_q4_k
                || !matches!(matrices.gate.tensor_type.0, 12 | 13)
                || matrices.up.tensor_type != matrices.gate.tensor_type
                || matrices.up.rows != matrices.gate.rows
                || matrices.up.columns != matrices.gate.columns
                || !matches!(matrices.down.tensor_type.0, 7 | 8 | 13 | 14))
        {
            return Err(BackendError::ExpertLoad(format!("CUDA expert L{layer} E{expert} arena 只支持完整 packed gate/up/down,禁止 F16 回退")));
        }
        let bytes = Self::required_bytes(&matrices, allow_q4_k)?;
        if bytes > self.max_bytes {
            return Err(BackendError::ExpertLoad(format!("CUDA expert L{layer} E{expert} 超过 cache 上限")));
        }
        let region_bytes = [matrices.gate.storage_len(), matrices.up.storage_len(), matrices.down.storage_len()].iter().map(|bytes| bytes.next_multiple_of(256)).sum();
        let mut region = self.arena.as_ref().map(|arena| arena.allocate(region_bytes)).transpose()?.flatten();
        let mut recycled = None;
        while self.bytes > self.max_bytes - bytes || (self.arena.is_some() && region.is_none()) {
            if std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() {
                // copy-on:块宿主在 copy 流,DMA 与 freeAsync 同流序化(无需 copy 栅栏);
                // 但读块的 kernel 在计算流,free 前让 copy 流等计算流,防止池复用后
                // 在途 kernel 读取被后续写入踩坏。
                ctx.fence_compute_before_copy().map_err(BackendError::ExpertLoad)?;
            }
            let oldest = self.oldest_key()?;
            let removed = self.cache.remove(&oldest).expect("CUDA expert LRU key 必须存在");
            self.bytes -= removed.bytes;
            if let Some(arena) = &self.arena {
                drop(removed);
                if region.is_none() {
                    region = arena.allocate(region_bytes)?;
                }
            } else {
                recycled = Some(removed);
            }
        }
        let mut weights = if allow_q4_k && recycled.as_mut().map(|weights| weights.reload_packed(ctx, &matrices)).transpose()?.unwrap_or(false) {
            recycled.unwrap()
        } else {
            drop(recycled);
            Self::load(ctx, matrices, layer, expert, allow_q4_k, region, overlap).map_err(|error| BackendError::ExpertLoad(format!("{error}; cache_bytes={} budget={} entries={}", self.bytes, self.max_bytes, self.cache.len())))?
        };
        self.loaded += 1;
        self.uploaded_bytes += weights.bytes as u64;
        debug_assert_eq!(weights.bytes, bytes);
        self.clock = self.clock.wrapping_add(1);
        weights.used_at = self.clock;
        weights.frequency = frequency;
        self.bytes += weights.bytes;
        self.cache.insert(key, weights);
        self.record_touch(key);
        Ok(())
    }
}

impl MoePrefillBackend for CudaContext {
    type MoeAccumulator = CudaMoeAccumulator;

    fn moe_route(&self, input: &CudaTensor, router_weight: &CudaWeight, router_bias: &CudaWeight, spec: &TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        if !matches!(spec.scoring_func, ScoringFunc::Softmax | ScoringFunc::SigmoidBias) {
            return Err(BackendError::Compute { msg: format!("CUDA MoE router 不支持 {:?} 评分", spec.scoring_func) });
        }
        if router_weight.rows != spec.num_experts || router_weight.cols != input.cols || router_bias.rows * router_bias.cols != spec.num_experts {
            return Err(BackendError::Compute {
                msg: format!("CUDA router shape input=[{},{}] weight=[{},{}] bias=[{},{}] experts={} 不一致", input.rows, input.cols, router_weight.rows, router_weight.cols, router_bias.rows, router_bias.cols, spec.num_experts,),
            });
        }
        let router_weight = router_weight.data_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "CUDA router weight 缺少 F32 device storage".to_owned() })?;
        let router_bias = router_bias.data_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "CUDA router bias 缺少 F32 device storage".to_owned() })?;
        let (expert_ids, weights) = match spec.scoring_func {
            ScoringFunc::SigmoidBias if std::env::var_os("ZLLM_CUDA_HOST_ROUTE").is_none() => {
                // sigmoid 路由恒按选中集合归一化(DeepSeek noaux_tc / Laguna 语义)。
                ops::routing::moe_route_sigmoid_bias_topk_f32(self, input, router_weight, router_bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor).map_err(|msg| BackendError::Compute { msg })?
            }
            ScoringFunc::SigmoidBias => {
                // 消元开关 ZLLM_CUDA_HOST_ROUTE:logits 走 cuBLAS,路由在 host 用
                // 参考实现(route_sigmoid_bias_logits),排除设备路由 kernel。
                let logits = crate::kernel::cuda::linear::cublas_matmul_control_f32(self, input, router_weight, spec.num_experts).map_err(|msg| BackendError::Compute { msg })?;
                let logits = logits.slice_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "host route 未返回 F32 logits".to_owned() })?;
                let host_logits = self.stream().clone_dtoh(logits).map_err(|msg| BackendError::Compute { msg: format!("{msg:?}") })?;
                let host_bias = self.stream().clone_dtoh(router_bias).map_err(|msg| BackendError::Compute { msg: format!("{msg:?}") })?;
                let mut ids = Vec::with_capacity(input.rows * spec.top_k);
                let mut ws = Vec::with_capacity(input.rows * spec.top_k);
                for row in 0..input.rows {
                    // 参考函数按单 token 设计,逐行调用。
                    let row_logits = &host_logits[row * spec.num_experts..(row + 1) * spec.num_experts];
                    let routing = crate::moe::routing::route_sigmoid_bias_logits(row_logits, &host_bias, spec.top_k, spec.routed_scaling_factor).map_err(|msg| BackendError::Compute { msg })?;
                    ids.extend(routing.experts.iter().map(|&expert| expert as u32));
                    ws.extend(routing.weights.iter().copied());
                }
                (ids, ws)
            }
            _ => ops::routing::moe_route_softmax_topk_f32(self, input, router_weight, router_bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected).map_err(|msg| BackendError::Compute { msg })?,
        };
        Ok(MoePrefillRouting { expert_ids, weights, rows: input.rows, top_k: spec.top_k })
    }

    fn moe_zeros(&self, rows: usize, cols: usize) -> Result<CudaMoeAccumulator, BackendError> {
        let count = rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: "CUDA MoE accumulator 大小溢出".to_owned() })?;
        let data = self.stream().alloc_zeros::<f32>(count).map_err(|error| BackendError::Compute { msg: format!("CUDA MoE accumulator 分配失败: {error:?}") })?;
        Ok(CudaMoeAccumulator { data, rows, cols })
    }

    fn moe_gather_rows(&self, input: &CudaTensor, rows: &[u32]) -> Result<CudaTensor, BackendError> {
        ops::routing::gather_rows_f16(self, input, rows).map_err(|msg| BackendError::Compute { msg })
    }

    fn moe_gather_rows_batch(&self, input: &CudaTensor, batches: &[Vec<u32>]) -> Result<Vec<CudaTensor>, BackendError> {
        batches.iter().map(|rows| self.moe_gather_rows(input, rows)).collect()
    }

    fn moe_scatter_add_rows(&self, output: &mut CudaMoeAccumulator, input: &CudaTensor, rows: &[u32], weights: &[f32]) -> Result<(), BackendError> {
        ops::routing::scatter_add_rows_f32(self, &output.data, output.rows, output.cols, input, rows, weights).map_err(|msg| BackendError::Compute { msg })
    }

    fn moe_scatter_add_rows_batch(&self, output: &mut CudaMoeAccumulator, inputs: &[CudaTensor], rows: &[Vec<u32>], weights: &[Vec<f32>]) -> Result<(), BackendError> {
        if inputs.len() != rows.len() || rows.len() != weights.len() {
            return Err(BackendError::Compute { msg: format!("CUDA MoE batch scatter 数量异常: inputs={}, rows={}, weights={}", inputs.len(), rows.len(), weights.len()) });
        }
        for ((input, rows), weights) in inputs.iter().zip(rows).zip(weights) {
            self.moe_scatter_add_rows(output, input, rows, weights)?;
        }
        Ok(())
    }

    fn moe_finish(&self, output: CudaMoeAccumulator) -> Result<CudaTensor, BackendError> {
        // f32 残差流:MoE 求和直接以 f32 张量交付,f16 悬崖(Laguna L46)从路径中消失。
        let placeholder = self.placeholder_f16().map_err(|msg| BackendError::Compute { msg })?;
        Ok(CudaTensor::new_f32_residual(output.data, placeholder, output.rows, output.cols))
    }
}

impl ExpertPrefillBackend for CudaContext {
    type PrefillExperts = CudaPrefillExperts;

    fn prefill_resident_routed_experts(
        &self,
        spec: &TopkMoeSpec,
        weights: crate::moe::topk_moe::RoutedMoeWeightsRef<'_, CudaWeight>,
        layer: usize,
        experts: &mut CudaPrefillExperts,
        inputs: crate::moe::topk_moe::RoutedMoeInputs<'_, CudaTensor>,
    ) -> Result<Option<CudaTensor>, BackendError> {
        let input = inputs.expert;
        if input.rows == 0
            || input.rows > 8
            || input.slice_f32.is_some()
            || inputs.route.slice_f32.is_some()
            || !matches!(spec.scoring_func, ScoringFunc::Softmax)
            || !matches!(spec.activation, Activation::Silu)
            || weights.selected_experts.is_some()
            || std::env::var_os("ZLLM_CUDA_F16_EXPERTS").is_some()
        {
            return Ok(None);
        }
        let first = experts.source.load_expert_gguf(layer, 0).map_err(BackendError::ExpertLoad)?;
        if first.gate.tensor_type.0 != 12 || first.up.tensor_type.0 != 12 || !matches!(first.down.tensor_type.0, 7 | 8) || spec.top_k * first.gate.rows * 2 > 48 * 1024 {
            return Ok(None);
        }
        let router = weights.router.data_f32.as_ref().ok_or_else(|| crate::backend::compute_error("CUDA MoE router 缺少 F32"))?;
        let bias = weights.bias.data_f32.as_ref().ok_or_else(|| crate::backend::compute_error("CUDA MoE bias 缺少 F32"))?;
        let (ids, route_weights) =
            ops::routing::moe_route_softmax_topk_device_f32(self, inputs.route, router, bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected).map_err(crate::backend::compute_error)?;
        if experts.state.transfer_group > 0 {
            return experts.state.packed_groups(self, experts.source.as_ref(), layer, input, &ids, &route_weights, spec.top_k, &first).map(Some);
        }
        let mut selected = ids.clone();
        selected.sort_unstable();
        selected.dedup();
        let mut active = HashMap::with_capacity(selected.len());
        for expert in selected {
            experts.state.ensure_on_stream(self, experts.source.as_ref(), layer, expert as usize, true)?;
            active.insert(expert, experts.state.cache[&(layer, expert as usize)].clone());
        }
        if std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() {
            self.wait_copy_fence().map_err(BackendError::ExpertLoad)?;
        }
        let mut references = Vec::with_capacity(ids.len());
        for (row, row_ids) in ids.chunks_exact(spec.top_k).enumerate() {
            let mut order: Vec<_> = row_ids.iter().enumerate().collect();
            order.sort_unstable_by_key(|(_, id)| **id);
            for (route, id) in order {
                let expert = &active[id];
                match (expert.gate_up.as_ref(), expert.down.as_ref()) {
                    (CudaGateUp::Q4K { gate, up, rows, cols }, CudaDown::Packed { data, rows: out, tensor_type }) if *rows == first.gate.rows && *cols == input.cols && *out == first.down.rows && *tensor_type == first.down.tensor_type.0 => {
                        references.push((&**gate, &**up, &**data, row * spec.top_k + route))
                    }
                    _ => return Ok(None),
                }
            }
        }
        ops::linear::moe_q4k_block32(self, input, &references, first.gate.rows, first.down.rows, first.down.tensor_type.0, &route_weights).map(Some).map_err(crate::backend::compute_error)
    }

    fn prefill_expert_batch(&self, spec: &TopkMoeSpec, layer: usize, experts: &mut CudaPrefillExperts, batch: Vec<ExpertPrefillBatch<CudaTensor>>) -> Result<Vec<CudaTensor>, BackendError> {
        batch
            .into_iter()
            .map(|item| {
                let allow_quantized = matches!(spec.activation, Activation::Silu) && std::env::var_os("ZLLM_CUDA_F16_EXPERTS").is_none();
                let source_matrix = experts.source.load_expert_gguf(layer, item.expert).map_err(BackendError::ExpertLoad)?;
                experts.state.ensure_on_stream(self, experts.source.as_ref(), layer, item.expert, allow_quantized)?;
                if std::env::var_os("ZLLM_CUDA_EXPERT_VERIFY").is_some() {
                    // 判别审计:比对缓存内 packed 字节与 GGUF 源字节(首 64 字节)。
                    let cached = experts.state.cache.get(&(layer, item.expert)).expect("ensure 后必须驻留");
                    let expect_gate = source_matrix.gate.bytes().map_err(BackendError::ExpertLoad)?;
                    match cached.gate_up.as_ref() {
                        CudaGateUp::Q4K { gate, .. } => {
                            let got = self.stream().clone_dtoh(&**gate).map_err(|e| BackendError::Compute { msg: format!("{e:?}") })?;
                            let mismatch = expect_gate.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
                            eprintln!(
                                "[expert-verify] L{layer} E{} gate 前64字节 mismatch={}/64 (源前4={:02x}{:02x}{:02x}{:02x} 缓存前4={:02x}{:02x}{:02x}{:02x})",
                                item.expert,
                                mismatch.min(64),
                                expect_gate.get(0).unwrap_or(&0),
                                expect_gate.get(1).unwrap_or(&0),
                                expect_gate.get(2).unwrap_or(&0),
                                expect_gate.get(3).unwrap_or(&0),
                                got.first().unwrap_or(&0),
                                got.get(1).unwrap_or(&0),
                                got.get(2).unwrap_or(&0),
                                got.get(3).unwrap_or(&0)
                            );
                        }
                        _ => eprintln!("[expert-verify] L{layer} E{} 非 Q4K 路径,跳过", item.expert),
                    }
                }
                if std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() {
                    self.wait_copy_fence().map_err(BackendError::ExpertLoad)?;
                }
                let weights = experts.state.cache.get(&(layer, item.expert)).expect("CUDA prefill expert ensure 后必须存在").clone();
                let activated = weights.gate_up.activated(self, &item.input, &spec.activation)?;
                weights.down.linear(self, &activated)
            })
            .collect()
    }
}

impl ExpertDecodeBackend for CudaContext {
    type MoeState = CudaMoeState;
    type DecodeRouting = CudaDecodeRouting;

    fn decode_route(&self, input: &CudaTensor, router_weight: &CudaWeight, router_bias: &CudaWeight, spec: &TopkMoeSpec) -> Result<(MoePrefillRouting, Self::DecodeRouting), BackendError> {
        if !matches!(spec.scoring_func, ScoringFunc::Softmax | ScoringFunc::SigmoidBias) {
            return Err(BackendError::Compute { msg: format!("CUDA MoE decode router 不支持 {:?} 评分", spec.scoring_func) });
        }
        if router_weight.rows != spec.num_experts || router_weight.cols != input.cols || router_bias.rows * router_bias.cols != spec.num_experts {
            return Err(BackendError::Compute { msg: "CUDA MoE decode router shape 不一致".to_owned() });
        }
        let router_weight = router_weight.data_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "CUDA decode router weight 缺少 F32 device storage".to_owned() })?;
        let router_bias = router_bias.data_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "CUDA decode router bias 缺少 F32 device storage".to_owned() })?;
        let (expert_ids, route_weights) = match spec.scoring_func {
            ScoringFunc::SigmoidBias => ops::routing::moe_route_sigmoid_bias_topk_device_f32(self, input, router_weight, router_bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor).map_err(|msg| BackendError::Compute { msg })?,
            _ => ops::routing::moe_route_softmax_topk_device_f32(self, input, router_weight, router_bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected).map_err(|msg| BackendError::Compute { msg })?,
        };
        let grouped = MoePrefillRouting { expert_ids: expert_ids.iter().map(|&expert| expert as u32).collect(), weights: vec![0.0; expert_ids.len()], rows: input.rows, top_k: spec.top_k };
        Ok((grouped, CudaDecodeRouting { expert_ids, route_weights }))
    }

    fn prefetch_experts(&self, spec: &TopkMoeSpec, state: &mut CudaMoeState, request: ExpertPrefetchRequest<'_>) -> Result<usize, BackendError> {
        let source = request.source.require_gguf().map_err(BackendError::ExpertLoad)?;
        let mut loaded = 0;
        for expert in request.experts {
            if !state.cache.contains_key(&(request.layer, expert)) {
                loaded += 1;
            }
            state.ensure_on_stream(self, source, request.layer, expert, matches!(spec.activation, Activation::Silu))?;
        }
        Ok(loaded)
    }

    fn decode_routed_experts<'a, F>(
        &self,
        spec: &TopkMoeSpec,
        layer: usize,
        source: ExpertSource<'_>,
        state: &mut CudaMoeState,
        input: &CudaTensor,
        assignments: &ExpertAssignments,
        routing: &Self::DecodeRouting,
        on_ready: F,
    ) -> Result<CudaTensor, BackendError>
    where
        F: FnOnce(&mut CudaMoeState) -> Result<Option<ExpertPrefetchRequest<'a>>, BackendError>,
    {
        let source = source.require_gguf().map_err(BackendError::ExpertLoad)?;
        let verify = std::env::var_os("ZLLM_CUDA_DECODE_EXPERT_VERIFY").is_some();
        if state.profile {
            self.synchronize().map_err(|error| crate::backend::compute_error(format!("CUDA profile 同步: {error:?}")))?;
        }
        if state.transfer_group > 0 && !verify && !state.profile && input.rows == 1 && matches!(spec.activation, Activation::Silu) {
            let first = source.load_expert_gguf(layer, 0).map_err(BackendError::ExpertLoad)?;
            if first.gate.tensor_type.0 == 12 && first.up.tensor_type.0 == 12 && matches!(first.down.tensor_type.0, 7 | 8) {
                let result = state.packed_groups(self, source, layer, input, &routing.expert_ids, &routing.route_weights, spec.top_k, &first)?;
                if let Some(request) = on_ready(state)? {
                    self.prefetch_experts(spec, state, request)?;
                }
                return Ok(result);
            }
        }
        let upload_start = std::time::Instant::now();
        let mut active = Vec::new();
        for (expert, assigned) in assignments.iter().enumerate() {
            if assigned.is_empty() {
                continue;
            }
            state.ensure_on_stream(self, source, layer, expert, matches!(spec.activation, Activation::Silu))?;
            let weights = state.cache.get(&(layer, expert)).expect("CUDA active expert 必须驻留").clone();
            let route = routing.expert_ids.iter().position(|&routed| routed as usize == expert).ok_or_else(|| BackendError::Compute { msg: format!("CUDA route 缺少 expert {expert}") })?;
            active.push((route, expert, weights));
        }
        if let Some(request) = on_ready(state)? {
            // 与 ROCm 路径一致:prefetch_experts 在当前 stream 上同步 ensure 预测专家,
            // 不能静默丢弃请求,否则预测预取在 CUDA 上完全不生效。
            self.prefetch_experts(spec, state, request)?;
        }

        if std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() {
            // expert 权重由 copy 流异步装载;进入计算前建立跨流依赖(幂等)。
            self.wait_copy_fence().map_err(BackendError::ExpertLoad)?;
        }
        if state.profile {
            self.synchronize().map_err(|error| crate::backend::compute_error(format!("CUDA profile 同步: {error:?}")))?;
            state.upload_wall += upload_start.elapsed().as_secs_f64();
        }
        let compute_start = std::time::Instant::now();
        if verify {
            // 取证:fence 后逐 expert 比对 packed 字节(clone_dtoh 走计算流,fence 有效
            // 则 mismatch 恒 0;mismatch>0 即 fence 失效/DMA 数据坏)。
            for (_, expert, weights) in &active {
                let matrices = source.load_expert_gguf(layer, *expert).map_err(BackendError::ExpertLoad)?;
                if let (CudaGateUp::Q4K { gate, .. }, Ok(expect)) = (weights.gate_up.as_ref(), matrices.gate.bytes()) {
                    let got = self.stream().clone_dtoh(&**gate).map_err(|e| BackendError::Compute { msg: format!("verify dtoh: {e:?}") })?;
                    let mismatch = expect.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
                    eprintln!("[decode-verify] L{layer} E{expert} gate mismatch={}/{}", mismatch, expect.len());
                }
                if let (CudaDown::Q5K { data, .. } | CudaDown::Q6K { data, .. }, Ok(expect)) = (weights.down.as_ref(), matrices.down.bytes()) {
                    let got = self.stream().clone_dtoh(&**data).map_err(|e| BackendError::Compute { msg: format!("verify dtoh: {e:?}") })?;
                    let mismatch = expect.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
                    eprintln!("[decode-verify] L{layer} E{expert} down mismatch={}/{}", mismatch, expect.len());
                }
            }
        }
        if !verify && input.rows == 1 {
            if let Some((_, _, first)) = active.first() {
                if let (CudaGateUp::Q4K { rows: intermediate, cols, .. }, CudaDown::Packed { rows: output_columns, tensor_type, .. }) = (first.gate_up.as_ref(), first.down.as_ref()) {
                    let references = active
                        .iter()
                        .map(|(route, _, expert)| match (expert.gate_up.as_ref(), expert.down.as_ref()) {
                            (CudaGateUp::Q4K { gate, up, rows, cols: c }, CudaDown::Packed { data, rows: out, tensor_type: t }) if rows == intermediate && c == cols && out == output_columns && t == tensor_type => {
                                Some((&**gate, &**up, &**data, *route))
                            }
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(references) = references.filter(|refs| refs.len() * intermediate * 2 <= 48 * 1024) {
                        let result = ops::linear::moe_q4k_block32(self, input, &references, *intermediate, *output_columns, *tensor_type, &routing.route_weights).map_err(crate::backend::compute_error)?;
                        if state.profile {
                            self.synchronize().map_err(|error| crate::backend::compute_error(format!("CUDA profile 同步: {error:?}")))?;
                            state.compute_wall += compute_start.elapsed().as_secs_f64();
                        }
                        return Ok(result);
                    }
                }
            }
        }
        let count = input.rows.checked_mul(input.cols).ok_or_else(|| BackendError::Compute { msg: "CUDA MoE accumulator 大小溢出".to_owned() })?;
        let data = match state.accumulator.take() {
            Some(data) if data.len() == count => data,
            Some(_) | None => self.stream().alloc_zeros::<f32>(count).map_err(|error| BackendError::Compute { msg: format!("CUDA MoE accumulator 分配失败: {error:?}") })?,
        };
        let mut routed = CudaMoeAccumulator { data, rows: input.rows, cols: input.cols };
        if verify {
            // 取证:expert 输入的类型标记与 nonfinite 状态(f32 residual 误入 Q4K kernel
            // 会读 placeholder 越界;乱字节分布也能区分旧字节 vs 坏输入)。
            let nonfinite = self.debug_last_row_f32(input).map(|values| values.iter().filter(|value| !value.is_finite()).count()).unwrap_or(usize::MAX);
            eprintln!("[decode-verify] L{layer} input f32={} nonfinite={}", input.slice_f32.is_some(), nonfinite);
        }
        for (route, expert_id, expert) in active {
            let activated = expert.gate_up.activated(self, input, &spec.activation)?;
            if verify {
                // 无条件打印:区分"仅首个 expert 坏(未写 buffer)"与"全部坏(输入坏)"。
                if let Ok(values) = self.debug_last_row_f32(&activated) {
                    let bad = values.iter().filter(|value| !value.is_finite()).count();
                    eprintln!("[decode-verify] L{layer} E{expert_id} activated nonfinite={}/{}", bad, values.len());
                }
            }
            expert.down.accumulate(self, &activated, &routed, &routing.route_weights, route)?;
        }
        // f32 交付:拷贝出结果(累加器 slice 要复用),另置零新 slice 供下个 token。
        let count = routed.rows.checked_mul(routed.cols).ok_or_else(|| BackendError::Compute { msg: "CUDA decode MoE 大小溢出".to_owned() })?;
        let output_data = ops::tensor::copy_f32_slice(self, &routed.data, count).map_err(|msg| BackendError::Compute { msg })?;
        let zeroed = self.stream().alloc_zeros::<f32>(count).map_err(|e| BackendError::Compute { msg: format!("CUDA decode 累加器清零失败: {e:?}") })?;
        drop(routed.data);
        state.accumulator = Some(zeroed);
        let placeholder = self.placeholder_f16().map_err(|msg| BackendError::Compute { msg })?;
        if state.profile {
            self.synchronize().map_err(|error| crate::backend::compute_error(format!("CUDA profile 同步: {error:?}")))?;
            state.compute_wall += compute_start.elapsed().as_secs_f64();
        }
        Ok(CudaTensor::new_f32_residual(output_data, placeholder, routed.rows, routed.cols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::container::gguf::{GgmlType, GgufReader};
    use cudarc::driver::safe::DevicePtr;

    #[test]
    fn recycled_expert_preserves_active_owners_and_encoded_bytes() {
        // 三份不同字节的 tiny GGUF,第三份切换 down 编码以覆盖不可复用分支。
        let mut file = Vec::from(*b"GGUF");
        file.extend(3u32.to_le_bytes());
        file.extend(9u64.to_le_bytes());
        file.extend(0u64.to_le_bytes());
        let mut payload = Vec::new();
        for expert in 0..3 {
            for (part, rows, columns, kind) in [("gate", 32usize, 256usize, 12), ("up", 32, 256, 12), ("down", 256, 32, if expert == 2 { 8 } else { 7 })] {
                let name = format!("{part}{expert}");
                file.extend((name.len() as u64).to_le_bytes());
                file.extend(name.as_bytes());
                file.extend(2u32.to_le_bytes());
                file.extend((columns as u64).to_le_bytes());
                file.extend((rows as u64).to_le_bytes());
                file.extend((kind as u32).to_le_bytes());
                file.extend((payload.len() as u64).to_le_bytes());
                payload.extend((0..GgmlType(kind).storage_bytes(rows * columns).unwrap()).map(|i| (i * 17 + expert * 59 + part.len()) as u8));
                payload.resize(payload.len().next_multiple_of(32), 0);
            }
        }
        file.resize(file.len().next_multiple_of(32), 0);
        file.extend(payload);
        let path = std::env::temp_dir().join(format!("zllm-cuda-recycle-{}.gguf", std::process::id()));
        std::fs::write(&path, file).unwrap();
        let reader = GgufReader::open(&path).unwrap();
        let source =
            (0..3).map(|id| GgufExpertWeights { gate: reader.read_matrix(&format!("gate{id}")).unwrap(), up: reader.read_matrix(&format!("up{id}")).unwrap(), down: reader.read_matrix(&format!("down{id}")).unwrap() }).collect::<Vec<_>>();
        struct Source(Vec<GgufExpertWeights>);
        impl GgufExpertSource for Source {
            fn intermediate(&self) -> usize {
                32
            }
            fn hidden(&self) -> usize {
                256
            }
            fn load_expert_gguf(&self, _: usize, expert: usize) -> Result<GgufExpertWeights, String> {
                Ok(self.0[expert].clone())
            }
        }
        let source = Source(source);
        let ctx = CudaContext::new_default().unwrap();
        let budget = CudaMoeState::required_bytes(&source.0[2], true).unwrap();
        for arena in [false, true] {
            let mut state = CudaMoeState::new(budget);
            if arena {
                state.reserve_arena(&ctx).unwrap();
            }
            let pointer = |weights: &CudaExpertWeights| match weights.down.as_ref() {
                CudaDown::Packed { data, .. } => data.device_ptr(ctx.stream()).0,
                _ => unreachable!(),
            };
            let check = |weights: &CudaExpertWeights, expert: usize| {
                let CudaGateUp::Q4K { gate, up, .. } = weights.gate_up.as_ref() else { unreachable!() };
                let CudaDown::Packed { data, .. } = weights.down.as_ref() else { unreachable!() };
                for (device, matrix) in [(gate, &source.0[expert].gate), (up, &source.0[expert].up), (data, &source.0[expert].down)] {
                    assert_eq!(ctx.stream().clone_dtoh(&**device).unwrap(), matrix.bytes().unwrap());
                }
            };
            state.ensure_on_stream(&ctx, &source, 0, 0, true).unwrap();
            let active = state.cache[&(0, 0)].clone();
            if arena {
                assert!(state.ensure_on_stream(&ctx, &source, 0, 1, true).is_err(), "活跃区域不能超过物理预算或被覆写");
                check(&active, 0);
                drop(active);
                state.ensure_on_stream(&ctx, &source, 0, 1, true).unwrap();
            } else {
                state.ensure_on_stream(&ctx, &source, 0, 1, true).unwrap();
                assert_ne!(pointer(&active), pointer(&state.cache[&(0, 1)]));
                check(&active, 0);
                drop(active);
            }
            let reused_pointer = pointer(&state.cache[&(0, 1)]);
            for round in 0..64 {
                let expert = round % 2;
                state.ensure_on_stream(&ctx, &source, 0, expert, true).unwrap();
                assert_eq!(pointer(&state.cache[&(0, expert)]), reused_pointer);
                check(&state.cache[&(0, expert)], expert);
                assert!(state.resident_bytes() <= budget);
            }
            state.ensure_on_stream(&ctx, &source, 0, 2, true).unwrap();
            check(&state.cache[&(0, 2)], 2);
            assert_eq!(state.cache.len(), 1);
            if arena {
                let pool = state.arena.as_ref().unwrap().clone();
                state.cache.clear();
                ctx.synchronize().unwrap();
                assert_eq!(*pool.free.lock().unwrap(), BTreeMap::from([(0, budget)]));
            }
            let mut state = CudaMoeState::new(budget * 2);
            if arena {
                state.reserve_arena(&ctx).unwrap();
            }
            for round in 0..300 {
                let id = (round * 17 + round / 7) % 3;
                let next_bytes = CudaMoeState::required_bytes(&source.0[id], true).unwrap();
                let evicted = (!state.cache.contains_key(&(0, id)) && state.bytes + next_bytes > state.max_bytes).then(|| *state.cache.iter().min_by_key(|(_, value)| value.used_at).unwrap().0);
                state.ensure_on_stream(&ctx, &source, 0, id, true).unwrap();
                if let Some(oldest) = evicted {
                    assert!(!state.cache.contains_key(&oldest));
                }
                assert!(state.lru.len() <= 4 * state.cache.len().max(1));
                if round % 50 == 0 {
                    check(&state.cache[&(0, id)], id);
                }
            }
        }
        std::fs::remove_file(path).unwrap();
    }
}
