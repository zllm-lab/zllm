//! CUDA 专家权重驻留、量化 kernel 与 routed batch capability。

use std::{collections::HashMap, sync::Arc};

use cudarc::driver::safe::{CudaSlice, CudaStream};
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

    pub fn into_decode_state(self) -> CudaMoeState {
        self.state
    }
}

#[derive(Clone)]
enum CudaGateUp {
    F16 { gate: CudaWeight, up: CudaWeight },
    Q4K { gate: CudaSlice<u8>, up: CudaSlice<u8>, rows: usize, cols: usize },
}

impl CudaGateUp {
    fn activated(&self, ctx: &CudaContext, input: &CudaTensor, activation: &Activation) -> Result<CudaTensor, BackendError> {
        match self {
            Self::F16 { gate, up } => ctx.gated_linear(input, gate, up, activation),
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
            Self::Q4K { gate, up, .. } => gate.len() + up.len(),
        }
    }
}

#[derive(Clone)]
enum CudaDown {
    F16(CudaWeight),
    Q5K { data: CudaSlice<u8>, rows: usize, cols: usize },
    Q6K { data: CudaSlice<u8>, rows: usize, cols: usize },
}

impl CudaDown {
    fn linear(&self, ctx: &CudaContext, input: &CudaTensor) -> Result<CudaTensor, BackendError> {
        match self {
            Self::F16(weight) => ctx.linear(input, weight),
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
            Self::Q5K { data, .. } | Self::Q6K { data, .. } => data.len(),
        }
    }
}

#[derive(Clone)]
struct CudaExpertWeights {
    gate_up: CudaGateUp,
    down: CudaDown,
    bytes: usize,
    used_at: u64,
}

pub struct CudaMoeState {
    cache: HashMap<(usize, usize), CudaExpertWeights>,
    bytes: usize,
    max_bytes: usize,
    clock: u64,
    accumulator: Option<CudaSlice<f32>>,
    /// 累计 miss(load)次数,用于命中率观测。
    pub loaded: usize,
}

impl CudaMoeState {
    pub fn new(max_bytes: usize) -> Self {
        Self { cache: HashMap::new(), bytes: 0, max_bytes, clock: 0, accumulator: None, loaded: 0 }
    }

    pub fn resident_bytes(&self) -> usize {
        self.bytes
    }

    fn required_bytes(matrices: &crate::weight::expert_source::GgufExpertWeights, allow_q4_k: bool) -> Result<usize, BackendError> {
        let f16_bytes = |rows: usize, columns: usize| rows.checked_mul(columns).and_then(|elements| elements.checked_mul(std::mem::size_of::<f16>())).ok_or_else(|| BackendError::ExpertLoad("CUDA expert 驻留大小溢出".to_owned()));
        let down = if allow_q4_k && matches!(matrices.down.tensor_type.0, 13 | 14) { matrices.down.storage_len() } else { f16_bytes(matrices.down.rows, matrices.down.columns)? };
        let gate_up = if allow_q4_k && matrices.gate.tensor_type.0 == 12 && matrices.up.tensor_type.0 == 12 && matrices.gate.rows == matrices.up.rows && matrices.gate.columns == matrices.up.columns {
            matrices.gate.storage_len().checked_add(matrices.up.storage_len())
        } else {
            f16_bytes(matrices.gate.rows, matrices.gate.columns)?.checked_add(f16_bytes(matrices.up.rows, matrices.up.columns)?)
        }
        .ok_or_else(|| BackendError::ExpertLoad("CUDA expert gate/up 驻留大小溢出".to_owned()))?;
        gate_up.checked_add(down).ok_or_else(|| BackendError::ExpertLoad("CUDA expert 总驻留大小溢出".to_owned()))
    }

    fn load(ctx: &CudaContext, matrices: GgufExpertWeights, layer: usize, expert: usize, allow_q4_k: bool) -> Result<CudaExpertWeights, BackendError> {
        // packed 字节走 pinned 中转:pageable clone_htod 在 PCIe3 上带宽仅 ~1/3
        // 且有 SyncOnDrop stall,expert 流式路径对 H2D 带宽敏感。
        let upload_u8 = |bytes: &[u8]| -> Result<cudarc::driver::safe::CudaSlice<u8>, BackendError> {
            // copy-on:DMA 目标块与 DMA 同在 copy 流 mallocAsync——块的 alloc/DMA/free
            // 依赖闭合在 copy 流 FIFO 内,池复用不会撞上在途 DMA。此前块在计算流分配、
            // copy 流裸 DMA 写入:池对跨流写入不可见,块驱逐后复用给 expert 计算输出,
            // 被迟到 DMA 写坏(实测 copy-on 首 decode 层 activated 686/1024 nonfinite
            // 而全部 expert 权重字节逐位完好)。
            let stream: &Arc<cudarc::driver::safe::CudaStream> = if std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() { ctx.copy_stream() } else { ctx.stream() };
            let mut slice = unsafe { stream.alloc::<u8>(bytes.len()) }.map_err(|error| BackendError::ExpertLoad(format!("分配 expert buffer: {error:?}")))?;
            ctx.upload_u8_pinned(bytes, &mut slice).map_err(BackendError::ExpertLoad)?;
            Ok(slice)
        };
        let decode = |matrix: crate::weight::container::gguf::GgufMatrix| {
            let rows = matrix.rows;
            let cols = matrix.columns;
            let data = matrix.decode().map_err(BackendError::ExpertLoad)?;
            let data = data.into_iter().map(f16::from_f32).collect::<Vec<_>>();
            // F16 回退(Q5_K gate/up 层)同样走 pinned 上传:pageable clone_htod 在
            // 与 copy 流 pinned DMA 并发时存在驱动 staging 干扰(实测 L47 全 NaN),
            // 且本身在重型流上有 SyncOnDrop stall。
            let mut slice = ctx.buffer_uninit::<f16>(data.len()).map_err(|error| BackendError::ExpertLoad(format!("分配 F16 expert buffer: {error:?}")))?;
            ctx.upload_f16_pinned(&data, &mut slice).map_err(BackendError::ExpertLoad)?;
            Ok(CudaWeight::new(slice, rows, cols))
        };
        let down = if allow_q4_k && matrices.down.tensor_type.0 == 13 {
            let data = upload_u8(matrices.down.bytes().map_err(BackendError::ExpertLoad)?)?;
            CudaDown::Q5K { data, rows: matrices.down.rows, cols: matrices.down.columns }
        } else if allow_q4_k && matrices.down.tensor_type.0 == 14 {
            let data = upload_u8(matrices.down.bytes().map_err(BackendError::ExpertLoad)?)?;
            CudaDown::Q6K { data, rows: matrices.down.rows, cols: matrices.down.columns }
        } else {
            CudaDown::F16(decode(matrices.down)?)
        };
        let gate_up = if allow_q4_k && matrices.gate.tensor_type.0 == 12 && matrices.up.tensor_type.0 == 12 && matrices.gate.rows == matrices.up.rows && matrices.gate.columns == matrices.up.columns {
            let gate = upload_u8(matrices.gate.bytes().map_err(BackendError::ExpertLoad)?)?;
            let up = upload_u8(matrices.up.bytes().map_err(BackendError::ExpertLoad)?)?;
            CudaGateUp::Q4K { gate, up, rows: matrices.gate.rows, cols: matrices.gate.columns }
        } else {
            CudaGateUp::F16 { gate: decode(matrices.gate)?, up: decode(matrices.up)? }
        };
        let bytes = gate_up.bytes() + down.bytes();
        Ok(CudaExpertWeights { gate_up, down, bytes, used_at: 0 })
    }

    fn ensure_on_stream(&mut self, ctx: &CudaContext, source: &dyn GgufExpertSource, layer: usize, expert: usize, allow_q4_k: bool) -> Result<(), BackendError> {
        let key = (layer, expert);
        if let Some(cached) = self.cache.get_mut(&key) {
            self.clock = self.clock.wrapping_add(1);
            cached.used_at = self.clock;
            return Ok(());
        }
        let matrices = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
        let bytes = Self::required_bytes(&matrices, allow_q4_k)?;
        if bytes > self.max_bytes {
            return Err(BackendError::ExpertLoad(format!("CUDA expert L{layer} E{expert} 超过 cache 上限")));
        }
        while self.bytes > self.max_bytes - bytes {
            if std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some() {
                // copy-on:块宿主在 copy 流,DMA 与 freeAsync 同流序化(无需 copy 栅栏);
                // 但读块的 kernel 在计算流,free 前让 copy 流等计算流,防止池复用后
                // 在途 kernel 读取被后续写入踩坏。
                ctx.fence_compute_before_copy().map_err(BackendError::ExpertLoad)?;
            }
            let oldest = self.cache.iter().min_by_key(|(_, cached)| cached.used_at).map(|(&key, _)| key).ok_or_else(|| BackendError::ExpertLoad("CUDA expert cache 无法释放足够空间".to_owned()))?;
            let removed = self.cache.remove(&oldest).expect("CUDA expert LRU key 必须存在");
            self.bytes -= removed.bytes;
        }
        let mut weights = Self::load(ctx, matrices, layer, expert, allow_q4_k)?;
        self.loaded += 1;
        debug_assert_eq!(weights.bytes, bytes);
        self.clock = self.clock.wrapping_add(1);
        weights.used_at = self.clock;
        self.bytes += weights.bytes;
        self.cache.insert(key, weights);
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
                    match &cached.gate_up {
                        CudaGateUp::Q4K { gate, .. } => {
                            let got = self.stream().clone_dtoh(gate).map_err(|e| BackendError::Compute { msg: format!("{e:?}") })?;
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
        if verify {
            // 取证:fence 后逐 expert 比对 packed 字节(clone_dtoh 走计算流,fence 有效
            // 则 mismatch 恒 0;mismatch>0 即 fence 失效/DMA 数据坏)。
            for (_, expert, weights) in &active {
                let matrices = source.load_expert_gguf(layer, *expert).map_err(BackendError::ExpertLoad)?;
                if let (CudaGateUp::Q4K { gate, .. }, Ok(expect)) = (&weights.gate_up, matrices.gate.bytes()) {
                    let got = self.stream().clone_dtoh(gate).map_err(|e| BackendError::Compute { msg: format!("verify dtoh: {e:?}") })?;
                    let mismatch = expect.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
                    eprintln!("[decode-verify] L{layer} E{expert} gate mismatch={}/{}", mismatch, expect.len());
                }
                if let (CudaDown::Q5K { data, .. } | CudaDown::Q6K { data, .. }, Ok(expect)) = (&weights.down, matrices.down.bytes()) {
                    let got = self.stream().clone_dtoh(data).map_err(|e| BackendError::Compute { msg: format!("verify dtoh: {e:?}") })?;
                    let mismatch = expect.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
                    eprintln!("[decode-verify] L{layer} E{expert} down mismatch={}/{}", mismatch, expect.len());
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
        Ok(CudaTensor::new_f32_residual(output_data, placeholder, routed.rows, routed.cols))
    }
}
