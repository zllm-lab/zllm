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
}

impl CudaMoeState {
    pub fn new(max_bytes: usize) -> Self {
        Self { cache: HashMap::new(), bytes: 0, max_bytes, clock: 0, accumulator: None }
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

    fn load(stream: &Arc<CudaStream>, matrices: GgufExpertWeights, layer: usize, expert: usize, allow_q4_k: bool) -> Result<CudaExpertWeights, BackendError> {
        let decode = |matrix: crate::weight::container::gguf::GgufMatrix| {
            let rows = matrix.rows;
            let cols = matrix.columns;
            let data = matrix.decode().map_err(BackendError::ExpertLoad)?;
            let data = data.into_iter().map(f16::from_f32).collect::<Vec<_>>();
            let data = stream.clone_htod::<f16, _>(&data).map_err(|error| BackendError::ExpertLoad(format!("上传 F16 expert weight: {error:?}")))?;
            Ok(CudaWeight::new(data, rows, cols))
        };
        let down = if allow_q4_k && matrices.down.tensor_type.0 == 13 {
            let data = stream.clone_htod::<u8, _>(matrices.down.bytes().map_err(BackendError::ExpertLoad)?).map_err(|error| BackendError::ExpertLoad(format!("上传 Q5_K down L{layer} E{expert}: {error:?}")))?;
            CudaDown::Q5K { data, rows: matrices.down.rows, cols: matrices.down.columns }
        } else if allow_q4_k && matrices.down.tensor_type.0 == 14 {
            let data = stream.clone_htod::<u8, _>(matrices.down.bytes().map_err(BackendError::ExpertLoad)?).map_err(|error| BackendError::ExpertLoad(format!("上传 Q6_K down L{layer} E{expert}: {error:?}")))?;
            CudaDown::Q6K { data, rows: matrices.down.rows, cols: matrices.down.columns }
        } else {
            CudaDown::F16(decode(matrices.down)?)
        };
        let gate_up = if allow_q4_k && matrices.gate.tensor_type.0 == 12 && matrices.up.tensor_type.0 == 12 && matrices.gate.rows == matrices.up.rows && matrices.gate.columns == matrices.up.columns {
            let gate = stream.clone_htod::<u8, _>(matrices.gate.bytes().map_err(BackendError::ExpertLoad)?).map_err(|error| BackendError::ExpertLoad(format!("上传 Q4_K gate L{layer} E{expert}: {error:?}")))?;
            let up = stream.clone_htod::<u8, _>(matrices.up.bytes().map_err(BackendError::ExpertLoad)?).map_err(|error| BackendError::ExpertLoad(format!("上传 Q4_K up L{layer} E{expert}: {error:?}")))?;
            CudaGateUp::Q4K { gate, up, rows: matrices.gate.rows, cols: matrices.gate.columns }
        } else {
            CudaGateUp::F16 { gate: decode(matrices.gate)?, up: decode(matrices.up)? }
        };
        let bytes = gate_up.bytes() + down.bytes();
        Ok(CudaExpertWeights { gate_up, down, bytes, used_at: 0 })
    }

    fn ensure_on_stream(&mut self, stream: &Arc<CudaStream>, source: &dyn GgufExpertSource, layer: usize, expert: usize, allow_q4_k: bool) -> Result<(), BackendError> {
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
            let oldest = self.cache.iter().min_by_key(|(_, cached)| cached.used_at).map(|(&key, _)| key).ok_or_else(|| BackendError::ExpertLoad("CUDA expert cache 无法释放足够空间".to_owned()))?;
            let removed = self.cache.remove(&oldest).expect("CUDA expert LRU key 必须存在");
            self.bytes -= removed.bytes;
        }
        let mut weights = Self::load(stream, matrices, layer, expert, allow_q4_k)?;
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
        if spec.scoring_func != ScoringFunc::Softmax {
            return Err(BackendError::Compute { msg: "CUDA MoE router 当前只支持 Softmax".to_owned() });
        }
        if router_weight.rows != spec.num_experts || router_weight.cols != input.cols || router_bias.rows * router_bias.cols != spec.num_experts {
            return Err(BackendError::Compute {
                msg: format!("CUDA router shape input=[{},{}] weight=[{},{}] bias=[{},{}] experts={} 不一致", input.rows, input.cols, router_weight.rows, router_weight.cols, router_bias.rows, router_bias.cols, spec.num_experts,),
            });
        }
        let router_weight = router_weight.data_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "CUDA router weight 缺少 F32 device storage".to_owned() })?;
        let router_bias = router_bias.data_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "CUDA router bias 缺少 F32 device storage".to_owned() })?;
        let (expert_ids, weights) =
            ops::routing::moe_route_softmax_topk_f32(self, input, router_weight, router_bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected).map_err(|msg| BackendError::Compute { msg })?;
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
        ops::routing::f32_to_f16(self, &output.data, output.rows, output.cols, false).map_err(|msg| BackendError::Compute { msg })
    }
}

impl ExpertPrefillBackend for CudaContext {
    type PrefillExperts = CudaPrefillExperts;

    fn prefill_expert_batch(&self, spec: &TopkMoeSpec, layer: usize, experts: &mut CudaPrefillExperts, batch: Vec<ExpertPrefillBatch<CudaTensor>>) -> Result<Vec<CudaTensor>, BackendError> {
        batch
            .into_iter()
            .map(|item| {
                let allow_quantized = matches!(spec.activation, Activation::Silu);
                experts.state.ensure_on_stream(self.stream(), experts.source.as_ref(), layer, item.expert, allow_quantized)?;
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
        if spec.scoring_func != ScoringFunc::Softmax {
            return Err(BackendError::Compute { msg: "CUDA MoE router 当前只支持 Softmax".to_owned() });
        }
        if router_weight.rows != spec.num_experts || router_weight.cols != input.cols || router_bias.rows * router_bias.cols != spec.num_experts {
            return Err(BackendError::Compute { msg: "CUDA MoE decode router shape 不一致".to_owned() });
        }
        let router_weight = router_weight.data_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "CUDA decode router weight 缺少 F32 device storage".to_owned() })?;
        let router_bias = router_bias.data_f32.as_ref().ok_or_else(|| BackendError::Compute { msg: "CUDA decode router bias 缺少 F32 device storage".to_owned() })?;
        let (expert_ids, route_weights) =
            ops::routing::moe_route_softmax_topk_device_f32(self, input, router_weight, router_bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected).map_err(|msg| BackendError::Compute { msg })?;
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
            state.ensure_on_stream(self.stream(), source, request.layer, expert, matches!(spec.activation, Activation::Silu))?;
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
        let mut active = Vec::new();
        for (expert, assigned) in assignments.iter().enumerate() {
            if assigned.is_empty() {
                continue;
            }
            state.ensure_on_stream(self.stream(), source, layer, expert, matches!(spec.activation, Activation::Silu))?;
            let weights = state.cache.get(&(layer, expert)).expect("CUDA active expert 必须驻留").clone();
            let route = routing.expert_ids.iter().position(|&routed| routed as usize == expert).ok_or_else(|| BackendError::Compute { msg: format!("CUDA route 缺少 expert {expert}") })?;
            active.push((route, weights));
        }
        if let Some(request) = on_ready(state)? {
            // 与 ROCm 路径一致:prefetch_experts 在当前 stream 上同步 ensure 预测专家,
            // 不能静默丢弃请求,否则预测预取在 CUDA 上完全不生效。
            self.prefetch_experts(spec, state, request)?;
        }

        let count = input.rows.checked_mul(input.cols).ok_or_else(|| BackendError::Compute { msg: "CUDA MoE accumulator 大小溢出".to_owned() })?;
        let data = match state.accumulator.take() {
            Some(data) if data.len() == count => data,
            Some(_) | None => self.stream().alloc_zeros::<f32>(count).map_err(|error| BackendError::Compute { msg: format!("CUDA MoE accumulator 分配失败: {error:?}") })?,
        };
        let routed = CudaMoeAccumulator { data, rows: input.rows, cols: input.cols };
        for (route, expert) in active {
            let activated = expert.gate_up.activated(self, input, &spec.activation)?;
            expert.down.accumulate(self, &activated, &routed, &routing.route_weights, route)?;
        }
        let output = ops::routing::f32_to_f16(self, &routed.data, routed.rows, routed.cols, true).map_err(|msg| BackendError::Compute { msg })?;
        state.accumulator = Some(routed.data);
        Ok(output)
    }
}
