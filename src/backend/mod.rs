//! 计算后端：实现基础算子，并封装设备资源、权重驻留和执行调度能力。

pub mod cpu;
#[cfg(feature = "with-cuda")]
pub mod cuda;
pub mod huawei;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(feature = "with-rocm")]
pub mod rocm;
#[cfg(all(target_os = "android", feature = "with-vulkan"))]
pub mod vulkan;

use std::sync::Arc;

use crate::{
    attention::block::BlockAttentionSpec,
    attention::dsa::DsaSpec,
    attention::gqa::GqaSpec,
    attention::mla::MlaSpec,
    diffusion::ModulationSegment,
    moe::{
        Activation,
        routing::ExpertAssignments,
        topk_moe::{RoutedMoeInputs, RoutedMoeWeightsRef, SharedExpertRef, TopkMoeSpec},
    },
    vae::{Conv1dSpec, Conv3dSpec, PixelShuffleSpec},
    weight::expert_source::ExpertSource,
    weight::format::mxfp8::Mxfp8Matrix,
    weight::{Fp8Matrix, container::gguf::GgufMatrix},
};

pub use crate::moe::routing::{ExpertPrefillBatch, MoePrefillRouting};

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("不支持模型层 L{layer}")]
    UnsupportedLayer { layer: usize },
    #[error("加载 expert 失败: {0}")]
    ExpertLoad(String),
    #[error("后端计算失败: {msg}")]
    Compute { msg: String },
}

/// backend 内存池的逻辑用途。用途决定资源能否与另一类 allocation 复用，
/// 但不暴露 HIP、Metal、CUDA 等平台句柄。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Scratch,
    Activation,
    Cache,
    Weight,
    Transfer,
}

/// allocation 的最长逻辑生命周期。物理内存可以在逻辑 owner 退出后留在
/// backend pool 中，但必须等待对应设备 completion 后才能再次借出。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryLifetime {
    Operation,
    Stage,
    Session,
    Model,
}

/// backend 无关的内存申请。`tag`/`slot` 区分同尺寸但可能同时存活的 workspace；
/// backend 可以选择 best-fit、精确尺寸或用途槽位复用，不要求平台采用同一种池。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryRequest {
    pub bytes: usize,
    pub alignment: usize,
    pub kind: MemoryKind,
    pub lifetime: MemoryLifetime,
    pub tag: Option<&'static str>,
    pub slot: usize,
}

impl MemoryRequest {
    pub const fn new(bytes: usize, kind: MemoryKind, lifetime: MemoryLifetime) -> Self {
        Self { bytes, alignment: 1, kind, lifetime, tag: None, slot: 0 }
    }

    pub const fn scratch(tag: &'static str, bytes: usize) -> Self {
        Self { bytes, alignment: 1, kind: MemoryKind::Scratch, lifetime: MemoryLifetime::Operation, tag: Some(tag), slot: 0 }
    }

    pub const fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    pub const fn with_slot(mut self, slot: usize) -> Self {
        self.slot = slot;
        self
    }

    pub fn validate(self) -> Result<Self, BackendError> {
        if !self.alignment.is_power_of_two() {
            return Err(BackendError::Compute { msg: format!("memory pool alignment={} 不是 2 的幂", self.alignment) });
        }
        Ok(self)
    }
}

/// backend 无关的内存池能力。公共层统一描述用途和生命周期；具体 storage、
/// completion、pending→available 推进与物理释放仍由 backend 实现。
///
/// 返回的 `Memory` 必须是 RAII handle：clone 延长 allocation 生命周期，最后一个
/// handle 退出后由 backend 决定进入 pending、available 或直接释放。
pub trait MemoryPool {
    type Memory: Clone + Send + Sync + 'static;

    fn allocate_memory(&self, request: MemoryRequest) -> Result<Self::Memory, BackendError>;
    fn memory_bytes(&self, memory: &Self::Memory) -> u64;
}

/// 单个 token 行的采样参数。temperature=0 表示确定性 argmax。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub random: f32,
}

/// 单个 token 行的生成围栏。围栏只描述选择约束，不感知模型、tokenizer 或
/// 工具协议；请求层负责根据当前语法状态生成它。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenFence {
    excluded: Arc<[u32]>,
    forced: Option<u32>,
}

impl TokenFence {
    pub fn excluding(excluded: impl IntoIterator<Item = u32>) -> Self {
        let mut excluded = excluded.into_iter().collect::<Vec<_>>();
        excluded.sort_unstable();
        excluded.dedup();
        Self { excluded: excluded.into(), forced: None }
    }

    pub fn forcing(token: u32) -> Self {
        Self { excluded: Arc::from([]), forced: Some(token) }
    }

    pub fn constrained(forced: Option<u32>, excluded: impl IntoIterator<Item = u32>) -> Self {
        let mut fence = Self::excluding(excluded);
        fence.forced = forced;
        fence
    }

    pub fn excluded(&self) -> &[u32] {
        &self.excluded
    }

    pub fn forced(&self) -> Option<u32> {
        self.forced
    }

    pub fn is_open(&self) -> bool {
        self.excluded.is_empty() && self.forced.is_none()
    }
}

/// 事件驱动 stage 调度所需的设备能力。
///
/// runtime 只依赖完成事件与可用资源，不感知 HIP、Metal command buffer 或其他
/// 平台对象。模型只提供工作分类和 batch 执行，事件生命周期由 backend 持有。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageSubmissionKind {
    Latency,
    Background,
}

pub trait StageExecutionBackend {
    type Completion: Send;

    /// 外部异步 profiler 的 stage work 关联区间。runtime 只传稳定的工作标识，
    /// backend 决定是否写 marker；默认实现完全无开销。
    fn trace_stage_work_begin(&self, _label: &str) -> Result<(), BackendError> {
        Ok(())
    }
    fn trace_stage_work_end(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// 低开销 stage 设备时间线。默认 backend 不采集；设备 backend 仅在自己的
    /// profile 开关启用时记录 event，模型 runtime 不感知具体平台对象。
    fn profile_stage_begin(&self, _detailed_eligible: bool) -> Result<(), BackendError> {
        Ok(())
    }
    fn profile_stage_end(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn stage_available_bytes(&self) -> Result<usize, BackendError>;
    /// 当前设备的物理显存总量。runtime 用 available/total 计算模型无关的资源水位，
    /// 不直接接触 HIP、Metal 或 CUDA 查询接口。
    fn stage_total_bytes(&self) -> Result<usize, BackendError> {
        // 不提供默认值：用 available 冒充 total 会让水位恒为 1，runtime 拿到假数据。
        Err(BackendError::Compute { msg: "stage_total_bytes 未实现：backend 必须显式提供设备物理显存总量".to_owned() })
    }
    /// 为本次 submission 选择设备执行队列。默认 backend 只有一条有序队列；
    /// 支持独立后台队列的 backend 必须同时允许 completion 跨队列乱序退休。
    fn activate_stage_submission(&self, _kind: StageSubmissionKind) -> Result<(), BackendError> {
        Ok(())
    }
    fn supports_concurrent_stage_submissions(&self) -> bool {
        false
    }
    /// 同一条 latency 设备队列允许提前排入的 submission 数。默认保持既有
    /// 调度策略；只有明确需要 completion 节拍限流的 backend 才收紧。
    fn max_queued_latency_submissions(&self) -> usize {
        usize::MAX
    }
    fn begin_stage_submission(&self) -> Result<(), BackendError> {
        Ok(())
    }
    fn abort_stage_submission(&self) -> Result<(), BackendError> {
        Ok(())
    }
    fn record_stage_completion(&self) -> Result<Self::Completion, BackendError>;
    fn stage_completion_ready(&self, completion: &Self::Completion) -> Result<bool, BackendError>;
    /// CU 正在执行且没有空闲 execution slot 时等待本 stage 完成。默认实现只依赖
    /// 非阻塞查询；设备 backend 应覆盖为原生 event wait，避免轮询尾延迟。
    fn wait_stage_completion(&self, completion: &Self::Completion) -> Result<(), BackendError> {
        while !self.stage_completion_ready(completion)? {
            std::thread::yield_now();
        }
        Ok(())
    }
    /// ordered handoff 的链尾已完成时，runtime 用它退休被链尾覆盖的 completion。
    /// 默认实现保持 completion 自身的析构语义；异步 backend 可据此避免重复查询。
    fn retire_ordered_stage_completion(&self, _completion: &Self::Completion) -> Result<(), BackendError> {
        Ok(())
    }
    /// session 关闭且设备工作完成后释放线程本地临时资源；权重与 session state 不在此生命周期内。
    fn finish_stage_session(&self) -> Result<(), BackendError> {
        Ok(())
    }
}

/// stage 间张量迁移与紧凑表示能力。runtime 只描述边界，不接触设备指针。
pub trait StageTensorBackend: SegmentedTensorBackend + StageExecutionBackend {
    fn stage_label(&self) -> String;
    fn activate_stage(&self) -> Result<(), BackendError>;
    fn stage_tensor_from_f32(&self, values: Vec<f32>, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError>;
    fn move_tensor_to_stage(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError>;
    /// 输入可能仍在源执行队列上计算；backend 必须建立源计算、搬运与目标计算的
    /// completion 顺序。runtime 对所有 backend 使用同一条有序交接路径。
    fn move_tensor_to_stage_ordered(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn stabilize_stage_tensor(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn compact_stage_tensor(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn warmup_stage(&self) -> Result<(), BackendError> {
        Ok(())
    }
}

/// DSA 模型在连续 stage 间传递 selection 与创建独立 session 的后端能力。
pub trait DsaStageBackend: DsaPrefillBackend + StageTensorBackend {
    type DsaSelection: Send;

    fn new_stage_cache(&self, layer_count: usize, max_seq_len: usize) -> Result<Self::Cache, BackendError>;
    fn new_stage_dsa(&self, layer_count: usize, max_seq_len: usize, head_dim: usize, top_k: usize) -> Result<Self::DsaState, BackendError>;
    /// 告知当前设备 stage 已进入 decode 的 session 数量。后端只能把它作为
    /// 不改变数学语义的 dispatch hint；默认实现保持原路径。
    fn set_stage_decode_parallelism(&self, dsa: &mut Self::DsaState, sessions: usize) {
        let _ = (dsa, sessions);
    }
    /// 在下一份工作开始前原地回退 session 的逻辑长度；已分配设备页保持不变。
    fn truncate_stage_state(&self, cache: &mut Self::Cache, dsa: &mut Self::DsaState, rows: usize) -> Result<(), BackendError>;
    fn stage_cache_allocated_bytes(&self, cache: &Self::Cache, dsa: &Self::DsaState) -> u64;
    fn import_stage_selection(&self, state: &mut Self::DsaState, selection: Option<Self::DsaSelection>) -> Result<(), BackendError>;
    fn export_stage_selection(&self, state: &Self::DsaState) -> Option<Self::DsaSelection>;
    fn move_selection_to_stage_ordered(&self, selection: Self::DsaSelection) -> Result<Self::DsaSelection, BackendError>;
    fn stabilize_stage_selection(&self, selection: Self::DsaSelection) -> Result<Self::DsaSelection, BackendError>;
}

/// 后端资源、权重驻留与提交生命周期。这里不定义任何模型计算算子。
pub trait BackendResources {
    type Tensor;
    type Weight;
    type Cache;
    type LayerScope<'a>
    where
        Self: 'a;

    /// 一层模型计算的资源作用域；具体同步与临时资源回收由 backend 决定。
    fn layer_scope(&self) -> Self::LayerScope<'_>;
    fn token_rows(&self, tensor: &Self::Tensor) -> usize;
    fn token_cols(&self, tensor: &Self::Tensor) -> usize;
    /// tensor 当前持有的底层 allocation 字节数；逻辑 view 必须返回 owner
    /// allocation，供模型无关的 session cache 做显存核算。
    fn tensor_allocated_bytes(&self, tensor: &Self::Tensor) -> u64;

    /// 开始一段 GPU-only 算子批次；CPU backend 无需处理。
    fn begin_batch(&self);
    /// Decode 的 GPU-only 数据链；backend 可使用与 prefill 不同的提交粒度。
    fn begin_decode_batch(&self) {
        self.begin_batch();
    }
    /// 提交当前设备批次但不等待完成；用于把 GPU 工作与随后的 host I/O 重叠。
    fn submit_batch(&self) {}
    /// 结束当前设备批次并恢复 backend 状态；公开执行边界在成功和失败路径都必须调用。
    fn finish_batch(&self);
    /// finish_batch 的推迟同步变体:恢复状态但不等待 GPU,供逐 token decode 流水线
    /// 把输出步/下一轮接进同一提交窗口;等待点由调用方自行管理(如 Metal 的 CB 句柄)。
    fn finish_batch_deferred(&self) {
        self.finish_batch();
    }
    /// 等待并释放当前流式权重块；需要保持外层 batch 状态的 backend 必须覆盖默认实现。
    fn finish_stream_chunk(&self) {
        self.finish_batch();
    }
    /// 等待当前设备队列完成；仅用于显式 profile/诊断边界。
    fn synchronize(&self) -> Result<(), BackendError> {
        Ok(())
    }
    /// 可选设备时间线 marker；默认 backend 不采集，ROCm profiling 路径异步记录 event。
    fn profile_device_operator(&self, _label: &'static str) -> Result<(), BackendError> {
        Ok(())
    }
    fn prepare_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError>;
    /// 专家权重可由 backend 保留适合 grouped/fused kernel 的原始布局；默认与
    /// 普通线性层相同，避免模型层感知设备侧编码。
    fn prepare_expert_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        self.prepare_weight(weight, rows, cols)
    }
    /// 从输出矩阵抽取任意行并保持原顺序。draft vocabulary 只缩小 greedy
    /// LM head；backend 应尽量保留原量化布局，默认实现提供正确性回退。
    fn prepare_weight_rows(&self, weight: LinearWeight<'_>, source_rows: usize, cols: usize, selected_rows: &[u32]) -> Result<Self::Weight, BackendError> {
        prepare_weight_rows_default(self, weight, source_rows, cols, selected_rows)
    }
    /// 后端可为相关权重选择更合适的驻留布局；默认仍分别准备，保持原语义。
    fn prepare_weight_pair(&self, first: LinearWeight<'_>, second: LinearWeight<'_>, rows: usize, cols: usize) -> Result<(Self::Weight, Self::Weight), BackendError> {
        Ok((self.prepare_weight(first, rows, cols)?, self.prepare_weight(second, rows, cols)?))
    }
    /// 把按输出行分组、按输入列分段消费的 BlockFP8 矩阵准备成等宽权重。
    /// 默认实现保持各组独立；resident backend 可让各组共享连续 allocation，
    /// 供 grouped decode kernel 一次提交。
    fn prepare_grouped_block_fp8(&self, matrix: &crate::weight::format::block_fp8::BlockFp8Matrix, groups: usize, rows_per_group: usize) -> Result<Vec<Self::Weight>, BackendError> {
        if groups == 0 || matrix.rows != groups * rows_per_group {
            return Err(BackendError::Compute { msg: format!("grouped BlockFP8 shape=[{},{}] groups={groups} rows={rows_per_group}", matrix.rows, matrix.cols) });
        }
        let (block_rows, block_cols) = matrix.block_shape();
        if !rows_per_group.is_multiple_of(block_rows) {
            return Err(BackendError::Compute { msg: format!("grouped BlockFP8 rows={rows_per_group} 不是 block_rows={block_rows} 的倍数") });
        }
        let scale_columns = matrix.cols.div_ceil(block_cols);
        let scale_rows_per_group = rows_per_group / block_rows;
        (0..groups)
            .map(|group| {
                let row_start = group * rows_per_group;
                let scale_start = group * scale_rows_per_group * scale_columns;
                let part = crate::weight::format::block_fp8::BlockFp8Matrix::new(
                    matrix.codes()[row_start * matrix.cols..(row_start + rows_per_group) * matrix.cols].to_vec(),
                    matrix.scales()[scale_start..scale_start + scale_rows_per_group * scale_columns].to_vec(),
                    rows_per_group,
                    matrix.cols,
                    block_rows,
                    block_cols,
                )
                .map_err(|msg| BackendError::Compute { msg })?;
                self.prepare_weight(LinearWeight::block_fp8(&part), rows_per_group, matrix.cols)
            })
            .collect()
    }
    fn prepare_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Weight, BackendError>;

    /// 准备 GemmaRMSNorm 权重。`values` 使用模型语义（零中心 gamma）；需要把
    /// GGUF 的 `gamma + 1` 编码交给设备原生 RMSNorm 的 backend 在这里完成。
    fn prepare_gemma_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        self.prepare_f32(values, rows, cols)
    }
}

/// 模型无关的基础张量算子。注意力、MoE、扩散等领域能力在下方独立 trait 中扩展。
pub trait Backend: BackendResources {
    fn linear(&self, input: &Self::Tensor, weight: &Self::Weight) -> Result<Self::Tensor, BackendError>;
    /// 把输入等宽分段后分别线性投影，再按列拼接输出。权重组之间没有依赖，
    /// backend 可合成 grouped GEMV；默认路径保持组合语义。
    fn grouped_linear_columns(&self, input: &Self::Tensor, weights: &[Self::Weight]) -> Result<Self::Tensor, BackendError> {
        if weights.is_empty() || !self.token_cols(input).is_multiple_of(weights.len()) {
            return Err(BackendError::Compute { msg: format!("grouped linear input columns={} groups={} 不兼容", self.token_cols(input), weights.len()) });
        }
        if weights.len() == 1 {
            return self.linear(input, &weights[0]);
        }
        let group_columns = self.token_cols(input) / weights.len();
        let (head, mut tail) = self.split_columns(input, group_columns)?;
        let mut projected = self.linear(&head, &weights[0])?;
        for weight in &weights[1..weights.len() - 1] {
            let (head, rest) = self.split_columns(&tail, group_columns)?;
            tail = rest;
            let output = self.linear(&head, weight)?;
            projected = self.concat_columns(&projected, &output)?;
        }
        let output = self.linear(&tail, weights.last().expect("grouped linear 非空已检查"))?;
        self.concat_columns(&projected, &output)
    }
    /// 精度敏感投影要求结果保持 F32；原生 tensor 已是 F32 的 backend 无需覆盖。
    fn linear_f32(&self, input: &Self::Tensor, weight: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        self.linear(input, weight)
    }

    /// 线性投影后直接加 residual；backend 可把 add 融入 matmul epilogue。
    fn linear_add(&self, input: &Self::Tensor, weight: &Self::Weight, residual: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        let output = self.linear(input, weight)?;
        self.add(residual, &output)
    }

    /// `value * sigmoid(linear(input, weight))`；backend 可融合标量 gate 投影与逐元素门控。
    fn linear_sigmoid_gate(&self, input: &Self::Tensor, weight: &Self::Weight, value: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        let gate = self.linear(input, weight)?;
        self.sigmoid_gate(value, &gate)
    }

    /// `gemma_rmsnorm(input + residual, weight, eps)`；backend 可融合 add + RMSNorm,
    /// 避免写出一份独立的 sum 中间 tensor。类 LLaMA 残差流每层 2 次 add + 2 次 RMSNorm,
    /// 融合后每层省 1 个 full pass over hidden 的读 + 1 个 sum tensor 的写。
    fn add_gemma_rmsnorm(&self, input: &Self::Tensor, residual: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        let sum = self.add(input, residual)?;
        self.gemma_rmsnorm_f32(&sum, weight, eps)
    }

    /// 同时返回 residual sum 与其 GemmaRMSNorm；需要继续保留 residual 的
    /// Transformer FFN 路径可由 backend 在一个 kernel 中生成两个输出。
    fn add_gemma_rmsnorm_pair(&self, input: &Self::Tensor, residual: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let sum = self.add(input, residual)?;
        let normalized = self.gemma_rmsnorm_f32(&sum, weight, eps)?;
        Ok((sum, normalized))
    }

    fn dual_linear(&self, input: &Self::Tensor, first: &Self::Weight, second: &Self::Weight) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        Ok((self.linear(input, first)?, self.linear(input, second)?))
    }

    /// 三个共享输入的独立投影(如 Q/K/V);默认逐个 linear,backend 可合并为单次 dispatch。
    fn triple_linear(&self, input: &Self::Tensor, first: &Self::Weight, second: &Self::Weight, third: &Self::Weight) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), BackendError> {
        Ok((self.linear(input, first)?, self.linear(input, second)?, self.linear(input, third)?))
    }

    /// 串行执行 RMSNorm，再让两路独立投影共享同一份归一化结果。
    /// backend 可按两路权重的真实消费精度直接生成原生布局，避免短命中间 tensor。
    fn rmsnorm_dual_linear(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, first: &Self::Weight, second: &Self::Weight) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let normalized = self.rmsnorm(input, norm_weight, eps)?;
        self.dual_linear(&normalized, first, second)
    }

    /// 串行执行 RMSNorm 与线性投影；backend 可让中间 tensor 保持设备原生精度。
    fn rmsnorm_linear(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, linear_weight: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        let normalized = self.rmsnorm(input, norm_weight, eps)?;
        self.linear(&normalized, linear_weight)
    }

    /// `activation(gate(input)) * up(input)`；backend 可融合两个投影与激活。
    fn gated_linear(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Weight, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        let (gate, up) = self.dual_linear(input, gate, up)?;
        self.gated_activation(&gate, &up, activation)
    }

    /// `down(activation(gate(input)) * up(input))`；backend 可把完整 dense FFN
    /// 合成一次设备提交，并让三个量化权重及全部中间结果保持 resident。
    fn gated_mlp(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Weight, down: &Self::Weight, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        let activated = self.gated_linear(input, gate, up, activation)?;
        self.linear(&activated, down)
    }

    /// `down(activation(gate(input)) * up(input)) + residual`；backend 可把 residual
    /// 加法融到 down 投影的 epilogue,消 1 个独立 dispatch 与一份中间输出。
    /// 默认实现等价 `gated_mlp + add`,所有 backend 立即满足;支持 fused epilogue
    /// 的 backend(Metal GGUF 量化路径)可重写为单次设备提交。
    fn gated_mlp_add_residual(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Weight, down: &Self::Weight, activation: &Activation, residual: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        let mlp = self.gated_mlp(input, gate, up, down, activation)?;
        self.add(residual, &mlp)
    }

    /// `activation(linear(input, gate)) * up`；backend 可融合线性投影与逐元素门控。
    fn linear_gated_activation(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Tensor, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        let gate = self.linear(input, gate)?;
        self.gated_activation(&gate, up, activation)
    }

    fn rmsnorm(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError>;
    /// 为后续量化投影准备一次可复用的低精度 activation；默认保持 backend 原生精度。
    fn rmsnorm_quantized(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        self.rmsnorm(input, weight, eps)
    }
    /// 同一份 RMSNorm 同时供精确路径和量化路径消费；不支持双输出的 backend 走原路径。
    fn rmsnorm_quantized_pair(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Option<(Self::Tensor, Self::Tensor)>, BackendError> {
        let _ = (input, weight, eps);
        Ok(None)
    }
    /// 路由等精度敏感路径可要求归一化结果保持 F32；原生 tensor 已是 F32 的 backend 无需覆盖。
    fn rmsnorm_f32(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        self.rmsnorm(input, weight, eps)
    }
    fn gemma_rmsnorm(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError>;
    /// GemmaRMSNorm 的精度敏感版本，与 `rmsnorm_f32` 保持相同控制面语义。
    fn gemma_rmsnorm_f32(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        self.gemma_rmsnorm(input, weight, eps)
    }
    fn layernorm_bias(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError>;
    fn split_columns(&self, input: &Self::Tensor, left_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError>;
    /// 消费 packed gate/up，backend 可融合拆分、激活与乘法并及时释放输入。
    fn split_gated_activation(&self, input: Self::Tensor, left_columns: usize, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        let (gate, up) = self.split_columns(&input, left_columns)?;
        self.gated_activation(&gate, &up, activation)
    }
    /// 把每个 `[left block, right block]` 交错列块拆成两个连续 tensor。
    fn split_interleaved_columns(&self, input: &Self::Tensor, block_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError>;
    fn concat_columns(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn rope(&self, input: &Self::Tensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<Self::Tensor, BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn rope_prefix(&self, input: &Self::Tensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<Self::Tensor, BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn rope_pair_prefix(
        &self,
        query: Self::Tensor,
        key: Self::Tensor,
        head_count: usize,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        position: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let query_output = self.rope_prefix(&query, head_count, rotary_dim, layout, position, cos, sin)?;
        let key_output = self.rope_prefix(&key, head_count, rotary_dim, layout, position, cos, sin)?;
        Ok((query_output, key_output))
    }
    /// decode 单行 Q/K RoPE 合并(Q/K 头数可不同,GQA 场景);默认两次独立调用,
    /// backend 可覆盖为单 dispatch 消除小算子延迟。
    #[allow(clippy::too_many_arguments)]
    fn rope_prefix_qk(
        &self,
        query: &Self::Tensor,
        key: &Self::Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        position: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let query_output = self.rope_prefix(query, num_heads, rotary_dim, layout, position, cos, sin)?;
        let key_output = self.rope_prefix(key, num_kv_heads, rotary_dim, layout, position, cos, sin)?;
        Ok((query_output, key_output))
    }
    fn add(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    /// 融合残差缩放：`(left + right) * scale`，避免单独提交逐元素 scale。
    fn add_scaled(&self, left: &Self::Tensor, right: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError>;
    /// `left + rmsnorm(right)` 的缩放残差；backend 可保留 RMSNorm 的中间精度语义并融合写回。
    fn rmsnorm_add_scaled(&self, left: &Self::Tensor, right: &Self::Tensor, weight: &Self::Weight, eps: f32, scale: f32) -> Result<Self::Tensor, BackendError> {
        let normalized = self.rmsnorm(right, weight, eps)?;
        self.add_scaled(left, &normalized, scale)
    }
    /// 把连续列分段后分别计算 `(left + rmsnorm(right, weight, eps)) * scale`。
    /// 默认实现保持组合语义；backend 可在连续存储上融合 reduction 与逐元素计算。
    #[allow(clippy::too_many_arguments)]
    fn segmented_rmsnorm_add_scaled(&self, left: &Self::Tensor, right: &Self::Tensor, weight: &Self::Weight, segments: usize, segment_columns: usize, eps: f32, scale: f32) -> Result<Vec<Self::Tensor>, BackendError> {
        let expected_columns = segments.checked_mul(segment_columns).ok_or_else(|| BackendError::Compute { msg: "segmented RMSNorm 列数溢出".to_owned() })?;
        if segments == 0 || self.token_rows(left) != self.token_rows(right) || self.token_cols(left) != expected_columns || self.token_cols(right) != expected_columns {
            return Err(BackendError::Compute {
                msg: format!("segmented RMSNorm shape 不兼容: left=[{},{}] right=[{},{}] segments={segments} columns={segment_columns}", self.token_rows(left), self.token_cols(left), self.token_rows(right), self.token_cols(right),),
            });
        }
        if segments == 1 {
            return Ok(vec![self.rmsnorm_add_scaled(left, right, weight, eps, scale)?]);
        }

        let (right_head, mut right_tail) = self.split_columns(right, segment_columns)?;
        let (left_head, mut left_tail) = self.split_columns(left, segment_columns)?;
        let mut output = Vec::with_capacity(segments);
        output.push(self.rmsnorm_add_scaled(&left_head, &right_head, weight, eps, scale)?);
        for _ in 1..segments - 1 {
            let (right_head, tail) = self.split_columns(&right_tail, segment_columns)?;
            right_tail = tail;
            let (left_head, tail) = self.split_columns(&left_tail, segment_columns)?;
            left_tail = tail;
            output.push(self.rmsnorm_add_scaled(&left_head, &right_head, weight, eps, scale)?);
        }
        output.push(self.rmsnorm_add_scaled(&left_tail, &right_tail, weight, eps, scale)?);
        Ok(output)
    }
    /// `input * sigmoid(gate)`；gate 可与 input 同 shape，也可每行只有一个标量。
    fn sigmoid_gate(&self, input: &Self::Tensor, gate: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    /// 在设备内取一行；prefill 只把最后一个 token 送入 LM head。
    fn select_row(&self, input: &Self::Tensor, row: usize) -> Result<Self::Tensor, BackendError>;
    /// 在设备内按给定顺序取多行；模型 runtime 用它构造移位后的 MTP 输入。
    fn select_rows(&self, input: &Self::Tensor, rows: &[u32]) -> Result<Self::Tensor, BackendError> {
        if rows.len() == 1 { self.select_row(input, rows[0] as usize) } else { Err(BackendError::Compute { msg: "当前 backend 不支持 select_rows".to_owned() }) }
    }
    /// 在设备内完成 logits argmax；host 只接收最终 token id。
    fn argmax(&self, input: &Self::Tensor) -> Result<u32, BackendError>;
    /// 在设备内完成带禁止 token 的 argmax；生成策略不需要把 logits 拷回 host。
    fn argmax_excluding(&self, input: &Self::Tensor, excluded: &[u32]) -> Result<u32, BackendError> {
        if excluded.is_empty() { self.argmax(input) } else { Err(BackendError::Compute { msg: "当前 backend 不支持带禁止 token 的 argmax".to_owned() }) }
    }
    /// 在设备内按 temperature/top-p 采样；host 只接收最终 token id。
    fn sample_top_p(&self, input: &Self::Tensor, temperature: f32, top_p: f32, random: f32) -> Result<u32, BackendError>;
    /// 带禁止 token 的 top-p；默认保持现有 backend 的空禁止集合能力。
    fn sample_top_p_excluding(&self, input: &Self::Tensor, temperature: f32, top_p: f32, random: f32, excluded: &[u32]) -> Result<u32, BackendError> {
        if excluded.is_empty() { self.sample_top_p(input, temperature, top_p, random) } else { Err(BackendError::Compute { msg: "当前 backend 不支持带禁止 token 的 top-p".to_owned() }) }
    }
    fn gated_activation(&self, gate: &Self::Tensor, up: &Self::Tensor, activation: &Activation) -> Result<Self::Tensor, BackendError>;
}

/// Host 权重视图，只在 decode layer 准备阶段使用。
#[derive(Clone, Copy)]
pub enum LinearWeight<'a> {
    F32(&'a [f32]),
    F16(&'a [half::f16]),
    Bf16Bytes(&'a [u8]),
    Quantized(crate::weight::format::quantization::QuantizedMatrixRef<'a>),
}

impl<'a> LinearWeight<'a> {
    pub fn block_fp8(weight: &'a crate::weight::format::block_fp8::BlockFp8Matrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::BlockFp8(weight))
    }

    pub fn mxfp4(weight: &'a crate::weight::format::mxfp4::Mxfp4Matrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Mxfp4(weight))
    }

    pub fn fp8(weight: &'a Fp8Matrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Fp8(weight))
    }

    pub fn per_tensor_fp8(weight: &'a crate::weight::PerTensorFp8Matrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::PerTensorFp8(weight))
    }

    pub fn mxfp8(weight: &'a Mxfp8Matrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Mxfp8(weight))
    }

    pub fn nvfp4(weight: &'a crate::weight::format::nvfp4::Nvfp4Matrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Nvfp4(weight))
    }

    pub fn w4a16(weight: &'a crate::weight::format::quantization::W4A16Matrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::W4A16(weight))
    }

    pub fn w8a16(weight: &'a crate::weight::format::quantization::W8A16Matrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::W8A16(weight))
    }

    pub fn mlx_affine(weight: &'a crate::weight::format::quantization::MlxAffineMatrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::MlxAffine(weight))
    }

    pub fn gguf(weight: &'a GgufMatrix) -> Self {
        Self::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(weight))
    }
}

fn prepare_weight_rows_default<B: BackendResources + ?Sized>(backend: &B, weight: LinearWeight<'_>, source_rows: usize, cols: usize, selected_rows: &[u32]) -> Result<B::Weight, BackendError> {
    if selected_rows.is_empty() || selected_rows.iter().any(|&row| row as usize >= source_rows) {
        return Err(BackendError::Compute { msg: format!("selected weight rows 非法: source_rows={source_rows} selected={}", selected_rows.len()) });
    }
    let select = |values: &[f32]| -> Result<Vec<f32>, BackendError> {
        if values.len() != source_rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: "selected weight shape 溢出".to_owned() })? {
            return Err(BackendError::Compute { msg: format!("selected F32 weight 元素数 {}，期望 {}", values.len(), source_rows * cols) });
        }
        let mut output = Vec::with_capacity(selected_rows.len() * cols);
        for &row in selected_rows {
            output.extend_from_slice(&values[row as usize * cols..(row as usize + 1) * cols]);
        }
        Ok(output)
    };
    match weight {
        LinearWeight::F32(values) => {
            let values = select(values)?;
            backend.prepare_f32(&values, selected_rows.len(), cols)
        }
        LinearWeight::F16(values) => {
            if values.len() != source_rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: "selected F16 weight shape 溢出".to_owned() })? {
                return Err(BackendError::Compute { msg: format!("selected F16 weight 元素数 {}，期望 {}", values.len(), source_rows * cols) });
            }
            let mut output = Vec::with_capacity(selected_rows.len() * cols);
            for &row in selected_rows {
                output.extend_from_slice(&values[row as usize * cols..(row as usize + 1) * cols]);
            }
            backend.prepare_weight(LinearWeight::F16(&output), selected_rows.len(), cols)
        }
        LinearWeight::Bf16Bytes(values) => {
            let row_bytes = cols.checked_mul(2).ok_or_else(|| BackendError::Compute { msg: "selected BF16 row bytes 溢出".to_owned() })?;
            if values.len() != source_rows.checked_mul(row_bytes).ok_or_else(|| BackendError::Compute { msg: "selected BF16 weight shape 溢出".to_owned() })? {
                return Err(BackendError::Compute { msg: format!("selected BF16 weight 字节数 {}，期望 {}", values.len(), source_rows * row_bytes) });
            }
            let mut output = Vec::with_capacity(selected_rows.len() * row_bytes);
            for &row in selected_rows {
                output.extend_from_slice(&values[row as usize * row_bytes..(row as usize + 1) * row_bytes]);
            }
            backend.prepare_weight(LinearWeight::Bf16Bytes(&output), selected_rows.len(), cols)
        }
        LinearWeight::Quantized(matrix) => {
            if matrix.rows() != source_rows || matrix.cols() != cols {
                return Err(BackendError::Compute { msg: format!("selected {} weight shape [{},{}]，期望 [{source_rows},{cols}]", matrix.name(), matrix.rows(), matrix.cols()) });
            }
            let decoded = matrix.decode().map_err(|msg| BackendError::Compute { msg })?;
            let values = select(&decoded)?;
            backend.prepare_f32(&values, selected_rows.len(), cols)
        }
    }
}

/// 视觉编码器需要的平台能力。模型专有的层顺序留在 `runtime::<model>`，
/// backend 只处理张量放置、普通 ViT 算子和 image embedding 写入。
pub trait VisionBackend: Backend {
    fn vision_tensor_from_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Tensor, BackendError>;
    fn vision_tensor_zeros(&self, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        let elements = rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: "视觉 zeros 大小溢出".to_owned() })?;
        self.vision_tensor_from_f32(&vec![0.0; elements], rows, cols)
    }
    fn vision_tensor_from_f32_bf16(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        self.vision_tensor_from_f32(values, rows, cols)
    }
    fn add_bias(&self, input: &Self::Tensor, bias: &Self::Weight) -> Result<Self::Tensor, BackendError>;
    fn gelu(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn vision_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, cos: &Self::Tensor, sin: &Self::Tensor, head_count: usize) -> Result<Self::Tensor, BackendError>;
    /// 二维 GPT-NeoX RoPE：head 前后两半分别使用 x/y 位置，各半内部再做 rotate-half。
    /// `score_scale` 显式传入，兼容 Gemma4V 的未缩放 attention。
    #[allow(clippy::too_many_arguments)]
    fn vision_attention_2d(&self, _query: &Self::Tensor, _key: &Self::Tensor, _value: &Self::Tensor, _cos: &Self::Tensor, _sin: &Self::Tensor, _head_count: usize, _score_scale: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持二维视觉 RoPE attention".to_owned() })
    }
    fn vision_clamp(&self, _input: &Self::Tensor, _minimum: f32, _maximum: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持视觉 clamp".to_owned() })
    }
    fn vision_quick_gelu_gated(&self, _gate: &Self::Tensor, _up: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 QuickGEGLU".to_owned() })
    }
    fn vision_average_pool(&self, _input: &Self::Tensor, _grid_height: usize, _grid_width: usize, _kernel_size: usize, _output_scale: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持视觉二维平均池化".to_owned() })
    }
    fn merge_spatial(&self, input: &Self::Tensor, merge_size: usize) -> Result<Self::Tensor, BackendError>;
    fn scatter_rows(&self, destination: &mut Self::Tensor, start_row: usize, source: &Self::Tensor) -> Result<(), BackendError>;
}

/// VAE 编解码器需要的平台能力。Conv3D/GroupNorm/像素shuffle 等算子，
/// 与 ViT 的 VisionBackend 平行独立，不合并。
pub trait VaeBackend: DiffusionBackend {
    fn vae_tensor_from_f32(&self, _values: Vec<f32>, _rows: usize, _cols: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE F32 tensor upload".to_owned() })
    }

    fn vae_tensor_to_f32(&self, _input: &Self::Tensor) -> Result<Vec<f32>, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE F32 tensor download".to_owned() })
    }

    fn layer_norm(&self, _input: &Self::Tensor, _weight: &Self::Weight, _bias: &Self::Weight, _eps: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE LayerNorm".to_owned() })
    }

    /// 对每个 attention head 做无仿射 RMSNorm，避免构造 host ones 权重。
    fn rms_norm_heads_unit(&self, _input: &Self::Tensor, _heads: usize, _head_dim: usize, _eps: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE unit head RMSNorm".to_owned() })
    }

    /// Q/K 同时做无仿射 head RMSNorm 与 SplitHalf prefix RoPE；默认组合已有能力。
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_rope_pair_unit(&self, query: &Self::Tensor, key: &Self::Tensor, heads: usize, head_dim: usize, rotary_dim: usize, eps: f32, cosine: &[f32], sine: &[f32]) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let query = self.rms_norm_heads_unit(query, heads, head_dim, eps)?;
        let key = self.rms_norm_heads_unit(key, heads, head_dim, eps)?;
        let query = self.rope_prefix(&query, heads, rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, 0, cosine, sine)?;
        let key = self.rope_prefix(&key, heads, rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, 0, cosine, sine)?;
        Ok((query, key))
    }

    /// `input + update * scale[column]`，scale 必须常驻设备。
    fn scaled_residual(&self, _input: &Self::Tensor, _update: &Self::Tensor, _scale: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE scaled residual".to_owned() })
    }

    /// `input + (update + bias) * scale[column]`；默认组合已有能力，设备可覆盖为单 kernel。
    fn scaled_residual_bias(&self, input: &Self::Tensor, update: &Self::Tensor, bias: &Self::Weight, scale: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        let update = self.add_row_bias(update, bias)?;
        self.scaled_residual(input, &update, scale)
    }

    /// packed gate/up 加 bias 后直接激活；默认组合已有能力，设备可避免完整 F32 bias 中间张量。
    fn split_gated_bias_activation(&self, input: Self::Tensor, bias: &Self::Weight, left_columns: usize, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        let input = self.add_row_bias(&input, bias)?;
        self.split_gated_activation(input, left_columns, activation)
    }

    /// row bias 与第一次列拆分合并；默认组合已有能力，设备可删除完整 bias 中间张量。
    fn split_columns_bias(&self, input: &Self::Tensor, bias: &Self::Weight, left_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let input = self.add_row_bias(input, bias)?;
        self.split_columns(&input, left_columns)
    }

    /// row bias 与三路等宽 Q/K/V 拆分合并；默认组合已有能力。
    #[allow(clippy::type_complexity)]
    fn split_three_columns_bias(&self, input: &Self::Tensor, bias: &Self::Weight, columns: usize) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), BackendError> {
        let input = self.add_row_bias(input, bias)?;
        let (first, tail) = self.split_columns(&input, columns)?;
        let (second, third) = self.split_columns(&tail, columns)?;
        Ok((first, second, third))
    }

    /// 将 DiT patch 行重排为 VAE voxel 行，同时执行 `latent * scale + bias`。
    fn unpatch_affine(&self, _input: &Self::Tensor, _scale: &Self::Weight, _bias: &Self::Weight, _shape: [usize; 3], _patch: [usize; 3], _channels: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE unpatch affine".to_owned() })
    }

    /// `[batch*time, channels]` DiT 音频行转成 `[batch*channels, time]`，并反归一化。
    fn audio_unpack_affine(&self, _input: &Self::Tensor, _scale: &Self::Weight, _bias: &Self::Weight, _batch: usize, _time: usize, _channels: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE audio unpack affine".to_owned() })
    }

    /// 输入按 `[batch*channels, time]` 保存；weight_v 可选 weight-norm 参数 weight_g。
    #[allow(clippy::too_many_arguments)]
    fn conv1d(&self, _input: &Self::Tensor, _weight_g: Option<&Self::Weight>, _weight_v: &Self::Weight, _bias: Option<&Self::Weight>, _spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE Conv1D".to_owned() })
    }

    fn conv1d_strided(&self, _input: &Self::Tensor, _weight_g: Option<&Self::Weight>, _weight_v: &Self::Weight, _bias: Option<&Self::Weight>, _spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持带 stride 的 VAE Conv1D".to_owned() })
    }

    fn snake(&self, _input: &Self::Tensor, _alpha: &Self::Weight, _channels: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE Snake".to_owned() })
    }

    fn vae_gelu(&self, _input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE GELU".to_owned() })
    }

    /// `[batch*channels, time] -> [batch*time, channels]`，供音频 encoder 进入 attention。
    fn channels_to_time(&self, _input: &Self::Tensor, _channels: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE channels-to-time transpose".to_owned() })
    }

    #[allow(clippy::too_many_arguments)]
    fn causal_attention(&self, _query: &Self::Tensor, _key: &Self::Tensor, _value: &Self::Tensor, _time: usize, _heads: usize, _head_dim: usize, _score_scale: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE causal attention".to_owned() })
    }

    fn conv_transpose1d(&self, _input: &Self::Tensor, _weight_g: &Self::Weight, _weight_v: &Self::Weight, _bias: &Self::Weight, _spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE ConvTranspose1D".to_owned() })
    }

    /// 官方 alias-free `upsample2 -> SnakeBeta -> downsample2`。
    #[allow(clippy::too_many_arguments)]
    fn snake_beta(&self, _input: &Self::Tensor, _alpha: &Self::Weight, _beta: &Self::Weight, _up_filter: &Self::Weight, _down_filter: &Self::Weight, _channels: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE SnakeBeta".to_owned() })
    }

    fn scale_tensor(&self, _input: &Self::Tensor, _scale: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE tensor scale".to_owned() })
    }

    fn tanh(&self, _input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE tanh".to_owned() })
    }

    /// 截取最前面的 token 行，供视频 VAE 丢弃 register/mask token。
    fn take_rows(&self, _input: &Self::Tensor, _rows: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE take rows".to_owned() })
    }

    /// 对每个 batch 独立截取前缀行。
    fn take_rows_batched(&self, _input: &Self::Tensor, _rows: usize, _batch: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 batched VAE take rows".to_owned() })
    }

    /// 将常驻设备的参数行追加到 activation，避免 register/mask token 经 host tensor 上传。
    fn concat_weight_rows(&self, _input: &Self::Tensor, _weight: &Self::Weight, _rows: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 VAE weight row concat".to_owned() })
    }

    /// 对每个 batch 重复追加同一组参数行。
    fn concat_weight_rows_batched(&self, _input: &Self::Tensor, _weight: &Self::Weight, _rows: usize, _batch: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 batched VAE weight row concat".to_owned() })
    }

    /// 时空 3D 卷积（视频 VAE 核心算子）。
    fn conv3d(&self, input: &Self::Tensor, weight: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv3dSpec) -> Result<Self::Tensor, BackendError>;

    /// Encoder Conv3D：空间 reflect padding，downsample 额外只补右/下边界。
    fn encoder_conv3d(&self, _input: &Self::Tensor, _weight: &Self::Weight, _bias: Option<&Self::Weight>, _spec: &Conv3dSpec, _spatial_pad_after: [usize; 2]) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 encoder Conv3D".to_owned() })
    }

    /// GroupNorm（VAE 专用归一化）。
    fn group_norm(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, num_groups: usize, eps: f32) -> Result<Self::Tensor, BackendError>;

    fn group_norm_time_isolated(&self, _input: &Self::Tensor, _weight: &Self::Weight, _bias: &Self::Weight, _time: usize, _num_groups: usize, _eps: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 time-isolated GroupNorm".to_owned() })
    }

    /// 像素 shuffle（空间上采样）。
    fn pixel_shuffle(&self, input: &Self::Tensor, spec: &PixelShuffleSpec) -> Result<Self::Tensor, BackendError>;
}

/// 扩散 Transformer 需要的平台能力。AdaLN 调制、时间步嵌入等，
/// 与标准 Backend 的 linear/norm 平行独立。
pub trait DiffusionBackend: Backend {
    /// 把 tensor 放到当前 backend 的设备；单设备 backend 默认无需搬运。
    fn transfer_tensor(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor)
    }

    fn silu(&self, _input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 diffusion SiLU".to_owned() })
    }

    fn add_row_bias(&self, _input: &Self::Tensor, _bias: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 diffusion row bias".to_owned() })
    }

    /// 沿 token 行拼接两个同列张量，保持扩散模型打包序列常驻设备。
    fn concat_rows(&self, _left: &Self::Tensor, _right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 diffusion row concat".to_owned() })
    }

    /// flow-matching velocity 的 Euler 更新：`sample + scale * velocity`。
    fn flow_step(&self, _sample: &Self::Tensor, _velocity: &Self::Tensor, _scale: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 diffusion flow step".to_owned() })
    }

    /// `[time, modalities*chunks*hidden] -> chunks 个 [time*modalities, hidden]`。
    fn modulation_chunks(&self, _input: &Self::Tensor, _modalities: usize, _chunks: usize, _hidden: usize) -> Result<Vec<Self::Tensor>, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 modulation chunks".to_owned() })
    }

    /// Q/K 每个 head 独立做普通 RMSNorm。
    fn rmsnorm_heads(&self, _input: &Self::Tensor, _weight: &Self::Weight, _head_count: usize, _head_dim: usize, _eps: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 per-head RMSNorm".to_owned() })
    }

    /// 消费 fused QKV，完成 Q/K per-head RMSNorm、前缀 SplitHalf RoPE 与全序列 attention。
    #[allow(clippy::too_many_arguments)]
    fn full_attention_qkv(
        &self,
        qkv: Self::Tensor,
        query_norm: &Self::Weight,
        key_norm: &Self::Weight,
        head_count: usize,
        head_dim: usize,
        rotary_dim: usize,
        eps: f32,
        cosine: &[f32],
        sine: &[f32],
        score_scale: f32,
    ) -> Result<Self::Tensor, BackendError> {
        let attention_dim = head_count.checked_mul(head_dim).ok_or_else(|| BackendError::Compute { msg: "full attention QKV columns 溢出".to_owned() })?;
        let (query, key_value) = self.split_columns(&qkv, attention_dim)?;
        let (key, value) = self.split_columns(&key_value, attention_dim)?;
        let query = self.rmsnorm_heads(&query, query_norm, head_count, head_dim, eps)?;
        let key = self.rmsnorm_heads(&key, key_norm, head_count, head_dim, eps)?;
        let (query, key) = self.rope_pair_prefix(query, key, head_count, rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, 0, cosine, sine)?;
        self.full_attention(query, key, value, head_count, head_dim, score_scale)
    }

    /// 无 mask 的全序列 self-attention；不物化 `S×S` score 矩阵。
    fn full_attention(&self, _query: Self::Tensor, _key: Self::Tensor, _value: Self::Tensor, _head_count: usize, _head_dim: usize, _score_scale: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 full self-attention".to_owned() })
    }

    /// 连续存放多个等长序列，attention 在 batch 边界处严格隔离。
    #[allow(clippy::too_many_arguments)]
    fn full_attention_batched(&self, query: Self::Tensor, key: Self::Tensor, value: Self::Tensor, batch: usize, rows: usize, head_count: usize, head_dim: usize, score_scale: f32) -> Result<Self::Tensor, BackendError> {
        if batch == 1 {
            return self.full_attention(query, key, value, head_count, head_dim, score_scale);
        }
        Err(BackendError::Compute { msg: format!("当前 backend 不支持 batch={batch} full self-attention rows={rows}") })
    }

    /// 自适应 LayerNorm：`out = input * (1 + scale) + shift`。
    fn adaln_modulate(&self, input: &Self::Tensor, shift: &Self::Tensor, scale: &Self::Tensor) -> Result<Self::Tensor, BackendError>;

    /// 从 host F32 构造扩散模型 tensor；pruned 时间曲线只上传极小的插值结果。
    fn diffusion_tensor_from_f32(&self, _values: &[f32], _rows: usize, _cols: usize) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 diffusion F32 tensor upload".to_owned() })
    }

    /// 正弦时间步嵌入（sinusoidal positional embedding）。
    fn timestep_embedding(&self, timesteps: &[f32], dim: usize) -> Result<Self::Tensor, BackendError>;

    /// 按连续 token 段广播 AdaLN 参数。
    fn adaln_modulate_segmented(&self, _input: &Self::Tensor, _shift: &Self::Tensor, _scale: &Self::Tensor, _segments: &[ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 segmented AdaLN".to_owned() })
    }

    /// 普通 RMSNorm 后按连续 token 段应用 AdaLN。默认保持组合语义；扩散模型
    /// 热路径可覆盖为单 kernel，避免写出再读回完整 normalized tensor。
    fn rmsnorm_adaln_modulate_segmented(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, shift: &Self::Tensor, scale: &Self::Tensor, segments: &[ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        let normalized = self.rmsnorm(input, norm_weight, eps)?;
        self.adaln_modulate_segmented(&normalized, shift, scale, segments)
    }

    /// `residual + update * gate`，gate 按连续 token 段广播。
    fn gated_residual_segmented(&self, _residual: &Self::Tensor, _update: &Self::Tensor, _gate: &Self::Tensor, _segments: &[ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "当前 backend 不支持 segmented gated residual".to_owned() })
    }
}

/// MLA prefill 的平台能力：runtime 负责投影与 RoPE，backend 负责 cache 驻留和 attention kernel 选择。
pub trait MlaPrefillBackend: Backend {
    #[allow(clippy::too_many_arguments)]
    fn mla_prefill_attention(&self, query: &Self::Tensor, latent: &Self::Tensor, k_rope: &Self::Tensor, kv_b: &Self::Weight, cache: Option<&mut Self::Cache>, layer: usize, spec: &MlaSpec) -> Result<Self::Tensor, BackendError>;
    /// K-RoPE 与 cache append 的融合入口。默认保持先旋转、再执行 prefill attention 的组合语义。
    #[allow(clippy::too_many_arguments)]
    fn mla_prefill_attention_rope(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        cache: Option<&mut Self::Cache>,
        layer: usize,
        position: usize,
        cos: &[f32],
        sin: &[f32],
        spec: &MlaSpec,
    ) -> Result<Self::Tensor, BackendError> {
        let k_rope = self.rope(k_rope, 1, spec.qk_rope_head_dim, spec.rotary_layout, position, cos, sin)?;
        self.mla_prefill_attention(query, latent, &k_rope, kv_b, cache, layer, spec)
    }
}

pub trait DecodeBackend: Backend {
    type DsaState;
    /// 准备 MLA 吸收式 KV-B 权重。模型只标注逻辑用途；是否保留量化、展开精度
    /// 以及设备驻留布局由实际消费 attention kernel 的 backend 决定。
    fn prepare_mla_kv_b(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        self.prepare_weight(weight, rows, cols)
    }
    fn append_mla(&self, cache: &mut Self::Cache, layer: usize, latent: &Self::Tensor, rope: &Self::Tensor) -> Result<(), BackendError>;
    /// decode 的 K-RoPE 与 cache append 可共享一次提交；默认保持先旋转、后写 cache 的组合语义。
    #[allow(clippy::too_many_arguments)]
    fn append_mla_rope(
        &self,
        cache: &mut Self::Cache,
        layer: usize,
        latent: &Self::Tensor,
        rope: &Self::Tensor,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        position: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(), BackendError> {
        let rope = self.rope(rope, 1, rotary_dim, layout, position, cos, sin)?;
        self.append_mla(cache, layer, latent, &rope)
    }
    fn mla_decode_attention(&self, query: &Self::Tensor, cache: &Self::Cache, kv_b: &Self::Weight, layer: usize, position: usize, spec: &MlaSpec) -> Result<Self::Tensor, BackendError>;

    fn dsa_can_append(&self, state: &mut Self::DsaState, layer: usize, position: usize, spec: &DsaSpec) -> bool;
    fn append_dsa_keys(&self, state: &mut Self::DsaState, layer: usize, position: usize, keys: &Self::Tensor, spec: &DsaSpec) -> Result<(), BackendError>;
    /// segmented 路径在写任何 cache 前，用这个只读能力查询保证融合可以整批启用。
    #[allow(clippy::too_many_arguments)]
    fn supports_dsa_keys_layernorm_rope(&self, state: &Self::DsaState, keys: &Self::Tensor, norm_weight: &Self::Weight, norm_bias: &Self::Weight, spec: &DsaSpec) -> bool {
        let _ = (state, keys, norm_weight, norm_bias, spec);
        false
    }
    /// decode indexer 的 key prologue 融合入口。返回 false 时调用方保持
    /// LayerNorm+bias、RoPE、cache append 的通用三算子路径。
    #[allow(clippy::too_many_arguments)]
    fn append_dsa_keys_layernorm_rope(
        &self,
        state: &mut Self::DsaState,
        layer: usize,
        position: usize,
        keys: &Self::Tensor,
        norm_weight: &Self::Weight,
        norm_bias: &Self::Weight,
        eps: f32,
        cos: &[f32],
        sin: &[f32],
        spec: &DsaSpec,
    ) -> Result<bool, BackendError> {
        let _ = (state, layer, position, keys, norm_weight, norm_bias, eps, cos, sin, spec);
        Ok(false)
    }
    /// kpool 池化(glm5_next DSA)的打包追加:key 与 gate 同步落盘。
    /// 默认拒绝,backend 需显式支持。
    fn append_dsa_keys_gated(&self, state: &mut Self::DsaState, layer: usize, position: usize, keys: &Self::Tensor, gate: &Self::Tensor, spec: &DsaSpec) -> Result<(), BackendError> {
        let _ = (state, layer, position, keys, gate, spec);
        Err(BackendError::Compute { msg: "backend 未实现 kpool 打包追加".to_owned() })
    }
    fn dsa_select_topk(&self, state: &mut Self::DsaState, layer: usize, query: &Self::Tensor, head_weights: &Self::Tensor, spec: &DsaSpec) -> Result<(), BackendError>;
    /// selection 与后续独立设备工作重叠的入口。默认后端保持同步语义；只有真正
    /// 拥有异步 CPU/device 实现的 backend 才需要覆盖 begin/finish。
    fn dsa_select_topk_begin(&self, state: &mut Self::DsaState, layer: usize, query: &Self::Tensor, head_weights: &Self::Tensor, spec: &DsaSpec) -> Result<(), BackendError> {
        self.dsa_select_topk(state, layer, query, head_weights, spec)
    }
    /// 在 selected attention 消费 selection 前退休 begin 提交的工作。
    fn dsa_select_topk_finish(&self, _state: &mut Self::DsaState) -> Result<(), BackendError> {
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_decode_attention_selected(&self, query: &Self::Tensor, cache: &Self::Cache, kv_b: &Self::Weight, layer: usize, position: usize, mla: &MlaSpec, dsa: &DsaSpec, state: &Self::DsaState) -> Result<Self::Tensor, BackendError>;
}

/// DSA prefill 的批量选择与稀疏 MLA 能力。模型层分布与 selection 复用由 runtime 决定。
pub trait DsaPrefillBackend: MlaPrefillBackend + DecodeBackend {
    fn dsa_select_prefill(&self, state: &mut Self::DsaState, layer: usize, query: &Self::Tensor, head_weights: &Self::Tensor, spec: &DsaSpec) -> Result<(), BackendError>;

    #[allow(clippy::too_many_arguments)]
    fn mla_prefill_attention_selected(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        cache: Option<&mut Self::Cache>,
        layer: usize,
        mla: &MlaSpec,
        dsa: &DsaSpec,
        state: &Self::DsaState,
    ) -> Result<Self::Tensor, BackendError>;

    /// 稀疏 MLA 的 K-RoPE 与 cache append 融合入口；默认保持旧的组合路径。
    #[allow(clippy::too_many_arguments)]
    fn mla_prefill_attention_selected_rope(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        cache: Option<&mut Self::Cache>,
        layer: usize,
        position: usize,
        cos: &[f32],
        sin: &[f32],
        mla: &MlaSpec,
        dsa: &DsaSpec,
        state: &Self::DsaState,
    ) -> Result<Self::Tensor, BackendError> {
        let k_rope = self.rope(k_rope, 1, mla.qk_rope_head_dim, mla.rotary_layout, position, cos, sin)?;
        self.mla_prefill_attention_selected(query, latent, &k_rope, kv_b, cache, layer, mla, dsa, state)
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_prefill_attention_selected_segmented(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        layer: usize,
        mla: &MlaSpec,
        dsa: &DsaSpec,
        segments: &mut [DsaPrefillSegment<'_, Self>],
    ) -> Result<Self::Tensor, BackendError> {
        let _ = (query, latent, k_rope, kv_b, layer, mla, dsa, segments);
        Err(BackendError::Compute { msg: "backend 未实现 segmented DSA prefill".to_owned() })
    }
}

pub struct DsaPrefillSegment<'a, B: DsaPrefillBackend + ?Sized> {
    pub rows: usize,
    pub cache: &'a mut B::Cache,
    pub state: &'a B::DsaState,
}

#[derive(Clone, Copy)]
pub struct TokenSegment {
    pub position: usize,
    pub rows: usize,
}

/// 连续 batch 的 token 行操作。runtime 只描述 sequence 边界，backend 决定是否用 view 或复制。
pub trait SegmentedTensorBackend: Backend {
    fn concat_token_rows(&self, tensors: &[&Self::Tensor]) -> Result<Self::Tensor, BackendError>;
    /// 为持续增长的行序列预留容量。默认保持普通 concat；设备后端可在已有
    /// prefix 占用同一 allocation 起点时只追加 suffix。
    fn concat_token_rows_reserved(&self, tensors: &[&Self::Tensor], capacity_rows: usize) -> Result<Self::Tensor, BackendError> {
        let _ = capacity_rows;
        self.concat_token_rows(tensors)
    }
    /// 消费 prefix 后追加 suffix。拥有唯一 allocation 的后端可原地扩展；默认
    /// 保持 concat 语义，供没有可变 host allocation 的设备后端复用。
    fn append_token_rows_reserved(&self, prefix: Self::Tensor, suffix: &Self::Tensor, capacity_rows: usize) -> Result<Self::Tensor, BackendError> {
        self.concat_token_rows_reserved(&[&prefix, suffix], capacity_rows)
    }
    fn slice_token_rows(&self, tensor: &Self::Tensor, row_start: usize, rows: usize) -> Result<Self::Tensor, BackendError>;
    /// 逐行选择 token；默认保持单行语义，设备后端可合并 kernel 和 host 同步。
    fn argmax_rows_excluding(&self, input: &Self::Tensor, excluded: &[u32]) -> Result<Vec<u32>, BackendError> {
        let rows = self.token_rows(input);
        if rows == 0 {
            return Err(BackendError::Compute { msg: "argmax rows 输入不能为空".to_owned() });
        }
        (0..rows).map(|row| self.slice_token_rows(input, row, 1).and_then(|row| self.argmax_excluding(&row, excluded))).collect()
    }
    fn argmax_rows_fenced(&self, input: &Self::Tensor, fences: &[TokenFence]) -> Result<Vec<u32>, BackendError> {
        let rows = self.token_rows(input);
        if rows == 0 || rows != fences.len() {
            return Err(BackendError::Compute { msg: format!("argmax fenced rows shape 不兼容: rows={rows} fences={}", fences.len()) });
        }
        if let Some(first) = fences.first().filter(|first| fences.iter().all(|fence| fence == *first)) {
            if let Some(token) = first.forced() {
                return Ok(vec![token; rows]);
            }
            return self.argmax_rows_excluding(input, first.excluded());
        }
        (0..rows)
            .map(|row| {
                if let Some(token) = fences[row].forced() {
                    return Ok(token);
                }
                self.slice_token_rows(input, row, 1).and_then(|input| self.argmax_excluding(&input, fences[row].excluded()))
            })
            .collect()
    }
    /// 按行使用独立策略选择 token；设备后端可合并 kernel 和 host 同步。
    fn sample_rows_excluding(&self, input: &Self::Tensor, sampling: &[TokenSampling], excluded: &[u32]) -> Result<Vec<u32>, BackendError> {
        let rows = self.token_rows(input);
        if rows == 0 || rows != sampling.len() {
            return Err(BackendError::Compute { msg: format!("sample rows shape 不兼容: rows={rows} sampling={}", sampling.len()) });
        }
        (0..rows)
            .map(|row| {
                let input = self.slice_token_rows(input, row, 1)?;
                let sample = sampling[row];
                if sample.temperature == 0.0 { self.argmax_excluding(&input, excluded) } else { self.sample_top_p_excluding(&input, sample.temperature, sample.top_p, sample.random, excluded) }
            })
            .collect()
    }
    /// 每行使用独立生成围栏。默认实现保持正确性；设备后端可在围栏相同时复用
    /// 批量 kernel，或实现逐行 mask kernel。
    fn sample_rows_fenced(&self, input: &Self::Tensor, sampling: &[TokenSampling], fences: &[TokenFence]) -> Result<Vec<u32>, BackendError> {
        let rows = self.token_rows(input);
        if rows == 0 || rows != sampling.len() || rows != fences.len() {
            return Err(BackendError::Compute { msg: format!("sample fenced rows shape 不兼容: rows={rows} sampling={} fences={}", sampling.len(), fences.len()) });
        }
        if let Some(first) = fences.first().filter(|first| fences.iter().all(|fence| fence == *first)) {
            if let Some(token) = first.forced() {
                return Ok(vec![token; rows]);
            }
            return self.sample_rows_excluding(input, sampling, first.excluded());
        }
        (0..rows)
            .map(|row| {
                if let Some(token) = fences[row].forced() {
                    return Ok(token);
                }
                let input = self.slice_token_rows(input, row, 1)?;
                let sample = sampling[row];
                let excluded = fences[row].excluded();
                if sample.temperature == 0.0 { self.argmax_excluding(&input, excluded) } else { self.sample_top_p_excluding(&input, sample.temperature, sample.top_p, sample.random, excluded) }
            })
            .collect()
    }
    #[allow(clippy::too_many_arguments)]
    fn rope_segmented(
        &self,
        input: &Self::Tensor,
        head_count: usize,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        prefix: bool,
        segments: &[TokenSegment],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Self::Tensor, BackendError> {
        let rows = segments.iter().try_fold(0usize, |rows, segment| rows.checked_add(segment.rows).ok_or_else(|| BackendError::Compute { msg: "segmented RoPE rows 溢出".to_owned() }))?;
        if segments.is_empty() || segments.iter().any(|segment| segment.rows == 0) || rows != self.token_rows(input) {
            return Err(BackendError::Compute { msg: format!("segmented RoPE segments={} rows={rows}，tensor rows={}", segments.len(), self.token_rows(input)) });
        }
        let mut offset = 0usize;
        let mut outputs = Vec::with_capacity(segments.len());
        for segment in segments {
            let input = self.slice_token_rows(input, offset, segment.rows)?;
            let output = if prefix { self.rope_prefix(&input, head_count, rotary_dim, layout, segment.position, cos, sin)? } else { self.rope(&input, head_count, rotary_dim, layout, segment.position, cos, sin)? };
            outputs.push(output);
            offset += segment.rows;
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        self.concat_token_rows(&outputs)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_segmented_pair(
        &self,
        query: &Self::Tensor,
        query_head_count: usize,
        key: &Self::Tensor,
        key_head_count: usize,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        prefix: bool,
        segments: &[TokenSegment],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let rows = segments.iter().try_fold(0usize, |rows, segment| rows.checked_add(segment.rows).ok_or_else(|| BackendError::Compute { msg: "segmented RoPE rows 溢出".to_owned() }))?;
        if self.token_rows(query) != self.token_rows(key) {
            return Err(BackendError::Compute { msg: format!("segmented RoPE Q/K rows 不一致: {} != {}", self.token_rows(query), self.token_rows(key)) });
        }
        if segments.is_empty() || segments.iter().any(|segment| segment.rows == 0) || rows != self.token_rows(query) {
            return Err(BackendError::Compute { msg: format!("segmented RoPE segments={} rows={rows}，Q/K rows={}", segments.len(), self.token_rows(query)) });
        }
        let mut offset = 0usize;
        let mut queries = Vec::with_capacity(segments.len());
        let mut keys = Vec::with_capacity(segments.len());
        for segment in segments {
            let query = self.slice_token_rows(query, offset, segment.rows)?;
            let key = self.slice_token_rows(key, offset, segment.rows)?;
            let (query, key) = if prefix {
                (self.rope_prefix(&query, query_head_count, rotary_dim, layout, segment.position, cos, sin)?, self.rope_prefix(&key, key_head_count, rotary_dim, layout, segment.position, cos, sin)?)
            } else {
                (self.rope(&query, query_head_count, rotary_dim, layout, segment.position, cos, sin)?, self.rope(&key, key_head_count, rotary_dim, layout, segment.position, cos, sin)?)
            };
            queries.push(query);
            keys.push(key);
            offset += segment.rows;
        }
        let queries = queries.iter().collect::<Vec<_>>();
        let keys = keys.iter().collect::<Vec<_>>();
        Ok((self.concat_token_rows(&queries)?, self.concat_token_rows(&keys)?))
    }
}

pub trait GqaPrefillBackend: Backend {
    fn rmsnorm_heads(&self, _input: &Self::Tensor, _weight: &Self::Weight, _head_count: usize, _head_dim: usize, _eps: f32) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Compute { msg: "backend 未实现逐 head RMSNorm".to_owned() })
    }
    fn gemma_rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError>;
    fn gemma_rmsnorm_heads_f32(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        self.gemma_rmsnorm_heads(input, weight, head_count, head_dim, eps)
    }
    fn gqa_prefill_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, spec: &GqaSpec) -> Result<Self::Tensor, BackendError>;
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<Self::Tensor, BackendError>;

    #[allow(clippy::too_many_arguments)]
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
        let _ = (cache, layer, position, query, key, value, spec, visible_ends, retain_full_cache);
        Err(BackendError::Compute { msg: "backend 未实现带视觉块可见性的 GQA prefill".to_owned() })
    }

    fn gqa_prefill_attention_cached_from(&self, cache: &Self::Cache, source_layer: usize, position: usize, query: &Self::Tensor, spec: &GqaSpec) -> Result<Self::Tensor, BackendError>;

    /// `rmsnorm -> Q/K/V gemv -> 逐头 RMSNorm -> Split-half RoPE` 单次
    /// launch。默认不支持，由具体 backend 提供；模型 runtime 决定是否使用。
    #[allow(clippy::too_many_arguments)]
    fn fused_rmsnorm_qkv_head_norm_rope(
        &self,
        hidden: &Self::Tensor,
        input_norm: &Self::Weight,
        query: &Self::Weight,
        key: &Self::Weight,
        value: &Self::Weight,
        query_norm: &Self::Weight,
        key_norm: &Self::Weight,
        cos: &[f32],
        sin: &[f32],
        head_count: usize,
        kv_head_count: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        eps: f32,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), BackendError> {
        let _ = (hidden, input_norm, query, key, value, query_norm, key_norm, cos, sin, head_count, kv_head_count, head_dim, rotary_dim, position, eps);
        Err(BackendError::Compute { msg: "backend 未提供融合 RMSNorm+QKV+head norm+RoPE".to_owned() })
    }

    /// Q/K/V 三路逐头 RMSNorm；backend 可用一个 encoder 合并提交。
    #[allow(clippy::too_many_arguments)]
    fn qkv_head_norms(
        &self,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        query_norm: &Self::Weight,
        key_norm: &Self::Weight,
        value_norm: &Self::Weight,
        head_count: usize,
        kv_head_count: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), BackendError> {
        let query = self.gemma_rmsnorm_heads(query, query_norm, head_count, head_dim, eps)?;
        let key = self.gemma_rmsnorm_heads(key, key_norm, kv_head_count, head_dim, eps)?;
        let value = self.gemma_rmsnorm_heads(value, value_norm, kv_head_count, head_dim, eps)?;
        Ok((query, key, value))
    }

    /// rmsnorm → 三路独立投影(Q/K/V)共享一个 encoder,省一个边界。
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_triple_linear(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, first: &Self::Weight, second: &Self::Weight, third: &Self::Weight) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), BackendError> {
        let normed = self.rmsnorm(input, norm_weight, eps)?;
        self.triple_linear(&normed, first, second, third)
    }

    /// rmsnorm → gated gemv(gate/up+激活)共享一个 encoder,省一个边界。
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_gated_linear(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, gate: &Self::Weight, up: &Self::Weight, activation: &crate::moe::Activation) -> Result<Self::Tensor, BackendError> {
        let normed = self.rmsnorm(input, norm_weight, eps)?;
        self.gated_linear(&normed, gate, up, activation)
    }

    fn gqa_prefill_attention_cached_from_visible(&self, cache: &Self::Cache, source_layer: usize, position: usize, query: &Self::Tensor, spec: &GqaSpec, visible_ends: &[u32]) -> Result<Self::Tensor, BackendError> {
        let _ = (cache, source_layer, position, query, spec, visible_ends);
        Err(BackendError::Compute { msg: "backend 未实现 shared-KV 视觉块可见性".to_owned() })
    }
}

/// query/KV 长度与可见范围显式分离的模型无关块注意力能力。
pub trait BlockAttentionBackend: Backend {
    fn block_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, spec: &BlockAttentionSpec) -> Result<Self::Tensor, BackendError>;

    /// 多段 K/V 保持原始存储，不为 batch session 物化一份巨大的连续历史。
    /// 默认实现兼容现有 backend；CPU/设备后端可直接分段读取。
    fn block_attention_segments(&self, query: &Self::Tensor, keys: &[&Self::Tensor], values: &[&Self::Tensor], spec: &BlockAttentionSpec) -> Result<Self::Tensor, BackendError>
    where
        Self: SegmentedTensorBackend,
    {
        if keys.len() != values.len() || keys.is_empty() {
            return Err(BackendError::Compute { msg: format!("block attention segments K/V 数量={}/{}", keys.len(), values.len()) });
        }
        let key = self.concat_token_rows(keys)?;
        let value = self.concat_token_rows(values)?;
        self.block_attention(query, &key, &value, spec)
    }

    /// 在不物化连续 K/V 的情况下读取常驻前缀和本轮短后缀。默认实现保持
    /// 参考语义，后端可覆盖为分段读取 kernel。
    fn block_attention_prefix_suffix(
        &self,
        query: &Self::Tensor,
        prefix_key: &Self::Tensor,
        prefix_value: &Self::Tensor,
        suffix_key: &Self::Tensor,
        suffix_value: &Self::Tensor,
        spec: &BlockAttentionSpec,
    ) -> Result<Self::Tensor, BackendError>
    where
        Self: SegmentedTensorBackend,
    {
        let key = self.concat_token_rows(&[prefix_key, suffix_key])?;
        let value = self.concat_token_rows(&[prefix_value, suffix_value])?;
        self.block_attention(query, &key, &value, spec)
    }
}

/// MoE prefill 的通用张量编排能力；专家权重的加载策略由 runtime closure 决定。
pub trait MoePrefillBackend: Backend {
    type MoeAccumulator;

    fn moe_route(&self, input: &Self::Tensor, router_weight: &Self::Weight, router_bias: &Self::Weight, spec: &TopkMoeSpec) -> Result<MoePrefillRouting, BackendError>;

    /// 模型已经给出专家 ID 时，只计算这些专家对应的路由权重。
    fn moe_route_selected(&self, input: &Self::Tensor, router_weight: &Self::Weight, selected_experts: &[u32], spec: &TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        let _ = (input, router_weight, selected_experts, spec);
        Err(BackendError::Compute { msg: "backend 未实现固定专家路由".to_owned() })
    }

    fn moe_zeros(&self, rows: usize, cols: usize) -> Result<Self::MoeAccumulator, BackendError>;
    fn moe_gather_rows(&self, input: &Self::Tensor, rows: &[u32]) -> Result<Self::Tensor, BackendError>;
    fn moe_gather_rows_batch(&self, input: &Self::Tensor, batches: &[Vec<u32>]) -> Result<Vec<Self::Tensor>, BackendError> {
        batches.iter().map(|rows| self.moe_gather_rows(input, rows)).collect()
    }
    fn moe_scatter_add_rows(&self, output: &mut Self::MoeAccumulator, input: &Self::Tensor, rows: &[u32], weights: &[f32]) -> Result<(), BackendError>;
    fn moe_scatter_add_rows_batch(&self, output: &mut Self::MoeAccumulator, inputs: &[Self::Tensor], rows: &[Vec<u32>], weights: &[Vec<f32>]) -> Result<(), BackendError> {
        if inputs.len() != rows.len() || rows.len() != weights.len() {
            return Err(BackendError::Compute { msg: format!("MoE batch scatter 数量异常: inputs={}, rows={}, weights={}", inputs.len(), rows.len(), weights.len(),) });
        }
        for ((input, rows), weights) in inputs.iter().zip(rows).zip(weights) {
            self.moe_scatter_add_rows(output, input, rows, weights)?;
        }
        Ok(())
    }
    fn moe_finish(&self, output: Self::MoeAccumulator) -> Result<Self::Tensor, BackendError>;
}

/// 专家 kernel 能力。routing、分组、批次和 scatter 统一由 runtime 编排。
pub trait ExpertPrefillBackend: MoePrefillBackend {
    type PrefillExperts;

    /// 同一 MLA 层已经为两张卡准备 query-head / KV-B / O-proj 分片时返回 true。
    fn supports_cooperative_mla_prefill(&self, _layer: usize, _experts: &Self::PrefillExperts) -> bool {
        false
    }

    /// cooperative attention 已绑定 peer 时，从首次写入起把 DSA key history
    /// 按固定 block parity 落到唯一物理 owner。返回 true 表示 append 已完成。
    #[allow(clippy::too_many_arguments)]
    fn cooperative_dsa_append_keys_layernorm_rope(
        &self,
        _layer: usize,
        _experts: &Self::PrefillExperts,
        _state: &mut <Self as DecodeBackend>::DsaState,
        _position: usize,
        _keys: &Self::Tensor,
        _norm_weight: &Self::Weight,
        _norm_bias: &Self::Weight,
        _eps: f32,
        _cosine: &[f32],
        _sine: &[f32],
        _spec: &DsaSpec,
    ) -> Result<bool, BackendError>
    where
        Self: DsaPrefillBackend,
    {
        Ok(false)
    }

    /// DSA prefill 的 query 行彼此独立时，允许 cooperative peer 分担一部分
    /// query，并把完整 selection 按原行序归并回 owner。返回 true 表示 selection
    /// 已写入 state；默认后端保持单卡路径。
    fn cooperative_dsa_select_prefill(&self, _layer: usize, _experts: &Self::PrefillExperts, _state: &mut <Self as DecodeBackend>::DsaState, _query: &Self::Tensor, _head_weights: &Self::Tensor, _spec: &DsaSpec) -> Result<bool, BackendError>
    where
        Self: DsaPrefillBackend,
    {
        Ok(false)
    }

    /// cooperative peer 已常驻 Indexer query 投影权重时，直接按 token 行拆分
    /// q_lora，在两卡分别完成 wq_b、RoPE 与完整历史扫描。返回 true 表示
    /// selection 已写入 state；默认后端保持单卡投影与 selection。
    #[allow(clippy::too_many_arguments)]
    fn cooperative_dsa_project_select_prefill(
        &self,
        _layer: usize,
        _experts: &Self::PrefillExperts,
        _state: &mut <Self as DecodeBackend>::DsaState,
        _q_lora: &Self::Tensor,
        _owner_wq_b: &Self::Weight,
        _head_weights: &Self::Tensor,
        _position: usize,
        _cosine: &[f32],
        _sine: &[f32],
        _spec: &DsaSpec,
    ) -> Result<bool, BackendError>
    where
        Self: DsaPrefillBackend,
    {
        Ok(false)
    }

    /// 两卡共同完成 q_b → MLA → o_proj，并把两份 full-hidden partial 与 residual
    /// 在 owner 上融合。KV/DSA 的物理 ownership 与同步由 backend 负责。
    #[allow(clippy::too_many_arguments)]
    fn cooperative_mla_prefill_add(
        &self,
        layer: usize,
        experts: &Self::PrefillExperts,
        normalized_q_lora: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        residual: &Self::Tensor,
        cache: Option<&mut Self::Cache>,
        dsa_state: Option<&<Self as DecodeBackend>::DsaState>,
        position: usize,
        cosine: &[f32],
        sine: &[f32],
        mla: &MlaSpec,
        dsa: &DsaSpec,
    ) -> Result<Self::Tensor, BackendError>
    where
        Self: DsaPrefillBackend,
    {
        let _ = (layer, experts, normalized_q_lora, latent, k_rope, residual, cache, dsa_state, position, cosine, sine, mla, dsa);
        Err(BackendError::Compute { msg: "backend 未实现 cooperative MLA prefill".to_owned() })
    }

    /// resident routed 与 shared expert 可以共享同一份输入和显式工作区时，
    /// 直接完成整层 MoE 与 residual 合并；不支持时返回 `None`。
    fn prefill_resident_moe_add(
        &self,
        spec: &TopkMoeSpec,
        weights: RoutedMoeWeightsRef<'_, Self::Weight>,
        shared_experts: &[SharedExpertRef<'_, Self::Weight>],
        layer: usize,
        experts: &mut Self::PrefillExperts,
        inputs: RoutedMoeInputs<'_, Self::Tensor>,
        residual: &Self::Tensor,
    ) -> Result<Option<Self::Tensor>, BackendError> {
        let _ = (spec, weights, shared_experts, layer, experts, inputs, residual);
        Ok(None)
    }

    /// 单 token 且完整 expert archive 已驻留时，将 routing 与 routed expert 全程留在设备端。
    fn prefill_resident_routed_experts(
        &self,
        spec: &TopkMoeSpec,
        weights: RoutedMoeWeightsRef<'_, Self::Weight>,
        layer: usize,
        experts: &mut Self::PrefillExperts,
        inputs: RoutedMoeInputs<'_, Self::Tensor>,
    ) -> Result<Option<Self::Tensor>, BackendError> {
        let _ = (spec, weights, layer, experts, inputs);
        Ok(None)
    }

    /// 确定性 routed 归约时直接合并 shared output；不支持时返回 `None`。
    fn prefill_resident_routed_experts_add_shared(
        &self,
        spec: &TopkMoeSpec,
        weights: RoutedMoeWeightsRef<'_, Self::Weight>,
        layer: usize,
        experts: &mut Self::PrefillExperts,
        inputs: RoutedMoeInputs<'_, Self::Tensor>,
        shared: &Self::Tensor,
    ) -> Result<Option<Self::Tensor>, BackendError> {
        let _ = (spec, weights, layer, experts, inputs, shared);
        Ok(None)
    }

    /// 确定性 routed 归约时直接合并 shared output 与层 residual；不支持时返回 `None`。
    fn prefill_resident_routed_experts_add(
        &self,
        spec: &TopkMoeSpec,
        weights: RoutedMoeWeightsRef<'_, Self::Weight>,
        layer: usize,
        experts: &mut Self::PrefillExperts,
        inputs: RoutedMoeInputs<'_, Self::Tensor>,
        shared: Option<&Self::Tensor>,
        residual: &Self::Tensor,
    ) -> Result<Option<Self::Tensor>, BackendError> {
        let _ = (spec, weights, layer, experts, inputs, shared, residual);
        Ok(None)
    }

    fn prefill_expert_batch(&self, spec: &TopkMoeSpec, layer: usize, experts: &mut Self::PrefillExperts, batch: Vec<ExpertPrefillBatch<Self::Tensor>>) -> Result<Vec<Self::Tensor>, BackendError> {
        let _ = (spec, layer, experts, batch);
        Err(BackendError::Compute { msg: "backend 未实现 expert batch capability".to_owned() })
    }

    fn prefill_routed_experts(&self, spec: &TopkMoeSpec, layer: usize, experts: &mut Self::PrefillExperts, input: &Self::Tensor, assignments: &ExpertAssignments) -> Result<Option<Self::Tensor>, BackendError> {
        let _ = (spec, layer, experts, input, assignments);
        Ok(None)
    }
}

/// MoE decode 的平台资源能力。
///
/// routing、分组、共享专家和最终合并属于 moe 通用算法；backend 只负责预取，
/// 以及根据已分组 assignments 管理权重驻留并计算 routed experts。
pub struct ExpertPrefetchRequest<'a> {
    pub layer: usize,
    pub source: ExpertSource<'a>,
    pub experts: Vec<usize>,
}

pub trait ExpertDecodeBackend: MoePrefillBackend {
    type MoeState;
    type DecodeRouting;

    /// 设备可同时索引完整 expert archive 时，将 routing 与 routed experts 留在设备端。
    #[allow(clippy::type_complexity)]
    fn decode_resident_routed_experts(
        &self,
        spec: &TopkMoeSpec,
        weights: RoutedMoeWeightsRef<'_, Self::Weight>,
        layer: usize,
        source: ExpertSource<'_>,
        state: &mut Self::MoeState,
        inputs: RoutedMoeInputs<'_, Self::Tensor>,
    ) -> Result<Option<(Self::Tensor, Option<Vec<u16>>)>, BackendError> {
        let _ = (spec, weights, layer, source, state, inputs);
        Ok(None)
    }

    fn decode_route(&self, input: &Self::Tensor, router_weight: &Self::Weight, router_bias: &Self::Weight, spec: &TopkMoeSpec) -> Result<(MoePrefillRouting, Self::DecodeRouting), BackendError>;

    fn decode_route_selected(&self, input: &Self::Tensor, router_weight: &Self::Weight, selected_experts: &[u32], spec: &TopkMoeSpec) -> Result<(MoePrefillRouting, Self::DecodeRouting), BackendError> {
        let _ = (input, router_weight, selected_experts, spec);
        Err(BackendError::Compute { msg: "backend 未实现固定专家 decode 路由".to_owned() })
    }

    fn prefetch_experts(&self, spec: &TopkMoeSpec, state: &mut Self::MoeState, request: ExpertPrefetchRequest<'_>) -> Result<usize, BackendError>;

    #[allow(clippy::too_many_arguments)]
    fn decode_routed_experts<'a, F>(
        &self,
        spec: &TopkMoeSpec,
        layer: usize,
        source: ExpertSource<'_>,
        state: &mut Self::MoeState,
        input: &Self::Tensor,
        assignments: &ExpertAssignments,
        routing: &Self::DecodeRouting,
        on_ready: F,
    ) -> Result<Self::Tensor, BackendError>
    where
        F: FnOnce(&mut Self::MoeState) -> Result<Option<ExpertPrefetchRequest<'a>>, BackendError>;
}

pub(crate) fn compute_error(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}

pub(crate) fn checked_elements(rows: usize, cols: usize, what: &str) -> Result<usize, BackendError> {
    rows.checked_mul(cols).ok_or_else(|| compute_error(format!("{what} shape [{rows},{cols}] 大小溢出")))
}

#[cfg(test)]
mod memory_pool_tests {
    use super::*;

    #[test]
    fn scratch_request_keeps_backend_independent_lifetime() {
        let request = MemoryRequest::scratch("attention", 4096).with_slot(2).with_alignment(256).validate().unwrap();
        assert_eq!(request.kind, MemoryKind::Scratch);
        assert_eq!(request.lifetime, MemoryLifetime::Operation);
        assert_eq!(request.tag, Some("attention"));
        assert_eq!(request.slot, 2);
        assert_eq!(request.alignment, 256);
    }

    #[test]
    fn request_rejects_non_power_of_two_alignment() {
        let error = MemoryRequest::new(1024, MemoryKind::Activation, MemoryLifetime::Stage).with_alignment(3).validate().unwrap_err();
        assert!(error.to_string().contains("不是 2 的幂"));
    }
}
/// 推测解码对 KV cache 的最小事务能力。
///
/// runtime 只提交 verifier 决定保留的输入行数，不感知 recent ring、压缩历史或
/// compressor pending state；同一 backend 可以为不同 cache 类型分别实现该能力。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeCacheCommit {
    retained_rows: usize,
}

impl SpeculativeCacheCommit {
    pub const fn retaining(retained_rows: usize) -> Self {
        Self { retained_rows }
    }

    pub const fn rollback() -> Self {
        Self::retaining(0)
    }

    pub const fn retained_rows(self) -> usize {
        self.retained_rows
    }
}

pub trait SpeculativeCacheBackend<C> {
    fn begin_speculative_cache(&self, cache: &mut C) -> Result<(), BackendError>;

    fn commit_speculative_cache(&self, cache: &mut C, commit: SpeculativeCacheCommit) -> Result<(), BackendError>;

    fn rollback_speculative_cache(&self, cache: &mut C) -> Result<(), BackendError> {
        self.commit_speculative_cache(cache, SpeculativeCacheCommit::rollback())
    }
}
