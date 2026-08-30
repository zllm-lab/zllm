//! Qwen3.6/3.8 DSpark drafter 与 Metal 的权重装配和单块执行。
//!
//! 复用模型无关的 `runtime::dspark` backbone 与 `runtime::speculative` 语义;
//! drafter 权重来自 llama.cpp `dflash` GGUF(Q8_0),embedding/lm_head 复用目标模型。

use crate::{
    attention::rope::RopeTable,
    backend::{
        Backend, BackendError, BackendResources, LinearWeight,
        metal::{MetalContext, MetalTensor, MetalWeight},
    },
    runtime::{
        dspark::{DsparkBackbone, DsparkLayer, DsparkTargetCache, backbone_forward},
        prepare_gguf_matrix, prepare_gguf_matrix_pair,
        speculative::SpeculativeBlock,
    },
    weight::{container::gguf::GgufReader, model::qwen36_dspark as tensors, model::qwen36_dspark::Qwen36DsparkSpec},
};

pub struct Qwen36DsparkRuntime {
    reader: GgufReader,
    pub spec: Qwen36DsparkSpec,
    backbone: DsparkBackbone<MetalWeight>,
    /// fc [hidden, 5×hidden]:5 份 capture 拼接后一次投影回 drafter hidden。
    fc: MetalWeight,
    aux_norm: MetalWeight,
    markov_projection: MetalWeight,
    pub draft_tokens: usize,
    rope: RopeTable,
}

fn compute(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}

/// 普通向量 norm(非 Gemma 偏移)的 F16 resident 装配,走 F16 gemv/kernel 直通。
fn prepare_dspark_norm_f16(backend: &MetalContext, reader: &GgufReader, name: &str) -> Result<MetalWeight, BackendError> {
    let values = reader.read_tensor_f32(name).map_err(|error| compute(format!("DSpark {name}: {error}")))?;
    let packed: Vec<half::f16> = values.iter().map(|&value| half::f16::from_f32(value)).collect();
    backend.prepare_weight(LinearWeight::F16(&packed), 1, values.len())
}

impl Qwen36DsparkRuntime {
    /// `max_seq_len` 是目标模型上下文上限;RoPE 表覆盖 block_size 冗余。
    pub fn load(ctx: &MetalContext, reader: GgufReader, max_seq_len: usize, draft_tokens: usize) -> Result<Self, BackendError> {
        let spec = Qwen36DsparkSpec::from_gguf(&reader).map_err(compute)?;
        spec.validate_tensors(&reader).map_err(compute)?;
        spec.block_spec(draft_tokens).map_err(compute)?;
        if draft_tokens == 0 || draft_tokens >= spec.block_size {
            return Err(compute(format!("DSpark draft_tokens={draft_tokens} 超出 block_size={}", spec.block_size)));
        }
        let mut layers = Vec::with_capacity(spec.layer_count);
        for layer in 0..spec.layer_count {
            let (gate, up) = prepare_gguf_matrix_pair(ctx, &reader, &tensors::layer_gate(layer), &tensors::layer_up(layer)).map_err(|error| compute(format!("DSpark blk.{layer} gate/up: {error:?}")))?;
            layers.push(DsparkLayer {
                input_norm: prepare_dspark_norm_f16(ctx, &reader, &tensors::layer_input_norm(layer))?,
                q_proj: prepare_gguf_matrix(ctx, &reader, &tensors::layer_query(layer))?,
                q_norm: prepare_dspark_norm_f16(ctx, &reader, &tensors::layer_query_norm(layer))?,
                k_proj: prepare_gguf_matrix(ctx, &reader, &tensors::layer_key(layer))?,
                k_norm: prepare_dspark_norm_f16(ctx, &reader, &tensors::layer_key_norm(layer))?,
                v_proj: prepare_gguf_matrix(ctx, &reader, &tensors::layer_value(layer))?,
                o_proj: prepare_gguf_matrix(ctx, &reader, &tensors::layer_output(layer))?,
                post_attention_norm: prepare_dspark_norm_f16(ctx, &reader, &tensors::layer_post_attention_norm(layer))?,
                gate_proj: gate,
                up_proj: up,
                down_proj: prepare_gguf_matrix(ctx, &reader, &tensors::layer_down(layer))?,
                sliding_window: None,
                sliding_window_non_causal: false,
            });
        }
        let backbone = DsparkBackbone { layers, output_norm: prepare_dspark_norm_f16(ctx, &reader, tensors::OUTPUT_NORM)?, head_count: spec.head_count, kv_head_count: spec.kv_head_count, head_dim: spec.head_dim, rms_eps: spec.rms_eps };
        let fc_matrix = reader.read_matrix(tensors::FC_WEIGHT).map_err(|error| compute(format!("DSpark fc: {error}")))?;
        if fc_matrix.rows != spec.hidden_size || fc_matrix.columns != spec.capture_boundaries.len() * spec.hidden_size {
            return Err(compute(format!("DSpark fc=[out={},in={}]，期望 [out={},in={}]", fc_matrix.rows, fc_matrix.columns, spec.hidden_size, spec.capture_boundaries.len() * spec.hidden_size)));
        }
        let fc = ctx.prepare_weight(LinearWeight::gguf(&fc_matrix), fc_matrix.rows, fc_matrix.columns)?;
        Ok(Self {
            markov_projection: prepare_gguf_matrix(ctx, &reader, tensors::MARKOV_W2)?,
            aux_norm: prepare_dspark_norm_f16(ctx, &reader, tensors::ENC_OUTPUT_NORM)?,
            backbone,
            fc,
            draft_tokens,
            rope: RopeTable::precompute(max_seq_len + spec.block_size, spec.head_dim, spec.rope_theta),
            spec,
            reader,
        })
    }

    pub fn capture_count(&self) -> usize {
        self.spec.capture_boundaries.len()
    }

    /// 全 full-attention drafter:target hidden 历史需全量保留(无滑窗可裁)。
    pub fn target_history_window(&self) -> Option<usize> {
        self.backbone.target_history_window()
    }

    /// 5 份 tap capture 拼接 → fc 投影 → RMSNorm,得到 drafter 的 target hidden。
    pub fn project_captures(&self, ctx: &MetalContext, captures: &[MetalTensor]) -> Result<MetalTensor, BackendError> {
        if captures.len() != self.capture_count() || captures.iter().any(|tensor| tensor.cols != self.spec.hidden_size) {
            return Err(compute(format!("DSpark captures={} 形状不符(期望 {} 份 [rows,{}])", captures.len(), self.capture_count(), self.spec.hidden_size)));
        }
        let mut concatenated = captures[0].clone();
        for capture in &captures[1..] {
            concatenated = ctx.concat_columns(&concatenated, capture).map_err(|error| compute(format!("DSpark capture 拼接: {error:?}")))?;
        }
        let projected = ctx.linear(&concatenated, &self.fc)?;
        ctx.rmsnorm(&projected, &self.aux_norm, self.backbone.rms_eps)
    }

    /// 只扩展 drafter 的 target KV cache(prefill 预热/verify 接受前缀推进),
    /// 不执行 backbone。`aux_hidden` 是 project_captures 的输出。
    pub fn extend_cache(&self, ctx: &MetalContext, cache: &mut DsparkTargetCache<MetalTensor>, aux_hidden: &MetalTensor, position: usize) -> Result<(), BackendError> {
        crate::runtime::dspark::warm_target_cache(ctx, &self.backbone, cache, aux_hidden, position, &self.rope.cos, &self.rope.sin)
    }

    /// 单请求 draft:整块 noise(anchor + mask)一次 backbone 前向,Markov 链逐深度
    /// 选 token;不消费 confidence(避免每深度一次投影与 GPU→CPU 同步)。
    #[allow(clippy::too_many_arguments)]
    pub fn draft_block(
        &self,
        ctx: &MetalContext,
        target: &GgufReader,
        lm_head: &MetalWeight,
        cache: &mut DsparkTargetCache<MetalTensor>,
        anchor: u32,
        target_hidden: &MetalTensor,
        target_position: usize,
        block_position: usize,
        vocab_size: usize,
    ) -> Result<SpeculativeBlock, BackendError> {
        // llama.cpp dspark 语义:noise 块共 draft_tokens 行(anchor + 其余 mask),
        // 位置从 block_position 起逐行递增;Markov 链行 i 预测第 i+1 个 draft。
        // 整条 drafter 链(~70 dispatch)延迟提交,消逐 op CPU↔GPU 同步;
        // 首个 argmax 的读回边界才真正等待。
        crate::backend::BackendResources::begin_batch(ctx);
        let debug_timing = std::env::var_os("ZLLM_QWEN36_DSPARK_DEBUG").is_some();
        let started = std::time::Instant::now();
        let block_rows = self.draft_tokens;
        let mut input_ids = vec![self.spec.mask_token_id; block_rows];
        input_ids[0] = anchor;
        let embedding = target.embedding_rows("token_embd.weight", &input_ids, self.spec.hidden_size, vocab_size).map_err(compute)?;
        let noise = ctx.tensor_from_f32(&embedding, block_rows, self.spec.hidden_size).map_err(|error| compute(format!("DSpark noise 上传: {error}")))?;
        let hidden = backbone_forward(ctx, &self.backbone, cache, noise, target_hidden, target_position, block_position, &self.rope.cos, &self.rope.sin)?;
        let base_logits = ctx.linear(&hidden, lm_head)?;
        let backbone_wall = started.elapsed();
        let markov_started = std::time::Instant::now();
        let mut previous = anchor;
        let mut drafts = Vec::with_capacity(self.draft_tokens);
        let mut first_argmax_wall = std::time::Duration::ZERO;
        for depth in 0..self.draft_tokens {
            let markov_row = self.reader.read_matrix_row_f32(tensors::MARKOV_W1, previous as usize).map_err(compute)?;
            let embedding = ctx.tensor_from_f32(&markov_row, 1, self.spec.markov_rank()).map_err(|error| compute(format!("DSpark markov 上传: {error}")))?;
            let bias = ctx.linear(&embedding, &self.markov_projection)?;
            let logits = ctx.add(&ctx.select_row(&base_logits, depth)?, &bias)?;
            let argmax_started = std::time::Instant::now();
            let token = ctx.argmax(&logits)?;
            if depth == 0 {
                first_argmax_wall = argmax_started.elapsed();
            }
            drafts.push(token);
            previous = token;
        }
        ctx.submit_batch();
        if debug_timing {
            eprintln!("[dspark-draft-time] backbone={:.3}s(含 lm_head 入队) markov={:.3}s(首 argmax={:.3}s)({} 深度)", backbone_wall.as_secs_f32(), markov_started.elapsed().as_secs_f32(), first_argmax_wall.as_secs_f32(), self.draft_tokens);
        }
        SpeculativeBlock::new(anchor, drafts, Vec::new())
    }
}
