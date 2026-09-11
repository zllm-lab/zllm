//! MiniCPM5-2B-DSpark drafter：直接加载 HuggingFace safetensors（布局按张量名
//! 映射，不做格式转换），复用模型无关的 `runtime::dspark` backbone 与
//! `runtime::speculative` 语义；embedding/lm_head 复用目标模型。
//!
//! 架构（与论文/QNN/Python oracle 一致）：
//! - fc [hidden, 5×hidden] 把 5 份 target 层 hidden 拼接投影回 drafter hidden，
//!   再 RMSNorm（hidden_norm）；
//! - 5 层 Qwen3 风格 drafter（q/k norm、GQA 16:2），块内对 target 上下文与
//!   块内均为双向注意力（用大窗口 non-causal 表达）；
//! - lm_head 共享目标模型；Markov 头逐深度修正 logits（w1 词表行 + w2 投影）。

use crate::{
    attention::rope::RopeTable,
    backend::{
        Backend, BackendError, BackendResources, LinearWeight,
        metal::{MetalContext, MetalTensor, MetalWeight},
    },
    runtime::{
        dspark::{DsparkBackbone, DsparkLayer, DsparkTargetCache, backbone_forward},
        speculative::SpeculativeBlock,
    },
};
use std::path::Path;

fn compute(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}

pub struct Minicpm5DsparkSpec {
    pub hidden_size: usize,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub layer_count: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub markov_rank: usize,
    pub vocab_size: usize,
    /// 5 份 target hidden 采集层（MiniCPM5-2B-DSpark: [1, 10, 20, 30, 39]）。
    pub target_layers: Vec<usize>,
}

pub struct Minicpm5DsparkRuntime {
    pub spec: Minicpm5DsparkSpec,
    backbone: DsparkBackbone<MetalWeight>,
    fc: MetalWeight,
    aux_norm: MetalWeight,
    markov_w2: MetalWeight,
    /// [vocab, rank] 行主序常驻 host，逐深度按 previous token 抠行。
    markov_w1: Vec<f32>,
    pub draft_tokens: usize,
    rope: RopeTable,
}

/// safetensors 张量按名读取为 f32（BF16/F32/F16 布局都接受）。
fn tensor_f32(file: &safetensors::SafeTensors, name: &str) -> Result<Vec<f32>, BackendError> {
    let view = file.tensor(name).map_err(|error| compute(format!("DSpark 缺少 {name}: {error}")))?;
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::BF16 => Ok(bytes.chunks_exact(2).map(|pair| half::bf16::from_le_bytes([pair[0], pair[1]]).to_f32()).collect()),
        safetensors::Dtype::F16 => Ok(bytes.chunks_exact(2).map(|pair| half::f16::from_le_bytes([pair[0], pair[1]]).to_f32()).collect()),
        safetensors::Dtype::F32 => Ok(bytes.chunks_exact(4).map(|pair| f32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]])).collect()),
        other => Err(compute(format!("DSpark {name} dtype {other:?} 暂不支持直接加载"))),
    }
}

fn shape(file: &safetensors::SafeTensors, name: &str) -> Result<Vec<usize>, BackendError> {
    Ok(file.tensor(name).map_err(|error| compute(format!("DSpark 读取 {name}: {error}")))?.shape().to_vec())
}

/// norm 向量按 F16 resident 装配（与 qwen36 drafter 同路径）。
fn prepare_norm_f16(ctx: &MetalContext, values: &[f32]) -> Result<MetalWeight, BackendError> {
    let packed: Vec<half::f16> = values.iter().map(|&value| half::f16::from_f32(value)).collect();
    ctx.prepare_weight(LinearWeight::F16(&packed), 1, values.len())
}

/// 线性矩阵 [out, in] 以 F32 resident 装配（drafter 体量小，精度优先）。
fn prepare_matrix_f32(ctx: &MetalContext, values: &[f32], rows: usize, columns: usize) -> Result<MetalWeight, BackendError> {
    if values.len() != rows * columns {
        return Err(compute(format!("DSpark 矩阵元素 {} != {rows}×{columns}", values.len())));
    }
    ctx.prepare_weight(LinearWeight::F32(values), rows, columns)
}

fn check_matrix(shape: &[usize], rows: usize, columns: usize, name: &str) -> Result<(), BackendError> {
    if shape != &[rows, columns] {
        return Err(compute(format!("DSpark {name} shape {shape:?}，期望 [{rows},{columns}]")));
    }
    Ok(())
}

impl Minicpm5DsparkRuntime {
    pub fn load(ctx: &MetalContext, directory: &Path, max_seq_len: usize) -> Result<Self, BackendError> {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("config.json")).map_err(|error| compute(format!("DSpark config.json: {error}")))?,
        )
        .map_err(|error| compute(format!("DSpark config.json 解析: {error}")))?;
        let num = |key: &str| config.get(key).and_then(|value| value.as_u64()).map(|value| value as usize);
        let spec = Minicpm5DsparkSpec {
            hidden_size: num("hidden_size").ok_or_else(|| compute("DSpark config 缺 hidden_size"))?,
            head_count: num("num_attention_heads").ok_or_else(|| compute("DSpark config 缺 num_attention_heads"))?,
            kv_head_count: num("num_key_value_heads").ok_or_else(|| compute("DSpark config 缺 num_key_value_heads"))?,
            head_dim: num("head_dim").ok_or_else(|| compute("DSpark config 缺 head_dim"))?,
            layer_count: num("num_hidden_layers").ok_or_else(|| compute("DSpark config 缺 num_hidden_layers"))?,
            rms_eps: config.get("rms_norm_eps").and_then(|value| value.as_f64()).unwrap_or(1e-6) as f32,
            rope_theta: config.pointer("/rope_parameters/rope_theta").and_then(|value| value.as_f64()).unwrap_or(5_000_000.0) as f32,
            block_size: num("block_size").ok_or_else(|| compute("DSpark config 缺 block_size"))?,
            mask_token_id: num("mask_token_id").ok_or_else(|| compute("DSpark config 缺 mask_token_id"))? as u32,
            markov_rank: num("markov_rank").ok_or_else(|| compute("DSpark config 缺 markov_rank"))?,
            vocab_size: num("draft_vocab_size").or_else(|| num("vocab_size")).ok_or_else(|| compute("DSpark config 缺 vocab_size"))?,
            target_layers: config.get("target_layer_ids").and_then(|value| value.as_array()).map(|items| items.iter().filter_map(|item| item.as_u64().map(|id| id as usize)).collect()).ok_or_else(|| compute("DSpark config 缺 target_layer_ids"))?,
        };
        let bytes = std::fs::read(directory.join("model.safetensors")).map_err(|error| compute(format!("DSpark safetensors: {error}")))?;
        let file = safetensors::SafeTensors::deserialize(&bytes).map_err(|error| compute(format!("DSpark safetensors 解析: {error}")))?;

        let mut layers = Vec::with_capacity(spec.layer_count);
        for layer in 0..spec.layer_count {
            let p = format!("layers.{layer}.");
            let expect = |name: &str, rows: usize, columns: usize| check_matrix(&shape(&file, name)?, rows, columns, name);
            expect(&(p.clone() + "self_attn.q_proj.weight"), spec.hidden_size, spec.hidden_size)?;
            expect(&(p.clone() + "self_attn.k_proj.weight"), spec.kv_head_count * spec.head_dim, spec.hidden_size)?;
            expect(&(p.clone() + "self_attn.v_proj.weight"), spec.kv_head_count * spec.head_dim, spec.hidden_size)?;
            expect(&(p.clone() + "self_attn.o_proj.weight"), spec.hidden_size, spec.hidden_size)?;
            expect(&(p.clone() + "mlp.gate_proj.weight"), num("intermediate_size").unwrap_or(6144), spec.hidden_size)?;
            let matrix = |name: &str| prepare_matrix_f32(ctx, &tensor_f32(&file, name)?, shape(&file, name)?[0], shape(&file, name)?[1]);
            layers.push(DsparkLayer {
                input_norm: prepare_norm_f16(ctx, &tensor_f32(&file, &(p.clone() + "input_layernorm.weight"))?)?,
                q_proj: matrix(&(p.clone() + "self_attn.q_proj.weight"))?,
                q_norm: prepare_norm_f16(ctx, &tensor_f32(&file, &(p.clone() + "self_attn.q_norm.weight"))?)?,
                k_proj: matrix(&(p.clone() + "self_attn.k_proj.weight"))?,
                k_norm: prepare_norm_f16(ctx, &tensor_f32(&file, &(p.clone() + "self_attn.k_norm.weight"))?)?,
                v_proj: matrix(&(p.clone() + "self_attn.v_proj.weight"))?,
                o_proj: matrix(&(p.clone() + "self_attn.o_proj.weight"))?,
                post_attention_norm: prepare_norm_f16(ctx, &tensor_f32(&file, &(p.clone() + "post_attention_layernorm.weight"))?)?,
                gate_proj: matrix(&(p.clone() + "mlp.gate_proj.weight"))?,
                up_proj: matrix(&(p.clone() + "mlp.up_proj.weight"))?,
                down_proj: matrix(&(p.clone() + "mlp.down_proj.weight"))?,
                // DSpark 块内双向注意力：大窗口 non-causal 可见性等价全量双向。
                sliding_window: Some(1 << 30),
                sliding_window_non_causal: true,
            });
        }
        let capture_count = spec.target_layers.len();
        check_matrix(&shape(&file, "fc.weight")?, spec.hidden_size, capture_count * spec.hidden_size, "fc.weight")?;
        let fc = prepare_matrix_f32(ctx, &tensor_f32(&file, "fc.weight")?, spec.hidden_size, capture_count * spec.hidden_size)?;
        check_matrix(&shape(&file, "markov_head.markov_w2.weight")?, spec.vocab_size, spec.markov_rank, "markov_w2")?;
        let markov_w2 = prepare_matrix_f32(ctx, &tensor_f32(&file, "markov_head.markov_w2.weight")?, spec.vocab_size, spec.markov_rank)?;
        check_matrix(&shape(&file, "markov_head.markov_w1.weight")?, spec.vocab_size, spec.markov_rank, "markov_w1")?;
        let markov_w1 = tensor_f32(&file, "markov_head.markov_w1.weight")?;
        let backbone = DsparkBackbone {
            output_norm: prepare_norm_f16(ctx, &tensor_f32(&file, "norm.weight")?)?,
            layers,
            head_count: spec.head_count,
            kv_head_count: spec.kv_head_count,
            head_dim: spec.head_dim,
            rms_eps: spec.rms_eps,
        };
        let draft_tokens = spec.block_size;
        Ok(Self {
            rope: RopeTable::precompute(max_seq_len + spec.block_size, spec.head_dim, spec.rope_theta),
            aux_norm: prepare_norm_f16(ctx, &tensor_f32(&file, "hidden_norm.weight")?)?,
            spec,
            backbone,
            fc,
            markov_w2,
            markov_w1,
            draft_tokens,
        })
    }

    pub fn capture_layers(&self) -> &[usize] {
        &self.spec.target_layers
    }

    /// 5 份 target hidden capture 拼接 → fc 投影 → RMSNorm，得到 drafter 的 aux hidden。
    pub fn project_captures(&self, ctx: &MetalContext, captures: &[MetalTensor]) -> Result<MetalTensor, BackendError> {
        if captures.len() != self.spec.target_layers.len() || captures.iter().any(|tensor| tensor.cols != self.spec.hidden_size) {
            return Err(compute(format!("DSpark captures={} 形状不符(期望 {} 份 [rows,{}])", captures.len(), self.spec.target_layers.len(), self.spec.hidden_size)));
        }
        let mut concatenated = captures[0].clone();
        for capture in &captures[1..] {
            concatenated = ctx.concat_columns(&concatenated, capture).map_err(|error| compute(format!("DSpark capture 拼接: {error:?}")))?;
        }
        let projected = ctx.linear(&concatenated, &self.fc)?;
        ctx.rmsnorm(&projected, &self.aux_norm, self.backbone.rms_eps)
    }

    /// 只扩展 drafter 的 target KV cache（prefill 预热/verify 接受前缀推进）。
    pub fn extend_cache(&self, ctx: &MetalContext, cache: &mut DsparkTargetCache<MetalTensor>, aux_hidden: &MetalTensor, position: usize) -> Result<(), BackendError> {
        crate::runtime::dspark::warm_target_cache(ctx, &self.backbone, cache, aux_hidden, position, &self.rope.cos, &self.rope.sin)
    }

    /// 单请求 draft：整块 noise（anchor + mask）一次 backbone 前向，Markov 链逐深度
    /// 选 token；confidence 头不参与贪心提案（与 qwen36 drafter 同语义）。
    #[allow(clippy::too_many_arguments)]
    pub fn draft_block(
        &self,
        ctx: &MetalContext,
        lm_head: &MetalWeight,
        cache: &mut DsparkTargetCache<MetalTensor>,
        anchor: u32,
        noise: MetalTensor,
        // 最近一次预热的 aux hidden 行（1 行）；backbone 的覆盖检测发现 cache
        // 已覆盖该行时会跳过重投影，等效零行 target_hidden。
        warm_aux_row: &MetalTensor,
        warm_position: usize,
        block_position: usize,
    ) -> Result<SpeculativeBlock, BackendError> {
        let block_rows = self.draft_tokens;
        if ctx.token_rows(&noise) != block_rows || ctx.token_cols(&noise) != self.spec.hidden_size {
            return Err(compute(format!("DSpark noise [{},{}]，期望 [{block_rows},{}]", ctx.token_rows(&noise), ctx.token_cols(&noise), self.spec.hidden_size)));
        }
        if ctx.token_rows(warm_aux_row) != 1 || ctx.token_cols(warm_aux_row) != self.spec.hidden_size {
            return Err(compute(format!("DSpark warm_aux_row [{},{}]，期望 [1,{}]", ctx.token_rows(warm_aux_row), ctx.token_cols(warm_aux_row), self.spec.hidden_size)));
        }
        let hidden = backbone_forward(ctx, &self.backbone, cache, noise, warm_aux_row, warm_position, block_position, &self.rope.cos, &self.rope.sin)?;
        let base_logits = ctx.linear(&hidden, lm_head)?;
        let mut previous = anchor;
        let mut drafts = Vec::with_capacity(block_rows);
        for depth in 0..block_rows {
            let start = previous as usize * self.spec.markov_rank;
            let row = &self.markov_w1[start..start + self.spec.markov_rank];
            let embedding = ctx.tensor_from_f32(row, 1, self.spec.markov_rank).map_err(|error| compute(format!("DSpark markov 上传: {error:?}")))?;
            let bias = ctx.linear(&embedding, &self.markov_w2)?;
            let logits = ctx.add(&ctx.select_row(&base_logits, depth)?, &bias)?;
            let token = ctx.argmax(&logits)?;
            drafts.push(token);
            previous = token;
        }
        SpeculativeBlock::new(anchor, drafts, Vec::new())
    }
}
