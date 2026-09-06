// Qwen3.6-27B 模型规格与执行组合。
pub mod cpu;
#[cfg(feature = "with-cuda")]
pub mod cuda;
#[cfg(feature = "with-cuda")]
pub mod cuda_hybrid;
#[cfg(feature = "with-cuda")]
pub mod cuda_node;
pub mod engine;
#[cfg(target_os = "macos")]
mod metal_session;
pub mod protocol;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_hybrid;
//
// 当前权重目录名为 Qwen3.6，配置里的架构标识仍是
// `Qwen3_5ForConditionalGeneration`。这里以真实权重和官方执行语义为准，
// 不把发布命名差异泄漏到 backend。

use std::ops::Range;

use crate::{
    attention::{
        AttentionSpec,
        gated_delta_net::{GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetState},
        gqa::{CausalWindow, GqaSpec},
        hybrid::{DeltaNetWeights, FullAttentionWeights, HybridAttention, HybridAttentionOptions, HybridTokenMixer},
        rope::RopeTable,
    },
    backend::{Backend, BackendError, LinearWeight},
    moe::{
        FeedforwardSpec,
        dense_mlp::{Activation, DenseMlpSpec},
    },
    norm::NormSpec,
    runtime::{LayerId, LayerSpec, Model, ModelError},
    tokenizer::Tokenizer,
    vision::ImageTensor,
    weight::container::gguf::GgufReader,
    weight::model::qwen36::Qwen36Weights,
};

pub use crate::model_spec::qwen36::{Qwen36AttentionKind, Qwen36Config, Qwen36VisionConfig};

/// 大批量 prefill 的临时 buffer 会随已提交 command buffer 保活；24GB UMA
/// 上限制为两层，避免文本或视觉塔把整模型的 activation 同时留在内存。
const MAX_IN_FLIGHT_PREFILL_LAYERS: usize = 2;

impl Qwen36VisionConfig {
    /// 转换为 Qwen3-VL vision config 以复用其 ViT 执行代码(架构完全相同)。
    /// Qwen3.6-27B 不使用 DeepStack,deepstack_visual_indexes 为空。
    pub fn to_qwen3vl(&self) -> super::qwen3_vl::Qwen3VlVisionConfig {
        super::qwen3_vl::Qwen3VlVisionConfig {
            depth: self.depth,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_heads,
            position_embeddings: self.position_embeddings,
            patch_size: self.patch_size,
            temporal_patch_size: self.temporal_patch_size,
            spatial_merge_size: self.spatial_merge_size,
            output_hidden_size: self.output_hidden_size,
            rope_theta: self.rope_theta,
            deepstack_visual_indexes: Vec::new(),
            min_pixels: self.min_pixels,
            max_pixels: self.max_pixels,
            max_aspect_ratio: self.max_aspect_ratio,
            image_mean: self.image_mean,
            image_std: self.image_std,
        }
    }
}

/// Qwen3.6 chat 模板。统一各入口的 prompt 包装语义:cpu/cuda/rocm/metal
/// 都通过它构造带角色的对话前缀,避免各入口模板漂移。
pub fn chat_prompt(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

// ============================================================================
// 多模态输入构造：多图 + 视频。渲染 placeholder、定位 token 区间并生成
// M-RoPE 三轴位置；视觉段位置按 HF 语义压缩(推进 max(t,h,w))。
// ============================================================================

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen36VisualKind {
    Image,
    Video,
}

/// 一个待注入的视觉输入(图像或视频)及其预处理结果。
pub struct Qwen36Visual {
    pub kind: Qwen36VisualKind,
    pub tensor: ImageTensor,
}

pub struct Qwen36MultimodalInput {
    pub token_ids: Vec<u32>,
    /// 与 visuals 顺序一一对应的 placeholder token 区间。
    pub visual_ranges: Vec<Range<usize>>,
    pub position_ids: [Vec<usize>; 3],
    pub rope_delta: i64,
}

/// 渲染多视觉 chat prompt 并 tokenize、定位 placeholder、生成三轴位置。
/// 视觉排在用户文本之前;图像/视频的先后顺序由调用方在 `visuals` 里决定。
pub fn qwen36_multimodal_input(tokenizer: &Tokenizer, config: &Qwen36Config, prompt: &str, visuals: &[Qwen36Visual]) -> Result<Qwen36MultimodalInput, String> {
    use super::qwen3_vl::{IMAGE_TOKEN, VIDEO_TOKEN, VISION_END_TOKEN, VISION_START_TOKEN};
    if prompt.contains(IMAGE_TOKEN) || prompt.contains(VIDEO_TOKEN) || prompt.contains(VISION_START_TOKEN) || prompt.contains(VISION_END_TOKEN) {
        return Err("用户文本包含 Qwen3.6 保留视觉 token".to_owned());
    }
    let mut rendered = String::from("<|im_start|>user\n");
    for (block, _, _) in qwen36_visual_blocks(config, visuals)? {
        rendered.push_str(&block);
    }
    rendered.push_str(prompt);
    rendered.push_str("<|im_end|>\n<|im_start|>assistant\n");
    qwen36_multimodal_input_rendered(tokenizer, config, &rendered, visuals)
}

/// 校验每个视觉输入的 grid/merge 并渲染 placeholder 块。
/// 返回 (完整块文本, placeholder 行数, 对应 token id)，顺序与 `visuals` 一致。
pub fn qwen36_visual_blocks(config: &Qwen36Config, visuals: &[Qwen36Visual]) -> Result<Vec<(String, usize, u32)>, String> {
    use super::qwen3_vl::{IMAGE_TOKEN, VIDEO_TOKEN, VISION_END_TOKEN, VISION_START_TOKEN};
    visuals
        .iter()
        .enumerate()
        .map(|(index, visual)| {
            let (grid, merge) = (visual.tensor.grid, visual.tensor.merge_size);
            if merge == 0 || !grid.height.is_multiple_of(merge) || !grid.width.is_multiple_of(merge) {
                return Err(format!("Qwen3.6 visual {index} grid={grid:?} merge={merge} 非法"));
            }
            let rows = grid.temporal.checked_mul(grid.height / merge).and_then(|value| value.checked_mul(grid.width / merge)).ok_or("Qwen3.6 visual token 数溢出")?;
            if rows == 0 {
                return Err(format!("Qwen3.6 visual {index} token 数为 0"));
            }
            let (token, token_id) = match visual.kind {
                Qwen36VisualKind::Image => (IMAGE_TOKEN, config.image_token_id),
                Qwen36VisualKind::Video => (VIDEO_TOKEN, config.video_token_id),
            };
            Ok((format!("{VISION_START_TOKEN}{}{VISION_END_TOKEN}", token.repeat(rows)), rows, token_id))
        })
        .collect()
}

/// 已内联渲染好视觉 placeholder 的完整 prompt 的多模态输入构造：
/// tokenize → 按出现顺序扫描定位每个视觉的 token 区间 → M-RoPE 三轴位置。
/// 多轮模板等自定义渲染顺序的调用方(node)使用；扫描会校验 placeholder
/// 数量与 token 类型，用户文本混入保留 token 时报错。
pub fn qwen36_multimodal_input_rendered(tokenizer: &Tokenizer, config: &Qwen36Config, rendered: &str, visuals: &[Qwen36Visual]) -> Result<Qwen36MultimodalInput, String> {
    let expected = qwen36_visual_blocks(config, visuals)?.into_iter().map(|(_, rows, token_id)| (rows, token_id)).collect::<Vec<_>>();
    let token_ids = tokenizer.tokenize(rendered.as_bytes());
    let mut visual_ranges = Vec::with_capacity(visuals.len());
    let mut cursor = 0usize;
    for (index, &(rows, token_id)) in expected.iter().enumerate() {
        let vision_start = token_ids[cursor..].iter().position(|&token| token == config.vision_start_token_id).map(|position| cursor + position).ok_or_else(|| format!("Qwen3.6 visual {index} 缺少 vision_start"))?;
        let start = vision_start + 1;
        let end = start.checked_add(rows).ok_or("Qwen3.6 visual range 溢出")?;
        if token_ids.get(start..end).is_none_or(|tokens| tokens.iter().any(|&token| token != token_id)) || token_ids.get(end) != Some(&config.vision_end_token_id) {
            return Err(format!("Qwen3.6 visual {index} placeholder 数量或类型不匹配"));
        }
        visual_ranges.push(start..end);
        cursor = end + 1;
    }
    let descriptors = visual_ranges.iter().cloned().zip(visuals.iter().map(|visual| (visual.tensor.grid, visual.tensor.merge_size))).collect::<Vec<_>>();
    let (position_ids, rope_delta) = super::qwen3_vl::multimodal_positions_many(token_ids.len(), &descriptors)?;
    Ok(Qwen36MultimodalInput { token_ids, visual_ranges, position_ids, rope_delta })
}

impl Qwen36Config {
    /// Qwen3.5-4B 文本主干。GGUF 的 block_count=33 包含最后一个 MTP block，
    /// 正常 prefill/decode 只执行前 32 层。
    pub fn standard_4b() -> Self {
        Self {
            vocab_size: 248_320,
            hidden_size: 2_560,
            intermediate_size: 9_216,
            num_layers: 32,
            num_attention_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            rope_dim: 64,
            rope_theta: 10_000_000.0,
            mrope_section: [11, 11, 10],
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262_144,
            full_attention_interval: 4,
            linear_key_heads: 16,
            linear_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_size: 4,
            mtp_layers: 1,
            bos_token_id: 248_044,
            eos_token_ids: vec![248_046, 248_044],
            image_token_id: 248_056,
            video_token_id: 248_057,
            vision_start_token_id: 248_053,
            vision_end_token_id: 248_054,
            vision: Qwen36VisionConfig {
                depth: 27,
                hidden_size: 1_152,
                intermediate_size: 4_304,
                num_heads: 16,
                position_embeddings: 2_304,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                output_hidden_size: 2_560,
                rope_theta: 10_000.0,
                min_pixels: 65_536,
                max_pixels: 16_777_216,
                max_aspect_ratio: 200.0,
                image_mean: [0.5; 3],
                image_std: [0.5; 3],
            },
        }
    }

    pub fn standard_27b() -> Self {
        Self {
            vocab_size: 248_320,
            hidden_size: 5_120,
            intermediate_size: 17_408,
            num_layers: 64,
            num_attention_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            rope_dim: 64,
            rope_theta: 10_000_000.0,
            mrope_section: [11, 11, 10],
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262_144,
            full_attention_interval: 4,
            linear_key_heads: 16,
            linear_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_size: 4,
            mtp_layers: 1,
            bos_token_id: 248_044,
            eos_token_ids: vec![248_046, 248_044],
            image_token_id: 248_056,
            video_token_id: 248_057,
            vision_start_token_id: 248_053,
            vision_end_token_id: 248_054,
            vision: Qwen36VisionConfig {
                depth: 27,
                hidden_size: 1_152,
                intermediate_size: 4_304,
                num_heads: 16,
                position_embeddings: 2_304,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                output_hidden_size: 5_120,
                rope_theta: 10_000.0,
                min_pixels: 65_536,
                max_pixels: 16_777_216,
                max_aspect_ratio: 200.0,
                image_mean: [0.5; 3],
                image_std: [0.5; 3],
            },
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.num_layers == 0 || self.full_attention_interval == 0 {
            return Err("Qwen3.6 层数和全注意力间隔必须大于 0".into());
        }
        if self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err("Qwen3.6 hidden/intermediate size 必须大于 0".into());
        }
        if self.num_attention_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err("Qwen3.5 attention heads/head_dim 必须大于 0".into());
        }
        if self.rope_dim > self.head_dim || self.mrope_section.iter().sum::<usize>() * 2 != self.rope_dim {
            return Err(format!("Qwen3.6 M-RoPE section {:?} 与 rope_dim {} 不一致", self.mrope_section, self.rope_dim));
        }
        let linear_qkv_size = self.linear_key_heads * self.linear_key_head_dim * 2 + self.linear_value_heads * self.linear_value_head_dim;
        if linear_qkv_size == 0 {
            return Err("Qwen3.5 DeltaNet QKV 投影宽度必须大于 0".into());
        }
        if self.vision.output_hidden_size != self.hidden_size {
            return Err("Qwen3.6 vision merger 输出必须等于文本 hidden size".into());
        }
        Ok(())
    }

    pub fn full_attention_spec(&self) -> GqaSpec {
        GqaSpec {
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            rope_theta: self.rope_theta,
            use_qk_norm: true,
            window: CausalWindow::Full,
            score_scale: 1.0 / (self.head_dim as f32).sqrt(),
            output_gate: true,
        }
    }

    pub fn gated_delta_net_spec(&self) -> GatedDeltaNetSpec {
        GatedDeltaNetSpec {
            key_heads: self.linear_key_heads,
            value_heads: self.linear_value_heads,
            key_head_dim: self.linear_key_head_dim,
            value_head_dim: self.linear_value_head_dim,
            conv_kernel: self.linear_conv_kernel_size,
            rms_eps: self.rms_norm_eps,
            output_gate: crate::attention::gated_delta_net::GdnOutputGate::Silu,
        }
    }
}

#[derive(Debug)]
pub struct Qwen36 {
    pub config: Qwen36Config,
    layer_specs: Vec<LayerSpec>,
}

impl Qwen36 {
    pub fn new(config: Qwen36Config) -> Result<Self, String> {
        config.validate()?;
        let layer_specs = (0..config.num_layers).map(|layer| Self::build_layer_spec(&config, layer)).collect();
        Ok(Self { config, layer_specs })
    }

    pub fn standard_27b() -> Self {
        Self::new(Qwen36Config::standard_27b()).expect("内置 Qwen3.6-27B 配置必须有效")
    }

    pub fn attention_kind(&self, layer: usize) -> Result<Qwen36AttentionKind, String> {
        if layer >= self.config.num_layers {
            return Err(format!("Qwen3.6 layer {} 越界，总层数 {}", layer, self.config.num_layers));
        }
        if (layer + 1).is_multiple_of(self.config.full_attention_interval) { Ok(Qwen36AttentionKind::FullAttention) } else { Ok(Qwen36AttentionKind::GatedDeltaNet) }
    }

    pub fn attention_spec(&self, layer: usize) -> Result<&AttentionSpec, String> {
        self.layer_specs.get(layer).map(|spec| &spec.attention).ok_or_else(|| format!("Qwen3.6 layer {layer} 越界，总层数 {}", self.config.num_layers))
    }

    fn build_layer_spec(config: &Qwen36Config, layer: usize) -> LayerSpec {
        let attention = if (layer + 1).is_multiple_of(config.full_attention_interval) { AttentionSpec::Gqa(config.full_attention_spec()) } else { AttentionSpec::GatedDeltaNet(config.gated_delta_net_spec()) };
        LayerSpec {
            attention,
            feedforward: FeedforwardSpec::Dense(DenseMlpSpec { intermediate_size: config.intermediate_size, activation: Activation::Silu }),
            input_norm: NormSpec::GemmaRms { eps: config.rms_norm_eps },
            post_attention_norm: NormSpec::GemmaRms { eps: config.rms_norm_eps },
            post_norm: None,
        }
    }
}

impl Default for Qwen36 {
    fn default() -> Self {
        Self::standard_27b()
    }
}

impl Model for Qwen36 {
    type Config = Qwen36Config;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn layer_count(&self) -> usize {
        self.config.num_layers
    }

    fn layer_spec(&self, layer: LayerId) -> Result<&LayerSpec, ModelError> {
        self.layer_specs.get(layer).ok_or(ModelError::LayerOutOfRange { layer, layer_count: self.config.num_layers })
    }
}

// ============================================================================
// 共享 attention 底座 —— Qwen3.6 与 Ornith 都基于这套混合注意力编排。
//
// GatedDeltaNet(线性注意力)与 GQA(全注意力)按固定周期交替。两种 mixer 共享
// backend 能力 trait,只有 FFN 由各自模型决定(Dense MLP / MoE)。
// ============================================================================

// ============================================================================
// Qwen3.6 Dense MLP 执行流程
// ============================================================================

/// Qwen3.6 已 prepare 到 backend 的 Dense MLP 权重。
#[derive(Debug)]
pub struct Qwen36RuntimeMlp<W> {
    pub gate: W,
    pub up: W,
    pub down: W,
}

/// Qwen3.6 已 prepare 到 backend 的一层权重。
#[derive(Debug)]
pub struct Qwen36RuntimeLayer<W> {
    pub input_norm: W,
    pub token_mixer: HybridTokenMixer<W>,
    pub post_attention_norm: W,
    pub mlp: Qwen36RuntimeMlp<W>,
}

/// Qwen3.6 MTP 层权重。
#[derive(Debug)]
pub struct Qwen36RuntimeMtp<W> {
    pub embedding_norm: W,
    pub hidden_norm: W,
    pub input_projection: W,
    pub layer: Qwen36RuntimeLayer<W>,
    pub output_norm: W,
}

pub type Qwen36OutputHead<W> = super::output::OutputHead<W>;

pub struct Qwen36Runtime<'a, B: Backend> {
    backend: &'a B,
    config: &'a Qwen36Config,
    layers: &'a [Qwen36RuntimeLayer<B::Weight>],
    attention: HybridAttention<'a, B>,
    mlp: DenseMlpSpec,
}

impl<'a, B: Backend> Qwen36Runtime<'a, B> {
    pub fn new(backend: &'a B, config: &'a Qwen36Config, layers: &'a [Qwen36RuntimeLayer<B::Weight>], rope: &'a RopeTable, attention_options: HybridAttentionOptions) -> Self {
        Self {
            backend,
            config,
            layers,
            attention: HybridAttention::new(backend, config.full_attention_spec(), config.gated_delta_net_spec(), config.rms_norm_eps, rope, attention_options),
            mlp: DenseMlpSpec { intermediate_size: config.intermediate_size, activation: Activation::Silu },
        }
    }

    pub fn at<'run>(&'run self, cache: &'run mut B::Cache, recurrent: &'run mut GatedDeltaNetState<B::GatedDeltaNetStorage>, position: usize) -> Qwen36Execution<'run, 'a, B>
    where
        B: crate::backend::GqaPrefillBackend + GatedDeltaNetKernel,
    {
        Qwen36Execution { runtime: self, cache, recurrent, position, capture: None, gdn_inputs: None }
    }

    /// 执行调用方按需准备的一层，供显存不足以常驻整模型的平台流式复用模型算法。
    pub fn prepared_layer(
        &self,
        cache: &mut B::Cache,
        recurrent: &mut GatedDeltaNetState<B::GatedDeltaNetStorage>,
        position: usize,
        layer: usize,
        weights: &Qwen36RuntimeLayer<B::Weight>,
        hidden: B::Tensor,
    ) -> Result<B::Tensor, BackendError>
    where
        B: crate::backend::GqaPrefillBackend + GatedDeltaNetKernel,
        B::Tensor: Clone,
    {
        if layer >= self.config.num_layers || self.backend.token_rows(&hidden) == 0 || self.backend.token_cols(&hidden) != self.config.hidden_size {
            return Err(BackendError::Compute { msg: format!("Qwen3.6 streamed L{layer} 输入 shape 异常: [{},{}]", self.backend.token_rows(&hidden), self.backend.token_cols(&hidden)) });
        }
        self.backend.begin_batch();
        let output = Qwen36Execution { runtime: self, cache, recurrent, position, capture: None, gdn_inputs: None }.run_layer(weights, layer, &hidden);
        self.backend.finish_batch();
        output
    }

    /// MTP 层固定使用 FullAttention，不创建或传递无用的 Gated DeltaNet state。
    fn mtp_decode_open(&self, weights: &Qwen36RuntimeMtp<B::Weight>, cache: &mut B::Cache, token_embedding: &B::Tensor, target_hidden: &B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: crate::backend::GqaPrefillBackend,
    {
        let _scope = self.backend.layer_scope();
        let fused = crate::runtime::mtp_project(
            self.backend,
            token_embedding,
            target_hidden,
            &weights.embedding_norm,
            &weights.hidden_norm,
            &weights.input_projection,
            self.config.hidden_size,
            NormSpec::GemmaRms { eps: self.config.rms_norm_eps },
            true,
        )?;
        let normed = self.backend.gemma_rmsnorm_f32(&fused, &weights.layer.input_norm, self.config.rms_norm_eps)?;
        let HybridTokenMixer::FullAttention(attention) = &weights.layer.token_mixer else {
            return Err(BackendError::Compute { msg: "Qwen3.6 MTP 必须使用 FullAttention".to_owned() });
        };
        let mixed = self.attention.full(attention, cache, self.config.num_layers, &normed, position)?;
        let residual = self.backend.add(&fused, &mixed)?;
        let ffn_input = self.backend.gemma_rmsnorm_f32(&residual, &weights.layer.post_attention_norm, self.config.rms_norm_eps)?;
        let hidden = self.backend.gated_mlp_add_residual(&ffn_input, &weights.layer.mlp.gate, &weights.layer.mlp.up, &weights.layer.mlp.down, &self.mlp.activation, &residual)?;
        self.backend.gemma_rmsnorm_f32(&hidden, &weights.output_norm, self.config.rms_norm_eps)
    }

    pub fn mtp_decode(&self, weights: &Qwen36RuntimeMtp<B::Weight>, cache: &mut B::Cache, token_embedding: &B::Tensor, target_hidden: &B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: crate::backend::GqaPrefillBackend,
    {
        self.backend.begin_decode_batch();
        let result = self.mtp_decode_open(weights, cache, token_embedding, target_hidden, position);
        self.backend.finish_batch();
        result
    }

    /// 把只消费 MTP hidden 的后处理留在同一个 decode batch，避免先等 MTP 层、再提交 lm_head。
    pub fn mtp_decode_then<R, F>(&self, weights: &Qwen36RuntimeMtp<B::Weight>, cache: &mut B::Cache, token_embedding: &B::Tensor, target_hidden: &B::Tensor, position: usize, then: F) -> Result<R, BackendError>
    where
        B: crate::backend::GqaPrefillBackend,
        F: FnOnce(&B::Tensor) -> Result<R, BackendError>,
    {
        self.backend.begin_decode_batch();
        let result = self.mtp_decode_open(weights, cache, token_embedding, target_hidden, position).and_then(|hidden| then(&hidden));
        self.backend.finish_batch();
        result
    }
}

pub struct Qwen36Execution<'run, 'model, B>
where
    B: crate::backend::GqaPrefillBackend + GatedDeltaNetKernel,
{
    runtime: &'run Qwen36Runtime<'model, B>,
    cache: &'run mut B::Cache,
    recurrent: &'run mut GatedDeltaNetState<B::GatedDeltaNetStorage>,
    position: usize,
    /// DSpark drafter 的 tap 层 hidden 捕获(边界处 push 整行张量)。
    capture: Option<(&'run crate::runtime::speculative::HiddenStateCapturePlan, &'run mut Vec<B::Tensor>)>,
    /// DSpark 部分接受后的 GDN-only 重放素材:各 DeltaNet 层的 normed 输入
    /// (按 GDN 层升序;verify 的前 R 行重放只需这些,不必重跑整条前向)。
    gdn_inputs: Option<&'run mut Vec<B::Tensor>>,
}

/// 保持 MLX affine 压缩布局，由 backend 决定原生消费或 reference 解码。
fn prepare_mlx_matrix<B: Backend>(backend: &B, matrix: &crate::weight::format::quantization::MlxAffineMatrix) -> Result<B::Weight, BackendError> {
    backend.prepare_weight(LinearWeight::mlx_affine(matrix), matrix.rows, matrix.cols)
}

/// 把 1D TensorData(BF16/F16/F32)解到 F32 向量并做 GemmaRMS 偏移(value - 1.0)。
fn prepare_gemma_vector<B: Backend>(backend: &B, tensor: &crate::weight::container::safetensor::TensorData) -> Result<B::Weight, BackendError> {
    let values = tensor.to_f32().map_err(|msg| BackendError::Compute { msg: format!("Qwen3.6 {msg}") })?;
    let shifted: Vec<f32> = values.iter().map(|value| value - 1.0).collect();
    backend.prepare_gemma_f32(&shifted, 1, shifted.len())
}

/// 把 1D TensorData 解到 F32 向量(不做偏移,用于 conv/a_log/dt_bias 等)。
fn prepare_f32_vector<B: Backend>(backend: &B, tensor: &crate::weight::container::safetensor::TensorData) -> Result<B::Weight, BackendError> {
    let values = tensor.to_f32().map_err(|msg| BackendError::Compute { msg: format!("Qwen3.6 {msg}") })?;
    backend.prepare_f32(&values, 1, values.len())
}

/// DeltaNet norm 权重也做 GemmaRMS 偏移(per-head RMSNorm 用 (1 + w))。
fn prepare_delta_norm_vector<B: Backend>(backend: &B, tensor: &crate::weight::container::safetensor::TensorData) -> Result<B::Weight, BackendError> {
    prepare_gemma_vector(backend, tensor)
}

/// prepare 一层权重到 backend。
pub fn prepare_qwen36_layer<B: Backend>(backend: &B, weights: &Qwen36Weights, layer: usize) -> Result<Qwen36RuntimeLayer<B::Weight>, BackendError> {
    let source = weights.layer(layer).map_err(|error| BackendError::Compute { msg: error })?;
    let token_mixer = match source.token_mixer {
        crate::weight::model::qwen36::Qwen36TokenMixerWeights::FullAttention(weights) => HybridTokenMixer::FullAttention(FullAttentionWeights {
            query_gate: prepare_mlx_matrix(backend, &weights.query_and_gate)?,
            query_norm: prepare_gemma_vector(backend, &weights.query_norm)?,
            key: prepare_mlx_matrix(backend, &weights.key)?,
            key_norm: prepare_gemma_vector(backend, &weights.key_norm)?,
            value: prepare_mlx_matrix(backend, &weights.value)?,
            output: prepare_mlx_matrix(backend, &weights.output)?,
        }),
        crate::weight::model::qwen36::Qwen36TokenMixerWeights::GatedDeltaNet(weights) => HybridTokenMixer::DeltaNet(DeltaNetWeights {
            qkv: prepare_mlx_matrix(backend, &weights.qkv)?,
            z: prepare_mlx_matrix(backend, &weights.z)?,
            alpha: prepare_mlx_matrix(backend, &weights.a)?,
            beta: prepare_mlx_matrix(backend, &weights.b)?,
            conv: prepare_f32_vector(backend, &weights.conv)?,
            a_log: prepare_f32_vector(backend, &weights.a_log)?,
            dt_bias: prepare_f32_vector(backend, &weights.dt_bias)?,
            norm: prepare_delta_norm_vector(backend, &weights.norm)?,
            output: prepare_mlx_matrix(backend, &weights.output)?,
        }),
    };
    Ok(Qwen36RuntimeLayer {
        input_norm: prepare_gemma_vector(backend, &source.input_norm)?,
        token_mixer,
        post_attention_norm: prepare_gemma_vector(backend, &source.post_attention_norm)?,
        mlp: Qwen36RuntimeMlp { gate: prepare_mlx_matrix(backend, &source.mlp.gate)?, up: prepare_mlx_matrix(backend, &source.mlp.up)?, down: prepare_mlx_matrix(backend, &source.mlp.down)? },
    })
}

pub fn prepare_qwen36_layers<B: Backend>(backend: &B, weights: &Qwen36Weights) -> Result<Vec<Qwen36RuntimeLayer<B::Weight>>, BackendError> {
    let config = weights.config();
    (0..config.num_layers).map(|layer| prepare_qwen36_layer(backend, weights, layer)).collect()
}

pub fn prepare_qwen36_output_head<B: Backend>(backend: &B, weights: &Qwen36Weights) -> Result<Qwen36OutputHead<B::Weight>, BackendError> {
    prepare_qwen36_output_head_quantized(backend, weights, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_qwen36_output_head_quantized<B: Backend>(backend: &B, weights: &Qwen36Weights, quantization: crate::weight::LmHeadQuantization) -> Result<Qwen36OutputHead<B::Weight>, BackendError> {
    let config = weights.config();
    let norm = weights.final_norm().map_err(|error| BackendError::Compute { msg: error })?;
    let norm_values = norm.to_f32().map_err(|msg| BackendError::Compute { msg: format!("Qwen3.6 {msg}") })?;
    let shifted: Vec<f32> = norm_values.iter().map(|value| value - 1.0).collect();
    let lm_head = weights.lm_head().map_err(|error| BackendError::Compute { msg: error })?;
    super::output::prepare_output_head_gemma_quantized(backend, &shifted, LinearWeight::mlx_affine(&lm_head), config.vocab_size, config.hidden_size, quantization)
}

/// Node decode 需要分别持有 norm 与 lm_head，以便 MTP/DSpark 复用完整输出头。
pub fn prepare_qwen36_output_parts_quantized<B: Backend>(backend: &B, weights: &Qwen36Weights, quantization: crate::weight::LmHeadQuantization) -> Result<(B::Weight, B::Weight), BackendError> {
    let config = weights.config();
    let norm = weights.final_norm().map_err(|error| BackendError::Compute { msg: error })?;
    let norm_values = norm.to_f32().map_err(|msg| BackendError::Compute { msg: format!("Qwen3.6 {msg}") })?;
    let shifted = norm_values.iter().map(|value| value - 1.0).collect::<Vec<_>>();
    let norm = backend.prepare_gemma_f32(&shifted, 1, config.hidden_size)?;
    let lm_head = weights.lm_head().map_err(|error| BackendError::Compute { msg: error })?;
    let lm_head = super::output::prepare_lm_head_weight(backend, LinearWeight::mlx_affine(&lm_head), config.vocab_size, config.hidden_size, quantization)?;
    Ok((norm, lm_head))
}

impl<'a, B> Qwen36Runtime<'a, B>
where
    B: crate::backend::GqaPrefillBackend + GatedDeltaNetKernel,
{
    /// DSpark GDN-only 重放:对给定 normed 输入只推进第 layer 个 DeltaNet 层的
    /// recurrent state(零 z 占位跳过 z/output 投影;不触碰 KV 与 MLP)。
    pub fn replay_delta_layer(&self, recurrent: &mut GatedDeltaNetState<B::GatedDeltaNetStorage>, layer: usize, input: &B::Tensor, position: usize, zero_z: &B::Tensor) -> Result<(), BackendError> {
        match self.layers.get(layer).map(|weights| &weights.token_mixer) {
            Some(HybridTokenMixer::DeltaNet(weights)) => {
                self.attention.delta_advance_state(weights, recurrent, layer, input, position, zero_z)?;
                Ok(())
            }
            _ => Err(BackendError::Compute { msg: format!("Qwen3.6 L{layer} 不是 DeltaNet 层,无法 GDN 重放") }),
        }
    }

    /// 本模型 DeltaNet 层的编号(升序);GDN 重放素材按此对位。
    pub fn delta_layer_ids(&self) -> Vec<usize> {
        self.layers.iter().enumerate().filter(|(_, weights)| matches!(weights.token_mixer, HybridTokenMixer::DeltaNet(_))).map(|(layer, _)| layer).collect()
    }
}

impl<'run, B> Qwen36Execution<'run, '_, B>
where
    B: crate::backend::GqaPrefillBackend + GatedDeltaNetKernel,
    B::Tensor: Clone,
{
    /// 记录 DSpark tap 层 hidden;prefill/decode 在计划边界处 push 张量副本。
    pub fn capture(mut self, plan: &'run crate::runtime::speculative::HiddenStateCapturePlan, captures: &'run mut Vec<B::Tensor>) -> Self {
        self.capture = Some((plan, captures));
        self
    }

    fn maybe_capture(&mut self, layer: usize, hidden: &B::Tensor) {
        if let Some((plan, captures)) = &mut self.capture {
            if plan.captures_layer_output(layer) {
                captures.push(hidden.clone());
            }
        }
    }

    /// 记录各 DeltaNet 层的 normed 输入(DSpark GDN-only 重放素材)。
    pub fn gdn_inputs(mut self, sink: &'run mut Vec<B::Tensor>) -> Self {
        self.gdn_inputs = Some(sink);
        self
    }

    fn maybe_collect_gdn_input(&mut self, layer: usize, normed: &B::Tensor) {
        if let Some(sink) = &mut self.gdn_inputs {
            if matches!(self.runtime.layers.get(layer).map(|weights| &weights.token_mixer), Some(HybridTokenMixer::DeltaNet(_))) {
                sink.push(normed.clone());
            }
        }
    }

    fn run_layer(&mut self, weights: &Qwen36RuntimeLayer<B::Weight>, layer: usize, hidden: &B::Tensor) -> Result<B::Tensor, BackendError> {
        let normed = self.runtime.backend.gemma_rmsnorm_f32(hidden, &weights.input_norm, self.runtime.config.rms_norm_eps)?;
        self.maybe_collect_gdn_input(layer, &normed);
        let mixed = match &weights.token_mixer {
            HybridTokenMixer::FullAttention(weights) => self.runtime.attention.full(weights, self.cache, layer, &normed, self.position)?,
            HybridTokenMixer::DeltaNet(weights) => self.runtime.attention.delta(weights, self.recurrent, layer, &normed, self.position)?,
        };
        let (residual, ffn_input) = self.runtime.backend.add_gemma_rmsnorm_pair(hidden, &mixed, &weights.post_attention_norm, self.runtime.config.rms_norm_eps)?;
        self.runtime.backend.gated_mlp_add_residual(&ffn_input, &weights.mlp.gate, &weights.mlp.up, &weights.mlp.down, &self.runtime.mlp.activation, &residual)
    }

    /// 执行单个 prefill 层。设备端 layer-major 调度用它记录逐层耗时，同时让
    /// hidden 直接作为下一层输入，不在层边界下载或重建张量。
    pub fn prefill_layer(mut self, layer: usize, hidden: B::Tensor) -> Result<B::Tensor, BackendError> {
        let backend = self.runtime.backend;
        let config = self.runtime.config;
        let weights = self.runtime.layers.get(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        let token_count = backend.token_rows(&hidden);
        if token_count == 0 || backend.token_cols(&hidden) != config.hidden_size {
            return Err(BackendError::Compute { msg: format!("Qwen3.6 L{layer} prefill 输入 shape 异常: hidden=[{},{}]", token_count, backend.token_cols(&hidden),) });
        }
        backend.begin_batch();
        let output = self.run_layer(weights, layer, &hidden);
        backend.finish_batch();
        let output = output?;
        if backend.token_rows(&output) != token_count || backend.token_cols(&output) != config.hidden_size {
            return Err(BackendError::Compute { msg: format!("Qwen3.6 L{layer} prefill 输出 shape 异常") });
        }
        Ok(output)
    }

    pub fn prefill(mut self, mut hidden: B::Tensor) -> Result<B::Tensor, BackendError> {
        let backend = self.runtime.backend;
        let config = self.runtime.config;
        let layers = self.runtime.layers;
        let token_count = backend.token_rows(&hidden);
        if layers.len() != config.num_layers || token_count == 0 || backend.token_cols(&hidden) != config.hidden_size {
            return Err(BackendError::Compute { msg: format!("Qwen3.6 prefill 输入不完整: layers={}/{}, hidden=[{},{}]", layers.len(), config.num_layers, token_count, backend.token_cols(&hidden)) });
        }
        let result = (|| {
            for (layer, weights) in layers.iter().enumerate() {
                let _scope = backend.layer_scope();
                backend.begin_batch();
                hidden = self.run_layer(weights, layer, &hidden)?;
                self.maybe_capture(layer, &hidden);
                // 下一层仍沿同一 queue 消费；每两层设置完成边界，及时释放
                // PendingMetalProfile 保活的大型 attention/MLP activation。
                // (按 chunk 行数放宽 in-flight 深度的实验无收益:gaps 的
                // 真因是每 CB ~2.8ms 的提交固定成本,不是 CPU 编码等待。)
                backend.submit_batch();
                if (layer + 1).is_multiple_of(MAX_IN_FLIGHT_PREFILL_LAYERS) {
                    backend.synchronize()?;
                }
                if backend.token_rows(&hidden) != token_count || backend.token_cols(&hidden) != config.hidden_size {
                    return Err(BackendError::Compute { msg: format!("Qwen3.6 L{layer} prefill 输出 shape 异常") });
                }
            }
            Ok(hidden)
        })();
        backend.finish_batch();
        result
    }

    pub fn decode(mut self, mut hidden: B::Tensor) -> Result<B::Tensor, BackendError> {
        let backend = self.runtime.backend;
        let config = self.runtime.config;
        let layers = self.runtime.layers;
        if layers.len() != config.num_layers || backend.token_rows(&hidden) != 1 || backend.token_cols(&hidden) != config.hidden_size {
            return Err(BackendError::Compute { msg: format!("Qwen3.6 decode 输入不完整: layers={}/{}, hidden=[{},{}]", layers.len(), config.num_layers, backend.token_rows(&hidden), backend.token_cols(&hidden)) });
        }
        let result = (|| {
            let _scope = backend.layer_scope();
            for (layer, weights) in layers.iter().enumerate() {
                backend.begin_decode_batch();
                hidden = self.run_layer(weights, layer, &hidden)?;
                self.maybe_capture(layer, &hidden);
                // decode 同样保持层间有序提交；完成同步留给整个 token 边界。
                backend.submit_batch();
            }
            Ok(hidden)
        })();
        backend.finish_batch();
        result
    }
}

pub fn qwen36_token_output<B: Backend>(backend: &B, cfg: &Qwen36Config, head: &Qwen36OutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    super::output::token_output(backend, head, hidden, &super::output::OutputPlan { eps: cfg.rms_norm_eps, norm: super::output::OutputNorm::GemmaRms, excluded_tokens: vec![cfg.image_token_id, cfg.video_token_id] })
}

// ============================================================================
// M-RoPE 预计算
//
// Qwen3.6 mrope_section = [11, 11, 10],rope_dim = 64,half = 32。三个轴按段切分:
// 前 11 对 → 轴 0 (temporal),次 11 对 → 轴 1 (height),末 10 对 → 轴 2 (width)。
// 纯文本三轴位置一致,等价标准 RoPE。
// ============================================================================

/// M-RoPE 表(平台无关的 [`RopeTable`],CPU/Metal/CUDA 共用)。
pub fn qwen36_mrope_table(config: &Qwen36Config, position_ids: &[Vec<usize>; 3]) -> Result<RopeTable, String> {
    let rows = position_ids[0].len();
    if rows == 0 || position_ids.iter().any(|axis| axis.len() != rows) {
        return Err("Qwen3.6 M-RoPE position shape 无效".to_owned());
    }
    let half = config.rope_dim / 2;
    let section_total: usize = config.mrope_section.iter().sum();
    if section_total != half {
        return Err(format!("Qwen3.6 mrope_section {:?} 总和 {} 与 rope_dim/2={} 不一致", config.mrope_section, section_total, half));
    }
    let section_ends = [config.mrope_section[0], config.mrope_section[0] + config.mrope_section[1], half];
    let mut cos = Vec::with_capacity(rows * half);
    let mut sin = Vec::with_capacity(rows * half);
    for row in 0..rows {
        for pair in 0..half {
            let axis = if pair < section_ends[0] {
                0
            } else if pair < section_ends[1] {
                1
            } else {
                2
            };
            let frequency = config.rope_theta.powf(-2.0 * pair as f32 / config.rope_dim as f32);
            let angle = position_ids[axis][row] as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok(RopeTable { cos, sin, rotary_dim: config.rope_dim, seq_len: rows })
}

/// decode 的 M-RoPE 表(纯文本三轴相等)。行 r 的位置取 r + rope_delta，
/// 行号本身仍是序列位置(KV 槽位与 DeltaNet 步进)。最后一行是 decode
/// 实际读取的行，必须有效；delta 为负时更早的行下溢钳到 0 占位。
pub fn qwen36_decode_rope_table(config: &Qwen36Config, sequence_position: usize, rope_delta: i64) -> Result<RopeTable, String> {
    let end = sequence_position.checked_add(1).ok_or("Qwen3.6 decode position 溢出")?;
    let half = config.rope_dim / 2;
    if config.rope_dim == 0 || !config.rope_dim.is_multiple_of(2) || !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
        return Err(format!("Qwen3.6 RoPE 参数无效: dim={} theta={}", config.rope_dim, config.rope_theta));
    }
    let mut cos = Vec::with_capacity(end.checked_mul(half).ok_or("Qwen3.6 decode RoPE 大小溢出")?);
    let mut sin = Vec::with_capacity(cos.capacity());
    for row in 0..end {
        let value = i64::try_from(row).ok().and_then(|row| row.checked_add(rope_delta));
        let position = value.map_or(0, |value| value.max(0) as usize);
        for pair in 0..half {
            let frequency = config.rope_theta.powf(-2.0 * pair as f32 / config.rope_dim as f32);
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    i64::try_from(sequence_position)
        .ok()
        .and_then(|position| position.checked_add(rope_delta))
        .and_then(|position| usize::try_from(position).ok())
        .ok_or_else(|| format!("Qwen3.6 decode position={sequence_position} delta={rope_delta} 无效"))?;
    // 早期负位置允许钳到 0 占位，但当前实际 decode 行必须有效。
    Ok(RopeTable { cos, sin, rotary_dim: config.rope_dim, seq_len: end })
}

// ============================================================================
// GGUF 权重 prepare —— 把 Qwen36Gguf 的 tensor 映射到 backend 常驻权重。
// 与 Ornith 共享同一套 attention 底座,tensor 命名一致;FFN 由 Dense MLP 决定。
// ============================================================================
// ============================================================================

pub fn prepare_qwen36_gguf_layers<B: Backend>(backend: &B, source: &GgufReader, cfg: &Qwen36Config) -> Result<Vec<Qwen36RuntimeLayer<B::Weight>>, BackendError> {
    (0..cfg.num_layers).map(|layer| prepare_qwen36_gguf_layer(backend, source, cfg, layer)).collect()
}

pub fn prepare_qwen36_gguf_layer<B: Backend>(backend: &B, source: &GgufReader, cfg: &Qwen36Config, layer: usize) -> Result<Qwen36RuntimeLayer<B::Weight>, BackendError> {
    // layer >= num_layers 是 MTP 块(blk.64，纯 FullAttention，无 ssm 权重)
    let kind = if layer >= cfg.num_layers || (layer + 1).is_multiple_of(cfg.full_attention_interval) { Qwen36AttentionKind::FullAttention } else { Qwen36AttentionKind::GatedDeltaNet };
    let prefix = format!("blk.{layer}");
    // norm 权重 F16 resident:F16-weight kernel 原生直通,层内数据流保持 F16
    let input_norm = prepare_gguf_gemma_vector_f16(backend, source, &format!("{prefix}.attn_norm.weight"))?;
    let post_attention_norm = prepare_gguf_gemma_vector_f16(backend, source, &format!("{prefix}.post_attention_norm.weight"))?;
    let token_mixer = match kind {
        Qwen36AttentionKind::FullAttention => HybridTokenMixer::FullAttention(FullAttentionWeights {
            query_gate: prepare_gguf_matrix(backend, source, &format!("{prefix}.attn_q.weight"))?,
            query_norm: prepare_gguf_gemma_vector_f16(backend, source, &format!("{prefix}.attn_q_norm.weight"))?,
            key: prepare_gguf_matrix(backend, source, &format!("{prefix}.attn_k.weight"))?,
            key_norm: prepare_gguf_gemma_vector_f16(backend, source, &format!("{prefix}.attn_k_norm.weight"))?,
            value: prepare_gguf_matrix(backend, source, &format!("{prefix}.attn_v.weight"))?,
            output: prepare_gguf_matrix(backend, source, &format!("{prefix}.attn_output.weight"))?,
        }),
        Qwen36AttentionKind::GatedDeltaNet => HybridTokenMixer::DeltaNet(DeltaNetWeights {
            qkv: prepare_gguf_matrix(backend, source, &format!("{prefix}.attn_qkv.weight"))?,
            z: prepare_gguf_matrix(backend, source, &format!("{prefix}.attn_gate.weight"))?,
            // 量化 GGUF(如 UD-Q3_K_XL 的 IQ4_XS)走原生量化 gemv;dense 则 F32。
            alpha: prepare_gguf_matrix(backend, source, &format!("{prefix}.ssm_alpha.weight"))?,
            beta: prepare_gguf_matrix(backend, source, &format!("{prefix}.ssm_beta.weight"))?,
            conv: prepare_gguf_matrix(backend, source, &format!("{prefix}.ssm_conv1d.weight"))?,
            a_log: prepare_gguf_a_log_vector(backend, source, &format!("{prefix}.ssm_a"))?,
            dt_bias: prepare_gguf_f32_vector(backend, source, &format!("{prefix}.ssm_dt.bias"))?,
            // ssm_norm 是普通 RMSNorm(不做 GemmaRMS 偏移)，保持 GGUF 的 F32 控制精度。
            norm: prepare_gguf_f32_vector(backend, source, &format!("{prefix}.ssm_norm.weight"))?,
            output: prepare_gguf_matrix(backend, source, &format!("{prefix}.ssm_out.weight"))?,
        }),
    };
    let (mlp_gate, mlp_up) = prepare_gguf_matrix_pair(backend, source, &format!("{prefix}.ffn_gate.weight"), &format!("{prefix}.ffn_up.weight"))?;
    let _ = cfg;
    Ok(Qwen36RuntimeLayer { input_norm, token_mixer, post_attention_norm, mlp: Qwen36RuntimeMlp { gate: mlp_gate, up: mlp_up, down: prepare_gguf_matrix(backend, source, &format!("{prefix}.ffn_down.weight"))? } })
}

/// MTP 块权重(blk.{num_layers}.nextn.* + 同名层权重)。GGUF 不含 MTP 块时返回 None。
pub fn prepare_qwen36_mtp<B: Backend>(backend: &B, source: &GgufReader, cfg: &Qwen36Config) -> Result<Option<Qwen36RuntimeMtp<B::Weight>>, BackendError> {
    let layer = cfg.num_layers;
    let prefix = format!("blk.{layer}");
    if source.tensor(&format!("{prefix}.nextn.eh_proj.weight")).is_none() {
        return Ok(None);
    }
    Ok(Some(Qwen36RuntimeMtp {
        embedding_norm: prepare_gguf_gemma_vector_f16(backend, source, &format!("{prefix}.nextn.enorm.weight"))?,
        hidden_norm: prepare_gguf_gemma_vector_f16(backend, source, &format!("{prefix}.nextn.hnorm.weight"))?,
        input_projection: prepare_gguf_matrix(backend, source, &format!("{prefix}.nextn.eh_proj.weight"))?,
        layer: prepare_qwen36_gguf_layer(backend, source, cfg, layer)?,
        output_norm: prepare_gguf_gemma_vector_f16(backend, source, &format!("{prefix}.nextn.shared_head_norm.weight"))?,
    }))
}

/// 返回 (final_norm, lm_head) 两个独立权重,供 decode 直接调用 backend 算子。
pub fn prepare_qwen36_gguf_output<B: Backend>(backend: &B, source: &GgufReader) -> Result<(B::Weight, B::Weight), BackendError> {
    prepare_qwen36_gguf_output_quantized(backend, source, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_qwen36_gguf_output_quantized<B: Backend>(backend: &B, source: &GgufReader, quantization: crate::weight::LmHeadQuantization) -> Result<(B::Weight, B::Weight), BackendError> {
    let norm = prepare_gguf_gemma_vector_f16(backend, source, "output_norm.weight")?;
    let output_name = if source.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let matrix = source.read_matrix(output_name).map_err(crate::runtime::compute_error)?;
    let head = super::output::prepare_lm_head_weight(backend, LinearWeight::gguf(&matrix), matrix.rows, matrix.columns, quantization)?;
    Ok((norm, head))
}

/// 单独准备 token embedding，供能够在设备上直接做 packed lookup 的 backend
/// 与 output head 并行持有；decode 不再回到 GGUF 读取单行。
pub fn prepare_qwen36_gguf_embedding<B: Backend>(backend: &B, source: &GgufReader) -> Result<B::Weight, BackendError> {
    prepare_gguf_matrix(backend, source, "token_embd.weight")
}

// ============================================================================
// Vision encoder —— ViT 架构与 Qwen3-VL 完全相同,复用其执行函数。
// 仅权重 prepare 从 mmproj GGUF 的 v.* 命名加载(而非 safetensors model.visual.*)。
// ============================================================================

use super::{prepare_gguf_a_log_vector, prepare_gguf_f32_vector, prepare_gguf_gemma_vector_f16, prepare_gguf_matrix, prepare_gguf_matrix_pair};
use crate::backend::VisionBackend;
use crate::weight::container::safetensor::TensorData;
use crate::weight::model::vision::LinearWeights;

/// mmproj F16 矩阵数据封装为 TensorData(shape 为 [rows, cols] 行主序)。
fn tensor_data_from_f16(bytes: &[u8], rows: usize, cols: usize) -> TensorData {
    TensorData { name: String::new(), dtype: "F16".to_owned(), shape: vec![rows, cols], data: bytes.to_vec() }
}

fn tensor_data_f32(values: &[f32]) -> TensorData {
    TensorData { name: String::new(), dtype: "F32".to_owned(), shape: vec![1, values.len()], data: f32_to_bytes(values) }
}

/// 视觉塔当前只为 F16 矩阵提供原生装配；量化 mmproj 必须在服务启动时拒绝，
/// 不能在首次图像请求时按 F16 字节误读或临时反量化。
pub fn validate_qwen36_mmproj_formats(mmproj: &GgufReader, cfg: &Qwen36Config) -> Result<(), String> {
    let require = |name: &str, allowed: &[u32]| -> Result<(), String> {
        let tensor = mmproj.tensor(name).ok_or_else(|| format!("Qwen3.6 mmproj 缺少 tensor {name}"))?;
        if !allowed.contains(&tensor.tensor_type.0) {
            return Err(format!("Qwen3.6 mmproj tensor {name}={} 缺少原生视觉算子，当前只接受 {:?}", tensor.tensor_type.name(), allowed));
        }
        Ok(())
    };
    for name in ["v.patch_embd.weight", "v.patch_embd.weight.1"].into_iter().take(cfg.vision.temporal_patch_size) {
        require(name, &[1])?;
    }
    for layer in 0..cfg.vision.depth {
        let prefix = format!("v.blk.{layer}");
        for suffix in ["attn_qkv.weight", "attn_out.weight", "ffn_up.weight", "ffn_down.weight"] {
            require(&format!("{prefix}.{suffix}"), &[1])?;
        }
        for suffix in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "attn_qkv.bias", "attn_out.bias", "ffn_up.bias", "ffn_down.bias"] {
            require(&format!("{prefix}.{suffix}"), &[0, 1, 30])?;
        }
    }
    for name in ["mm.0.weight", "mm.2.weight"] {
        require(name, &[1])?;
    }
    for name in ["v.patch_embd.bias", "v.position_embd.weight", "v.post_ln.weight", "v.post_ln.bias", "mm.0.bias", "mm.2.bias"] {
        require(name, &[0, 1, 30])?;
    }
    Ok(())
}

/// ViT 层权重(直接用 qwen3_vl 的 PreparedVisionLayer)。mmproj 是独立的视觉 GGUF reader。
pub fn prepare_qwen36_vision_layer<B: Backend>(backend: &B, mmproj: &GgufReader, cfg: &Qwen36Config, layer: usize) -> Result<super::qwen3_vl::PreparedVisionLayer<B::Weight>, BackendError> {
    let v = &cfg.vision;
    let hidden = v.hidden_size;
    let three_hidden = hidden * 3;
    let inter = v.intermediate_size;
    let prefix = format!("v.blk.{layer}");
    let ln1_w = mmproj.read_tensor_f32(&format!("{prefix}.ln1.weight")).map_err(crate::runtime::compute_error)?;
    let ln1_b = mmproj.read_tensor_f32(&format!("{prefix}.ln1.bias")).map_err(crate::runtime::compute_error)?;
    let ln2_w = mmproj.read_tensor_f32(&format!("{prefix}.ln2.weight")).map_err(crate::runtime::compute_error)?;
    let ln2_b = mmproj.read_tensor_f32(&format!("{prefix}.ln2.bias")).map_err(crate::runtime::compute_error)?;
    Ok(super::qwen3_vl::PreparedVisionLayer {
        input_norm: super::qwen3_vl::prepare_norm(backend, &crate::weight::model::vision::LayerNormWeights { weight: tensor_data_f32(&ln1_w), bias: tensor_data_f32(&ln1_b) })?,
        qkv: super::qwen3_vl::prepare_linear(
            backend,
            &LinearWeights {
                weight: tensor_data_from_f16(&mmproj.read_tensor(&format!("{prefix}.attn_qkv.weight")).map_err(crate::runtime::compute_error)?, three_hidden, hidden),
                bias: Some(tensor_data_f32(&mmproj.read_tensor_f32(&format!("{prefix}.attn_qkv.bias")).map_err(crate::runtime::compute_error)?)),
            },
        )?,
        output: super::qwen3_vl::prepare_linear(
            backend,
            &LinearWeights {
                weight: tensor_data_from_f16(&mmproj.read_tensor(&format!("{prefix}.attn_out.weight")).map_err(crate::runtime::compute_error)?, hidden, hidden),
                bias: Some(tensor_data_f32(&mmproj.read_tensor_f32(&format!("{prefix}.attn_out.bias")).map_err(crate::runtime::compute_error)?)),
            },
        )?,
        post_attention_norm: super::qwen3_vl::prepare_norm(backend, &crate::weight::model::vision::LayerNormWeights { weight: tensor_data_f32(&ln2_w), bias: tensor_data_f32(&ln2_b) })?,
        mlp_input: super::qwen3_vl::prepare_linear(
            backend,
            &LinearWeights {
                weight: tensor_data_from_f16(&mmproj.read_tensor(&format!("{prefix}.ffn_up.weight")).map_err(crate::runtime::compute_error)?, inter, hidden),
                bias: Some(tensor_data_f32(&mmproj.read_tensor_f32(&format!("{prefix}.ffn_up.bias")).map_err(crate::runtime::compute_error)?)),
            },
        )?,
        mlp_output: super::qwen3_vl::prepare_linear(
            backend,
            &LinearWeights {
                weight: tensor_data_from_f16(&mmproj.read_tensor(&format!("{prefix}.ffn_down.weight")).map_err(crate::runtime::compute_error)?, hidden, inter),
                bias: Some(tensor_data_f32(&mmproj.read_tensor_f32(&format!("{prefix}.ffn_down.bias")).map_err(crate::runtime::compute_error)?)),
            },
        )?,
    })
}

pub fn prepare_qwen36_vision_layers<B: Backend>(backend: &B, mmproj: &GgufReader, cfg: &Qwen36Config) -> Result<Vec<super::qwen3_vl::PreparedVisionLayer<B::Weight>>, BackendError> {
    (0..cfg.vision.depth).map(|layer| prepare_qwen36_vision_layer(backend, mmproj, cfg, layer)).collect()
}

/// merger: v.post_ln + mm.0 + mm.2(norm_after_merge=false)。
pub fn prepare_qwen36_vision_merger<B: Backend>(backend: &B, mmproj: &GgufReader, cfg: &Qwen36Config) -> Result<super::qwen3_vl::PreparedVisionMerger<B::Weight>, BackendError> {
    let v = &cfg.vision;
    let merged = v.hidden_size * v.spatial_merge_size * v.spatial_merge_size;
    let output_hidden = v.output_hidden_size;
    let norm_w = mmproj.read_tensor_f32("v.post_ln.weight").map_err(crate::runtime::compute_error)?;
    let norm_b = mmproj.read_tensor_f32("v.post_ln.bias").map_err(crate::runtime::compute_error)?;
    Ok(super::qwen3_vl::PreparedVisionMerger {
        norm: super::qwen3_vl::prepare_norm(backend, &crate::weight::model::vision::LayerNormWeights { weight: tensor_data_f32(&norm_w), bias: tensor_data_f32(&norm_b) })?,
        input: super::qwen3_vl::prepare_linear(
            backend,
            &LinearWeights {
                weight: tensor_data_from_f16(&mmproj.read_tensor("mm.0.weight").map_err(crate::runtime::compute_error)?, merged, merged),
                bias: Some(tensor_data_f32(&mmproj.read_tensor_f32("mm.0.bias").map_err(crate::runtime::compute_error)?)),
            },
        )?,
        output: super::qwen3_vl::prepare_linear(
            backend,
            &LinearWeights {
                weight: tensor_data_from_f16(&mmproj.read_tensor("mm.2.weight").map_err(crate::runtime::compute_error)?, output_hidden, merged),
                bias: Some(tensor_data_f32(&mmproj.read_tensor_f32("mm.2.bias").map_err(crate::runtime::compute_error)?)),
            },
        )?,
        norm_after_merge: false,
    })
}

/// patch embedding：GGUF 把 conv3d kernel 按时间片拆成 `[kW,kH,in,out]` 两个
/// tensor(GGML ne 反转 PyTorch `[out,in,kH,kW]`，内存序保持 PyTorch 行主序)，
/// 这里重排为 HF 线性矩阵 `[hidden, in*kT*kH*kW]`，输入向量顺序 `[c][t][y][x]`
/// 与 PatchImageProcessor 一致。静图两时间片像素相同，t 先后不影响结果。
pub fn prepare_qwen36_patch_embedding<B: Backend>(backend: &B, mmproj: &GgufReader, cfg: &Qwen36Config) -> Result<super::qwen3_vl::PreparedLinear<B::Weight>, BackendError> {
    let v = &cfg.vision;
    let hidden = v.hidden_size;
    let patch = v.patch_size;
    let spatial = patch.checked_mul(patch).ok_or_else(|| BackendError::Compute { msg: format!("Qwen3.6 patch_size={patch} 溢出") })?;
    let slice_input = 3 * spatial;
    let temporal_stride = v.temporal_patch_size * spatial;
    let patch_input = 3 * temporal_stride;
    let mut weight = vec![0.0f32; hidden.checked_mul(patch_input).ok_or_else(|| BackendError::Compute { msg: "Qwen3.6 patch embedding 大小溢出".to_owned() })?];
    for (t, name) in ["v.patch_embd.weight", "v.patch_embd.weight.1"].iter().enumerate().take(v.temporal_patch_size) {
        let bytes = mmproj.read_tensor(name).map_err(crate::runtime::compute_error)?;
        if bytes.len() != hidden * slice_input * 2 {
            return Err(BackendError::Compute { msg: format!("Qwen3.6 patch embedding {name} bytes={}，期望 {}", bytes.len(), hidden * slice_input * 2) });
        }
        let values = bytes.chunks_exact(2).map(|pair| half::f16::from_le_bytes([pair[0], pair[1]]).to_f32()).collect::<Vec<_>>();
        for channel in 0..3 {
            for index in 0..spatial {
                // 切片内输入列 [in][kH*kW] → 全矩阵输入列 [in][kT*kH*kW]
                let source = channel * spatial + index;
                let target = channel * temporal_stride + t * spatial + index;
                for out in 0..hidden {
                    weight[out * patch_input + target] = values[out * slice_input + source];
                }
            }
        }
    }
    let bias = mmproj.read_tensor_f32("v.patch_embd.bias").map_err(crate::runtime::compute_error)?;
    super::qwen3_vl::prepare_linear(backend, &LinearWeights { weight: TensorData { name: String::new(), dtype: "F32".to_owned(), shape: vec![hidden, patch_input], data: f32_to_bytes(&weight) }, bias: Some(tensor_data_f32(&bias)) })
}

/// 完整 ViT 前向:patch embed → pos embed → 27 层 ViT → merger → [visual_tokens, hidden]。
pub fn qwen36_encode_image<B>(
    backend: &B,
    cfg: &Qwen36Config,
    patch: &super::qwen3_vl::PreparedLinear<B::Weight>,
    position: &[f32],
    layers: &[super::qwen3_vl::PreparedVisionLayer<B::Weight>],
    merger: &super::qwen3_vl::PreparedVisionMerger<B::Weight>,
    image: &ImageTensor,
) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend,
{
    let vision_cfg = cfg.vision.to_qwen3vl();
    let hidden_size = vision_cfg.hidden_size;
    let rows = image.rows;
    let expected_cols = 3 * vision_cfg.temporal_patch_size * vision_cfg.patch_size * vision_cfg.patch_size;
    if image.cols != expected_cols || image.data.len() != rows * image.cols {
        return Err(BackendError::Compute { msg: format!("Qwen3.6 vision 输入 shape [{rows},{}] 与期望 [{rows},{expected_cols}] 不符", image.cols) });
    }

    backend.begin_batch();
    let result = (|| {
        let _scope = backend.layer_scope();
        // patch embed + pos embed
        let input = backend.vision_tensor_from_f32(&image.data, rows, image.cols)?;
        // position embedding 需要按图像 grid 双线性插值(原始是 [position_embeddings, hidden] 的学习表)
        let position = super::qwen3_vl::interpolate_position_embedding(
            &vision_cfg,
            image.grid,
            &crate::weight::container::safetensor::TensorData { name: String::new(), dtype: "F32".to_owned(), shape: vec![vision_cfg.position_embeddings, hidden_size], data: f32_to_bytes(position) },
        )
        .map_err(crate::runtime::compute_error)?;
        let position_tensor = backend.vision_tensor_from_f32(&position, rows, hidden_size)?;
        let embedded = super::qwen3_vl::linear(backend, &input, patch)?;
        let mut hidden = backend.add(&embedded, &position_tensor)?;

        // ViT RoPE 预计算(head_dim = hidden / num_heads = 72)
        let head_dim = hidden_size / vision_cfg.num_heads;
        let (cos, sin) = super::qwen3_vl::vision_rope(image.grid, vision_cfg.spatial_merge_size, head_dim, vision_cfg.rope_theta).map_err(crate::runtime::compute_error)?;
        let cos = backend.vision_tensor_from_f32(&cos, rows, head_dim)?;
        let sin = backend.vision_tensor_from_f32(&sin, rows, head_dim)?;

        // batch_keep_alive 会把临时 buffer 保留到 synchronize；每层先提交，最多
        // 允许两层在飞，兼顾 GPU 流水与 24GB UMA 上的确定峰值。
        for (layer, layer_weights) in layers.iter().enumerate() {
            hidden = super::qwen3_vl::vision_layer(backend, &vision_cfg, layer_weights, &hidden, &cos, &sin)?;
            backend.submit_batch();
            if (layer + 1).is_multiple_of(MAX_IN_FLIGHT_PREFILL_LAYERS) {
                backend.synchronize()?;
            }
        }

        super::qwen3_vl::vision_merger(backend, &vision_cfg, merger, &hidden)
    })();
    backend.finish_batch();
    result
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::format::quantization::{MlxAffineMatrix, QuantizedMatrixRef, ScaleDType};

    #[test]
    fn mlx_linear_weight_preserves_packed_storage() {
        let matrix = MlxAffineMatrix::new(vec![0; 4], 1.0_f32.to_le_bytes().to_vec(), 0.0_f32.to_le_bytes().to_vec(), ScaleDType::F32, 4, 8, 1, 8).unwrap();
        assert!(matches!(LinearWeight::mlx_affine(&matrix), LinearWeight::Quantized(QuantizedMatrixRef::MlxAffine(_))));
    }

    #[test]
    fn standard_config_validates() {
        assert!(Qwen36Config::standard_27b().validate().is_ok());
    }

    #[test]
    fn layer_attention_schedule() {
        let model = Qwen36::standard_27b();
        // 3 层 GatedDeltaNet + 1 层 FullAttention 的固定周期。
        assert_eq!(model.attention_kind(0).unwrap(), Qwen36AttentionKind::GatedDeltaNet);
        assert_eq!(model.attention_kind(2).unwrap(), Qwen36AttentionKind::GatedDeltaNet);
        assert_eq!(model.attention_kind(3).unwrap(), Qwen36AttentionKind::FullAttention);
        assert_eq!(model.attention_kind(63).unwrap(), Qwen36AttentionKind::FullAttention);
    }

    #[test]
    fn mrope_text_collapses_to_standard_rope() {
        let config = Qwen36Config::standard_27b();
        // rope_dim=64, half=32;1 个 token → 32 对 cos/sin。
        // position 0 上的 cos(0)=1, sin(0)=0。
        let rope0 = qwen36_decode_rope_table(&config, 0, 0).unwrap();
        assert_eq!(rope0.seq_len, 1);
        assert_eq!(rope0.cos.len(), 32);
        assert_eq!(rope0.sin.len(), 32);
        assert!((rope0.cos[0] - 1.0).abs() < 1e-5);
        assert!(rope0.sin[0].abs() < 1e-5);
        // 非零 position 给非零 phase,首对 cos 应 < 1。
        let rope7 = qwen36_decode_rope_table(&config, 7, 0).unwrap();
        assert_eq!(rope7.seq_len, 8);
        assert!(rope7.cos[7 * 32].abs() < 1.0);
        // 三轴文本位置一致时(8 token 预填),与标准 RoPE 逐元素相等。
        let positions: [Vec<usize>; 3] = std::array::from_fn(|_| (0..8).collect());
        let mrope_text = qwen36_mrope_table(&config, &positions).unwrap();
        let standard = RopeTable::precompute(8, config.rope_dim, config.rope_theta);
        assert_eq!(mrope_text.cos.len(), standard.cos.len());
        for index in 0..mrope_text.cos.len() {
            assert!((mrope_text.cos[index] - standard.cos[index]).abs() < 1e-5, "index={index} cos 文本 M-RoPE vs 标准 RoPE 不一致");
            assert!((mrope_text.sin[index] - standard.sin[index]).abs() < 1e-5, "index={index} sin 文本 M-RoPE vs 标准 RoPE 不一致");
        }

        let shifted = qwen36_decode_rope_table(&config, 7, -2).unwrap();
        let shifted_positions = (0_usize..8).map(|position| position.saturating_sub(2)).collect::<Vec<_>>();
        let shifted_mrope = qwen36_mrope_table(&config, &[shifted_positions.clone(), shifted_positions.clone(), shifted_positions]).unwrap();
        assert_eq!(shifted.cos, shifted_mrope.cos);
        assert_eq!(shifted.sin, shifted_mrope.sin);
        assert!(qwen36_decode_rope_table(&config, 1, -2).is_err());
    }

    #[test]
    fn mrope_table_respects_section_boundaries() {
        let config = Qwen36Config::standard_27b();
        // 3 个 token,三轴位置各不相同(0,0,0) (10,20,30) (100,200,300)。
        let positions = [vec![0, 10, 100], vec![0, 20, 200], vec![0, 30, 300]];
        let rope = qwen36_mrope_table(&config, &positions).unwrap();
        assert_eq!(rope.seq_len, 3);
        assert_eq!(rope.cos.len(), 3 * 32);
        assert_eq!(rope.sin.len(), 3 * 32);
        // 三轴不同 → angle 不同 → cos/sin 不是常量。
        let span = rope.cos.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - rope.cos.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(span > 0.1, "三轴不同位置应给出非平凡 cos 范围,实际 span={span}");
    }

    #[test]
    fn visual_blocks_render_placeholder_counts() {
        use crate::runtime::qwen3_vl::{IMAGE_TOKEN, VISION_END_TOKEN, VISION_START_TOKEN};
        let config = Qwen36Config::standard_27b();
        // grid 2x4、merge 2 → (2/2)*(4/2) = 2 个视觉 token
        let tensor = crate::vision::ImageTensor { data: Vec::new(), rows: 4, cols: 0, grid: crate::vision::VisionGrid { temporal: 1, height: 2, width: 4 }, merge_size: 2 };
        let visuals = vec![Qwen36Visual { kind: Qwen36VisualKind::Image, tensor }];
        let blocks = qwen36_visual_blocks(&config, &visuals).unwrap();
        assert_eq!(blocks.len(), 1);
        let (block, rows, token_id) = &blocks[0];
        assert_eq!(*rows, 2);
        assert_eq!(*token_id, config.image_token_id);
        assert!(block.starts_with(VISION_START_TOKEN) && block.ends_with(VISION_END_TOKEN));
        assert_eq!(block.matches(IMAGE_TOKEN).count(), 2);
        // grid 不能按 merge 整除时报错
        let bad = crate::vision::ImageTensor { data: Vec::new(), rows: 4, cols: 0, grid: crate::vision::VisionGrid { temporal: 1, height: 3, width: 4 }, merge_size: 2 };
        assert!(qwen36_visual_blocks(&config, &[Qwen36Visual { kind: Qwen36VisualKind::Image, tensor: bad }]).is_err());
    }
}
#[cfg(target_os = "macos")]
pub mod dspark_metal;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod node;
#[cfg(all(target_os = "android", feature = "with-vulkan"))]
pub mod vulkan;
