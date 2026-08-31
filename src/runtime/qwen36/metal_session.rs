use std::{
    cell::OnceCell,
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use crate::{
    attention::{AttentionSpec, gated_delta_net::GatedDeltaNetState, hybrid::HybridAttentionOptions, rope::RopeTable},
    backend::{
        Backend,
        metal::{MetalContext, MetalGatedDeltaNetStorage, MetalKvCache, MetalTensor, MetalTensorDType, MetalWeight},
    },
    kv_cache::{DEFAULT_GROUP_SIZE, KvCacheFormat, KvCacheLayerMap, KvCacheLayout, KvCacheSpec},
    runtime::{
        dspark::DsparkTargetCache,
        output::{DraftHead, load_draft_vocabulary, normalized_draft_token_id, prepare_draft_head},
        qwen36::{self, Qwen36Config, Qwen36Runtime, dspark_metal::Qwen36DsparkRuntime},
    },
    tokenizer::{Detokenizer, Tokenizer},
    vision::RgbImage,
    weight::{container::gguf::GgufReader, format::mlx_affine::MlxAffineSource, model::qwen36::Qwen36Weights},
};

pub struct Qwen36Sequence {
    pub cache: MetalKvCache,
    pub recurrent: GatedDeltaNetState<MetalGatedDeltaNetStorage>,
    pub hidden: MetalTensor,
    pub tokens: Vec<u32>,
    /// 多模态会话 decode/续写的 M-RoPE 位置偏移；纯文本为 0(用 session 顺序表)。
    pub rope_delta: i64,
    /// rope_delta != 0 时预计算的 decode RoPE 表(行 r 的位置 = r + delta)，
    /// 首轮 prefill 或快照恢复时一次建好，decode 每步零重建。
    pub decode_rope: OnceCell<RopeTable>,
    /// DSpark drafter 的 target KV cache(按会话生命周期;快照恢复后为空,
    /// 该会话退回普通 decode 直到下一轮全量 prefill)。
    pub dspark_cache: DsparkTargetCache<MetalTensor>,
    /// prefill 预热后最后一位的 aux hidden 行 + 位置;首轮 draft 幂等重喂
    /// (backbone 的 cache 更新需要至少一行,空 tensor 无法在 Metal 分配)。
    pub dspark_last_aux: Option<(MetalTensor, usize)>,
}

impl Qwen36Sequence {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

/// mmproj 视觉塔常驻 Metal 权重；首次图像请求时懒加载(~0.8GB F16)，
/// 纯文本部署不占内存。
struct Qwen36NodeVision {
    patch: crate::runtime::qwen3_vl::PreparedLinear<MetalWeight>,
    position_embedding: Vec<f32>,
    layers: Vec<crate::runtime::qwen3_vl::PreparedVisionLayer<MetalWeight>>,
    merger: crate::runtime::qwen3_vl::PreparedVisionMerger<MetalWeight>,
}

/// 视觉 embedding 的 CPU 侧覆盖描述：token 区间可跨 prefill chunk，
/// 行数据为读回的 F32(f16 读回再上传无损，与 GPU scatter 数值一致)。
struct VisualOverlay {
    tokens: Range<usize>,
    values: Vec<f32>,
}

/// 把 overlay 落在 chunk 内的行复制进 chunk 的 token embedding。
fn overlay_visual_rows(embedding: &mut [f32], hidden: usize, chunk_offset: usize, overlays: &[VisualOverlay]) -> Result<(), String> {
    let chunk_tokens = embedding.len() / hidden;
    let chunk_end = chunk_offset.checked_add(chunk_tokens).ok_or("Qwen3.6 视觉 overlay chunk 区间溢出")?;
    for overlay in overlays {
        let start = overlay.tokens.start.max(chunk_offset);
        let end = overlay.tokens.end.min(chunk_end);
        if start >= end {
            continue;
        }
        let source = (start - overlay.tokens.start).checked_mul(hidden).ok_or("Qwen3.6 视觉 overlay 源偏移溢出")?;
        let target = (start - chunk_offset).checked_mul(hidden).ok_or("Qwen3.6 视觉 overlay 目标偏移溢出")?;
        let count = (end - start).checked_mul(hidden).ok_or("Qwen3.6 视觉 overlay 长度溢出")?;
        embedding[target..target + count].copy_from_slice(&overlay.values[source..source + count]);
    }
    Ok(())
}

enum Qwen36NodeWeights {
    Gguf(GgufReader),
    Mlx { weights: Qwen36Weights, bytes: u64 },
}

impl Qwen36NodeWeights {
    fn embedding_rows(&self, tokens: &[u32], hidden: usize, vocab: usize) -> Result<Vec<f32>, String> {
        match self {
            Self::Gguf(weights) => weights.embedding_rows("token_embd.weight", tokens, hidden, vocab),
            Self::Mlx { weights, .. } => weights.embedding_rows_f32(tokens),
        }
    }

    fn gguf(&self) -> Option<&GgufReader> {
        match self {
            Self::Gguf(weights) => Some(weights),
            Self::Mlx { .. } => None,
        }
    }

    fn bytes(&self) -> u64 {
        match self {
            Self::Gguf(weights) => weights.file_len(),
            Self::Mlx { bytes, .. } => *bytes,
        }
    }

    fn format(&self) -> &'static str {
        match self {
            Self::Gguf(_) => "gguf-mixed",
            Self::Mlx { .. } => "mlx-affine",
        }
    }
}

fn directory_bytes(root: &Path) -> u64 {
    if root.is_file() {
        return root.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    }
    std::fs::read_dir(root).into_iter().flatten().filter_map(Result::ok).map(|entry| directory_bytes(&entry.path())).sum()
}

pub struct Qwen36MetalSession {
    context: Arc<MetalContext>,
    cfg: Qwen36Config,
    weights: Qwen36NodeWeights,
    layers: Vec<qwen36::Qwen36RuntimeLayer<MetalWeight>>,
    mtp: Option<qwen36::Qwen36RuntimeMtp<MetalWeight>>,
    final_norm: MetalWeight,
    output_head: MetalWeight,
    draft_head: Option<DraftHead<MetalWeight>>,
    rope: RopeTable,
    /// 模型目录内 mmproj-*.gguf 路径；存在即具备图像输入能力。
    mmproj: Option<PathBuf>,
    /// 视觉塔常驻权重，首次图像请求懒加载；失败也缓存，避免反复重试。
    vision: OnceLock<Result<Qwen36NodeVision, String>>,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    attention_options: HybridAttentionOptions,
    kv_f16: bool,
    max_seq_len: usize,
    prefill_chunk_size: usize,
    model_format: &'static str,
    /// DSpark drafter(可选);embedding/lm_head 复用主模型。
    dspark: Option<Qwen36DsparkRuntime>,
}

impl Qwen36MetalSession {
    pub fn load(
        model_path: &Path,
        max_seq_len: usize,
        precise_gqa_prefill: bool,
        kv_f16: bool,
        prefill_chunk_size: usize,
        vision_max_tokens: Option<usize>,
        want_mtp: bool,
        mtp_draft_vocabulary: Option<&Path>,
        dspark_directory: Option<&Path>,
        dspark_draft_tokens: usize,
        replay_enabled: bool,
        lm_head_quantization: crate::weight::LmHeadQuantization,
    ) -> Result<Self, String> {
        let mut cfg = Qwen36Config::standard_27b();
        if let Some(max_tokens) = vision_max_tokens {
            let factor = cfg.vision.patch_size.checked_mul(cfg.vision.spatial_merge_size).ok_or("Qwen3.6 视觉 resize factor 溢出")?;
            let max_pixels = max_tokens.checked_mul(factor).and_then(|value| value.checked_mul(factor)).ok_or("Qwen3.6 vision_max_tokens 换算像素溢出")?;
            if max_pixels < cfg.vision.min_pixels {
                return Err(format!("Qwen3.6 vision_max_tokens={max_tokens} 低于最小像素预算 {} tokens", cfg.vision.min_pixels.div_ceil(factor * factor)));
            }
            cfg.vision.max_pixels = cfg.vision.max_pixels.min(max_pixels);
            eprintln!("[qwen36-node-vision] max_tokens={max_tokens} max_pixels={}", cfg.vision.max_pixels);
        }
        // 模型同目录的 mmproj-*.gguf 提供视觉塔；没有则节点保持纯文本。
        let model_directory = if model_path.is_dir() { Some(model_path) } else { model_path.parent() };
        let mmproj = model_directory.and_then(|directory| {
            std::fs::read_dir(directory).ok()?.filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("mmproj") && name.ends_with(".gguf")))
        });
        crate::runtime::validate_max_sequence_length("Qwen3.6", max_seq_len, cfg.max_position_embeddings)?;
        let weights = match MlxAffineSource::open(model_path) {
            Ok(source) => Qwen36NodeWeights::Mlx { weights: Qwen36Weights::new(source, cfg.clone()).map_err(|error| format!("加载 Qwen3.6 MLX affine 权重: {error}"))?, bytes: directory_bytes(model_path) },
            Err(mlx_error) => {
                let located = GgufReader::locate(model_path).map_err(|gguf_error| format!("Qwen3.6 权重既不是 MLX affine({mlx_error})也不是 GGUF({gguf_error})"))?;
                let reader = GgufReader::open(&located)?;
                reader.expect_metadata_str("general.architecture", "qwen35")?;
                reader.expect_metadata_u64("qwen35.embedding_length", cfg.hidden_size as u64)?;
                // GGUF 可能带 MTP block(block_count = num_layers + mtp_layers)，生成只执行前 num_layers 层。
                let block_count = reader.metadata_u64("qwen35.block_count")?;
                if block_count != cfg.num_layers as u64 && block_count != (cfg.num_layers + cfg.mtp_layers) as u64 {
                    return Err(format!("GGUF metadata qwen35.block_count={block_count}，期望 {} 或 {}", cfg.num_layers, cfg.num_layers + cfg.mtp_layers));
                }
                Qwen36NodeWeights::Gguf(reader)
            }
        };
        if !matches!(weights, Qwen36NodeWeights::Gguf(_)) && (want_mtp || mtp_draft_vocabulary.is_some() || dspark_directory.is_some()) {
            return Err("Qwen3.6 MLX affine Metal 当前不含 GGUF nextn/DSpark 权重，必须关闭 MTP、draft vocabulary 与 DSpark".to_owned());
        }
        if let Some(path) = &mmproj {
            let reader = GgufReader::open(path).map_err(|error| format!("打开 Qwen3.6 mmproj {}: {error}", path.display()))?;
            qwen36::validate_qwen36_mmproj_formats(&reader, &cfg).map_err(|error| format!("Qwen3.6 mmproj 格式预检失败: {error}"))?;
            eprintln!("[qwen36-node-vision] 检测到视觉权重 {}", path.display());
        }
        let (tokenizer, detokenizer) = match &weights {
            Qwen36NodeWeights::Gguf(weights) => (weights.bpe_tokenizer()?, weights.bpe_detokenizer()?),
            Qwen36NodeWeights::Mlx { .. } => crate::tokenizer::load_bpe_directory(model_path).map_err(|error| format!("加载 Qwen3.6 tokenizer: {error}"))?,
        };
        let context = Arc::new(MetalContext::new_default_with_replay(replay_enabled).map_err(|error| format!("MetalContext 初始化失败: {error}"))?);
        // 诊断模式:批折叠降到 1 op/command,拿 per-kernel 真实耗时(decode 会显著变慢)
        if std::env::var_os("ZLLM_QWEN36_PROFILE").is_some() {
            context.set_decode_batch_max_operations(1);
            context.set_deferred_batch_max_operations(1);
        }
        let (layers, final_norm, output_head) = match &weights {
            Qwen36NodeWeights::Gguf(weights) => {
                let layers = qwen36::prepare_qwen36_gguf_layers(context.as_ref(), weights, &cfg).map_err(|error| format!("准备 Qwen3.6 GGUF Metal 层: {error:?}"))?;
                let (norm, head) = qwen36::prepare_qwen36_gguf_output_quantized(context.as_ref(), weights, lm_head_quantization).map_err(|error| format!("准备 Qwen3.6 GGUF output: {error:?}"))?;
                (layers, norm, head)
            }
            Qwen36NodeWeights::Mlx { weights, .. } => {
                let layers = qwen36::prepare_qwen36_layers(context.as_ref(), weights).map_err(|error| format!("准备 Qwen3.6 MLX affine Metal 层: {error:?}"))?;
                let (norm, head) = qwen36::prepare_qwen36_output_parts_quantized(context.as_ref(), weights, lm_head_quantization).map_err(|error| format!("准备 Qwen3.6 MLX affine output: {error:?}"))?;
                (layers, norm, head)
            }
        };
        let draft_head = mtp_draft_vocabulary
            .map(|path| {
                let token_ids = load_draft_vocabulary(path, "qwen36", cfg.vocab_size, &cfg.eos_token_ids)?;
                let weights = weights.gguf().ok_or("Qwen3.6 MLX affine 不支持 GGUF draft vocabulary")?;
                let output_name = if weights.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
                let matrix = weights.read_matrix(output_name)?;
                let head = prepare_draft_head(context.as_ref(), crate::backend::LinearWeight::gguf(&matrix), cfg.vocab_size, cfg.hidden_size, token_ids).map_err(|error| format!("准备 Qwen3.6 FR-Spec draft head: {error:?}"))?;
                eprintln!("[qwen36-mtp-draft-head] vocabulary={} source={} format={}", head.token_ids().len(), path.display(), matrix.tensor_type.name());
                Ok::<_, String>(head)
            })
            .transpose()?;
        let mtp = if want_mtp { qwen36::prepare_qwen36_mtp(context.as_ref(), weights.gguf().ok_or("Qwen3.6 MTP 需要 GGUF 权重")?, &cfg).map_err(|error| format!("准备 Qwen3.6 MTP: {error:?}"))? } else { None };
        let dspark = match dspark_directory {
            Some(path) => {
                let started = std::time::Instant::now();
                let located = GgufReader::locate(path).map_err(|error| format!("定位 DSpark drafter {}: {error}", path.display()))?;
                let reader = GgufReader::open(&located).map_err(|error| format!("打开 DSpark drafter {}: {error}", located.display()))?;
                let runtime = Qwen36DsparkRuntime::load(context.as_ref(), reader, max_seq_len, dspark_draft_tokens).map_err(|error| format!("准备 DSpark drafter: {error:?}"))?;
                eprintln!("[qwen36-dspark] drafter={} block={} layers={} tokens={} wall={:.3}s", path.display(), runtime.spec.block_size, runtime.spec.layer_count, runtime.draft_tokens, started.elapsed().as_secs_f64());
                Some(runtime)
            }
            None => None,
        };
        // 文本路径用顺序 RoPE 表；多模态请求另建 M-RoPE / decode-delta 表。
        let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
        let attention_options = HybridAttentionOptions { precise_prefill: precise_gqa_prefill };
        let model_format = weights.format();
        Ok(Self { context, cfg, weights, layers, mtp, final_norm, output_head, draft_head, rope, mmproj, vision: OnceLock::new(), tokenizer, detokenizer, attention_options, kv_f16, max_seq_len, prefill_chunk_size, model_format, dspark })
    }

    fn runtime_with<'a>(&'a self, rope: &'a RopeTable) -> Qwen36Runtime<'a, crate::backend::metal::MetalContext> {
        Qwen36Runtime::new(&self.context, &self.cfg, &self.layers, rope, self.attention_options)
    }

    /// 模型目录存在 mmproj-*.gguf 即可接收图像请求；权重本身懒加载。
    pub fn vision_available(&self) -> bool {
        self.mmproj.is_some()
    }

    pub fn vision_initialized(&self) -> bool {
        self.vision.get().is_some()
    }

    /// 只物化跨请求长期驻留的视觉权重；请求图像 tensor 仍留在请求生命周期。
    pub fn ensure_vision(&self) -> Result<(), String> {
        self.vision_weights().map(|_| ())
    }

    /// 首次图像请求时 prepare 视觉塔并缓存；失败结果同样缓存。
    fn vision_weights(&self) -> Result<&Qwen36NodeVision, String> {
        self.vision
            .get_or_init(|| {
                let path = self.mmproj.as_deref().ok_or_else(|| "模型目录缺少 mmproj-*.gguf 视觉权重".to_owned())?;
                let started = std::time::Instant::now();
                let mmproj = GgufReader::open(path).map_err(|error| format!("打开 mmproj {}: {error}", path.display()))?;
                let patch = qwen36::prepare_qwen36_patch_embedding(self.context.as_ref(), &mmproj, &self.cfg).map_err(|error| format!("准备 Qwen3.6 patch embed: {error:?}"))?;
                let position_embedding = mmproj.read_tensor_f32("v.position_embd.weight").map_err(|error| format!("Qwen3.6 position embedding: {error}"))?;
                let layers = qwen36::prepare_qwen36_vision_layers(self.context.as_ref(), &mmproj, &self.cfg).map_err(|error| format!("准备 Qwen3.6 vision layers: {error:?}"))?;
                let merger = qwen36::prepare_qwen36_vision_merger(self.context.as_ref(), &mmproj, &self.cfg).map_err(|error| format!("准备 Qwen3.6 vision merger: {error:?}"))?;
                eprintln!("[qwen36-node-vision] mmproj prepare layers={} wall={:.3}s", layers.len(), started.elapsed().as_secs_f64());
                Ok(Qwen36NodeVision { patch, position_embedding, layers, merger })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// 图像预处理(smart resize + Pillow bicubic + patch 化)。
    pub fn preprocess_image(&self, image: &RgbImage) -> Result<qwen36::Qwen36Visual, String> {
        use crate::runtime::qwen3_vl::Qwen3VlImageProcessor;
        use crate::vision::ImageProcessor;
        let processor = Qwen3VlImageProcessor::new(&self.cfg.vision.to_qwen3vl())?;
        let tensor = processor.preprocess(image)?;
        Ok(qwen36::Qwen36Visual { kind: qwen36::Qwen36VisualKind::Image, tensor })
    }

    /// 已内联渲染占位符的 prompt 的多模态输入构造(tokenize + 区间扫描 + M-RoPE)。
    pub fn multimodal_input(&self, rendered: &str, visuals: &[qwen36::Qwen36Visual]) -> Result<qwen36::Qwen36MultimodalInput, String> {
        qwen36::qwen36_multimodal_input_rendered(&self.tokenizer, &self.cfg, rendered, visuals)
    }

    pub fn context(&self) -> &MetalContext {
        &self.context
    }

    pub fn context_handle(&self) -> Arc<MetalContext> {
        self.context.clone()
    }

    pub fn config(&self) -> &Qwen36Config {
        &self.cfg
    }

    pub fn kv_f16(&self) -> bool {
        self.kv_f16
    }

    pub fn mtp_available(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn model_bytes(&self) -> u64 {
        self.weights.bytes()
    }

    pub fn model_format(&self) -> &'static str {
        self.model_format
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn session_residency_bytes(&self) -> Result<usize, String> {
        let attention = AttentionSpec::Gqa(self.cfg.full_attention_spec());
        let spec = KvCacheSpec::from_attention(&attention).map_err(|error| format!("Qwen3.6 KV cache spec: {error}"))?;
        let mut layers: Vec<usize> = (0..self.cfg.num_layers).filter(|&layer| (layer + 1).is_multiple_of(self.cfg.full_attention_interval)).collect();
        if self.mtp.is_some() {
            layers.push(self.cfg.num_layers);
        }
        let logical_layers = self.cfg.num_layers + usize::from(self.mtp.is_some());
        let map = KvCacheLayerMap::from_cached_layers(logical_layers, layers.into_iter()).map_err(|error| format!("Qwen3.6 KV layer map: {error}"))?;
        let format = if self.kv_f16 { KvCacheFormat::F16 } else { KvCacheFormat::Int8 };
        let kv = KvCacheLayout::new_mapped(spec, format, map, self.max_seq_len, DEFAULT_GROUP_SIZE)?.total_bytes();
        let recurrent_layers = (0..self.cfg.num_layers).filter(|&layer| !(layer + 1).is_multiple_of(self.cfg.full_attention_interval)).count();
        let recurrent_per_layer = self.cfg.gated_delta_net_spec().conv_state_elements().saturating_add(self.cfg.gated_delta_net_spec().recurrent_elements()).saturating_mul(std::mem::size_of::<f32>());
        let recurrent = recurrent_layers.saturating_mul(recurrent_per_layer);
        // DSpark target K/V 属于每个会话，不能混入已在 load 时扣除的 drafter 权重。
        // 当前 Metal target tensor 为 F32；全 attention drafter 按 max_seq_len 整块预算。
        let dspark = self.dspark.as_ref().map_or(0, |dspark| {
            dspark
                .spec
                .layer_count
                .saturating_mul(2)
                .saturating_mul(self.max_seq_len)
                .saturating_mul(dspark.spec.kv_head_count)
                .saturating_mul(dspark.spec.head_dim)
                .saturating_mul(std::mem::size_of::<f32>())
                .saturating_add(dspark.spec.hidden_size.saturating_mul(std::mem::size_of::<f32>()))
        });
        Ok(kv.saturating_add(recurrent).saturating_add(dspark))
    }

    pub fn layer_count(&self) -> usize {
        self.cfg.num_layers
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.tokenize(text.as_bytes())
    }

    pub fn decode_bytes(&self, token: u32) -> Result<Vec<u8>, String> {
        self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("Qwen3.6 detokenize {token}: {error}"))
    }

    pub fn is_eos(&self, token: u32) -> bool {
        self.cfg.eos_token_ids.contains(&token)
    }

    fn new_cache(&self) -> Result<MetalKvCache, String> {
        let attention = AttentionSpec::Gqa(self.cfg.full_attention_spec());
        let cache_spec = KvCacheSpec::from_attention(&attention).map_err(|error| format!("Qwen3.6 KV cache spec: {error}"))?;
        // 每 full_attention_interval 层才有 1 层 FullAttention 需要 KV cache;
        // MTP 块(num_layers 位)是 FullAttention,投机解码时需要自己的 slot。
        let mut full_attention_layers: Vec<usize> = (0..self.cfg.num_layers).filter(|&layer| (layer + 1) % self.cfg.full_attention_interval == 0).collect();
        if self.mtp.is_some() {
            full_attention_layers.push(self.cfg.num_layers);
        }
        let logical_layers = self.cfg.num_layers + usize::from(self.mtp.is_some());
        let cache_layers = KvCacheLayerMap::from_cached_layers(logical_layers, full_attention_layers.into_iter()).map_err(|error| format!("Qwen3.6 KV cache layer map: {error}"))?;
        if self.kv_f16 { MetalKvCache::new_f16_mapped(&self.context, cache_spec, cache_layers, self.max_seq_len) } else { MetalKvCache::new_mapped(&self.context, cache_spec, cache_layers, self.max_seq_len) }
            .map_err(|error| format!("Qwen3.6 KV cache: {error}"))
    }

    pub fn prefill(&self, tokens: Vec<u32>) -> Result<Qwen36Sequence, String> {
        if tokens.is_empty() {
            return Err("Qwen3.6 prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.max_seq_len {
            return Err(format!("Qwen3.6 prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let cache = self.new_cache()?;
        let recurrent = GatedDeltaNetState::<MetalGatedDeltaNetStorage>::new(self.cfg.num_layers, self.cfg.gated_delta_net_spec()).map_err(|error| format!("Qwen3.6 DeltaNet state: {error:?}"))?;
        let mut sequence = Qwen36Sequence { cache, recurrent, hidden: self.context.tensor_zeros(1, 1), tokens: Vec::new(), rope_delta: 0, decode_rope: OnceCell::new(), dspark_cache: DsparkTargetCache::new(), dspark_last_aux: None };
        self.extend(&mut sequence, &tokens)?;
        Ok(sequence)
    }

    /// 多模态首轮 prefill：逐图 ViT encode → 视觉 embedding 读回 CPU →
    /// chunked prefill 按区间覆盖 embedding 行(区间可跨 chunk)。
    /// rope_delta != 0 时预建 decode 表，decode/续写/MTP 全部复用。
    pub fn prefill_visual(&self, input: &qwen36::Qwen36MultimodalInput, visuals: &[qwen36::Qwen36Visual]) -> Result<Qwen36Sequence, String> {
        if input.token_ids.is_empty() {
            return Err("Qwen3.6 prompt 不能为空".to_owned());
        }
        if input.token_ids.len() >= self.max_seq_len {
            return Err(format!("Qwen3.6 prompt {} tokens 超过 max_seq_len {}", input.token_ids.len(), self.max_seq_len));
        }
        let vision = self.vision_weights()?;
        let vision_started = std::time::Instant::now();
        let mut overlays = Vec::with_capacity(visuals.len());
        for (index, (range, visual)) in input.visual_ranges.iter().zip(visuals).enumerate() {
            let embedding = qwen36::qwen36_encode_image(self.context.as_ref(), &self.cfg, &vision.patch, &vision.position_embedding, &vision.layers, &vision.merger, &visual.tensor)
                .map_err(|error| format!("Qwen3.6 visual {index} encode: {error:?}"))?;
            // Metal 视觉链全程 F16;同步后读回 F32,f32→f16 回传无损。
            self.context.synchronize();
            let values = self.context.read_f16_to_f32(&embedding.buffer, embedding.len());
            let expected = range.len().checked_mul(self.cfg.hidden_size).ok_or("Qwen3.6 视觉 embedding 大小溢出")?;
            if values.len() != expected {
                return Err(format!("Qwen3.6 visual {index} embedding values={}，期望 {expected}", values.len()));
            }
            overlays.push(VisualOverlay { tokens: range.clone(), values });
        }
        eprintln!("[qwen36-node-vision] visuals={} encode={:.3}s tokens={} rope_delta={}", overlays.len(), vision_started.elapsed().as_secs_f64(), input.token_ids.len(), input.rope_delta);
        let prefill_rope = qwen36::qwen36_mrope_table(&self.cfg, &input.position_ids).map_err(|error| format!("Qwen3.6 M-RoPE 表: {error}"))?;
        let cache = self.new_cache()?;
        let recurrent = GatedDeltaNetState::<MetalGatedDeltaNetStorage>::new(self.cfg.num_layers, self.cfg.gated_delta_net_spec()).map_err(|error| format!("Qwen3.6 DeltaNet state: {error:?}"))?;
        let mut sequence =
            Qwen36Sequence { cache, recurrent, hidden: self.context.tensor_zeros(1, 1), tokens: Vec::new(), rope_delta: input.rope_delta, decode_rope: OnceCell::new(), dspark_cache: DsparkTargetCache::new(), dspark_last_aux: None };
        if input.rope_delta != 0 {
            // decode 后三轴重新一致,一张 rope_delta 常驻顺序表覆盖整个生成期
            let table = qwen36::qwen36_decode_rope_table(&self.cfg, self.max_seq_len - 1, input.rope_delta).map_err(|error| format!("Qwen3.6 decode rope: {error}"))?;
            let _ = sequence.decode_rope.set(table);
        }
        self.extend_with(&mut sequence, &input.token_ids, Some(&prefill_rope), &overlays)?;
        Ok(sequence)
    }

    /// 在已有 sequence 的 cache 与 DeltaNet state 上续写 suffix（前缀缓存 resume 路径）。
    pub fn extend(&self, sequence: &mut Qwen36Sequence, suffix: &[u32]) -> Result<(), String> {
        self.extend_with(sequence, suffix, None, &[])
    }

    /// chunked prefill 的共用实现。rope: 本次 extension 使用的 RoPE 表
    /// (多模态首轮传 M-RoPE 表)；None 时用 sequence 的 decode 表(多模态
    /// resume 续写)或 session 顺序表(纯文本)。
    fn extend_with(&self, sequence: &mut Qwen36Sequence, suffix: &[u32], rope: Option<&RopeTable>, overlays: &[VisualOverlay]) -> Result<(), String> {
        if suffix.is_empty() {
            return Ok(());
        }
        let end = sequence.tokens.len().checked_add(suffix.len()).ok_or("Qwen3.6 序列长度溢出")?;
        if end >= self.max_seq_len {
            return Err(format!("Qwen3.6 会话状态 {end} tokens 超过 max_seq_len {}", self.max_seq_len));
        }
        let offset = sequence.tokens.len();
        // rope 引用与 cache/recurrent 的可变借用按字段分离，closure 内共存
        let Qwen36Sequence { cache, recurrent, hidden, tokens, rope_delta: _, decode_rope, dspark_cache, dspark_last_aux } = sequence;
        let fallback: &RopeTable = decode_rope.get().unwrap_or(&self.rope);
        let rope = rope.unwrap_or(fallback);
        // DSpark drafter 在 prefill 期间同步预热 target KV cache(逐 chunk 消费)
        let dspark_plan = self.dspark_available().then(|| self.dspark_capture_plan()).transpose()?;
        crate::runtime::prefill::run_chunked_prefill(offset, suffix.len(), self.prefill_chunk_size, |range, position| {
            let chunk_offset = offset + range.start;
            let chunk = &suffix[range];
            let profile_prefill = std::env::var_os("ZLLM_QWEN36_PROFILE").is_some();
            if profile_prefill {
                self.context.reset_gpu_stats();
            }
            let started = std::time::Instant::now();
            crate::backend::BackendResources::begin_batch(self.context());
            let mut embedding = self.weights.embedding_rows(chunk, self.cfg.hidden_size, self.cfg.vocab_size)?;
            if !overlays.is_empty() {
                overlay_visual_rows(&mut embedding, self.cfg.hidden_size, chunk_offset, overlays)?;
            }
            let input = self.context.tensor_from_f32(&embedding, chunk.len(), self.cfg.hidden_size).map_err(|error| format!("上传 Qwen3.6 embedding: {error}"))?;
            let mut captures = Vec::new();
            let runtime = self.runtime_with(rope);
            let execution = runtime.at(cache, recurrent, position);
            let output = match &dspark_plan {
                Some(plan) => execution.capture(plan, &mut captures).prefill(input),
                None => execution.prefill(input),
            }
            .map_err(|error| format!("Qwen3.6 Metal prefill position={position}: {error:?}"))?;
            if dspark_plan.is_some() {
                let (rows, aux) = self.dspark_extend(dspark_cache, &captures, position)?;
                // 保留最后一位 aux 行:首轮 draft 的幂等重喂输入
                let last = self.context.select_row(&aux, rows - 1).map_err(|error| format!("DSpark 最后 aux 行: {error:?}"))?;
                *dspark_last_aux = Some((last, position + rows - 1));
            }
            *hidden = self.context.select_row(&output, chunk.len() - 1).map_err(|error| format!("Qwen3.6 选择最后 token: {error:?}"))?;
            tokens.extend_from_slice(chunk);
            if profile_prefill {
                self.context.synchronize();
                let gpu = self.context.gpu_stats();
                eprintln!(
                    "[qwen36-node-prefill] tokens={} wall={:.3}s gpu={:.3}s commands={} submit_wait={:.3}s gaps={:.3}s",
                    chunk.len(),
                    started.elapsed().as_secs_f64(),
                    gpu.seconds,
                    gpu.command_buffers,
                    gpu.submit_wait_seconds,
                    gpu.inter_command_gap_seconds
                );
                for operator in self.context.gpu_profile().into_iter().take(16) {
                    eprintln!("  qwen36 prefill gpu {:>9.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
                }
            }
            Ok(())
        })
    }

    pub fn decode_token(&self, sequence: &mut Qwen36Sequence, token: u32) -> Result<(), String> {
        let Qwen36Sequence { cache, recurrent, hidden, tokens, rope_delta: _, decode_rope, dspark_cache: _, dspark_last_aux: _ } = sequence;
        if tokens.len() >= self.max_seq_len {
            return Err(format!("Qwen3.6 会话状态 {} tokens 超过 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let embedding = self.weights.embedding_rows(&[token], self.cfg.hidden_size, self.cfg.vocab_size)?;
        let input = self.context.tensor_from_f32(&embedding, 1, self.cfg.hidden_size).map_err(|error| format!("上传 Qwen3.6 decode embedding: {error}"))?;
        let position = tokens.len();
        *hidden = self.runtime_with(decode_rope.get().unwrap_or(&self.rope)).at(cache, recurrent, position).decode(input).map_err(|error| format!("Qwen3.6 Metal decode position={position}: {error:?}"))?;
        tokens.push(token);
        Ok(())
    }

    pub fn token_output(&self, sequence: &Qwen36Sequence) -> Result<u32, String> {
        let normalized = self.context.gemma_rmsnorm_f32(&sequence.hidden, &self.final_norm, self.cfg.rms_norm_eps).map_err(|error| format!("Qwen3.6 output norm: {error:?}"))?;
        let logits = self.context.linear(&normalized, &self.output_head).map_err(|error| format!("Qwen3.6 output head: {error:?}"))?;
        self.sample_token(&logits)
    }

    /// 采样时排除图像/视频 placeholder token,多模态请求不会把它们当文字吐出。
    fn sample_token(&self, logits: &MetalTensor) -> Result<u32, String> {
        self.context.argmax_excluding(logits, &[self.cfg.image_token_id, self.cfg.video_token_id]).map_err(|error| format!("Qwen3.6 argmax: {error:?}"))
    }

    /// MTP 链式 draft(EAGLE 式自回归):首步吃 (待前向 token 的 embedding,
    /// 主干对前一 token 的 hidden),之后每步吃上一步候选 token 与 MTP 层自己
    /// 的 normed 输出;每步推进 MTP 自身 KV slot。返回 count 个候选 token。
    pub fn mtp_draft_chain(&self, sequence: &mut Qwen36Sequence, pending: u32, hidden_prev: &MetalTensor, count: usize) -> Result<Vec<u32>, String> {
        let mtp = self.mtp.as_ref().ok_or("Qwen3.6 MTP 权重未加载")?;
        let Qwen36Sequence { cache, recurrent: _, hidden: _, tokens: _, rope_delta: _, decode_rope, dspark_cache: _, dspark_last_aux: _ } = sequence;
        let rope = decode_rope.get().unwrap_or(&self.rope);
        let runtime = self.runtime_with(rope);
        let mut drafts = Vec::with_capacity(count);
        let mut token = pending;
        let mut hidden = hidden_prev.clone();
        for _ in 0..count {
            let mtp_position = cache.layer_len(self.cfg.num_layers);
            let embedding = self.weights.embedding_rows(&[token], self.cfg.hidden_size, self.cfg.vocab_size)?;
            let token_embedding = self.context.tensor_from_f32(&embedding, 1, self.cfg.hidden_size).map_err(|error| format!("上传 Qwen3.6 MTP embedding: {error}"))?;
            let (next, normed) = runtime
                .mtp_decode_then(mtp, cache, &token_embedding, &hidden, mtp_position, |normed| {
                    let token = if let Some(draft_head) = &self.draft_head {
                        normalized_draft_token_id(self.context.as_ref(), draft_head, normed, &[self.cfg.image_token_id, self.cfg.video_token_id])?
                    } else {
                        let logits = self.context.linear(normed, &self.output_head)?;
                        self.context.argmax_excluding(&logits, &[self.cfg.image_token_id, self.cfg.video_token_id])?
                    };
                    Ok((token, normed.clone()))
                })
                .map_err(|error| format!("Qwen3.6 MTP draft/head: {error:?}"))?;
            drafts.push(next);
            token = next;
            hidden = normed;
        }
        Ok(drafts)
    }

    /// 主干多行前向 + GDN 层 normed 输入捕获(MTP 链部分接受后的 GDN-only
    /// 回滚重放素材,与 DSpark 共用语义)。
    pub fn forward_rows_with_gdn_inputs(&self, sequence: &mut Qwen36Sequence, tokens: &[u32], gdn_inputs: &mut Vec<MetalTensor>) -> Result<MetalTensor, String> {
        self.forward_rows_capturing(sequence, tokens, None, Some(gdn_inputs))
    }

    /// 多行前向的共用实现。captures/gdn_inputs 是可选推理捕获 sink(DSpark
    /// tap hidden / GDN-only 重放素材),None 时零开销直通。
    fn forward_rows_capturing(
        &self,
        sequence: &mut Qwen36Sequence,
        tokens: &[u32],
        captures: Option<(&crate::runtime::speculative::HiddenStateCapturePlan, &mut Vec<MetalTensor>)>,
        gdn_inputs: Option<&mut Vec<MetalTensor>>,
    ) -> Result<MetalTensor, String> {
        let Qwen36Sequence { cache, recurrent, hidden: _, tokens: sequence_tokens, rope_delta: _, decode_rope, dspark_cache: _, dspark_last_aux: _ } = sequence;
        let position = sequence_tokens.len();
        if position + tokens.len() >= self.max_seq_len {
            return Err(format!("Qwen3.6 会话状态 {} tokens 超过 max_seq_len {}", position + tokens.len(), self.max_seq_len));
        }
        let embedding = self.weights.embedding_rows(tokens, self.cfg.hidden_size, self.cfg.vocab_size)?;
        let input = self.context.tensor_from_f32(&embedding, tokens.len(), self.cfg.hidden_size).map_err(|error| format!("上传 Qwen3.6 embedding: {error}"))?;
        let rope = decode_rope.get().unwrap_or(&self.rope);
        let runtime = self.runtime_with(rope);
        let mut execution = runtime.at(cache, recurrent, position);
        if let Some((plan, sink)) = captures {
            execution = execution.capture(plan, sink);
        }
        let output = match gdn_inputs {
            Some(sink) => execution.gdn_inputs(sink).prefill(input),
            None => execution.prefill(input),
        }
        .map_err(|error| format!("Qwen3.6 verify prefill position={position}: {error:?}"))?;
        sequence_tokens.extend_from_slice(tokens);
        Ok(output)
    }

    /// 多行 hidden 的逐行采样(DSpark verify)。norm 逐行归约 + 一次多行
    /// lm_head GEMM(权重读共享,5 行省 4 份 lm_head 读),逐行 argmax 统一
    /// 延迟提交后读回。F16 归约顺序与单行 decode 存在同性质分叉(多行
    /// verify 前向已接受,llama.cpp batched verify 同)。
    pub fn rows_token_output(&self, hidden: &MetalTensor) -> Result<Vec<u32>, String> {
        crate::backend::BackendResources::begin_batch(self.context());
        let normalized = self.context.gemma_rmsnorm_f32(hidden, &self.final_norm, self.cfg.rms_norm_eps).map_err(|error| format!("Qwen3.6 DSpark 行 norm: {error:?}"))?;
        let logits = self.context.linear(&normalized, &self.output_head).map_err(|error| format!("Qwen3.6 DSpark 行 logits: {error:?}"))?;
        let row_logits = (0..hidden.rows).map(|row| self.context.select_row(&logits, row).map_err(|error| format!("Qwen3.6 DSpark 行 {row} 切分: {error:?}"))).collect::<Result<Vec<_>, _>>()?;
        self.context.submit_batch();
        row_logits.iter().map(|logits| self.sample_token(logits)).collect()
    }

    /// DeltaNet state 的 CPU 侧快照(shared 内存直拷,reject 后整块恢复)。
    /// full attention 层没有 DeltaNet state,写空占位。
    pub fn snapshot_delta(&self, sequence: &Qwen36Sequence) -> Result<Vec<Vec<u8>>, String> {
        (0..self.cfg.num_layers)
            .map(|layer| match sequence.recurrent.layer_storage(layer) {
                Some(storage) => {
                    let conv = unsafe { std::slice::from_raw_parts(storage.conv_buffer().contents() as *const u8, storage.conv_buffer().length() as usize) };
                    let recurrent = unsafe { std::slice::from_raw_parts(storage.recurrent_buffer().contents() as *const u8, storage.recurrent_buffer().length() as usize) };
                    Ok([conv, recurrent].concat())
                }
                None => Ok(Vec::new()),
            })
            .collect()
    }

    pub fn restore_delta(&self, sequence: &mut Qwen36Sequence, snapshot: &[Vec<u8>], position: usize) -> Result<(), String> {
        let spec = self.cfg.gated_delta_net_spec();
        let conv_bytes = spec.conv_state_elements() * 4;
        let recurrent_bytes = spec.recurrent_elements() * 4;
        for layer in 0..self.cfg.num_layers {
            let bytes = snapshot.get(layer).ok_or("Qwen3.6 MTP 快照层数不足")?;
            if bytes.is_empty() {
                continue;
            }
            if bytes.len() != conv_bytes + recurrent_bytes {
                return Err("Qwen3.6 MTP 快照大小不符".to_owned());
            }
            let storage = sequence.recurrent.layer_storage(layer).ok_or(format!("Qwen3.6 MTP 恢复要求 DeltaNet L{layer} 已初始化"))?;
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), storage.conv_buffer().contents() as *mut u8, conv_bytes);
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(conv_bytes), storage.recurrent_buffer().contents() as *mut u8, recurrent_bytes);
            }
            sequence.recurrent.rewind_layer(layer, position).map_err(|error| format!("Qwen3.6 MTP 回滚 L{layer}: {error:?}"))?;
        }
        Ok(())
    }

    /// full attention 层的有效长度快照(reject 后恢复游标,行数据由重放覆盖)。
    pub fn snapshot_kv_lengths(&self, sequence: &Qwen36Sequence) -> Vec<usize> {
        (0..self.cfg.num_layers).map(|layer| sequence.cache.layer_len(layer)).collect()
    }

    pub fn restore_kv_lengths(&self, sequence: &mut Qwen36Sequence, lengths: &[usize]) -> Result<(), String> {
        // 只恢复有物理槽的 full attention 层(快照值非零即有槽)
        for (layer, len) in lengths.iter().enumerate() {
            if *len > 0 {
                sequence.cache.set_layer_len(layer, *len)?;
            }
        }
        Ok(())
    }

    pub fn dspark_available(&self) -> bool {
        self.dspark.is_some()
    }

    /// DSpark 会话可 draft 的前提:drafter 装配且 target cache 已预热
    /// (fresh prefill 会预热;terminal cache resume 的会话为空,退普通 decode)。
    pub fn dspark_ready(&self, sequence: &Qwen36Sequence) -> bool {
        match self.dspark.as_ref() {
            Some(dspark) => sequence.dspark_cache.warmed_layers() == dspark.spec.layer_count,
            None => false,
        }
    }

    fn dspark_capture_plan(&self) -> Result<crate::runtime::speculative::HiddenStateCapturePlan, String> {
        let dspark = self.dspark.as_ref().ok_or("Qwen3.6 DSpark 未装配")?;
        dspark.spec.capture_plan(self.cfg.num_layers).map_err(|error| format!("DSpark capture 计划: {error:?}"))
    }

    /// 多行前向并在 tap 边界捕获 hidden(DSpark verify 与重放共用)。
    pub fn forward_rows_with_capture(&self, sequence: &mut Qwen36Sequence, tokens: &[u32], captures: &mut Vec<MetalTensor>, gdn_inputs: &mut Vec<MetalTensor>) -> Result<MetalTensor, String> {
        let plan = self.dspark_capture_plan()?;
        let profile_verify = std::env::var_os("ZLLM_QWEN36_PROFILE").is_some();
        let started = std::time::Instant::now();
        if profile_verify {
            self.context.reset_gpu_stats();
        }
        let output = self.forward_rows_capturing(sequence, tokens, Some((&plan, captures)), Some(gdn_inputs))?;
        if profile_verify {
            self.context.synchronize();
            let gpu = self.context.gpu_stats();
            eprintln!("[qwen36-dspark-verify] rows={} wall={:.3}s gpu={:.3}s commands={}", tokens.len(), started.elapsed().as_secs_f32(), gpu.seconds, gpu.command_buffers);
            for operator in self.context.gpu_profile().into_iter().take(8) {
                eprintln!("  dspark verify gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
            }
        }
        Ok(output)
    }

    /// tap captures → fc 投影+norm → 扩展 drafter target cache。
    /// 返回 (推进行数, aux hidden)。
    pub fn dspark_extend(&self, sequence_dspark_cache: &mut DsparkTargetCache<MetalTensor>, captures: &[MetalTensor], position: usize) -> Result<(usize, MetalTensor), String> {
        let dspark = self.dspark.as_ref().ok_or("Qwen3.6 DSpark 未装配")?;
        let rows = captures.first().map(|tensor| tensor.rows).unwrap_or(0);
        if rows == 0 || captures.len() != dspark.capture_count() || captures.iter().any(|tensor| tensor.rows != rows) {
            return Err(format!("DSpark captures={} rows={} 形状不符(期望 {} 份等行)", captures.len(), rows, dspark.capture_count()));
        }
        let aux = dspark.project_captures(&self.context, captures).map_err(|error| format!("DSpark capture 投影: {error:?}"))?;
        dspark.extend_cache(&self.context, sequence_dspark_cache, &aux, position).map_err(|error| format!("DSpark cache 扩展: {error:?}"))?;
        Ok((rows, aux))
    }

    /// captures → fc 投影 → draft(draft 内把接受行消费进 target cache)。
    pub fn dspark_draft_with_captures(&self, sequence: &mut Qwen36Sequence, anchor: u32, captures: &[MetalTensor], aux_position: usize, block_position: usize) -> Result<crate::runtime::speculative::SpeculativeBlock, String> {
        let dspark = self.dspark.as_ref().ok_or("Qwen3.6 DSpark 未装配")?;
        let rows = captures.first().map(|tensor| tensor.rows).unwrap_or(0);
        if rows == 0 || captures.len() != dspark.capture_count() || captures.iter().any(|tensor| tensor.rows != rows) {
            return Err(format!("DSpark captures={} rows={} 形状不符(期望 {} 份等行)", captures.len(), rows, dspark.capture_count()));
        }
        let aux = dspark.project_captures(&self.context, captures).map_err(|error| format!("DSpark capture 投影: {error:?}"))?;
        self.dspark_draft(sequence, anchor, Some((&aux, aux_position)), block_position)
    }

    /// 部分接受后的 GDN-only 重放:从快照恢复的 state 起重放前 rows 行的
    /// DeltaNet 链(读 GDN 层 normed 输入,不重跑 attention/MLP)。
    pub fn replay_gdn(&self, sequence: &mut Qwen36Sequence, gdn_inputs: &[MetalTensor], rows: usize, position: usize) -> Result<(), String> {
        let Qwen36Sequence { cache: _, recurrent, hidden: _, tokens: _, rope_delta: _, decode_rope, dspark_cache: _, dspark_last_aux: _ } = sequence;
        let runtime = self.runtime_with(decode_rope.get().unwrap_or(&self.rope));
        let layer_ids = runtime.delta_layer_ids();
        if gdn_inputs.len() != layer_ids.len() {
            return Err(format!("DSpark GDN 重放素材 {} 层,期望 {}", gdn_inputs.len(), layer_ids.len()));
        }
        use crate::backend::SegmentedTensorBackend as _;
        // 零 z 占位(全部层共用):z 门控只影响层输出,重放丢弃输出只推进 state。
        let zero_z = self.context.tensor_pooled("dspark_replay_zero_z", rows, self.cfg.gated_delta_net_spec().value_dim());
        crate::backend::BackendResources::begin_batch(self.context());
        for (input, &layer) in gdn_inputs.iter().zip(&layer_ids) {
            if input.rows < rows {
                return Err(format!("DSpark GDN 重放素材 rows={} 不足 {rows}", input.rows));
            }
            let slice = if input.rows == rows { input.clone() } else { self.context.slice_token_rows(input, 0, rows).map_err(|error| format!("DSpark GDN 重放切片 L{layer}: {error:?}"))? };
            runtime.replay_delta_layer(recurrent, layer, &slice, position, &zero_z).map_err(|error| format!("DSpark GDN 重放 L{layer}: {error:?}"))?;
        }
        self.context.submit_batch();
        Ok(())
    }

    /// full attention 层的 KV 游标推进到 length(行已由 verify 写入)。
    pub fn advance_kv_lengths(&self, sequence: &mut Qwen36Sequence, length: usize) -> Result<(), String> {
        for layer in 0..self.cfg.num_layers {
            if sequence.cache.layer_len(layer) > 0 {
                sequence.cache.set_layer_len(layer, length)?;
            }
        }
        Ok(())
    }

    /// 单请求 DSpark draft 整块。`aux` 是上一轮接受前缀的 capture 投影输入
    /// (None = cache 已是当前位,prefill 刚预热完的首轮)。
    pub fn dspark_draft(&self, sequence: &mut Qwen36Sequence, anchor: u32, aux: Option<(&MetalTensor, usize)>, block_position: usize) -> Result<crate::runtime::speculative::SpeculativeBlock, String> {
        let dspark = self.dspark.as_ref().ok_or("Qwen3.6 DSpark 未装配")?;
        let weights = self.weights.gguf().ok_or("Qwen3.6 DSpark 需要 GGUF 主模型")?;
        let Qwen36Sequence { dspark_cache, dspark_last_aux, .. } = sequence;
        let (aux_hidden, target_position) = match aux {
            Some((hidden, position)) => (hidden, position),
            // 首轮:幂等重喂 prefill 留下的最后 aux 行(cache 不增长)
            None => {
                let (last, position) = dspark_last_aux.as_ref().ok_or("Qwen3.6 DSpark 首轮缺少 prefill aux 行")?;
                (last, *position)
            }
        };
        dspark.draft_block(&self.context, weights, &self.output_head, dspark_cache, anchor, aux_hidden, target_position, block_position, self.cfg.vocab_size).map_err(|error| format!("DSpark draft: {error:?}"))
    }
}

/// 终点会话：KV cache + DeltaNet state + hidden 的完整快照。
/// resident 复用时零开销；换出/落盘按字节编码到 fjall。
pub struct Qwen36TerminalState {
    pub sequence: Qwen36Sequence,
    pub pending: Vec<u32>,
    pub info: crate::kv_cache::terminal_cache::TerminalInfo,
}

/// decode 端重建 cache/state 所需的固定资源。
pub struct Qwen36SnapshotResources {
    pub context: Arc<MetalContext>,
    pub cfg: Qwen36Config,
    pub kv_f16: bool,
    pub max_seq_len: usize,
    /// MTP 开启时快照含第 num_layers 个 KV slot,resume 后 draft 从其长度续起。
    pub mtp: bool,
}

fn read_u64(cursor: &mut &[u8]) -> Result<u64, String> {
    if cursor.len() < 8 {
        return Err("Qwen3.6 快照字节不足".to_owned());
    }
    let value = u64::from_le_bytes(cursor[..8].try_into().expect("u64 字节"));
    *cursor = &cursor[8..];
    Ok(value)
}

/// GPU 已同步(生成结束，argmax 等待过队列)，Shared 内存直读。
fn buffer_bytes(buffer: &crate::backend::metal::api::Buffer) -> &[u8] {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, buffer.length() as usize) }
}

impl crate::kv_cache::terminal_cache::TerminalSnapshot for Qwen36TerminalState {
    type Resources = Qwen36SnapshotResources;

    fn encode(&self) -> Result<Vec<u8>, String> {
        let mut writer = crate::kv_cache::terminal_cache::SnapshotWriter::new();
        // v2:在 v1 基础上追加 rope_delta,多模态会话 resume 后 decode 位置才能对齐
        writer.u32(2);
        writer.u32s(&self.sequence.tokens)?;
        writer.u32s(&self.pending)?;
        writer.i64(self.sequence.rope_delta);
        let mut buffer = writer.into_inner();
        let hidden = &self.sequence.hidden;
        let (dtype_code, element_bytes) = match hidden.dtype {
            MetalTensorDType::F16 => (0u32, 2usize),
            MetalTensorDType::Bf16 => (1u32, 2),
            MetalTensorDType::F32 => (2u32, 4),
        };
        buffer.extend(&dtype_code.to_le_bytes());
        buffer.extend(&(hidden.rows as u32).to_le_bytes());
        buffer.extend(&(hidden.cols as u32).to_le_bytes());
        let hidden_bytes = hidden.rows.checked_mul(hidden.cols).and_then(|count| count.checked_mul(element_bytes)).ok_or("Qwen3.6 快照 hidden 大小溢出")?;
        buffer.extend_from_slice(&buffer_bytes(&hidden.buffer)[..hidden_bytes]);
        for layer in 0..self.sequence.recurrent.layer_count() {
            match self.sequence.recurrent.layer_storage(layer) {
                Some(storage) => {
                    buffer.extend(&storage.conv_buffer().length().to_le_bytes());
                    buffer.extend_from_slice(buffer_bytes(storage.conv_buffer()));
                    buffer.extend(&storage.recurrent_buffer().length().to_le_bytes());
                    buffer.extend_from_slice(buffer_bytes(storage.recurrent_buffer()));
                }
                // full attention 层没有 DeltaNet state，写零长度占位。
                None => buffer.extend(&[0u8; 16]),
            }
        }
        let cache = &self.sequence.cache;
        let columns = cache.gqa_columns()?;
        let groups = cache.gqa_groups_per_token()?;
        let kv_f16 = matches!(cache.format(), crate::kv_cache::KvCacheFormat::F16);
        let kv_bytes = buffer_bytes(cache.buffer());
        // MTP slot 的 KV 一并快照,draft resume 后从其长度续起
        let kv_layer_end = self.sequence.recurrent.layer_count() + usize::from(cache.layer_len(self.sequence.recurrent.layer_count()) > 0);
        for layer in 0..kv_layer_end {
            let len = cache.layer_len(layer);
            buffer.extend(&(len as u64).to_le_bytes());
            if len == 0 {
                continue;
            }
            let segments: Vec<(usize, usize)> = if kv_f16 {
                let row_bytes = len.checked_mul(columns).and_then(|count| count.checked_mul(2)).ok_or("Qwen3.6 快照 KV 大小溢出")?;
                vec![(cache.layer_gqa_key_offset(layer)?, row_bytes), (cache.layer_gqa_value_offset(layer)?, row_bytes)]
            } else {
                let codes = len.checked_mul(columns).ok_or("Qwen3.6 快照 KV 大小溢出")?;
                let scales = len.checked_mul(groups).and_then(|count| count.checked_mul(2)).ok_or("Qwen3.6 快照 KV 大小溢出")?;
                vec![(cache.layer_gqa_key_offset(layer)?, codes), (cache.layer_gqa_key_scale_offset(layer)?, scales), (cache.layer_gqa_value_offset(layer)?, codes), (cache.layer_gqa_value_scale_offset(layer)?, scales)]
            };
            for (offset, bytes) in segments {
                let end = offset.checked_add(bytes).ok_or("Qwen3.6 快照 KV 偏移溢出")?;
                if end > kv_bytes.len() {
                    return Err(format!("Qwen3.6 快照 L{layer} 段越界 offset={offset} bytes={bytes}"));
                }
                buffer.extend_from_slice(&kv_bytes[offset..end]);
            }
        }
        Ok(buffer)
    }

    fn decode(bytes: &[u8], resources: &Self::Resources) -> Result<Self, String> {
        let mut reader = crate::kv_cache::terminal_cache::SnapshotReader::new(bytes);
        if reader.u32("Qwen3.6 版本")? != 2 {
            return Err("Qwen3.6 快照版本不支持(v1 已失效,可清空 node cache_directory 重来)".to_owned());
        }
        let tokens = reader.u32s("Qwen3.6 tokens")?;
        let pending = reader.u32s("Qwen3.6 pending")?;
        let rope_delta = reader.i64("Qwen3.6 rope_delta")?;
        let ctx: &MetalContext = &resources.context;
        let dtype_code = reader.u32("Qwen3.6 hidden dtype")?;
        let rows = reader.u32("Qwen3.6 hidden rows")? as usize;
        let cols = reader.u32("Qwen3.6 hidden cols")? as usize;
        let element_bytes = match dtype_code {
            0 | 1 => 2usize,
            2 => 4,
            _ => return Err(format!("Qwen3.6 快照 hidden dtype={dtype_code} 未知")),
        };
        let hidden_count = rows.checked_mul(cols).and_then(|count| count.checked_mul(element_bytes)).ok_or("Qwen3.6 快照 hidden 大小溢出")?;
        let hidden_bits = reader.take(hidden_count, "Qwen3.6 hidden")?;
        let hidden = match dtype_code {
            0 => ctx.tensor_from_f16_bits(hidden_bits, rows, cols)?,
            1 => ctx.tensor_from_bf16_bits(hidden_bits, rows, cols)?,
            _ => {
                let values: Vec<f32> = hidden_bits.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 字节"))).collect();
                ctx.tensor_from_f32(&values, rows, cols)?
            }
        };
        let mut cursor = reader.remaining();
        let spec = resources.cfg.gated_delta_net_spec();
        let conv_bytes = spec.conv_state_elements().checked_mul(4).ok_or("Qwen3.6 快照 conv 大小溢出")?;
        let recurrent_bytes = spec.recurrent_elements().checked_mul(4).ok_or("Qwen3.6 快照 recurrent 大小溢出")?;
        let mut recurrent = GatedDeltaNetState::<MetalGatedDeltaNetStorage>::new(resources.cfg.num_layers, spec).map_err(|error| format!("Qwen3.6 快照重建 DeltaNet: {error:?}"))?;
        let read_storage_buffer = |cursor: &mut &[u8], expected: usize| -> Result<Option<crate::backend::metal::api::Buffer>, String> {
            let len = read_u64(cursor)? as usize;
            if len == 0 {
                return Ok(None);
            }
            if len != expected || cursor.len() < len {
                return Err(format!("Qwen3.6 快照 DeltaNet buffer 长度 {len} 与期望 {expected} 不符或字节不足"));
            }
            let target = ctx.shared_buffer_zeros(len);
            unsafe { std::ptr::copy_nonoverlapping(cursor.as_ptr(), target.contents() as *mut u8, len) };
            *cursor = &cursor[len..];
            Ok(Some(target))
        };
        for layer in 0..resources.cfg.num_layers {
            let full_attention = (layer + 1).is_multiple_of(resources.cfg.full_attention_interval);
            let conv = read_storage_buffer(&mut cursor, conv_bytes)?;
            let recurrent_layer = read_storage_buffer(&mut cursor, recurrent_bytes)?;
            match (conv, recurrent_layer, full_attention) {
                // full attention 层两个 buffer 都是零长度占位，保持懒分配的 None 槽。
                (None, None, true) => {}
                (Some(conv), Some(recurrent_layer), false) => {
                    recurrent.restore_layer(layer, tokens.len(), MetalGatedDeltaNetStorage::from_buffers(conv, recurrent_layer)).map_err(|error| format!("Qwen3.6 快照恢复 DeltaNet L{layer}: {error:?}"))?;
                }
                _ => return Err(format!("Qwen3.6 快照 L{layer} DeltaNet 段与层类型不符")),
            }
        }
        let attention = AttentionSpec::Gqa(resources.cfg.full_attention_spec());
        let cache_spec = KvCacheSpec::from_attention(&attention).map_err(|error| format!("Qwen3.6 快照 KV spec: {error}"))?;
        let mut full_attention_layers: Vec<usize> = (0..resources.cfg.num_layers).filter(|&layer| (layer + 1) % resources.cfg.full_attention_interval == 0).collect();
        if resources.mtp {
            full_attention_layers.push(resources.cfg.num_layers);
        }
        let logical_layers = resources.cfg.num_layers + usize::from(resources.mtp);
        let cache_layers = KvCacheLayerMap::from_cached_layers(logical_layers, full_attention_layers.into_iter()).map_err(|error| format!("Qwen3.6 快照 KV layer map: {error}"))?;
        let mut cache = if resources.kv_f16 { MetalKvCache::new_f16_mapped(ctx, cache_spec, cache_layers, resources.max_seq_len) } else { MetalKvCache::new_mapped(ctx, cache_spec, cache_layers, resources.max_seq_len) }
            .map_err(|error| format!("Qwen3.6 快照重建 cache: {error}"))?;
        let columns = cache.gqa_columns()?;
        let groups = cache.gqa_groups_per_token()?;
        let kv_f16 = matches!(cache.format(), crate::kv_cache::KvCacheFormat::F16);
        let kv_target = cache.buffer().contents() as *mut u8;
        let kv_layer_end = resources.cfg.num_layers + usize::from(resources.mtp);
        for layer in 0..kv_layer_end {
            let len = read_u64(&mut cursor)? as usize;
            if len == 0 {
                continue;
            }
            if len > cache.capacity() {
                return Err(format!("Qwen3.6 快照 L{layer} 长度 {len} 超过 capacity {}", cache.capacity()));
            }
            let segments: Vec<(usize, usize)> = if kv_f16 {
                let row_bytes = len.checked_mul(columns).and_then(|count| count.checked_mul(2)).ok_or("Qwen3.6 快照 KV 大小溢出")?;
                vec![(cache.layer_gqa_key_offset(layer)?, row_bytes), (cache.layer_gqa_value_offset(layer)?, row_bytes)]
            } else {
                let codes = len.checked_mul(columns).ok_or("Qwen3.6 快照 KV 大小溢出")?;
                let scales = len.checked_mul(groups).and_then(|count| count.checked_mul(2)).ok_or("Qwen3.6 快照 KV 大小溢出")?;
                vec![(cache.layer_gqa_key_offset(layer)?, codes), (cache.layer_gqa_key_scale_offset(layer)?, scales), (cache.layer_gqa_value_offset(layer)?, codes), (cache.layer_gqa_value_scale_offset(layer)?, scales)]
            };
            for (offset, bytes) in segments {
                if cursor.len() < bytes {
                    return Err(format!("Qwen3.6 快照 L{layer} 段字节不足"));
                }
                unsafe { std::ptr::copy_nonoverlapping(cursor.as_ptr(), kv_target.add(offset), bytes) };
                cursor = &cursor[bytes..];
            }
            cache.set_layer_len(layer, len)?;
        }
        let decode_rope = OnceCell::new();
        if rope_delta != 0 {
            // 多模态会话 resume:decode/续写用 rope_delta 偏移的常驻表
            let table = qwen36::qwen36_decode_rope_table(&resources.cfg, resources.max_seq_len - 1, rope_delta).map_err(|error| format!("Qwen3.6 快照 decode rope: {error}"))?;
            let _ = decode_rope.set(table);
        }
        Ok(Self {
            sequence: Qwen36Sequence { cache, recurrent, hidden, tokens, rope_delta, decode_rope, dspark_cache: DsparkTargetCache::new(), dspark_last_aux: None },
            pending,
            info: crate::kv_cache::terminal_cache::TerminalInfo::default(),
        })
    }

    fn terminal_tokens(&self) -> &[u32] {
        &self.sequence.tokens
    }

    fn info(&self) -> &crate::kv_cache::terminal_cache::TerminalInfo {
        &self.info
    }

    fn set_info(&mut self, info: crate::kv_cache::terminal_cache::TerminalInfo) {
        self.info = info;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_rows_span_chunk_boundaries() {
        // chunk 覆盖 token [2,4):第一个 overlay 区间 [1,3) 跨进来,
        // 只有交集行被替换;第二个 overlay 无交集不动。
        let mut embedding = vec![0.0f32; 2 * 2];
        let overlays = [VisualOverlay { tokens: 1..3, values: vec![1.0, 1.0, 2.0, 2.0] }, VisualOverlay { tokens: 5..6, values: vec![9.0; 2] }];
        overlay_visual_rows(&mut embedding, 2, 2, &overlays).unwrap();
        assert_eq!(embedding, vec![2.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn overlay_replaces_full_chunk_rows() {
        // overlay 完整覆盖 chunk [0,2),hidden=1,两行各自替换
        let mut embedding = vec![0.0f32; 2];
        let overlays = [VisualOverlay { tokens: 0..2, values: vec![7.0, 8.0] }];
        overlay_visual_rows(&mut embedding, 1, 0, &overlays).unwrap();
        assert_eq!(embedding, vec![7.0, 8.0]);
    }
}
