use std::path::{Path, PathBuf};

use half::bf16;

use crate::{
    backend::metal::{MetalContext, MetalKvCache, MetalTensor, MetalWeight},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::gemma4::{
        self, Gemma4, Gemma4RopeTables, gemma4_decode_round, gemma4_decode_round_deferred, gemma4_embedding_rows, gemma4_last_token_output, gemma4_per_layer_embedding_rows, gemma4_prefill_hidden, gemma4_prefill_hidden_with_visibility,
        gemma4_token_output,
        multimodal::{Gemma4MultimodalInput, Gemma4MultimodalModel, gemma4_multimodal_embedding, gemma4_multimodal_input_from_parts, prepare_gemma4_multimodal_model},
        prepare_gemma4_layers, prepare_gemma4_per_layer_model,
    },
    tokenizer::{Detokenizer, Tokenizer},
    weight::model::gemma4::Gemma4Weights,
};

use super::metal::{gemma4_metal_per_layer_inputs, prepare_gemma4_metal_output_head};

/// 设备端 embedding gather 来源:tied lm_head 是常驻 Q4_K 时零拷贝复用。
pub struct Gemma4EmbeddingSource {
    blob: crate::backend::metal::api::Buffer,
    row_bytes: usize,
}

/// 一次已提交未读回的输出步:final norm + lm_head + argmax 写入设备 id buffer,
/// `command` 是精确等待点(queue FIFO ⇒ 它完成即此前工作全部完成)。
pub struct Gemma4PendingToken {
    /// 该 token 的 id 在 readback 中的字节偏移(每位置唯一,不会被覆写)。
    id_offset: u64,
    command: crate::backend::metal::api::CommandBuffer,
    profiles_through: usize,
    /// 该 token 在 sequence.tokens 的占位下标(submit_step 消费时登记)。
    placeholder: Option<usize>,
}

pub struct Gemma4Sequence {
    pub cache: MetalKvCache,
    pub hidden: MetalTensor,
    pub tokens: Vec<u32>,
}

impl Gemma4Sequence {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

pub struct Gemma4MetalSession {
    pub(crate) context: std::sync::Arc<MetalContext>,
    model: Gemma4,
    weights: Gemma4Weights,
    layers: Vec<gemma4::Gemma4Layer<MetalWeight>>,
    per_layer_model: Option<gemma4::Gemma4PerLayerModel<MetalWeight>>,
    output_head: gemma4::Gemma4OutputHead<MetalWeight>,
    rope: Gemma4RopeTables,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    /// GGUF 自带的 Jinja chat template；非 GGUF 来源为 None，走硬编码模板。
    chat_template: Option<crate::runtime::chat_template::ChatTemplate>,
    embedding_scale: f32,
    max_seq_len: usize,
    prefill_chunk_size: usize,
    model_bytes: u64,
    model_format: String,
    /// argmax 读回区:max_seq_len+1 个 u32,按 token 位置写入、永不覆写
    /// (2-slot 交替存在实测竞态,见 minicpm5 竞态定位记录)。
    token_readback: crate::backend::metal::api::Buffer,
    /// 设备端 embedding gather 来源;None(无 PLE 之外的原因:独立 output.weight、
    /// lm_head 再量化)时异步流水线不可用,回落同步 decode_token。
    embedding_source: Option<Gemma4EmbeddingSource>,
    /// E4B 的 Q5_K per-layer token embedding；只给设备闭环重放使用。
    per_layer_embedding_source: Option<Gemma4EmbeddingSource>,
    /// 双缓冲命令重放引擎(首个请求惰性录制);GGUF Q4_K tied lm_head 接管 decode。
    replay: Option<crate::runtime::gemma4::metal_replay::Gemma4ReplayEngine>,
    /// 官方 MTP 投机头(惰性);启用时 decode 走 draft/verify 循环而非单步重放。
    mtp: Option<crate::runtime::gemma4::metal_mtp::Gemma4MtpModel>,
    /// MTP draft 的命令重放(串行单份表);draft 步 44→17.6ms。
    mtp_replay: Option<crate::runtime::gemma4::metal_mtp::Gemma4MtpReplay>,
    /// MTP verify 的命令重放(K+1 行多行 position 原语)。
    verify_replay: Option<crate::runtime::gemma4::metal_replay::Gemma4VerifyReplay>,
    /// config.execution.mtp_weights;None 即纯文本部署不受影响。
    mtp_weights: Option<PathBuf>,
    mtp_draft_tokens: usize,
    replay_decode: bool,
    /// 视觉塔/投影权重,首个图文请求时惰性加载(safetensors checkpoint 可能不含
    /// vision_embedder 张量,启动期加载会让纯文本部署直接失败)。
    multimodal_model: Option<Gemma4MultimodalModel<MetalWeight>>,
    accepts_images: bool,
}

impl Gemma4MetalSession {
    pub fn load(model_path: &Path, max_seq_len: usize, prefill_chunk_size: usize, lm_head_quantization: crate::weight::LmHeadQuantization, mtp_weights: Option<PathBuf>, mtp_draft_tokens: usize, replay_decode: bool) -> Result<Self, String> {
        let gguf = Gemma4Weights::is_gguf(model_path);
        let mlx_affine = !gguf && Gemma4Weights::is_mlx_affine(model_path)?;
        let model = Gemma4::new(Gemma4Weights::select_config(model_path)?).map_err(|error| format!("Gemma4 规格无效: {error:?}"))?;
        let cfg = model.config();
        crate::runtime::validate_max_sequence_length("Gemma4", max_seq_len, cfg.max_position_embeddings)?;
        let weights = Gemma4Weights::open(model_path, cfg.clone())?;
        let accepts_images = weights.has_vision_weights();
        // GGUF 自带 tokenizer / chat template；其余来源读权重目录的 tokenizer.json。
        let (tokenizer, detokenizer, chat_template) = if let Some(reader) = weights.gguf_reader() {
            let tokenizer = reader.bpe_tokenizer().map_err(|error| format!("Gemma4 GGUF tokenizer: {error}"))?;
            let detokenizer = reader.bpe_detokenizer().map_err(|error| format!("Gemma4 GGUF detokenizer: {error}"))?;
            let chat_template = reader
                .metadata("tokenizer.chat_template")
                .and_then(crate::weight::container::gguf::GgufValue::as_str)
                .map(|template| {
                    let mut compiled = crate::runtime::chat_template::ChatTemplate::new(template).map_err(|error| format!("Gemma4 GGUF chat template: {error}"))?;
                    compiled.set_special_tokens("<bos>", "<eos>");
                    Ok::<_, String>(compiled)
                })
                .transpose()?;
            (tokenizer, detokenizer, chat_template)
        } else {
            let tokenizer_path = model_path.join("tokenizer.json");
            let tokenizer = Tokenizer::new(&tokenizer_path).map_err(|error| format!("Gemma4 tokenizer: {error}"))?;
            let detokenizer = Detokenizer::load(&tokenizer_path).map_err(|error| format!("Gemma4 detokenizer: {error}"))?;
            (tokenizer, detokenizer, None)
        };
        let context = std::sync::Arc::new(MetalContext::new_default().map_err(|error| format!("MetalContext 初始化失败: {error}"))?);
        let backend = context.as_ref();
        let rope = Gemma4RopeTables::new(cfg, max_seq_len).map_err(|error| format!("Gemma4 RoPE: {error:?}"))?;
        let layers = prepare_gemma4_layers(backend, &model, &weights).map_err(|error| format!("准备 Gemma4 Metal 层: {error:?}"))?;
        let per_layer_model = prepare_gemma4_per_layer_model(backend, cfg, &weights).map_err(|error| format!("准备 Gemma4 per-layer model: {error:?}"))?;
        let output_head = prepare_gemma4_metal_output_head(backend, cfg, &weights, lm_head_quantization)?;
        let embedding_scale = bf16::from_f32(cfg.embedding_scale()).to_f32();
        let model_bytes = if model_path.is_file() { model_path.metadata().map(|metadata| metadata.len()).unwrap_or(0) } else { directory_bytes(model_path) };
        let model_format = if gguf {
            "gguf"
        } else if mlx_affine {
            "mlx-affine"
        } else {
            "safetensors"
        }
        .to_owned();
        let token_readback = context.shared_buffer_uninit((max_seq_len + 1) * std::mem::size_of::<u32>());
        // tied lm_head 常驻 Q4_K 时,decode 的 embedding 直接在设备端从该 blob 抠行,
        // 消掉每 token 的 GPU→CPU id 同步 + CPU dequant + 上传。
        let embedding_source = match output_head.lm_head() {
            MetalWeight::Gguf { blob, tensor_type: 12, row_bytes, rows, cols } if *rows == cfg.vocab_size && *cols == cfg.hidden_size => Some(Gemma4EmbeddingSource { blob: blob.clone(), row_bytes: *row_bytes }),
            _ => None,
        };
        let per_layer_embedding_source = weights
            .load_per_layer_embedding_gguf()?
            .map(|matrix| {
                if matrix.tensor_type.0 != 13 {
                    return Err(format!("Gemma4 Metal 重放暂只支持 Q5_K per-layer embedding，实际 {}", matrix.tensor_type.name()));
                }
                let resident = MetalWeight::allocate_gguf(backend, &matrix, matrix.rows, matrix.columns)?;
                resident.fill_gguf(&matrix)?;
                match resident {
                    MetalWeight::Gguf { blob, row_bytes, .. } => Ok(Gemma4EmbeddingSource { blob, row_bytes }),
                    _ => unreachable!("allocate_gguf 必然返回 GGUF"),
                }
            })
            .transpose()?;
        Ok(Self {
            context,
            model,
            weights,
            layers,
            per_layer_model,
            output_head,
            rope,
            tokenizer,
            detokenizer,
            chat_template,
            embedding_scale,
            max_seq_len,
            prefill_chunk_size,
            model_bytes,
            model_format,
            token_readback,
            embedding_source,
            per_layer_embedding_source,
            multimodal_model: None,
            accepts_images,
            replay: None,
            mtp: None,
            mtp_replay: None,
            verify_replay: None,
            mtp_weights,
            mtp_draft_tokens,
            replay_decode,
        })
    }

    pub fn context(&self) -> &MetalContext {
        &self.context
    }

    pub fn context_handle(&self) -> std::sync::Arc<MetalContext> {
        self.context.clone()
    }

    pub fn hybrid_gqa_spec(&self) -> crate::attention::gqa::HybridGqaSpec {
        self.model.hybrid_gqa().clone()
    }

    pub fn session_residency_bytes(&self) -> Result<usize, String> {
        crate::kv_cache::HybridGqaCacheLayout::new(self.model.hybrid_gqa(), self.max_seq_len)?.total_elements().checked_mul(std::mem::size_of::<half::f16>()).ok_or_else(|| "Gemma4 KV resident 字节数溢出".to_owned())
    }

    pub fn layer_count(&self) -> usize {
        self.model.layer_count()
    }

    pub fn model_bytes(&self) -> u64 {
        self.model_bytes
    }

    pub fn model_format(&self) -> &str {
        &self.model_format
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.tokenize(text.as_bytes())
    }

    pub fn chat_template(&self) -> Option<&crate::runtime::chat_template::ChatTemplate> {
        self.chat_template.as_ref()
    }

    pub fn decode_bytes(&self, token: u32) -> Result<Vec<u8>, String> {
        self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("Gemma4 detokenize {token}: {error}"))
    }

    pub fn is_eos(&self, token: u32) -> bool {
        self.model.config().eos_token_ids.contains(&token)
    }

    /// 当前 checkpoint 是否实际包含本后端可装配的视觉权重；决定能力上报。
    pub fn accepts_images(&self) -> bool {
        self.accepts_images
    }

    /// 图文混排 parts → soft-token 编排(图像就地展开,与一次性执行器同一套预处理)。
    pub fn multimodal_input(&self, parts: &[crate::vision::ContentPart<'_>]) -> Result<Gemma4MultimodalInput, String> {
        if !self.accepts_images {
            return Err("Gemma4 当前权重没有兼容的视觉塔".to_owned());
        }
        gemma4_multimodal_input_from_parts(&self.tokenizer, self.model.config(), parts)
    }

    /// 视觉 checkpoint 必须先装视觉常驻资源再录制 decode replay；反向顺序会
    /// 让转录期临时资源占住视觉 prefill 的 buffer 生命周期，首 token 随之漂移。
    pub fn prepare_multimodal_resources(&mut self) -> Result<(), String> {
        if !self.accepts_images || self.multimodal_model.is_some() {
            return Ok(());
        }
        let source = self.weights.load_multimodal_weights().map_err(|error| format!("加载 Gemma4 multimodal 权重: {error}"))?;
        self.multimodal_model = Some(prepare_gemma4_multimodal_model(self.context(), self.model.config(), &source).map_err(|error| format!("准备 Gemma4 multimodal 权重: {error:?}"))?);
        Ok(())
    }

    /// 图文请求的 prefill:chunk 边界对齐 soft-token 区间,图像行用视觉塔输出
    /// 替换文本 embedding,图像区间内双向可见。多模态权重首个图文请求惰性加载。
    pub fn prefill_multimodal(&mut self, input: &Gemma4MultimodalInput) -> Result<Gemma4Sequence, String> {
        if input.token_ids.is_empty() {
            return Err("Gemma4 prompt 不能为空".to_owned());
        }
        if input.token_ids.len() >= self.max_seq_len {
            return Err(format!("Gemma4 prompt {} tokens 超过 max_seq_len {}", input.token_ids.len(), self.max_seq_len));
        }
        self.prepare_multimodal_resources()?;
        let multimodal_model = self.multimodal_model.as_ref().expect("multimodal model 已加载");
        let cfg = self.model.config();
        let cache = MetalKvCache::new_hybrid_gqa(self.context(), self.model.hybrid_gqa().clone(), self.max_seq_len).map_err(|error| format!("Gemma4 hybrid KV cache: {error}"))?;
        let mut sequence = Gemma4Sequence { cache, hidden: self.context.tensor_zeros(1, 1), tokens: Vec::new() };
        let mut position = 0usize;
        while position < input.token_ids.len() {
            crate::backend::BackendResources::begin_batch(self.context());
            let preferred_end = position.saturating_add(self.prefill_chunk_size).min(input.token_ids.len());
            let end = input.chunk_end(position, preferred_end);
            let chunk = &input.token_ids[position..end];
            let embedding = gemma4_embedding_rows(&self.weights, &input.embedding_token_ids[position..end], cfg.hidden_size, self.embedding_scale)?;
            let chunk_hidden = gemma4_multimodal_embedding(self.context(), cfg, multimodal_model, input, &embedding, position..end).map_err(|error| format!("Gemma4 multimodal embedding: {error:?}"))?;
            let per_layer_inputs = gemma4_metal_per_layer_inputs(self.context(), cfg, &self.weights, self.per_layer_model.as_ref(), &chunk_hidden, chunk)?;
            sequence.hidden = if input.chunk_has_visual_visibility(position..end) {
                gemma4_prefill_hidden_with_visibility(self.context(), &mut sequence.cache, &self.model, &self.layers, &self.rope, chunk_hidden, per_layer_inputs.as_deref(), position, &input.visible_ends[position..end])
            } else {
                gemma4_prefill_hidden(self.context(), &mut sequence.cache, &self.model, &self.layers, &self.rope, chunk_hidden, per_layer_inputs.as_deref(), position)
            }
            .map_err(|error| format!("Gemma4 Metal prefill position={position}: {error:?}"))?;
            sequence.tokens.extend_from_slice(chunk);
            position = end;
        }
        Ok(sequence)
    }

    pub fn prefill(&self, tokens: Vec<u32>) -> Result<Gemma4Sequence, String> {
        if tokens.is_empty() {
            return Err("Gemma4 prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.max_seq_len {
            return Err(format!("Gemma4 prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let cache = MetalKvCache::new_hybrid_gqa(self.context(), self.model.hybrid_gqa().clone(), self.max_seq_len).map_err(|error| format!("Gemma4 hybrid KV cache: {error}"))?;
        let mut sequence = Gemma4Sequence { cache, hidden: self.context.tensor_zeros(1, 1), tokens: Vec::new() };
        self.extend(&mut sequence, &tokens)?;
        Ok(sequence)
    }

    /// 在已有 sequence 的 cache 上续写 suffix(前缀缓存 resume 路径)。
    pub fn extend(&self, sequence: &mut Gemma4Sequence, suffix: &[u32]) -> Result<(), String> {
        if suffix.is_empty() {
            return Ok(());
        }
        if sequence.tokens.len() + suffix.len() >= self.max_seq_len {
            return Err(format!("Gemma4 会话状态 {} + suffix {} 超过 max_seq_len {}", sequence.tokens.len(), suffix.len(), self.max_seq_len));
        }
        let cfg = self.model.config();
        let offset = sequence.tokens.len();
        crate::runtime::prefill::run_chunked_prefill(offset, suffix.len(), self.prefill_chunk_size, |range, position| {
            let chunk = &suffix[range];
            crate::backend::BackendResources::begin_batch(self.context());
            let embedding = gemma4_embedding_rows(&self.weights, chunk, cfg.hidden_size, self.embedding_scale)?;
            let chunk_hidden = self.context().tensor_from_f32(&embedding, chunk.len(), cfg.hidden_size).map_err(|error| format!("上传 Gemma4 embedding: {error}"))?;
            let per_layer_inputs = gemma4_metal_per_layer_inputs(self.context(), cfg, &self.weights, self.per_layer_model.as_ref(), &chunk_hidden, chunk)?;
            sequence.hidden = gemma4_prefill_hidden(self.context(), &mut sequence.cache, &self.model, &self.layers, &self.rope, chunk_hidden, per_layer_inputs.as_deref(), position)
                .map_err(|error| format!("Gemma4 Metal prefill position={position}: {error:?}"))?;
            sequence.tokens.extend_from_slice(chunk);
            Ok(())
        })
    }

    /// prefill 后的首个输出与 final norm 后的 hidden(input 为多行 normed,
    /// 末行对应首 token 的上文;MTP draft 的首轮 inp_h 取末行)。
    pub fn first_token_output(&self, sequence: &Gemma4Sequence) -> Result<crate::runtime::output::OutputResult<MetalTensor>, String> {
        let cfg = self.model.config();
        let plan = crate::runtime::output::OutputPlan { eps: cfg.rms_eps, norm: crate::runtime::output::OutputNorm::GemmaRms, excluded_tokens: vec![cfg.end_image_token_id, cfg.end_audio_token_id] };
        crate::runtime::output::last_token_output(self.context(), &self.output_head, &sequence.hidden, sequence.hidden.rows - 1, &plan).map_err(|error| format!("Gemma4 输出步: {error:?}"))
    }

    /// prefill 后的首个输出 token；hidden 是最后 chunk 的多行，取末行 logits。
    pub fn first_token(&self, sequence: &Gemma4Sequence) -> Result<u32, String> {
        gemma4_last_token_output(self.context(), self.model.config(), &self.output_head, &sequence.hidden, sequence.hidden.rows - 1).map(|output| output.token_id).map_err(|error| format!("Gemma4 output head: {error:?}"))
    }

    /// decode round 后的输出 token；此时 hidden 是单行。
    pub fn step_token(&self, sequence: &Gemma4Sequence) -> Result<u32, String> {
        gemma4_token_output(self.context(), self.model.config(), &self.output_head, &sequence.hidden).map(|output| output.token_id).map_err(|error| format!("Gemma4 output head: {error:?}"))
    }

    pub fn decode_token(&self, sequence: &mut Gemma4Sequence, token: u32) -> Result<(), String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("Gemma4 会话状态 {} tokens 超过 max_seq_len {}", sequence.tokens.len(), self.max_seq_len));
        }
        let cfg = self.model.config();
        // embedding 上传与 PLE 链有 40+ 个小算子;必须在延迟批次内编码,
        // 否则每个算子一次 commit+同步,单 token 白付 ~10ms GPU 往返
        crate::backend::BackendResources::begin_decode_batch(self.context());
        let embedding = gemma4_embedding_rows(&self.weights, &[token], cfg.hidden_size, self.embedding_scale)?;
        let input = self.context().tensor_from_f32(&embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Gemma4 decode embedding: {error}"))?;
        let per_layer_inputs = gemma4_metal_per_layer_inputs(self.context(), cfg, &self.weights, self.per_layer_model.as_ref(), &input, &[token])?;
        let position = sequence.tokens.len();
        sequence.hidden =
            gemma4_decode_round(self.context(), &mut sequence.cache, &self.model, &self.layers, &self.rope, input, per_layer_inputs.as_deref(), position).map_err(|error| format!("Gemma4 Metal decode position={position}: {error:?}"))?;
        sequence.tokens.push(token);
        Ok(())
    }

    /// 双缓冲命令重放是否接管 decode，由 execution.replay 明确决定。
    pub fn replay_decode_available(&self) -> bool {
        let per_layer_ready = self.model.config().per_layer_input_size == 0 || self.per_layer_embedding_source.is_some();
        self.replay_decode && self.embedding_source.is_some() && per_layer_ready
    }

    pub fn replay_resources_prepared(&self) -> bool {
        (!self.replay_decode_available() || self.replay.is_some()) && (!self.mtp_enabled() || self.mtp_replay.is_some() && self.verify_replay.is_some())
    }

    /// 首个请求时录制 A/B 命令表(一次 ~20ms);cache 必须与后续请求同规格,
    /// 重放表内 per-layer KV 偏移按录制期布局固化,由 bind_cache 换指针复用。
    pub fn ensure_replay(&mut self, cache: &MetalKvCache) -> Result<(), String> {
        match self.replay.as_mut() {
            None => {
                let Some(source) = &self.embedding_source else {
                    return Err("Gemma4 重放需要设备端 embedding 来源(tied Q4_K lm_head)".to_owned());
                };
                let per_layer_source = self.per_layer_embedding_source.as_ref();
                let engine = crate::runtime::gemma4::metal_replay::Gemma4ReplayEngine::record(
                    self.context(),
                    cache,
                    &self.model,
                    &self.layers,
                    self.per_layer_model.as_ref(),
                    &self.rope,
                    &self.output_head,
                    &source.blob,
                    source.row_bytes,
                    per_layer_source.map(|source| &source.blob),
                    per_layer_source.map(|source| source.row_bytes),
                    self.embedding_scale,
                )
                .map_err(|error| format!("Gemma4 重放录制: {error:?}"))?;
                self.replay = Some(engine);
            }
            Some(engine) => engine.bind_cache(cache),
        }
        Ok(())
    }

    /// 单 token 的 embedding 行(CPU 侧解 Q4_K;重放步首写入 input)。
    pub fn embedding_row(&self, token: u32) -> Result<Vec<f32>, String> {
        gemma4_embedding_rows(&self.weights, &[token], self.model.config().hidden_size, self.embedding_scale)
    }

    pub fn replay(&self) -> &crate::runtime::gemma4::metal_replay::Gemma4ReplayEngine {
        self.replay.as_ref().expect("replay 已初始化")
    }

    /// MTP 投机模式是否启用(config 指定 mtp-*.gguf 时)。
    pub fn mtp_available(&self) -> bool {
        self.mtp.is_some()
    }

    /// MTP 头是否已加载(决定重放录制时机)。
    pub fn mtp_enabled(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn mtp_draft_tokens(&self) -> usize {
        self.mtp_draft_tokens
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        self.model.config().eos_token_ids.clone()
    }

    pub fn mtp(&self) -> &crate::runtime::gemma4::metal_mtp::Gemma4MtpModel {
        self.mtp.as_ref().expect("MTP 已初始化")
    }

    /// MTP draft 重放:首请求录制,后续请求 bind 主干 KV。
    pub fn ensure_mtp_replay(&mut self, cache: &MetalKvCache) -> Result<(), String> {
        let Some(model) = &self.mtp else { return Ok(()) };
        match self.mtp_replay.as_mut() {
            None => {
                let replay = model.record_replay(self.context(), cache).map_err(|error| format!("Gemma4 MTP 重放录制: {error:?}"))?;
                self.mtp_replay = Some(replay);
            }
            Some(replay) => replay.bind_backbone(cache),
        }
        Ok(())
    }

    pub fn mtp_replay(&self) -> &crate::runtime::gemma4::metal_mtp::Gemma4MtpReplay {
        self.mtp_replay.as_ref().expect("MTP replay 已初始化")
    }

    /// verify 重放:首请求录制(K+1 行),后续请求 bind KV。
    pub fn ensure_verify_replay(&mut self, cache: &MetalKvCache, rows: usize) -> Result<(), String> {
        match self.verify_replay.as_mut() {
            None => {
                let final_norm = self.weights.final_norm()?;
                let replay = crate::runtime::gemma4::metal_replay::Gemma4VerifyReplay::record(self.context(), cache, &self.model, &self.layers, self.per_layer_model.as_ref(), &self.rope, &self.output_head, rows, &final_norm)
                    .map_err(|error| format!("Gemma4 verify 重放录制: {error:?}"))?;
                self.verify_replay = Some(replay);
            }
            Some(replay) => replay.bind_cache(cache),
        }
        Ok(())
    }

    pub fn verify_replay(&self) -> &crate::runtime::gemma4::metal_replay::Gemma4VerifyReplay {
        self.verify_replay.as_ref().expect("verify replay 已初始化")
    }

    /// verify 重放一步(含 embedding 上传)。
    pub fn verify_tokens_replay(&self, sequence: &mut Gemma4Sequence, tokens: &[u32]) -> Result<(Vec<u32>, Vec<f32>), String> {
        if tokens.is_empty() {
            return Err("Gemma4 verify 输入不能为空".to_owned());
        }
        let replay = self.verify_replay();
        if replay.rows() != tokens.len() {
            return Err(format!("Gemma4 verify 重放行数 {} 与输入 {} 不符", replay.rows(), tokens.len()));
        }
        let cfg = self.model.config();
        let embedding = gemma4_embedding_rows(&self.weights, tokens, cfg.hidden_size, self.embedding_scale)?;
        replay.write_input(&embedding).map_err(|error| format!("Gemma4 verify 重放输入: {error:?}"))?;
        if cfg.per_layer_input_size != 0 {
            let values = gemma4_per_layer_embedding_rows(cfg, &self.weights, tokens)?;
            replay.write_token_inputs(&values).map_err(|error| format!("Gemma4 verify per-layer 输入: {error:?}"))?;
        }
        let position = sequence.tokens.len();
        let trace = std::env::var_os("ZLLM_GEMMA4_VREPLAY_TRACE").is_some();
        if trace {
            self.context().reset_gpu_stats();
        }
        let (targets, normed) = replay.step(self.context(), &mut sequence.cache, position).map_err(|error| format!("Gemma4 verify 重放步: {error:?}"))?;
        if trace {
            let gpu = self.context().gpu_stats();
            eprintln!("[vr-gpu] gpu={:.3}s commands={} targets={:?}", gpu.seconds, gpu.command_buffers, targets);
            for profile in self.context().gpu_profile().into_iter().take(10) {
                eprintln!("  vr gpu {:>8.3} ms x{} | {} {}", profile.gpu_seconds * 1.0e3, profile.calls, profile.operator, profile.shape);
            }
        }
        sequence.tokens.extend_from_slice(tokens);
        Ok((targets, normed))
    }

    /// 首请求惰性加载 MTP 头(ZLLM_GEMMA4_MTP 路径);RoPE 表按 8K 上限,
    /// 超长上下文请求应回落重放路径。
    pub fn ensure_mtp(&mut self) -> Result<(), String> {
        if self.mtp.is_none() {
            let Some(path) = self.mtp_weights.clone() else {
                return Ok(());
            };
            let weights = crate::weight::model::gemma4::Gemma4MtpWeights::open(&path)?;
            let backbone_types: Vec<bool> = (0..self.model.layer_count()).map(|layer| self.model.layer_spec(layer).expect("层规格").attention.hybrid.window == crate::attention::gqa::CausalWindow::Full).collect();
            let model = crate::runtime::gemma4::metal_mtp::Gemma4MtpModel::prepare(self.context(), &weights, &backbone_types, self.model.config().num_kv_shared_layers, self.max_seq_len.min(8192))
                .map_err(|error| format!("准备 Gemma4 MTP: {error:?}"))?;
            eprintln!("[gemma4] MTP 投机解码已启用(4 层 draft 头)");
            self.mtp = Some(model);
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn config_pub(&self) -> &crate::runtime::gemma4::Gemma4Config {
        self.model.config()
    }

    #[cfg(test)]
    pub fn output_head_pub(&self) -> &crate::runtime::gemma4::Gemma4OutputHead<MetalWeight> {
        &self.output_head
    }

    #[cfg(test)]
    pub fn decode_token_pub(&self, sequence: &mut Gemma4Sequence, token: u32) -> Result<(), String> {
        self.decode_token(sequence, token)
    }

    #[cfg(test)]
    pub fn verify_tokens_pub(&self, sequence: &mut Gemma4Sequence, tokens: &[u32]) -> Result<(Vec<u32>, MetalTensor), String> {
        self.verify_tokens(sequence, tokens)
    }

    /// 多行 verify:tokens 在当前游标后前向并 append KV,返回每行 argmax
    /// (行 i 的结果 = tokens[0..=i] 之后主干认为的正确 token)与 final norm
    /// 后的多行 hidden(draft 的 inp_h 用 normed,与 llama.cpp nextn 语义一致)。
    pub fn verify_tokens(&self, sequence: &mut Gemma4Sequence, tokens: &[u32]) -> Result<(Vec<u32>, MetalTensor), String> {
        if tokens.is_empty() {
            return Err("Gemma4 verify 输入不能为空".to_owned());
        }
        if sequence.tokens.len() + tokens.len() >= self.max_seq_len {
            return Err(format!("Gemma4 verify {} + {} 超过 max_seq_len {}", sequence.tokens.len(), tokens.len(), self.max_seq_len));
        }
        let cfg = self.model.config();
        crate::backend::BackendResources::begin_batch(self.context());
        let embedding = gemma4_embedding_rows(&self.weights, tokens, cfg.hidden_size, self.embedding_scale)?;
        let input = self.context().tensor_from_f32(&embedding, tokens.len(), cfg.hidden_size).map_err(|error| format!("上传 Gemma4 verify embedding: {error}"))?;
        let per_layer_inputs = gemma4_metal_per_layer_inputs(self.context(), cfg, &self.weights, self.per_layer_model.as_ref(), &input, tokens)?;
        let position = sequence.tokens.len();
        let verify_profile = std::env::var_os("ZLLM_GEMMA4_VERIFY_TRACE").is_some();
        if verify_profile {
            self.context().reset_gpu_stats();
        }
        sequence.hidden =
            gemma4_prefill_hidden(self.context(), &mut sequence.cache, &self.model, &self.layers, &self.rope, input, per_layer_inputs.as_deref(), position).map_err(|error| format!("Gemma4 verify prefill position={position}: {error:?}"))?;
        sequence.tokens.extend_from_slice(tokens);
        let plan = crate::runtime::output::OutputPlan { eps: cfg.rms_eps, norm: crate::runtime::output::OutputNorm::GemmaRms, excluded_tokens: vec![cfg.end_image_token_id, cfg.end_audio_token_id] };
        let (normed, logits) = crate::runtime::output::norm_and_lm_head(self.context(), &self.output_head, &sequence.hidden, &plan).map_err(|error| format!("Gemma4 verify 输出: {error:?}"))?;
        let tokens = crate::backend::SegmentedTensorBackend::argmax_rows_excluding(self.context(), &logits, &plan.excluded_tokens).map_err(|error| format!("Gemma4 verify argmax: {error:?}"))?;
        if verify_profile {
            self.context().synchronize();
            let gpu = self.context().gpu_stats();
            eprintln!("[verify-profile] gpu={:.3}s commands={} rows={}", gpu.seconds, gpu.command_buffers, tokens.len());
            for profile in self.context().gpu_profile().into_iter().take(14) {
                eprintln!("  verify gpu {:>8.3} ms x{} | {} {}", profile.gpu_seconds * 1.0e3, profile.calls, profile.operator, profile.shape);
            }
        }
        Ok((tokens, normed))
    }

    /// 主干 norm 后 hidden 张量的指定行读回为 CPU f32(draft 的 inp_h 输入)。
    pub fn normed_hidden_row_f32(&self, normed: &MetalTensor, row: usize) -> Result<Vec<f32>, String> {
        if row >= normed.rows {
            return Err(format!("Gemma4 normed hidden 行 {row} 越界 {}", normed.rows));
        }
        let selected = crate::backend::Backend::select_row(self.context(), normed, row).map_err(|error| format!("select_row: {error:?}"))?;
        Ok(self.context().tensor_to_f32(&selected))
    }

    /// 进入流水线提交窗口:defer 全部 GPU 等待,等待点改为 Gemma4PendingToken 的 CB 句柄。
    /// CB 粒度保持默认(16 算子/CB):批内 CB 生命周期短,buffer 可在轮内复用;
    /// 实测整轮单 CB(1024)因中间 buffer 无法复用、每轮换页,GPU 步时反而更差。
    pub fn begin_async_decode(&self) {
        self.context.set_deferred_layer_scope_sync(true);
        self.context.set_deferred_waits(true);
    }

    /// 退出流水线窗口:恢复同步语义并 drain 全部在飞工作(含被 EOS 浪费的推测轮)。
    pub fn end_async_decode(&self) {
        self.context.set_deferred_waits(false);
        self.context.set_deferred_layer_scope_sync(false);
    }

    /// 提交输出步(不等待):final norm + lm_head + argmax 按 token 位置写入读回区。
    pub fn submit_output(&self, hidden: &MetalTensor, position: usize) -> Result<Gemma4PendingToken, String> {
        let cfg = self.model.config();
        let plan = crate::runtime::output::OutputPlan { eps: cfg.rms_eps, norm: crate::runtime::output::OutputNorm::GemmaRms, excluded_tokens: vec![cfg.end_image_token_id, cfg.end_audio_token_id] };
        let (_normed, logits) = crate::runtime::output::norm_and_lm_head(self.context(), &self.output_head, hidden, &plan).map_err(|error| format!("Gemma4 输出步: {error:?}"))?;
        let id_offset = (position * std::mem::size_of::<u32>()) as u64;
        crate::kernel::metal::moe::argmax_tensor_into_offset(self.context(), &logits, &plan.excluded_tokens, &self.token_readback, id_offset)?;
        let (command, through) = self.context.submit_batch_and_last_command().ok_or("Gemma4 输出步没有已提交命令")?;
        Ok(Gemma4PendingToken { id_offset, command, profiles_through: through, placeholder: None })
    }

    /// 推测提交一轮:从 pending 的读回位置设备端 gather embedding → decode round →
    /// 输出步。全程无 CPU 同步;pending 的 token 在 sequence.tokens 先占位,wait 时回填。
    /// 读回区按 token 位置寻址,不存在覆写(见 token_readback 注释)。
    pub fn submit_step(&self, sequence: &mut Gemma4Sequence, pending: &mut Gemma4PendingToken) -> Result<Gemma4PendingToken, String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("Gemma4 会话状态 {} tokens 超过 max_seq_len {}", sequence.tokens.len(), self.max_seq_len));
        }
        let Some(source) = &self.embedding_source else {
            return Err("Gemma4 异步 decode 缺少设备端 embedding 来源".to_owned());
        };
        let cfg = self.model.config();
        let input = crate::kernel::metal::gguf::gguf_gather_row_q4k_tensor_offset(self.context(), &self.token_readback, pending.id_offset, &source.blob, cfg.vocab_size, cfg.hidden_size, source.row_bytes, self.embedding_scale)?;
        let position = sequence.tokens.len();
        sequence.tokens.push(0);
        pending.placeholder = Some(position);
        sequence.hidden = gemma4_decode_round_deferred(self.context(), &mut sequence.cache, &self.model, &self.layers, &self.rope, input, None, position).map_err(|error| format!("Gemma4 Metal decode position={position}: {error:?}"))?;
        // 本轮产出的是 position+1 处的下一个 token;读回区下标必须与占位/产出对齐。
        self.submit_output(&sequence.hidden, position + 1)
    }

    /// 精确等待 pending 的 argmax CB 并读回 token id;已提交的推测轮次继续在 GPU 上跑。
    pub fn wait_token(&self, sequence: &mut Gemma4Sequence, pending: &Gemma4PendingToken) -> u32 {
        pending.command.wait_until_completed();
        self.context.complete_profiles_through(pending.profiles_through);
        let token = unsafe { *self.token_readback.contents().cast::<u8>().add(pending.id_offset as usize).cast::<u32>() };
        if let Some(position) = pending.placeholder
            && let Some(slot) = sequence.tokens.get_mut(position)
        {
            *slot = token;
        }
        token
    }

    /// 提前停止(EOS/stop 词/取消)时,停止 token 已被推测前向但不属于会话:
    /// 从 tokens 与 KV cache 游标丢弃(cache 字节保留但不可见,续写时原位覆写)。
    pub fn discard_forwarded_token(&self, sequence: &mut Gemma4Sequence) -> Result<(), String> {
        sequence.tokens.pop();
        if let Some(state) = sequence.cache.hybrid_gqa_state_mut() {
            state.truncate(sequence.tokens.len())?;
        }
        Ok(())
    }
}

/// 终点会话状态:sequence(cache+hidden+已处理 token)+ 已采样未前向的 pending token。
pub struct Gemma4TerminalState {
    pub sequence: Gemma4Sequence,
    pub pending: Vec<u32>,
    pub info: CacheInfo,
}

/// 快照恢复所需的引擎资源;与 session 共享同一 MetalContext,
/// 恢复出的 cache 与后续算子落在同一 queue,保持执行顺序。
pub struct Gemma4SnapshotResources {
    pub context: std::sync::Arc<MetalContext>,
    pub hybrid_gqa: crate::attention::gqa::HybridGqaSpec,
    pub max_seq_len: usize,
}

impl crate::kv_cache::terminal_cache::TerminalSnapshot for Gemma4TerminalState {
    type Resources = Gemma4SnapshotResources;

    fn encode(&self) -> Result<Vec<u8>, String> {
        let mut writer = crate::kv_cache::terminal_cache::SnapshotWriter::new();
        writer.u32(2);
        writer.u32s(&self.sequence.tokens)?;
        writer.u32s(&self.pending)?;
        let mut buffer = writer.into_inner();
        let hidden_dtype = match self.sequence.hidden.dtype {
            crate::backend::metal::MetalTensorDType::F16 => 0u32,
            crate::backend::metal::MetalTensorDType::Bf16 => 1,
            crate::backend::metal::MetalTensorDType::F32 => return Err("Gemma4 快照暂不支持 F32 hidden".to_owned()),
        };
        buffer.extend(&hidden_dtype.to_le_bytes());
        buffer.extend(&(self.sequence.hidden.rows as u32).to_le_bytes());
        buffer.extend(&(self.sequence.hidden.cols as u32).to_le_bytes());
        let contents = self.sequence.hidden.buffer.contents() as *const u8;
        let hidden_bytes = self.sequence.hidden.rows * self.sequence.hidden.cols * 2;
        // 恢复时 GPU 已同步(生成结束),共享内存直读
        buffer.extend(unsafe { std::slice::from_raw_parts(contents, hidden_bytes) });
        let state = self.sequence.cache.hybrid_gqa_state().ok_or("Gemma4 快照要求 hybrid cache")?;
        let layout = state.layout();
        let cache_bytes = self.sequence.cache.buffer().contents() as *const u8;
        for layer in 0..layout.layer_count() {
            let spec = layout.layer(layer)?;
            let range = state.retained_range(layer)?;
            buffer.extend(&(range.start as u64).to_le_bytes());
            buffer.extend(&(range.end as u64).to_le_bytes());
            let rows = range.end.min(spec.capacity);
            let elements = rows.checked_mul(spec.columns).ok_or("Gemma4 快照层元素溢出")?;
            let key = unsafe { std::slice::from_raw_parts(cache_bytes.add(spec.key_offset * 2), elements * 2) };
            let value = unsafe { std::slice::from_raw_parts(cache_bytes.add(spec.value_offset * 2), elements * 2) };
            buffer.extend(&(elements as u32).to_le_bytes());
            buffer.extend(key);
            buffer.extend(value);
        }
        Ok(buffer)
    }

    fn decode(bytes: &[u8], resources: &Self::Resources) -> Result<Self, String> {
        let mut reader = crate::kv_cache::terminal_cache::SnapshotReader::new(bytes);
        if reader.u32("Gemma4 版本")? != 2 {
            return Err("Gemma4 快照版本不支持".to_owned());
        }
        let tokens = reader.u32s("Gemma4 tokens")?;
        let pending = reader.u32s("Gemma4 pending")?;
        let hidden_dtype = reader.u32("Gemma4 hidden dtype")?;
        let hidden_rows = reader.u32("Gemma4 hidden rows")? as usize;
        let hidden_cols = reader.u32("Gemma4 hidden cols")? as usize;
        // encode 布局是 [header|hidden|layers...]：先在 cursor 上消耗 hidden,
        // 再读各层 KV;否则 L0 的 range/elements 会读到 hidden 字节。
        let hidden_byte_len = hidden_rows.checked_mul(hidden_cols).and_then(|count| count.checked_mul(2)).ok_or("Gemma4 快照 hidden 大小溢出")?;
        let hidden_bits = reader.take(hidden_byte_len, "Gemma4 hidden")?;
        let hidden = match hidden_dtype {
            0 => resources.context.tensor_from_f16_bits(hidden_bits, hidden_rows, hidden_cols)?,
            1 => resources.context.tensor_from_bf16_bits(hidden_bits, hidden_rows, hidden_cols)?,
            other => return Err(format!("Gemma4 快照 hidden dtype={other} 未知")),
        };
        let mut cache = MetalKvCache::new_hybrid_gqa(&resources.context, resources.hybrid_gqa.clone(), resources.max_seq_len).map_err(|error| format!("Gemma4 快照重建 cache: {error}"))?;
        let mut ranges = Vec::new();
        let cache_bytes = cache.buffer().contents() as *mut u8;
        let layout = cache.hybrid_gqa_state().expect("hybrid cache 布局").layout().clone();
        for layer in 0..layout.layer_count() {
            let spec = layout.layer(layer)?;
            let start = reader.u64(&format!("Gemma4 L{layer} start"))? as usize;
            let end = reader.u64(&format!("Gemma4 L{layer} end"))? as usize;
            let elements = reader.u32(&format!("Gemma4 L{layer} elements"))? as usize;
            if elements != end.min(spec.capacity) * spec.columns {
                return Err(format!("Gemma4 快照 L{layer} 元素数 {elements} 与范围不符"));
            }
            let bytes = reader.take(elements.checked_mul(4).ok_or_else(|| format!("Gemma4 L{layer} KV 大小溢出"))?, &format!("Gemma4 L{layer} KV"))?;
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), cache_bytes.add(spec.key_offset * 2), elements * 2);
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(elements * 2), cache_bytes.add(spec.value_offset * 2), elements * 2);
            }
            ranges.push(start..end);
        }
        reader.finish()?;
        let Some(state_mut) = cache.hybrid_gqa_state_mut() else { return Err("Gemma4 快照恢复游标失败".to_owned()) };
        state_mut.restore_ranges(ranges)?;
        Ok(Self { sequence: Gemma4Sequence { cache, hidden, tokens }, pending, info: crate::kv_cache::terminal_cache::TerminalInfo::default() })
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

/// 上报用：权重目录所有分片文件大小之和。
fn directory_bytes(root: &Path) -> u64 {
    std::fs::read_dir(root).map(|entries| entries.filter_map(Result::ok).map(|entry| entry.metadata().map(|metadata| metadata.len()).unwrap_or(0)).sum()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 重放步时真机计时:验证平铺重编码的 CPU 步时数量级(生产化收益依据)。
    /// `ZLLM_GEMMA4_GGUF=/path/to/gemma4.gguf cargo test --release --lib replay_step_timing -- --nocapture`
    #[test]
    fn replay_step_timing() {
        let Some(root) = std::env::var_os("ZLLM_GEMMA4_GGUF").map(std::path::PathBuf::from) else { return };
        let session = Gemma4MetalSession::load(&root, 2048, 2048, crate::weight::LmHeadQuantization::Native, None, 3, false).expect("session load");
        let mut sequence = session.prefill(session.tokenize("Hello, tell me a story.")).expect("prefill");
        let record_started = std::time::Instant::now();
        let mut replay =
            crate::runtime::gemma4::metal_replay::Gemma4DecodeReplay::record(session.context(), &mut sequence.cache, &session.model, &session.layers, &session.rope, &session.weights, session.per_layer_model.as_ref(), &session.output_head)
                .expect("record");
        println!("[replay-timing] record wall={:.3}s commands={}", record_started.elapsed().as_secs_f32(), replay.command_count());
        let mut token = 5u32;
        let mut position = sequence.tokens.len();
        for _ in 0..4 {
            token = replay.step(token, position).expect("warmup step");
            position += 1;
        }
        let steps = 50usize;
        let started = std::time::Instant::now();
        for _ in 0..steps {
            token = replay.step(token, position).expect("replay step");
            position += 1;
        }
        let wall = started.elapsed().as_secs_f32();
        println!("[replay-timing] step wall={:.1}ms ({:.1} tok/s) token={token}", wall / steps as f32 * 1.0e3, steps as f32 / wall);
        // 消融分档:gemv-only / gemv+attn / full,差分出小算子与 attention 的占比
        type Op = crate::backend::metal::api::RecordedComputeOp;
        let tiers: [(&str, fn(&Op) -> bool); 3] = [("gemv-only", |op: &Op| op.threads.width == 64), ("gemv+attn", |op: &Op| op.threads.width == 64 || (op.threads.width == 256 && op.groups.width > 1)), ("full", |_| true)];
        for (name, keep) in tiers.iter() {
            let kept = replay.ops().iter().filter(|op| keep(op)).count();
            let ablation_started = std::time::Instant::now();
            for step in 0..steps {
                token = replay.step_filtered(token, position + step, keep).expect("消融步");
            }
            position += steps;
            println!("[replay-ablation] {name}: kept={kept} 平均 {:.2} ms/步", ablation_started.elapsed().as_secs_f64() / steps as f64 * 1.0e3);
        }
        // 按 pipeline 实例分组消融:每类 kernel 的净耗时(其余组跳过)。
        let mut pipelines: Vec<crate::backend::metal::api::ComputePipelineState> = Vec::new();
        for op in replay.ops() {
            if !pipelines.iter().any(|pipeline| pipeline.same_handle(&op.pipeline)) {
                pipelines.push(op.pipeline.clone());
            }
        }
        let names: Vec<String> = pipelines.iter().map(|pipeline| session.context().cached_pipeline_name(pipeline).unwrap_or_else(|| "?".to_owned())).collect();
        drop(pipelines.iter());
        println!("[replay-kernels] 唯一 pipeline 数={}", pipelines.len());
        let mut timings: Vec<(usize, usize, f64)> = Vec::new();
        for (index, pipeline) in pipelines.iter().enumerate() {
            let keep = |op: &Op| pipeline.same_handle(&op.pipeline);
            let count = replay.ops().iter().filter(|op| keep(op)).count();
            let ablation_started = std::time::Instant::now();
            for step in 0..steps {
                token = replay.step_filtered(token, position + step, &keep).expect("分组消融步");
            }
            position += steps;
            timings.push((index, count, ablation_started.elapsed().as_secs_f64() / steps as f64 * 1.0e3));
        }
        timings.sort_by(|left, right| right.2.partial_cmp(&left.2).unwrap_or(std::cmp::Ordering::Equal));
        let shapes: Vec<(u64, u64, u64)> = {
            let mut shapes = vec![(0u64, 0u64, 0u64); pipelines.len()];
            for op in replay.ops() {
                for (index, pipeline) in pipelines.iter().enumerate() {
                    if pipeline.same_handle(&op.pipeline) {
                        shapes[index] = (op.threads.width, op.groups.width, op.groups.height);
                        break;
                    }
                }
            }
            shapes
        };
        for (index, count, ms) in timings.iter().take(16) {
            let (threads, groups_w, groups_h) = shapes[*index];
            println!("[replay-kernels] #{index:2} ×{count:3} = {ms:6.2} ms/步 | threads={threads} groups={}x{} | {}", groups_w, groups_h, names[*index]);
        }
    }
}
