use super::*;
use crate::attention::AttentionSpec;

pub struct MistralMetalSequence {
    pub cache: MetalKvCache,
    pub hidden: MetalTensor,
    pub tokens: Vec<u32>,
}

impl MistralMetalSequence {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

pub struct MistralMetalSession {
    context: MetalContext,
    config: MistralConfig,
    weights: Arc<MistralWeights>,
    layers: Vec<MistralTextLayer<MetalWeight>>,
    output_head: MistralOutputHead<MetalWeight>,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    rope: crate::attention::rope::RopeTable,
    cache_spec: KvCacheSpec,
    kv_f16: bool,
    max_seq_len: usize,
}

impl MistralMetalSession {
    pub fn load(model_path: &Path, max_seq_len: usize, kv_f16: bool, lm_head_quantization: crate::weight::LmHeadQuantization) -> Result<Self, String> {
        let weights = Arc::new(MistralWeights::open(model_path).map_err(|error| format!("Mistral GGUF 打开失败: {error}"))?);
        let config = *weights.config();
        crate::runtime::validate_max_sequence_length("Mistral", max_seq_len, config.max_position_embeddings)?;
        let context = MetalContext::new_default().map_err(|error| format!("MetalContext 初始化失败: {error}"))?;
        // 层与 output head 都在加载期 prepare；decode 不得逐 token 重读 GGUF/上传权重。
        let layers = mistral::prepare_mistral_layers(&context, weights.as_ref()).map_err(|error| format!("准备 Mistral layers: {error:?}"))?;
        let output_head = mistral::prepare_mistral_output_head_quantized(&context, &config, weights.as_ref(), lm_head_quantization).map_err(|error| format!("准备 Mistral output head: {error:?}"))?;
        let tokenizer = weights.tokenizer().map_err(|error| format!("Mistral tokenizer: {error}"))?;
        let detokenizer = weights.detokenizer().map_err(|error| format!("Mistral detokenizer: {error}"))?;
        let rope = mistral::mistral_rope_table(&config, max_seq_len);
        // 1 层 1 map,dense GQA 无共享;kv cache layer map 每层 1 行
        let _cache_layers = KvCacheLayerMap::dense(config.layer_count);
        let cache_spec = KvCacheSpec::from_attention(&AttentionSpec::Gqa(crate::attention::gqa::GqaSpec {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            rope_dim: config.head_dim,
            rope_theta: config.rope_theta,
            use_qk_norm: false,
            window: CausalWindow::Full,
            score_scale: 1.0 / (config.head_dim as f32).sqrt(),
            output_gate: false,
        }))
        .map_err(|error| format!("Mistral KV cache spec: {error}"))?;
        Ok(Self { context, config, weights, layers, output_head, tokenizer, detokenizer, rope, cache_spec, kv_f16, max_seq_len })
    }

    pub fn context(&self) -> &MetalContext {
        &self.context
    }
    pub fn model_bytes(&self) -> u64 {
        self.weights.reader().file_len()
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    pub fn session_residency_bytes(&self) -> Result<usize, String> {
        let format = if self.kv_f16 { crate::kv_cache::KvCacheFormat::F16 } else { crate::kv_cache::KvCacheFormat::Int8 };
        crate::kv_cache::KvCacheLayout::new(self.cache_spec.clone(), format, self.config.layer_count, self.max_seq_len, crate::kv_cache::DEFAULT_GROUP_SIZE).map(|layout| layout.total_bytes())
    }
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.tokenize(text.as_bytes())
    }
    pub fn decode_bytes(&self, token: u32) -> Result<Vec<u8>, String> {
        self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("Mistral detokenize {token}: {error}"))
    }
    pub fn is_eos(&self, token: u32) -> bool {
        token == mistral::MISTRAL_EOS_TOKEN_ID
    }

    /// 单段 prompt prefill；返回 last hidden + 新建 cache（每请求独立）。
    pub fn prefill(&self, tokens: Vec<u32>) -> Result<MistralMetalSequence, String> {
        if tokens.is_empty() {
            return Err("Mistral prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.max_seq_len {
            return Err(format!("Mistral prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let mut cache = if self.kv_f16 {
            MetalKvCache::new_f16(&self.context, self.cache_spec.clone(), self.config.layer_count, self.max_seq_len)
        } else {
            MetalKvCache::new(&self.context, self.cache_spec.clone(), self.config.layer_count, self.max_seq_len)
        }
        .map_err(|error| format!("Mistral KV cache 分配: {error}"))?;
        let embedding = self.weights.embedding_rows_f32(&tokens).map_err(|error| format!("Mistral embedding: {error}"))?;
        let hidden = self.context.tensor_from_f32_preserve(&embedding, tokens.len(), self.config.hidden_size).map_err(|error| format!("上传 Mistral embedding: {error}"))?;
        let output = mistral::mistral_text_hidden(&self.context, &self.config, &self.layers, Some(&mut cache), hidden, &self.rope, 0).map_err(|error| format!("Mistral Metal prefill: {error:?}"))?;
        let last = self.context.select_row(&output, tokens.len() - 1).map_err(|error| format!("Mistral 选择最后 token: {error:?}"))?;
        Ok(MistralMetalSequence { cache, hidden: last, tokens })
    }

    /// 取 prefill 末位 hidden → output head → sample
    pub fn next_token(&self, sequence: &MistralMetalSequence) -> Result<u32, String> {
        mistral::mistral_token_output(&self.context, &self.config, &self.output_head, &sequence.hidden).map(|output| output.token_id).map_err(|error| format!("Mistral output: {error:?}"))
    }

    /// 单 token decode 步：embedding → forward → 更新 sequence.hidden
    pub fn decode_token(&self, sequence: &mut MistralMetalSequence, token: u32) -> Result<(), String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("Mistral 会话状态 {} tokens 超过 max_seq_len {}", sequence.tokens.len(), self.max_seq_len));
        }
        let embedding = self.weights.embedding_rows_f32(&[token]).map_err(|error| format!("Mistral decode embedding: {error}"))?;
        let input = self.context.tensor_from_f32_preserve(&embedding, 1, self.config.hidden_size).map_err(|error| format!("上传 Mistral decode embedding: {error}"))?;
        let position = sequence.tokens.len();
        sequence.hidden = mistral::mistral_decode_round(&self.context, &self.config, &self.layers, &mut sequence.cache, input, &self.rope, position).map_err(|error| format!("Mistral decode position={position}: {error:?}"))?;
        sequence.tokens.push(token);
        Ok(())
    }
}
