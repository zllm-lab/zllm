use super::*;
use crate::attention::AttentionSpec;
use crate::backend::{BackendError, TokenSampling};

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
    /// GGUF 自带 chat template(K2-Horizon 必备;Mistral 走字面拼接,保持已验证行为)。
    chat_template: Option<crate::runtime::chat_template::ChatTemplate>,
    rope: crate::attention::rope::RopeTable,
    cache_spec: KvCacheSpec,
    kv_f16: bool,
    max_seq_len: usize,
}

impl MistralMetalSession {
    pub fn load_with_replay(model_path: &Path, max_seq_len: usize, kv_f16: bool, replay_enabled: bool, lm_head_quantization: crate::weight::LmHeadQuantization) -> Result<Self, String> {
        let weights = Arc::new(MistralWeights::open(model_path).map_err(|error| format!("Mistral GGUF 打开失败: {error}"))?);
        let config = *weights.config();
        crate::runtime::validate_max_sequence_length("Mistral", max_seq_len, config.max_position_embeddings)?;
        let context = MetalContext::new_default_with_replay(replay_enabled).map_err(|error| format!("MetalContext 初始化失败: {error}"))?;
        // 层与 output head 都在加载期 prepare；decode 不得逐 token 重读 GGUF/上传权重。
        let layers = mistral::prepare_mistral_layers(&context, weights.as_ref()).map_err(|error| format!("准备 Mistral layers: {error:?}"))?;
        let output_head = mistral::prepare_mistral_output_head_quantized(&context, &config, weights.as_ref(), lm_head_quantization).map_err(|error| format!("准备 Mistral output head: {error:?}"))?;
        let tokenizer = weights.tokenizer().map_err(|error| format!("Mistral tokenizer: {error}"))?;
        let detokenizer = weights.detokenizer().map_err(|error| format!("Mistral detokenizer: {error}"))?;
        let chat_template = match config.architecture {
            crate::weight::model::mistral::DenseGqaArchitecture::K2Horizon => {
                let source = weights.reader().metadata("tokenizer.chat_template").and_then(crate::weight::container::gguf::GgufValue::as_str).ok_or("K2-Horizon GGUF 缺少 tokenizer.chat_template")?;
                let mut compiled = crate::runtime::chat_template::ChatTemplate::new(source)?;
                // 模板里的 {{ bos_token }} 需要展开成词表真实 token 串,id 从 metadata 取,
                // 串经 detokenizer 还原(特殊 token 均为 ASCII,无损)。
                let bos = Self::reader_token_string(weights.reader(), &detokenizer, "tokenizer.ggml.bos_token_id", 0)?;
                let eos = Self::reader_token_string(weights.reader(), &detokenizer, "tokenizer.ggml.eos_token_id", config.eos_token_id)?;
                compiled.set_special_tokens(bos, eos);
                Some(compiled)
            }
            crate::weight::model::mistral::DenseGqaArchitecture::Mistral => None,
        };
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
        Ok(Self { context, config, weights, layers, output_head, tokenizer, detokenizer, chat_template, rope, cache_spec, kv_f16, max_seq_len })
    }

    /// GGUF metadata 里的 token id → 词表 token 串(detokenizer 还原)。
    fn reader_token_string(reader: &crate::weight::container::gguf::GgufReader, detokenizer: &Detokenizer, key: &str, fallback: u32) -> Result<String, String> {
        let id = reader.metadata(key).and_then(crate::weight::container::gguf::GgufValue::as_u64).map(|value| value as u32).unwrap_or(fallback);
        // 模板需要特殊 token 的原文；跳过它会把 BOS/EOS 渲染成空串，改变输入序列。
        let bytes = detokenizer.decode_bytes(&[id], false).map_err(|error| format!("还原 {key}={id} 的 token 串: {error}"))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
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
        token == self.config.eos_token_id
    }

    /// 请求 → prompt 字符串:有 GGUF chat template 时走模板渲染(数据驱动、
    /// 模型无关),否则回退 Mistral 字面拼接(已验证路径)。
    pub fn render_request_prompt(&self, request: &serde_json::Value) -> Result<String, String> {
        match &self.chat_template {
            Some(template) => template.render(request),
            None => mistral::mistral_request_prompt(request),
        }
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

    /// 同上,但按 temperature/top_p 采样(console 官方参考参数走这里)。
    pub fn next_token_sampled(&self, sequence: &MistralMetalSequence, sampling: &TokenSampling) -> Result<u32, String> {
        let output = mistral::mistral_token_output(&self.context, &self.config, &self.output_head, &sequence.hidden).map_err(|error| format!("Mistral output: {error:?}"))?;
        let token = self.context.sample_top_p(&output.logits, sampling.temperature, sampling.top_p, sampling.random).map_err(|error| format!("Mistral sample: {error:?}"))?;
        if std::env::var_os("ZLLM_SAMPLE_DEBUG").is_some() {
            self.audit_sampled_token(sequence, &output.logits, token, sampling);
        }
        Ok(token)
    }

    /// 采样审计(临时诊断,ZLLM_SAMPLE_DEBUG 门控):回读整行 logits,
    /// 报告被采样 token 的真实排名/概率/nucleus 大小,定位分布是否被采歪。
    fn audit_sampled_token(&self, sequence: &MistralMetalSequence, logits: &crate::backend::metal::MetalTensor, token: u32, sampling: &TokenSampling) {
        let length = logits.rows * logits.cols;
        let staging = self.context.shared_buffer_zeros(length * 2);
        let command = self.context.command_buffer();
        let encoder = command.new_blit_command_encoder();
        encoder.copy_from_buffer(&logits.buffer, 0, &staging, 0, staging.length());
        encoder.end_encoding();
        self.context.commit_and_wait(command.as_ref());
        let halves = unsafe { std::slice::from_raw_parts(staging.contents().cast::<half::f16>(), length) };
        let values: Vec<f32> = halves.iter().map(|value| value.to_f32()).collect();
        let mut order: Vec<u32> = (0..length as u32).collect();
        order.sort_unstable_by(|left, right| values[*right as usize].total_cmp(&values[*left as usize]));
        let rank = order.iter().position(|candidate| *candidate == token).map(|index| index + 1).unwrap_or(0);
        let max = values[order[0] as usize];
        let weights: Vec<f32> = order.iter().map(|index| (values[*index as usize] - max).exp()).collect();
        let total: f32 = weights.iter().sum();
        let mut nucleus = 0usize;
        let mut cumulative = 0.0f32;
        for weight in &weights {
            nucleus += 1;
            cumulative += weight;
            if cumulative >= sampling.top_p * total {
                break;
            }
        }
        let probability = weights.get(rank - 1).copied().unwrap_or(0.0) / total;
        eprintln!("[sample-debug] pos={} sampled={token} rank={rank} p={probability:.5} nucleus={nucleus} argmax={} argmax_p={:.3}", sequence.tokens.len(), order[0], weights[0] / total);
    }

    /// 引擎上报的 model_key:K2-Horizon 与 Mistral 共用 runtime 但分属两个模型,
    /// 采样参数/能力上报按各自 model_key 区分(见 runtime::official_sampling)。
    pub fn model_key(&self) -> &'static str {
        match self.config.architecture {
            crate::weight::model::mistral::DenseGqaArchitecture::K2Horizon => "k2-horizon",
            crate::weight::model::mistral::DenseGqaArchitecture::Mistral => "mistral",
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_preserves_special_tokens() {
        // 只写元数据，不依赖模型权重或 Metal 设备。
        let key = "tokenizer.ggml.bos_token_id";
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        let path = std::env::temp_dir().join(format!("zllm-metal-special-{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let reader = crate::weight::container::gguf::GgufReader::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let tokens = vec!["<|ifm|begin_of_text|>".to_owned(), "<|ifm|end_of_text|>".to_owned()];
        let detokenizer = Detokenizer::from_bpe_tokens(&tokens, &[true, true]).unwrap();
        let bos = MistralMetalSession::reader_token_string(&reader, &detokenizer, key, 1).unwrap();
        let eos = MistralMetalSession::reader_token_string(&reader, &detokenizer, "tokenizer.ggml.eos_token_id", 1).unwrap();
        let mut template = crate::runtime::chat_template::ChatTemplate::new("{{ bos_token }}{{ messages[0].content }}{{ eos_token }}").unwrap();
        template.set_special_tokens(bos, eos);
        let prompt = template.render(&serde_json::json!({"messages": [{"role": "user", "content": "你好"}]})).unwrap();
        assert_eq!(prompt, "<|ifm|begin_of_text|>你好<|ifm|end_of_text|>");
    }
}
