use crate::{
    attention::AttentionSpec,
    backend::metal::{MetalContext, MetalKvCache, MetalMoeDecodeState, MetalPrefillExperts, MetalTensor},
    backend::{Backend, TokenSampling},
    kv_cache::{KvCacheLayerMap, KvCacheSpec},
    moe::expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
    runtime::{
        expert_pipeline::ExpertDecodePipeline,
        k2_horizon::{
            K2HorizonConfig, K2HorizonGguf,
            metal::{self, K2Layer, K2OutputHead},
        },
    },
    tokenizer::{Detokenizer, Tokenizer},
    weight::expert_source::GgufExpertSource,
};
use std::{path::Path, sync::Arc};

pub struct K2MetalSequence {
    pub cache: MetalKvCache,
    pub hidden: MetalTensor,
    pub tokens: Vec<u32>,
}

impl K2MetalSequence {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

pub struct K2MetalDecoder {
    experts: ExpertDecodePipeline<MetalMoeDecodeState>,
}

pub struct K2MetalSession {
    context: MetalContext,
    config: K2HorizonConfig,
    weights: Arc<K2HorizonGguf>,
    layers: Vec<K2Layer<crate::backend::metal::MetalWeight>>,
    output_head: K2OutputHead,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    chat_template: crate::runtime::chat_template::ChatTemplate,
    rope: crate::attention::rope::RopeTable,
    cache_spec: KvCacheSpec,
    expert_state: Option<MetalMoeDecodeState>,
    replay: Option<crate::runtime::k2_horizon::metal_replay::K2DecodeReplay>,
    kv_f16: bool,
    max_seq_len: usize,
}

impl K2MetalSession {
    pub fn load(model_path: &Path, max_seq_len: usize, kv_f16: bool, replay_enabled: bool, expert_cache_gib: usize, lm_head_quantization: crate::weight::LmHeadQuantization) -> Result<Self, String> {
        let weights = Arc::new(K2HorizonGguf::open(model_path).map_err(|error| format!("K2-Horizon GGUF 打开失败: {error}"))?);
        let config = *weights.config();
        crate::runtime::validate_max_sequence_length("K2-Horizon", max_seq_len, config.max_position_embeddings)?;
        let context = MetalContext::new_default_with_replay(replay_enabled).map_err(|error| format!("MetalContext 初始化失败: {error}"))?;
        let layers = metal::prepare_layers(&context, weights.as_ref()).map_err(|error| format!("准备 K2-Horizon layers: {error:?}"))?;
        let output_head = metal::prepare_output(&context, weights.as_ref(), lm_head_quantization).map_err(|error| format!("准备 K2-Horizon output: {error:?}"))?;
        let tokenizer = weights.tokenizer()?;
        let detokenizer = weights.detokenizer()?;
        let template_source = weights.reader().metadata("tokenizer.chat_template").and_then(crate::weight::container::gguf::GgufValue::as_str).ok_or("K2-Horizon GGUF 缺少 tokenizer.chat_template")?;
        let mut chat_template = crate::runtime::chat_template::ChatTemplate::new(template_source)?;
        let bos = Self::token_string(weights.reader(), &detokenizer, "tokenizer.ggml.bos_token_id", config.bos_token_id)?;
        let eos = Self::token_string(weights.reader(), &detokenizer, "tokenizer.ggml.eos_token_id", config.eos_token_id)?;
        chat_template.set_special_tokens(bos, eos);
        let rope = metal::rope_table(&config, max_seq_len)?;
        let _cache_layers = KvCacheLayerMap::dense(config.layer_count);
        let cache_spec = KvCacheSpec::from_attention(&AttentionSpec::Gqa(config.gqa_spec())).map_err(|error| format!("K2-Horizon KV cache spec: {error}"))?;
        MetalMoeDecodeState::validate_gguf_expert_range(weights.as_ref(), config.leading_dense_layer_count, config.layer_count, config.expert_count).map_err(|error| format!("K2-Horizon expert 格式: {error:?}"))?;
        let expert_state = Some(MetalMoeDecodeState::with_gguf_cache_gib(expert_cache_gib)?);
        Ok(Self { context, config, weights, layers, output_head, tokenizer, detokenizer, chat_template, rope, cache_spec, expert_state, replay: None, kv_f16, max_seq_len })
    }

    fn token_string(reader: &crate::weight::container::gguf::GgufReader, detokenizer: &Detokenizer, key: &str, fallback: u32) -> Result<String, String> {
        let id = reader.metadata(key).and_then(crate::weight::container::gguf::GgufValue::as_u64).map(|value| value as u32).unwrap_or(fallback);
        let bytes = detokenizer.decode_bytes(&[id], false).map_err(|error| format!("还原 {key}={id}: {error}"))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn runtime(&self) -> metal::K2MetalRuntime<'_> {
        metal::K2MetalRuntime::new(&self.context, &self.config, &self.layers, &self.rope)
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
    pub fn model_key(&self) -> &'static str {
        "k2-horizon"
    }
    pub fn session_residency_bytes(&self) -> Result<usize, String> {
        let format = if self.kv_f16 { crate::kv_cache::KvCacheFormat::F16 } else { crate::kv_cache::KvCacheFormat::Int8 };
        crate::kv_cache::KvCacheLayout::new(self.cache_spec.clone(), format, self.config.layer_count, self.max_seq_len, crate::kv_cache::DEFAULT_GROUP_SIZE).map(|layout| layout.total_bytes())
    }
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.tokenize(text.as_bytes())
    }
    pub fn decode_bytes(&self, token: u32) -> Result<Vec<u8>, String> {
        self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("K2-Horizon detokenize {token}: {error}"))
    }
    pub fn is_eos(&self, token: u32) -> bool {
        token == self.config.eos_token_id
    }
    pub fn render_request_prompt(&self, request: &serde_json::Value) -> Result<String, String> {
        self.chat_template.render(request)
    }

    pub fn preload_experts(&mut self) -> Result<(usize, usize), String> {
        let mut state = self.expert_state.take().ok_or("K2-Horizon expert state 不可用")?;
        let result =
            state.preload_gguf_expert_range(&self.context, self.weights.as_ref(), self.config.leading_dense_layer_count, self.config.layer_count, self.config.expert_count).map_err(|error| format!("K2-Horizon expert 常驻失败: {error:?}"));
        self.expert_state = Some(state);
        let (count, bytes) = result?;
        Ok((count, bytes))
    }

    pub fn prefill(&self, tokens: Vec<u32>) -> Result<K2MetalSequence, String> {
        if tokens.is_empty() || tokens.len() >= self.max_seq_len {
            return Err(format!("K2-Horizon prompt tokens={}，要求 1..{}", tokens.len(), self.max_seq_len));
        }
        let mut cache = if self.kv_f16 {
            MetalKvCache::new_f16(&self.context, self.cache_spec.clone(), self.config.layer_count, self.max_seq_len)
        } else {
            MetalKvCache::new(&self.context, self.cache_spec.clone(), self.config.layer_count, self.max_seq_len)
        }
        .map_err(|error| format!("K2-Horizon KV cache: {error}"))?;
        let source: Arc<dyn GgufExpertSource> = self.weights.clone();
        let mut experts = MetalPrefillExperts::gguf(source);
        let embedding = self.weights.embedding_rows(&tokens)?;
        let input = self.context.tensor_from_f32_preserve(&embedding, tokens.len(), self.config.hidden_size).map_err(|error| format!("上传 K2-Horizon embedding: {error}"))?;
        let output = self.runtime().prefill(&mut cache, &mut experts, input, 0).map_err(|error| format!("K2-Horizon prefill: {error:?}"))?;
        let hidden = self.context.select_row(&output, tokens.len() - 1).map_err(|error| format!("选择 K2-Horizon 最后 token: {error:?}"))?;
        Ok(K2MetalSequence { cache, hidden, tokens })
    }

    pub fn begin_decode(&mut self) -> Result<K2MetalDecoder, String> {
        let state = self.expert_state.take().ok_or("K2-Horizon expert state 不可用")?;
        let experts = ExpertDecodePipeline::new(
            state,
            ExpertPredictorConfig {
                first_layer: self.config.leading_dense_layer_count,
                layer_count: self.config.layer_count,
                expert_count: self.config.expert_count,
                routed_top_k: self.config.expert_top_k,
                prefetch_count: 0,
                weights: ExpertPredictorWeights::default(),
            },
        )?;
        Ok(K2MetalDecoder { experts })
    }

    pub fn replay_decode_available(&self, end_position: usize) -> bool {
        self.context.replay_enabled() && (self.kv_f16 || end_position <= 512)
    }

    pub fn ensure_replay(&mut self, sequence: &K2MetalSequence, decoder: &mut K2MetalDecoder) -> Result<(), String> {
        match self.replay.as_mut() {
            Some(replay) => replay.bind_cache(&sequence.cache),
            None => {
                let replay = crate::runtime::k2_horizon::metal_replay::K2DecodeReplay::record(
                    &self.context,
                    &sequence.cache,
                    &self.config,
                    &self.layers,
                    &self.rope,
                    &self.output_head,
                    self.weights.as_ref(),
                    &mut decoder.experts,
                    sequence.token_count(),
                )
                .map_err(|error| format!("K2-Horizon replay 录制: {error:?}"))?;
                eprintln!("[k2-horizon-replay] commands={}", replay.command_count());
                self.replay = Some(replay);
            }
        }
        Ok(())
    }

    /// 消费当前 token，重放一轮 decode + output，并返回下一 token。
    pub fn replay_decode_token(&mut self, sequence: &mut K2MetalSequence, token: u32) -> Result<u32, String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("K2-Horizon 会话已达 {} tokens", self.max_seq_len));
        }
        let embedding = self.weights.embedding_rows(&[token])?;
        let position = sequence.tokens.len();
        let next = self.replay.as_mut().ok_or("K2-Horizon replay 尚未录制")?.step(&self.context, &mut sequence.cache, &embedding, position).map_err(|error| format!("K2-Horizon replay position={position}: {error:?}"))?;
        sequence.tokens.push(token);
        Ok(next)
    }

    pub fn finish_decode(&mut self, decoder: K2MetalDecoder) {
        self.expert_state = Some(decoder.experts.into_backend_state());
    }

    pub fn next_token(&self, sequence: &K2MetalSequence) -> Result<u32, String> {
        metal::token_output(&self.context, &self.config, &self.output_head, &sequence.hidden).map(|output| output.token_id).map_err(|error| format!("K2-Horizon output: {error:?}"))
    }

    pub fn next_token_sampled(&self, sequence: &K2MetalSequence, sampling: &TokenSampling) -> Result<u32, String> {
        let output = metal::token_output(&self.context, &self.config, &self.output_head, &sequence.hidden).map_err(|error| format!("K2-Horizon output: {error:?}"))?;
        self.context.sample_top_p(&output.logits, sampling.temperature, sampling.top_p, sampling.random).map_err(|error| format!("K2-Horizon sample: {error:?}"))
    }

    pub fn decode_token(&self, sequence: &mut K2MetalSequence, decoder: &mut K2MetalDecoder, token: u32) -> Result<(), String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("K2-Horizon 会话已达 {} tokens", self.max_seq_len));
        }
        let embedding = self.weights.embedding_rows(&[token])?;
        let input = self.context.tensor_from_f32_preserve(&embedding, 1, self.config.hidden_size).map_err(|error| format!("上传 K2-Horizon decode embedding: {error}"))?;
        let position = sequence.tokens.len();
        sequence.hidden = self.runtime().decode(self.weights.as_ref(), &mut decoder.experts, &mut sequence.cache, input, position).map_err(|error| format!("K2-Horizon decode position={position}: {error:?}"))?;
        sequence.tokens.push(token);
        Ok(())
    }
}
