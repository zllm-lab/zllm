use super::OrnithOptions;

use std::{path::Path, sync::Arc};

use crate::{
    attention::{AttentionSpec, gated_delta_net::GatedDeltaNetState, rope::RopeTable},
    backend::{
        Backend,
        metal::{MetalContext, MetalGatedDeltaNetStorage, MetalKvCache, MetalMoeDecodeState, MetalPrefillExperts, MetalTensor, MetalWeight},
    },
    kv_cache::{KvCacheLayerMap, KvCacheSpec},
    moe::expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
    runtime::expert_pipeline::ExpertDecodePipeline,
    tokenizer::{Detokenizer, Tokenizer},
    weight::expert_source::GgufExpertSource,
};

use crate::runtime::ornith::{self, OrnithConfig, OrnithGguf, OrnithLayer, OrnithLayerKind, OrnithOutputHead};

pub struct OrnithMetalSequence {
    cache: MetalKvCache,
    recurrent: GatedDeltaNetState<MetalGatedDeltaNetStorage>,
    hidden: MetalTensor,
    tokens: Vec<u32>,
}

impl OrnithMetalSequence {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

pub struct OrnithMetalDecoder {
    experts: ExpertDecodePipeline<MetalMoeDecodeState>,
}

pub struct OrnithMetalSession {
    context: MetalContext,
    config: OrnithConfig,
    weights: Arc<OrnithGguf>,
    layers: Vec<OrnithLayer<MetalWeight>>,
    output_head: OrnithOutputHead<MetalWeight>,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    cache_spec: KvCacheSpec,
    cache_layers: KvCacheLayerMap,
    rope: RopeTable,
    expert_state: Option<MetalMoeDecodeState>,
    experts_preloaded: bool,
    resident_expert_bytes: u64,
    max_seq_len: usize,
    options: OrnithOptions,
}

impl OrnithMetalSession {
    fn runtime(&self) -> ornith::OrnithRuntime<'_, MetalContext> {
        ornith::OrnithRuntime::new(&self.context, &self.config, &self.layers, 0, &self.rope, self.options.runtime)
    }

    pub fn load_with_replay(model_path: &Path, max_seq_len: usize, options: OrnithOptions, replay_enabled: bool, lm_head_quantization: crate::weight::LmHeadQuantization) -> Result<Self, String> {
        let weights = Arc::new(OrnithGguf::open(model_path)?);
        let config = weights.config().clone();
        ornith::ensure_supported(&config).map_err(|error| format!("Ornith runtime 不支持: {error:?}"))?;
        MetalMoeDecodeState::validate_gguf_expert_formats(weights.as_ref(), config.layer_count, config.num_experts).map_err(|error| format!("Ornith Metal expert 格式预检失败: {error:?}"))?;
        let tokenizer = weights.tokenizer()?;
        let detokenizer = weights.detokenizer()?;
        let context = MetalContext::new_default_with_replay(replay_enabled).map_err(|error| format!("MetalContext 初始化失败: {error}"))?;
        let layers = ornith::prepare_ornith_layers(&context, weights.as_ref()).map_err(|error| format!("准备 Ornith 层失败: {error:?}"))?;
        let output_head = ornith::prepare_ornith_output_head_quantized(&context, weights.as_ref(), lm_head_quantization).map_err(|error| format!("准备 Ornith 输出头失败: {error:?}"))?;
        let cache_spec = KvCacheSpec::from_attention(&AttentionSpec::Gqa(config.full_attention_spec()))?;
        let cache_layers = ornith::kv_cache_layer_map(&config)?;
        let rope = RopeTable::precompute(max_seq_len, config.rope_dim, config.rope_theta);
        let expert_state = MetalMoeDecodeState::with_gguf_cache_gib(options.expert_cache_gib)?;
        Ok(Self { context, config, weights, layers, output_head, tokenizer, detokenizer, cache_spec, cache_layers, rope, expert_state: Some(expert_state), experts_preloaded: false, resident_expert_bytes: 0, max_seq_len, options })
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

    pub fn layer_count(&self) -> usize {
        self.config.layer_count
    }

    pub fn resident_expert_bytes(&self) -> u64 {
        self.resident_expert_bytes
    }

    /// 每个 sequence 实际长期持有的整块 KV + DeltaNet recurrent state。
    /// hidden 是每步更换的短生命 tensor，不纳入 session resident footprint。
    pub fn session_residency_bytes(&self) -> Result<usize, String> {
        let format = if self.options.kv_f16 { crate::kv_cache::KvCacheFormat::F16 } else { crate::kv_cache::KvCacheFormat::Int8 };
        let cache = crate::kv_cache::KvCacheLayout::new_mapped(self.cache_spec.clone(), format, self.cache_layers.clone(), self.max_seq_len, crate::kv_cache::DEFAULT_GROUP_SIZE)?.total_bytes();
        let recurrent_layers = (0..self.config.layer_count).filter(|&layer| self.config.ornith_layer_kind(layer) == Some(OrnithLayerKind::DeltaNet)).count();
        let spec = self.config.gated_delta_net_spec();
        let recurrent_per_layer = spec.conv_state_elements().checked_add(spec.recurrent_elements()).and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>())).ok_or("Ornith DeltaNet resident bytes 溢出")?;
        cache.checked_add(recurrent_layers.checked_mul(recurrent_per_layer).ok_or("Ornith DeltaNet 总 resident bytes 溢出")?).ok_or("Ornith session resident bytes 溢出".to_owned())
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.tokenize(text.as_bytes())
    }

    pub fn decode_bytes(&self, token: u32) -> Result<Vec<u8>, String> {
        self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("Ornith detokenize {token}: {error}"))
    }

    pub fn is_eos(&self, token: u32) -> bool {
        self.config.eos_token_ids.contains(&token)
    }

    pub fn prefill(&self, tokens: Vec<u32>) -> Result<OrnithMetalSequence, String> {
        if tokens.is_empty() {
            return Err("Ornith prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.max_seq_len {
            return Err(format!("Ornith prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let mut cache = if self.options.kv_f16 {
            MetalKvCache::new_f16_mapped(&self.context, self.cache_spec.clone(), self.cache_layers.clone(), self.max_seq_len)
        } else {
            MetalKvCache::new_mapped(&self.context, self.cache_spec.clone(), self.cache_layers.clone(), self.max_seq_len)
        }
        .map_err(|error| format!("Ornith KV cache: {error}"))?;
        let mut recurrent = GatedDeltaNetState::<MetalGatedDeltaNetStorage>::new(self.config.layer_count, self.config.gated_delta_net_spec()).map_err(|error| format!("Ornith DeltaNet state: {error:?}"))?;
        let source: Arc<dyn GgufExpertSource> = self.weights.clone();
        let mut experts = MetalPrefillExperts::gguf(source);
        let embedding = self.weights.embedding_rows(&tokens).map_err(|error| format!("Ornith embedding: {error}"))?;
        let input = self.context.tensor_from_f32_preserve(&embedding, tokens.len(), self.config.hidden_size).map_err(|error| format!("上传 Ornith embedding: {error}"))?;
        let output = self.runtime().at(&mut cache, &mut recurrent, 0).prefill(&mut experts, input).map_err(|error| format!("Ornith prefill: {error:?}"))?;
        let hidden = self.context.select_row(&output, tokens.len() - 1).map_err(|error| format!("Ornith 选择最后 token: {error:?}"))?;
        Ok(OrnithMetalSequence { cache, recurrent, hidden, tokens })
    }

    pub fn extend(&self, sequence: &mut OrnithMetalSequence, suffix: &[u32]) -> Result<(), String> {
        if suffix.is_empty() {
            return Ok(());
        }
        let end = sequence.tokens.len().checked_add(suffix.len()).ok_or("Ornith 序列长度溢出")?;
        if end >= self.max_seq_len {
            return Err(format!("Ornith 会话状态 {end} tokens 超过 max_seq_len {}", self.max_seq_len));
        }
        let source: Arc<dyn GgufExpertSource> = self.weights.clone();
        let mut experts = MetalPrefillExperts::gguf(source);
        let embedding = self.weights.embedding_rows(suffix).map_err(|error| format!("Ornith 增量 embedding: {error}"))?;
        let input = self.context.tensor_from_f32_preserve(&embedding, suffix.len(), self.config.hidden_size).map_err(|error| format!("上传 Ornith 增量 embedding: {error}"))?;
        let output = self.runtime().at(&mut sequence.cache, &mut sequence.recurrent, sequence.tokens.len()).prefill(&mut experts, input).map_err(|error| format!("Ornith 增量 prefill: {error:?}"))?;
        sequence.hidden = self.context.select_row(&output, suffix.len() - 1).map_err(|error| format!("Ornith 选择增量最后 token: {error:?}"))?;
        sequence.tokens.extend_from_slice(suffix);
        Ok(())
    }

    /// eager 模式在建立 admission 快照前把专家权重常驻；lazy 模式保留流式加载。
    pub fn preload_experts(&mut self) -> Result<Option<(usize, usize)>, String> {
        if self.experts_preloaded || self.options.lazy_experts {
            return Ok(None);
        }
        let mut state = self.expert_state.take().ok_or("Ornith expert state 不可用")?;
        let source: Arc<dyn GgufExpertSource> = self.weights.clone();
        let result = state.preload_gguf_experts(&self.context, source.as_ref(), self.config.layer_count, self.config.num_experts).map_err(|error| format!("Ornith expert 常驻失败: {error:?}"));
        self.expert_state = Some(state);
        let (experts, bytes) = result?;
        self.experts_preloaded = true;
        self.resident_expert_bytes = bytes as u64;
        Ok(Some((experts, bytes)))
    }

    pub fn begin_decode(&mut self) -> Result<OrnithMetalDecoder, String> {
        let state = self.expert_state.take().ok_or("Ornith expert state 不可用")?;
        let experts = ExpertDecodePipeline::new(
            state,
            ExpertPredictorConfig {
                first_layer: 0,
                layer_count: self.config.layer_count,
                expert_count: self.config.num_experts,
                routed_top_k: self.config.num_experts_per_tok,
                prefetch_count: self.options.expert_prefetch_count.unwrap_or(0).min(self.config.num_experts),
                weights: ExpertPredictorWeights::default(),
            },
        )
        .map_err(|error| format!("Ornith expert pipeline: {error}"))?;
        Ok(OrnithMetalDecoder { experts })
    }

    pub fn finish_decode(&mut self, decoder: OrnithMetalDecoder) {
        self.expert_state = Some(decoder.experts.into_backend_state());
    }

    pub fn next_token(&self, sequence: &OrnithMetalSequence) -> Result<u32, String> {
        ornith::ornith_token_output(&self.context, &self.config, &self.output_head, &sequence.hidden).map(|output| output.token_id).map_err(|error| format!("Ornith output: {error:?}"))
    }

    pub fn decode_token(&self, sequence: &mut OrnithMetalSequence, decoder: &mut OrnithMetalDecoder, token: u32) -> Result<(), String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("Ornith 会话状态 {} tokens 超过 max_seq_len {}", sequence.tokens.len(), self.max_seq_len));
        }
        let embedding = self.weights.embedding_rows(&[token]).map_err(|error| format!("Ornith decode embedding: {error}"))?;
        let input = self.context.tensor_from_f32_preserve(&embedding, 1, self.config.hidden_size).map_err(|error| format!("上传 Ornith decode embedding: {error}"))?;
        let position = sequence.tokens.len();
        sequence.hidden = self.runtime().at(&mut sequence.cache, &mut sequence.recurrent, position).decode(self.weights.as_ref(), &mut decoder.experts, input).map_err(|error| format!("Ornith decode position={position}: {error:?}"))?;
        sequence.tokens.push(token);
        Ok(())
    }
}
