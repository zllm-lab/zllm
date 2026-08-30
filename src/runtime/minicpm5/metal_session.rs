use super::*;
use crate::attention::AttentionSpec;

pub struct MiniCpm5MetalSequence {
    pub cache: MetalKvCache,
    pub hidden: MetalTensor,
    pub tokens: Vec<u32>,
}

impl MiniCpm5MetalSequence {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

/// 设备端 embedding gather 的来源。
pub enum EmbeddingSource {
    /// tied lm_head 是常驻 Q6_K:直接抠行 dequant,零额外显存。
    Q6k { blob: crate::backend::metal::api::Buffer, row_bytes: usize },
    /// untied(或 tied 但非 Q6_K):token_embd 一次性 dequant 常驻 F16,gather 退化为行拷贝。
    F16(MetalTensor),
}

/// 一次已提交未读回的输出步:norm + lm_head + argmax 写入设备 id buffer,
/// `command` 是精确等待点(queue FIFO ⇒ 它完成即此前工作全部完成)。
pub struct PendingToken {
    id_buffer: crate::backend::metal::api::Buffer,
    /// 该 token 的 id 在 readback 中的字节偏移(每位置唯一,不会被覆写)。
    id_offset: u64,
    command: crate::backend::metal::api::CommandBuffer,
    profiles_through: usize,
    /// 该 token 在 sequence.tokens 的占位下标(submit_step 消费时登记)。
    placeholder: Option<usize>,
}

pub struct MiniCpm5MetalSession {
    context: MetalContext,
    config: MiniCpm5Config,
    weights: Arc<MiniCpm5Weights>,
    /// 全部层 resident 权重(load 时一次 prepare;逐 token 重传是 12x 减速,见 mod.rs)。
    layers: Vec<minicpm5::MiniCpm5TextLayer<MetalWeight>>,
    output_head: MiniCpm5OutputHead<MetalWeight>,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    cache_spec: KvCacheSpec,
    rope: RopeTable,
    kv_f16: bool,
    max_seq_len: usize,
    /// 从 GGUF metadata `tokenizer.ggml.eos_token_id` 读(1B 是 `</s>` = 1)。
    eos_token_id: u32,
    /// ChatML 轮次终止符 `<|im_end|>` 的 token id,从词表反查;找不到则只靠 EOS。
    im_end_token_id: Option<u32>,
    /// argmax 读回区:max_seq_len 个 u32,按 token 位置写入、永不覆写。
    /// 此前用 2 个 slot 交替,argmax(N+2) 与 CPU 读回 slot(N) 存在实测竞态
    /// (高负载下输出分叉;4 slot 只能推迟不能消除),改为按位置寻址根除。
    token_readback: crate::backend::metal::api::Buffer,
    /// 诊断用:logits 保活堆(仅 ZLLM_DEBUG_KEEP_LOGITS 时写入)。
    logits_graveyard: Mutex<Vec<MetalTensor>>,
    /// 设备端 embedding gather 来源;None 时异步流水线不可用,回落同步 decode_token。
    pub embedding_source: Option<EmbeddingSource>,
}

impl MiniCpm5MetalSession {
    pub fn load(model_path: &Path, max_seq_len: usize, kv_f16: bool, lm_head_quantization: crate::weight::LmHeadQuantization) -> Result<Self, String> {
        let weights = Arc::new(MiniCpm5Weights::open(model_path).map_err(|error| format!("MiniCPM5 GGUF 打开失败: {error}"))?);
        let config = *weights.config();
        crate::runtime::validate_max_sequence_length("MiniCPM5", max_seq_len, config.max_position_embeddings)?;
        let context = MetalContext::new_default().map_err(|error| format!("MetalContext 初始化失败: {error}"))?;
        let output_head = minicpm5::prepare_minicpm5_output_head_quantized(&context, &config, weights.as_ref(), lm_head_quantization).map_err(|error| format!("准备 MiniCPM5 output head: {error:?}"))?;
        let layers = minicpm5::prepare_minicpm5_layers(&context, weights.as_ref()).map_err(|error| format!("准备 MiniCPM5 层权重: {error:?}"))?;
        let tokenizer = weights.tokenizer().map_err(|error| format!("MiniCPM5 tokenizer: {error}"))?;
        let detokenizer = weights.detokenizer().map_err(|error| format!("MiniCPM5 detokenizer: {error}"))?;
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
        .map_err(|error| format!("MiniCPM5 KV cache spec: {error}"))?;
        let rope = RopeTable::precompute(max_seq_len, config.head_dim, config.rope_theta);
        let eos_token_id = minicpm5::minicpm5_eos_token_id(weights.as_ref());
        // 特殊 token 单独成词表条目,encode 出来应恰好是单 token。
        let im_end_tokens = tokenizer.tokenize_with_special(b"<|im_end|>", true);
        let im_end_token_id = (im_end_tokens.len() == 1).then_some(im_end_tokens[0]);
        let token_readback = context.shared_buffer_uninit((max_seq_len + 1) * 4);
        // 设备端 gather 的 embedding 来源:tied 且 lm_head 是 Q6_K 时零拷贝复用;
        // untied(如本 1B GGUF 有独立 output.weight)必须把 token_embd 常驻
        // dequant 成 F16,否则 gather 到的是 logits 矩阵。
        let embedding_source = 'source: {
            if config.tied_embedding
                && let MetalWeight::Gguf { blob, tensor_type: 14, row_bytes, rows, cols } = output_head.lm_head()
                && *rows == config.vocab_size
                && *cols == config.hidden_size
            {
                break 'source Some(EmbeddingSource::Q6k { blob: blob.clone(), row_bytes: *row_bytes });
            }
            match weights.embedding_matrix().and_then(|matrix| {
                let resident = context.prepare_weight(LinearWeight::gguf(&matrix), matrix.rows, matrix.columns).map_err(|error| format!("{error:?}"))?;
                match resident {
                    MetalWeight::Gguf { blob, tensor_type, row_bytes, rows, cols } if rows == config.vocab_size && cols == config.hidden_size => {
                        crate::kernel::metal::gguf::gguf_dequant_matrix_f16_tensor(&context, &blob, tensor_type, row_bytes, rows, cols)
                    }
                    _ => Err("MiniCPM5 embedding 不是 GGUF 量化矩阵".to_owned()),
                }
            }) {
                Ok(tensor) => Some(EmbeddingSource::F16(tensor)),
                Err(error) => {
                    eprintln!("[minicpm5] embedding 常驻化失败,异步 decode 不可用: {error}");
                    None
                }
            }
        };
        Ok(Self { context, config, weights, layers, output_head, tokenizer, detokenizer, cache_spec, rope, kv_f16, max_seq_len, eos_token_id, im_end_token_id, token_readback, logits_graveyard: Mutex::new(Vec::new()), embedding_source })
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
    /// 单会话满上下文的 KV cache 字节数,与 prefill 实际分配口径一致。
    pub fn session_capacity_bytes(&self) -> usize {
        let format = if self.kv_f16 { crate::kv_cache::KvCacheFormat::F16 } else { crate::kv_cache::KvCacheFormat::Int8 };
        crate::kv_cache::KvCacheLayout::new(self.cache_spec.clone(), format, self.config.layer_count, self.max_seq_len, crate::kv_cache::DEFAULT_GROUP_SIZE).map(|layout| layout.total_bytes()).unwrap_or(0)
    }
    pub fn eos_token_ids(&self) -> Vec<u32> {
        let mut ids = vec![self.eos_token_id];
        if let Some(im_end) = self.im_end_token_id {
            ids.push(im_end);
        }
        ids
    }
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.tokenize(text.as_bytes())
    }
    pub fn decode_bytes(&self, token: u32) -> Result<Vec<u8>, String> {
        self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("MiniCPM5 detokenize {token}: {error}"))
    }
    pub fn is_eos(&self, token: u32) -> bool {
        token == self.eos_token_id || self.im_end_token_id == Some(token)
    }

    pub fn prefill(&self, tokens: Vec<u32>) -> Result<MiniCpm5MetalSequence, String> {
        if tokens.is_empty() {
            return Err("MiniCPM5 prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.max_seq_len {
            return Err(format!("MiniCPM5 prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let mut cache = if self.kv_f16 {
            MetalKvCache::new_f16(&self.context, self.cache_spec.clone(), self.config.layer_count, self.max_seq_len)
        } else {
            MetalKvCache::new(&self.context, self.cache_spec.clone(), self.config.layer_count, self.max_seq_len)
        }
        .map_err(|error| format!("MiniCPM5 KV cache 分配: {error}"))?;
        let rope = minicpm5::minicpm5_rope_table(&self.config, tokens.len());
        let embedding = self.weights.embedding_rows_f32(&tokens).map_err(|error| format!("MiniCPM5 embedding: {error}"))?;
        // hidden 全程 F16:F32 会让每层 gemv 前多一次 cast、残差 add 走 f32(双倍带宽)
        let hidden = self.context.tensor_from_f32(&embedding, tokens.len(), self.config.hidden_size).map_err(|error| format!("上传 MiniCPM5 embedding: {error}"))?;
        let output = minicpm5::minicpm5_text_hidden(&self.context, &self.config, &self.layers, Some(&mut cache), hidden, &rope, 0).map_err(|error| format!("MiniCPM5 Metal prefill: {error:?}"))?;
        let last = self.context.select_row(&output, tokens.len() - 1).map_err(|error| format!("MiniCPM5 选择最后 token: {error:?}"))?;
        Ok(MiniCpm5MetalSequence { cache, hidden: last, tokens })
    }

    /// 在已有 sequence 的 cache 上续写 suffix(terminal cache resume 路径):
    /// 位置偏移取当前 token 数,attention 读取已有前缀 KV,只算后缀。
    pub fn extend(&self, sequence: &mut MiniCpm5MetalSequence, suffix: &[u32]) -> Result<(), String> {
        if suffix.is_empty() {
            return Ok(());
        }
        let position = sequence.tokens.len();
        if position + suffix.len() >= self.max_seq_len {
            return Err(format!("MiniCPM5 续写后 {} tokens 超过 max_seq_len {}", position + suffix.len(), self.max_seq_len));
        }
        let rope = minicpm5::minicpm5_rope_table(&self.config, position + suffix.len());
        let embedding = self.weights.embedding_rows_f32(suffix).map_err(|error| format!("MiniCPM5 embedding: {error}"))?;
        let hidden = self.context.tensor_from_f32(&embedding, suffix.len(), self.config.hidden_size).map_err(|error| format!("上传 MiniCPM5 embedding: {error}"))?;
        let output = minicpm5::minicpm5_text_hidden(&self.context, &self.config, &self.layers, Some(&mut sequence.cache), hidden, &rope, position).map_err(|error| format!("MiniCPM5 Metal 续写 prefill: {error:?}"))?;
        sequence.hidden = self.context.select_row(&output, suffix.len() - 1).map_err(|error| format!("MiniCPM5 选择最后 token: {error:?}"))?;
        sequence.tokens.extend_from_slice(suffix);
        Ok(())
    }

    pub fn next_token(&self, sequence: &MiniCpm5MetalSequence, sampling: Option<&crate::backend::TokenSampling>) -> Result<u32, String> {
        let result = match sampling {
            // 采样路径不排除 eos/im_end：模型能自然停在 <|im_end|>
            Some(sampling) if sampling.temperature > 0.0 => minicpm5::minicpm5_sampled_token_output(&self.context, &self.config, &self.output_head, &sequence.hidden, sampling),
            _ => {
                let eos = self.eos_token_ids();
                minicpm5::minicpm5_token_output(&self.context, &self.config, &self.output_head, &sequence.hidden, &eos)
            }
        };
        result.map(|output| output.token_id).map_err(|error| format!("MiniCPM5 output: {error:?}"))
    }

    pub fn decode_token(&self, sequence: &mut MiniCpm5MetalSequence, token: u32) -> Result<(), String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("MiniCPM5 会话状态 {} tokens 超过 max_seq_len {}", sequence.tokens.len(), self.max_seq_len));
        }
        let embedding = self.weights.embedding_rows_f32(&[token]).map_err(|error| format!("MiniCPM5 decode embedding: {error}"))?;
        let input = self.context.tensor_from_f32(&embedding, 1, self.config.hidden_size).map_err(|error| format!("上传 MiniCPM5 decode embedding: {error}"))?;
        let position = sequence.tokens.len();
        // 复用 session 级全表(precompute 一次):逐 token 重建 position+1 行是
        // O(n²) CPU sin/cos,长上下文每 token 多花数毫秒;row 由 position 选取。
        sequence.hidden = minicpm5::minicpm5_decode_round(&self.context, &self.config, &self.layers, &mut sequence.cache, input, &self.rope, position).map_err(|error| format!("MiniCPM5 decode position={position}: {error:?}"))?;
        sequence.tokens.push(token);
        Ok(())
    }

    /// 异步流水线是否可用(需要设备端 embedding gather 来源)。
    pub fn async_decode_available(&self) -> bool {
        self.embedding_source.is_some()
    }

    /// 进入流水线提交窗口:defer 全部 GPU 等待,等待点改为 PendingToken 的 CB 句柄。
    pub fn begin_async_decode(&self) {
        self.context.set_deferred_layer_scope_sync(true);
        self.context.set_deferred_waits(true);
        // 一轮 decode 约 240 个算子;放大批上限让整轮只经输出步一个 flush 点,
        // 每 token 只提交 1 个 CB,省掉 13 次 CB 边界的 CPU commit 与 GPU 调度泡。
        if std::env::var_os("ZLLM_DEBUG_CB1").is_some() {
            self.context.set_deferred_batch_max_operations(1);
        } else {
            self.context.set_deferred_batch_max_operations(4096);
        }
    }

    /// 退出流水线窗口:恢复同步语义并 drain 全部在飞工作(含被 EOS 浪费的推测轮)。
    pub fn end_async_decode(&self) {
        self.context.set_deferred_batch_max_operations(16);
        self.context.set_deferred_waits(false);
        self.context.set_deferred_layer_scope_sync(false);
    }

    /// 提交输出步(不等待):rmsnorm + lm_head + argmax 按 token 位置写入读回区。
    pub fn submit_output(&self, hidden: &MetalTensor, position: usize) -> Result<PendingToken, String> {
        // EOS 由 wait 后的生成状态机截断；这里排除会让异步 greedy 永远不能自然结束。
        let plan = crate::runtime::output::OutputPlan { eps: self.config.rms_eps, norm: crate::runtime::output::OutputNorm::Rms, excluded_tokens: Vec::new() };
        let (_normed, logits) = crate::runtime::output::norm_and_lm_head(&self.context, &self.output_head, hidden, &plan).map_err(|error| format!("MiniCPM5 输出步: {error:?}"))?;
        // 诊断:logits 指纹 + 刀锋候选值,定位 argmax 输入是否一致
        if std::env::var_os("ZLLM_DEBUG_WAIT_RACE").is_some() {
            self.context.synchronize();
            let bits = unsafe { std::slice::from_raw_parts(logits.buffer.contents() as *const u16, logits.len()) };
            let xor = bits.iter().enumerate().fold(0u32, |acc, (index, value)| acc ^ (*value as u32).rotate_left((index % 31) as u32));
            let l15311 = half::f16::from_bits(bits[15311]).to_f32();
            let l22213 = half::f16::from_bits(bits[22213]).to_f32();
            eprintln!("[dbg] position={position} logits xor={xor:08x} l15311={l15311:.6} l22213={l22213:.6}");
        }
        let id_offset = (position * std::mem::size_of::<u32>()) as u64;
        let logits_keep = logits.clone();
        crate::kernel::metal::moe::argmax_tensor_into_offset(&self.context, &logits, &plan.excluded_tokens, &self.token_readback, id_offset)?;
        let (command, through) = self.context.submit_batch_and_last_command().ok_or("MiniCPM5 输出步没有已提交命令".to_owned())?;
        if std::env::var_os("ZLLM_DEBUG_KEEP_LOGITS").is_some() {
            self.logits_graveyard.lock().unwrap_or_else(|p| p.into_inner()).push(logits_keep);
        }
        Ok(PendingToken { id_buffer: self.token_readback.clone(), id_offset, command, profiles_through: through, placeholder: None })
    }

    /// 推测提交一轮:从 pending 的读回位置设备端 gather embedding → decode round →
    /// 输出步。全程无 CPU 同步;pending 的 token 在 sequence.tokens 先占位,wait 时回填。
    /// 读回区按 token 位置寻址,不存在覆写(见 token_readback 注释)。
    pub fn submit_step(&self, sequence: &mut MiniCpm5MetalSequence, pending: &mut PendingToken) -> Result<PendingToken, String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("MiniCPM5 会话状态 {} tokens 超过 max_seq_len {}", sequence.tokens.len(), self.max_seq_len));
        }
        let input = match &self.embedding_source {
            Some(source) => {
                // 诊断:gather 前全同步,验证 gather 是否在 argmax 完成前读 readback
                if std::env::var_os("ZLLM_DEBUG_SYNC_GATHER").is_some() {
                    self.context.synchronize();
                }
                match source {
                    EmbeddingSource::Q6k { blob, row_bytes } => {
                        crate::kernel::metal::gguf::gguf_gather_row_q6k_tensor_offset(&self.context, &pending.id_buffer, pending.id_offset, blob, self.config.vocab_size, self.config.hidden_size, *row_bytes)?
                    }
                    EmbeddingSource::F16(matrix) => crate::kernel::metal::shape::gather_row_f16_tensor_offset(&self.context, &pending.id_buffer, pending.id_offset, matrix)?,
                }
            }
            None => return Err("MiniCPM5 异步 decode 缺少设备端 embedding 来源".to_owned()),
        };
        let position = sequence.tokens.len();
        sequence.tokens.push(0);
        pending.placeholder = Some(position);
        sequence.hidden = minicpm5::minicpm5_decode_round_deferred(&self.context, &self.config, &self.layers, &mut sequence.cache, input, &self.rope, position).map_err(|error| format!("MiniCPM5 decode position={position}: {error:?}"))?;
        // 诊断:decode round 与输出步拆成两个 CB
        self.context.submit_batch();
        // 本轮产出的是 position+1 处的下一个 token;读回区下标必须与占位/产出对齐。
        self.submit_output(&sequence.hidden, position + 1)
    }

    /// 精确等待 pending 的 argmax CB 并读回 token id;已提交的推测轮次继续在 GPU 上跑。
    pub fn wait_token(&self, sequence: &mut MiniCpm5MetalSequence, pending: &PendingToken) -> u32 {
        pending.command.wait_until_completed();
        self.context.complete_profiles_through(pending.profiles_through);
        let token = unsafe { *pending.id_buffer.contents().cast::<u8>().add(pending.id_offset as usize).cast::<u32>() };
        // 临时:读回时机诊断——同步后复读,不一致说明等待点/写点有问题
        if std::env::var_os("ZLLM_DEBUG_WAIT_RACE").is_some() {
            self.context.synchronize();
            let settled = unsafe { *pending.id_buffer.contents().cast::<u8>().add(pending.id_offset as usize).cast::<u32>() };
            if settled != token {
                eprintln!("[race] wait_token 读到未结算值: offset={} 即时={token} 同步后={settled}", pending.id_offset);
            }
        }
        if let Some(position) = pending.placeholder
            && let Some(slot) = sequence.tokens.get_mut(position)
        {
            *slot = token;
        }
        token
    }

    /// 测试用:直接读 readback 区某位置的 token id。
    #[cfg(test)]
    pub fn readback_at(&self, position: usize) -> u32 {
        unsafe { *self.token_readback.contents().cast::<u8>().add(position * std::mem::size_of::<u32>()).cast::<u32>() }
    }

    /// 测试用:CPU 已确认 token 的串行推测步——embedding 走 CPU 上传(不经设备 gather),
    /// 其余(deferred batching、输出步并入提交窗口)与 submit_step 完全一致。
    /// 用于把"设备 gather/推测重叠"与"deferred 轮内容"两类竞态来源分开。
    #[cfg(test)]
    pub fn submit_step_cpu_embed(&self, sequence: &mut MiniCpm5MetalSequence, token: u32) -> Result<PendingToken, String> {
        if sequence.tokens.len() >= self.max_seq_len {
            return Err(format!("MiniCPM5 会话状态 {} tokens 超过 max_seq_len {}", sequence.tokens.len(), self.max_seq_len));
        }
        let embedding = self.weights.embedding_rows_f32(&[token]).map_err(|error| format!("MiniCPM5 decode embedding: {error}"))?;
        let input = self.context.tensor_from_f32(&embedding, 1, self.config.hidden_size).map_err(|error| format!("上传 MiniCPM5 decode embedding: {error}"))?;
        let position = sequence.tokens.len();
        sequence.tokens.push(token);
        sequence.hidden = minicpm5::minicpm5_decode_round_deferred(&self.context, &self.config, &self.layers, &mut sequence.cache, input, &self.rope, position).map_err(|error| format!("MiniCPM5 decode position={position}: {error:?}"))?;
        self.context.submit_batch();
        self.submit_output(&sequence.hidden, position + 1)
    }
}
