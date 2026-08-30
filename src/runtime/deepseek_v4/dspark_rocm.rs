//! DeepSeek-V4 内嵌 DSpark 的 ROCm 组合。
//!
//! block draft、验证和统计继续复用 `runtime::speculative`；本模块只拥有 DeepSeek
//! 特有的 capture 投影、CSA/mHC/MoE 三层主干与 Markov 修正。

use std::sync::Arc;

use crate::{
    attention::{compressed_sparse::CompressedSparseKernel, hyper_connection::HyperConnectionKernel},
    backend::rocm::{RocmCompressedKvSerde, RocmCompressedKvStorage, RocmContext, RocmPrefillExperts, RocmTensor, RocmWeight},
    backend::{Backend, BackendError, BackendResources, SegmentedTensorBackend, StageExecutionBackend, TokenFence},
    runtime::{LinearWeight, compute_error, speculative::SpeculativeBlock},
    weight::model::deepseek_v4_dspark::DeepSeekV4DsparkCheckpoint,
};

use super::{
    DeepSeekV4, DeepSeekV4Config, DeepSeekV4OutputHead, DeepSeekV4PrefillSegment, DeepSeekV4PreparedLayer, DeepSeekV4RopeTables, deepseek_v4_output_hidden, deepseek_v4_prefill_layer, deepseek_v4_prefill_layer_segmented,
    deepseek_v4_store_attention_kv, prepare_deepseek_v4_loaded_layer, prepare_dense_tensor, prepare_hyper_connection, prepare_unit_norm,
};

struct DsparkInput {
    projection: RocmWeight,
    norm: RocmWeight,
}

struct DsparkStage {
    context: RocmContext,
    layer: DeepSeekV4PreparedLayer<RocmWeight>,
    sessions: Vec<Option<DsparkSession>>,
    experts: RocmPrefillExperts,
}

struct DsparkSession {
    target_cache: RocmCompressedKvStorage,
    block_cache: RocmCompressedKvStorage,
}

pub struct RocmDeepSeekV4DsparkSession {
    stages: Vec<DsparkSession>,
}

pub struct RocmDeepSeekV4DsparkCache {
    pub stages: Vec<(RocmCompressedKvSerde, RocmCompressedKvSerde)>,
}

impl RocmDeepSeekV4DsparkSession {
    pub fn fork_session(&self) -> Result<Self, BackendError> {
        Ok(Self { stages: self.stages.iter().map(|stage| Ok(DsparkSession { target_cache: stage.target_cache.fork_session()?, block_cache: stage.block_cache.fork_session()? })).collect::<Result<_, BackendError>>()? })
    }

    pub fn reset_session(&mut self) {
        for stage in &mut self.stages {
            stage.target_cache.reset_session();
            stage.block_cache.reset_session();
        }
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.stages.iter().map(|stage| stage.target_cache.allocated_bytes().saturating_add(stage.block_cache.allocated_bytes())).sum()
    }

    pub fn download_cache(&self) -> Result<RocmDeepSeekV4DsparkCache, BackendError> {
        Ok(RocmDeepSeekV4DsparkCache { stages: self.stages.iter().map(|stage| Ok((stage.target_cache.download_snapshot()?, stage.block_cache.download_snapshot()?))).collect::<Result<_, BackendError>>()? })
    }
}

pub struct RocmDeepSeekV4Dspark {
    output_context: RocmContext,
    checkpoint: DeepSeekV4DsparkCheckpoint,
    config: DeepSeekV4Config,
    model: DeepSeekV4,
    rope: Arc<DeepSeekV4RopeTables>,
    input: DsparkInput,
    stages: Vec<DsparkStage>,
    head: DeepSeekV4OutputHead<RocmWeight>,
    markov_embedding: Vec<u16>,
    markov_projection: RocmWeight,
    confidence_projection: Option<RocmWeight>,
    confidence_threshold: Option<f32>,
    draft_tokens: usize,
    profile: bool,
    pending_target_prefills: Vec<(RocmContext, crate::backend::rocm::RocmStageCompletion)>,
}

impl RocmDeepSeekV4Dspark {
    pub fn restore_session(&self, snapshot: RocmDeepSeekV4DsparkCache) -> Result<RocmDeepSeekV4DsparkSession, BackendError> {
        if snapshot.stages.len() != self.stages.len() {
            return Err(compute_error(format!("DeepSeek-V4 DSpark cache stages={}，期望 {}", snapshot.stages.len(), self.stages.len())));
        }
        let mut restored = Vec::with_capacity(self.stages.len());
        for (layer, (target, block)) in snapshot.stages.into_iter().enumerate() {
            let stage = &self.stages[layer];
            let spec = self.model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?;
            stage.context.activate().map_err(compute_error)?;
            let mut target_cache = stage.context.allocate_compressed_kv(&spec.attention)?;
            let mut block_cache = stage.context.allocate_compressed_kv(&spec.attention)?;
            target_cache.upload_snapshot(target)?;
            block_cache.upload_snapshot(block)?;
            restored.push(DsparkSession { target_cache, block_cache });
        }
        self.output_context.activate().map_err(compute_error)?;
        Ok(RocmDeepSeekV4DsparkSession { stages: restored })
    }

    pub fn load(
        context: RocmContext,
        checkpoint: DeepSeekV4DsparkCheckpoint,
        target_head: &DeepSeekV4OutputHead<RocmWeight>,
        rope: Arc<DeepSeekV4RopeTables>,
        devices: &[i32],
        draft_tokens: Option<usize>,
        confidence_threshold: Option<f32>,
        profile: bool,
    ) -> Result<Self, BackendError> {
        let draft_tokens = draft_tokens.unwrap_or(checkpoint.config.dspark_block_size);
        if draft_tokens == 0 {
            return Err(compute_error("DeepSeek-V4 DSpark draft_tokens 必须大于 0"));
        }
        let mut config = checkpoint.target_weights().config().clone();
        config.layer_count = checkpoint.config.dspark_target_layer_ids.len();
        config.mtp_layer_count = 0;
        config.hash_layer_count = 0;
        config.compress_ratios = vec![0; config.layer_count];
        let model = DeepSeekV4::new(config.clone()).map_err(|error| compute_error(error.to_string()))?;
        let input = checkpoint.load_input().map_err(compute_error)?;
        if devices.len() != config.layer_count {
            return Err(compute_error(format!("DeepSeek-V4 DSpark devices={} layers={}", devices.len(), config.layer_count)));
        }
        let input = DsparkInput {
            projection: context.prepare_weight(LinearWeight::block_fp8(&input.projection), config.hidden_size, checkpoint.config.dspark_target_layer_ids.len() * config.hidden_size)?,
            norm: prepare_dense_tensor(&context, &input.norm)?,
        };

        let mut stages = Vec::with_capacity(config.layer_count);
        for layer in 0..config.layer_count {
            let layer_context = context.for_device(devices[layer]).map_err(compute_error)?;
            layer_context.activate().map_err(compute_error)?;
            let weights = checkpoint.load_layer(layer).map_err(compute_error)?;
            let prepared = prepare_deepseek_v4_loaded_layer(&layer_context, &config, &weights)?;
            let spec = model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?;
            let mut experts = RocmPrefillExperts::mxfp4(Arc::new(checkpoint.expert_source()));
            // DSpark 与主模型复用同一 grouped 执行形态，避免额外保留逐专家 MXFP4 副本。
            let _ = experts.mxfp4_grouped(&layer_context, layer, &spec.feedforward)?;
            let layer_context = layer_context.with_independent_stream().map_err(compute_error)?;
            stages.push(DsparkStage { context: layer_context, layer: prepared, sessions: Vec::new(), experts });
        }
        context.activate().map_err(compute_error)?;

        let dspark_head = checkpoint.load_head().map_err(compute_error)?;
        let copies = config.hyper_connection_copies;
        let expanded = copies.checked_mul(config.hidden_size).ok_or_else(|| compute_error("DeepSeek-V4 DSpark output hidden 宽度溢出"))?;
        let hyper_connection = prepare_hyper_connection(&context, &dspark_head.hyper_connection)?;
        let output_norm = prepare_dense_tensor(&context, &dspark_head.norm)?;
        let head = DeepSeekV4OutputHead {
            input_norm: prepare_unit_norm(&context, expanded)?,
            function: hyper_connection.function,
            base: hyper_connection.base,
            scale: hyper_connection.scale,
            output: target_head.output.clone_with_prepared_norm(output_norm, false),
        };
        if dspark_head.markov_embedding.dtype != "BF16" || dspark_head.markov_embedding.shape != [checkpoint.config.vocab_size, checkpoint.config.dspark_markov_rank] {
            return Err(compute_error(format!("DeepSeek-V4 DSpark Markov embedding dtype={} shape={:?} 非法", dspark_head.markov_embedding.dtype, dspark_head.markov_embedding.shape)));
        }
        let markov_embedding = dspark_head.markov_embedding.data.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect();
        let markov_projection = prepare_dense_tensor(&context, &dspark_head.markov_projection)?;
        let confidence_projection = confidence_threshold.map(|_| prepare_dense_tensor(&context, &dspark_head.confidence_projection)).transpose()?;
        let output_context = context.with_independent_stream().map_err(compute_error)?;
        Ok(Self { output_context, checkpoint, config, model, rope, input, stages, head, markov_embedding, markov_projection, confidence_projection, confidence_threshold, draft_tokens, profile, pending_target_prefills: Vec::new() })
    }

    fn retire_target_prefills(&mut self) -> Result<(), BackendError> {
        let mut index = 0;
        while index < self.pending_target_prefills.len() {
            let ready = {
                let (context, completion) = &self.pending_target_prefills[index];
                context.stage_completion_ready(completion)?
            };
            if ready {
                self.pending_target_prefills.swap_remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    /// DSpark 权重与 expert 全部共享；每个会话只拥有两份很小的 Q8 recent-ring。
    pub fn open_session(&mut self, session: usize) -> Result<(), BackendError> {
        let opened = self.stages.iter().filter(|stage| stage.sessions.get(session).and_then(Option::as_ref).is_some()).count();
        if opened == self.stages.len() {
            return self.output_context.activate().map_err(compute_error);
        }
        if opened != 0 {
            return Err(compute_error(format!("DSpark session={session} 仅打开 {opened}/{} 层", self.stages.len())));
        }
        self.open_session_with(session, None)
    }

    pub fn open_session_with(&mut self, session: usize, restored: Option<RocmDeepSeekV4DsparkSession>) -> Result<(), BackendError> {
        let restored = match restored {
            Some(restored) if restored.stages.len() == self.stages.len() => Some(restored.stages),
            Some(restored) => return Err(compute_error(format!("DSpark session stages={} 期望={}", restored.stages.len(), self.stages.len()))),
            None => None,
        };
        for (layer, stage) in self.stages.iter().enumerate() {
            if stage.sessions.get(session).and_then(Option::as_ref).is_some() {
                return Err(compute_error(format!("DSpark session={session} layer={layer} 已打开")));
            }
        }
        let opened = match restored {
            Some(restored) => restored,
            None => {
                let mut opened = Vec::with_capacity(self.stages.len());
                for layer in 0..self.stages.len() {
                    let spec = self.model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?;
                    let stage = &self.stages[layer];
                    stage.context.activate().map_err(compute_error)?;
                    opened.push(DsparkSession { target_cache: stage.context.allocate_compressed_kv(&spec.attention)?, block_cache: stage.context.allocate_compressed_kv(&spec.attention)? });
                }
                opened
            }
        };
        self.output_context.activate().map_err(compute_error)?;
        for (stage, opened) in self.stages.iter_mut().zip(opened) {
            if stage.sessions.len() <= session {
                stage.sessions.resize_with(session + 1, || None);
            }
            stage.sessions[session] = Some(opened);
        }
        Ok(())
    }

    pub fn take_session(&mut self, session: usize) -> Result<RocmDeepSeekV4DsparkSession, BackendError> {
        self.retire_target_prefills()?;
        for (layer, stage) in self.stages.iter().enumerate() {
            if stage.sessions.get(session).and_then(Option::as_ref).is_none() {
                return Err(compute_error(format!("DSpark session={session} layer={layer} 未打开")));
            }
        }
        let mut stages = Vec::with_capacity(self.stages.len());
        for stage in &mut self.stages {
            stages.push(stage.sessions[session].take().expect("已确认全部 DSpark stage 已打开"));
        }
        Ok(RocmDeepSeekV4DsparkSession { stages })
    }

    pub fn close_session(&mut self, session: usize) {
        // take_session 先完整校验再修改，失败不会制造半开/半关 session。
        if self.take_session(session).is_err() {
            for stage in &mut self.stages {
                if let Some(slot) = stage.sessions.get_mut(session) {
                    *slot = None;
                }
            }
        }
    }

    pub fn prefill_target(&mut self, captures: &[RocmTensor], positions: &[usize]) -> Result<(), BackendError> {
        self.prefill_target_session(0, captures, positions)
    }

    pub fn prefill_target_session(&mut self, session: usize, captures: &[RocmTensor], positions: &[usize]) -> Result<(), BackendError> {
        self.retire_target_prefills()?;
        self.open_session(session)?;
        if captures.len() != self.checkpoint.config.dspark_target_layer_ids.len() || positions.is_empty() {
            return Err(compute_error(format!("DeepSeek-V4 DSpark captures={} positions={}", captures.len(), positions.len())));
        }
        let rows = self.output_context.token_rows(&captures[0]);
        if rows != positions.len() || captures.iter().any(|capture| self.output_context.token_rows(capture) != rows || self.output_context.token_cols(capture) != self.config.hidden_size) {
            return Err(compute_error("DeepSeek-V4 DSpark capture shape 不一致"));
        }
        self.output_context.activate().map_err(compute_error)?;
        let projected = self.output_context.block_fp8_three_segment_linear([&captures[0], &captures[1], &captures[2]], &self.input.projection)?;
        let main = self.output_context.rmsnorm(&projected, &self.input.norm, self.config.rms_eps)?;
        // projection stream 先把结果稳定到显式池；各 DSpark stream 用 event 等待
        // 同一 producer 后直接从源卡 fan-out，不再形成 device2→3→4 的串行链。
        let main = self.output_context.tensor_to_stable_deferred(main)?;
        for layer in 0..self.stages.len() {
            let spec = self.model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?;
            let stage = &mut self.stages[layer];
            stage.context.activate().map_err(compute_error)?;
            let stage_main = stage.context.tensor_on_device_ordered(main.clone())?;
            deepseek_v4_store_attention_kv(&stage.context, &self.config, spec, &stage.layer.attention, &mut stage.sessions[session].as_mut().expect("DSpark session 已打开").target_cache, &stage_main, self.rope.layer(spec), positions)?;
            self.pending_target_prefills.push((stage.context, stage.context.record_stage_completion()?));
        }
        self.output_context.activate().map_err(compute_error)?;
        Ok(())
    }

    pub fn prefill_target_prefix(&mut self, captures: &[RocmTensor], positions: &[usize], rows: usize) -> Result<(), BackendError> {
        self.prefill_target_prefix_session(0, captures, positions, rows)
    }

    pub fn prefill_target_prefix_session(&mut self, session: usize, captures: &[RocmTensor], positions: &[usize], rows: usize) -> Result<(), BackendError> {
        if rows == 0 || rows > positions.len() {
            return Err(compute_error(format!("DeepSeek-V4 DSpark retained rows={rows} positions={}", positions.len())));
        }
        self.output_context.activate().map_err(compute_error)?;
        let captures = captures.iter().map(|capture| self.output_context.slice_token_rows(capture, 0, rows)).collect::<Result<Vec<_>, _>>()?;
        self.prefill_target_session(session, &captures, &positions[..rows])
    }

    pub fn target_window_size(&self) -> Result<usize, BackendError> {
        self.model.layer_spec(0).map(|spec| spec.attention.window_size).map_err(|error| compute_error(error.to_string()))
    }

    pub fn prefill_target_suffix(&mut self, captures: &[RocmTensor], positions: &[usize], from_row: usize) -> Result<(), BackendError> {
        self.prefill_target_suffix_session(0, captures, positions, from_row)
    }

    pub fn prefill_target_suffix_session(&mut self, session: usize, captures: &[RocmTensor], positions: &[usize], from_row: usize) -> Result<(), BackendError> {
        if from_row >= positions.len() {
            return Err(compute_error(format!("DeepSeek-V4 DSpark suffix from={from_row} positions={}", positions.len())));
        }
        if from_row != 0 {
            self.open_session(session)?;
            for stage in &mut self.stages {
                stage.context.activate().map_err(compute_error)?;
                stage.sessions[session].as_mut().expect("DSpark session 已打开").target_cache.reset_session();
            }
        }
        let rows = positions.len() - from_row;
        self.output_context.activate().map_err(compute_error)?;
        let captures = captures.iter().map(|capture| self.output_context.slice_token_rows(capture, from_row, rows)).collect::<Result<Vec<_>, _>>()?;
        self.prefill_target_session(session, &captures, &positions[from_row..])
    }

    pub fn draft(&mut self, anchor: u32, position: usize) -> Result<SpeculativeBlock, BackendError> {
        self.draft_session(0, anchor, position, None)
    }

    pub fn draft_session(&mut self, session: usize, anchor: u32, position: usize, mut fence: Option<&mut dyn crate::runtime::generation_guard::TokenFenceProgram>) -> Result<SpeculativeBlock, BackendError> {
        self.retire_target_prefills()?;
        self.open_session(session)?;
        let profile = self.profile;
        let block_size = self.draft_tokens;
        let mut tokens = vec![self.checkpoint.config.dspark_noise_token_id; block_size];
        tokens[0] = anchor;
        let positions = (position..position + block_size).collect::<Vec<_>>();
        for stage in &mut self.stages {
            if profile {
                stage.context.profile_scope_begin("dspark_cache")?;
            }
            stage.context.activate().map_err(compute_error)?;
            let state = stage.sessions[session].as_mut().expect("DSpark session 已打开");
            state.block_cache.copy_recent_from(&state.target_cache)?;
            if profile {
                stage.context.profile_scope_end()?;
            }
        }

        self.output_context.activate().map_err(compute_error)?;
        if profile {
            self.output_context.profile_scope_begin("dspark_embed")?;
        }
        let embedding = self.checkpoint.target_weights().embedding_rows_bf16(&tokens).map_err(compute_error)?;
        let embedding = embedding.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect();
        let hidden = self.output_context.tensor_from_bf16_bits(embedding, block_size, self.config.hidden_size)?;
        let mut hidden = self.output_context.hyper_connection_expand(&hidden, self.config.hyper_connection_copies)?;
        if profile {
            self.output_context.profile_scope_end()?;
        }
        for layer in 0..self.stages.len() {
            let spec = self.model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?;
            let stage = &mut self.stages[layer];
            stage.context.activate().map_err(compute_error)?;
            if profile {
                stage.context.profile_scope_begin("dspark_layer")?;
            }
            hidden = stage.context.tensor_on_device(hidden)?;
            hidden = deepseek_v4_prefill_layer(
                &stage.context,
                &self.config,
                spec,
                &stage.layer,
                &mut stage.sessions[session].as_mut().expect("DSpark session 已打开").block_cache,
                &mut stage.experts,
                layer,
                &hidden,
                self.rope.layer(spec),
                &positions,
                &tokens,
                false,
                &spec.hyper_connection,
            )?;
            hidden = stage.context.tensor_as_bf16(hidden)?;
            if profile {
                stage.context.profile_scope_end()?;
            }
        }
        self.output_context.activate().map_err(compute_error)?;
        if profile {
            self.output_context.profile_scope_begin("dspark_head_move")?;
        }
        let hidden = self.output_context.tensor_on_device(hidden)?;
        if profile {
            self.output_context.profile_scope_end()?;
            self.output_context.profile_scope_begin("dspark_head_hidden")?;
        }
        let output_hidden = deepseek_v4_output_hidden(&self.output_context, &self.config, &self.head, &hidden)?;
        if profile {
            self.output_context.profile_scope_end()?;
            self.output_context.profile_scope_begin("dspark_head_linear")?;
        }
        let (_, logits) = crate::runtime::output::norm_and_lm_head(
            &self.output_context,
            &self.head.output,
            &output_hidden,
            &crate::runtime::output::OutputPlan { eps: self.config.rms_eps, norm: crate::runtime::output::OutputNorm::Rms, excluded_tokens: Vec::new() },
        )?;
        if profile {
            self.output_context.profile_scope_end()?;
        }
        let mut previous = anchor;
        let mut drafts = Vec::with_capacity(block_size);
        let mut markov_rows = self.confidence_projection.as_ref().map(|_| Vec::with_capacity(block_size));
        if profile {
            self.output_context.profile_scope_begin("dspark_markov")?;
        }
        for row in 0..block_size {
            let markov = self.markov_row(previous)?;
            let bias = self.output_context.linear(&markov, &self.markov_projection)?;
            let row = u32::try_from(row).map_err(|_| compute_error("DSpark logits row 超过 u32"))?;
            let token_fence = fence.as_deref().map(|fence| fence.fence()).unwrap_or_default();
            let token = argmax_add_rows_fenced(&self.output_context, &logits, &[row], &bias, &[token_fence])?[0];
            drafts.push(token);
            if let Some(fence) = fence.as_deref_mut() {
                fence.advance(token);
            }
            if let Some(rows) = &mut markov_rows {
                rows.push(markov);
            }
            previous = token;
        }
        if profile {
            self.output_context.profile_scope_end()?;
        }
        let confidences = if let (Some(projection), Some(rows)) = (&self.confidence_projection, markov_rows) {
            if profile {
                self.output_context.profile_scope_begin("dspark_confidence")?;
            }
            let rows = rows.iter().collect::<Vec<_>>();
            let markov = self.output_context.concat_token_rows(&rows)?;
            let features = self.output_context.concat_columns(&output_hidden, &markov)?;
            let confidence = self.output_context.linear(&features, projection)?;
            let confidences = self
                .output_context
                .tensor_to_f32(&confidence)?
                .into_iter()
                .map(|raw| {
                    if raw >= 0.0 {
                        1.0 / (1.0 + (-raw).exp())
                    } else {
                        let exp = raw.exp();
                        exp / (1.0 + exp)
                    }
                })
                .collect();
            if profile {
                self.output_context.profile_scope_end()?;
            }
            confidences
        } else {
            Vec::new()
        };
        let block = SpeculativeBlock::new(anchor, drafts, confidences)?;
        // 默认与 GLM DSpark 一致固定验证整块；只有显式配置 threshold
        // 才装配 confidence 权重并引入分数回读。
        match self.confidence_threshold {
            Some(threshold) => block.retain_confident_prefix(threshold),
            None => Ok(block),
        }
    }

    /// 只合并本轮已经就绪的会话。单路与 confidence 路径保留原语义，多路固定块
    /// 通过 segmented forward 共享每层 GEMM/MoE，不等待额外请求。
    pub fn draft_batch(&mut self, batch: &[(usize, u32, usize)], fences: &mut [Option<&mut dyn crate::runtime::generation_guard::TokenFenceProgram>]) -> Result<Vec<SpeculativeBlock>, BackendError> {
        self.retire_target_prefills()?;
        if batch.len() != fences.len() {
            return Err(compute_error(format!("DeepSeek-V4 DSpark batch fences={} 期望={}", fences.len(), batch.len())));
        }
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        if batch.len() == 1 || self.confidence_threshold.is_some() {
            let mut outputs = Vec::with_capacity(batch.len());
            for (index, &(session, anchor, position)) in batch.iter().enumerate() {
                outputs.push(self.draft_session(session, anchor, position, fences[index].take())?);
            }
            return Ok(outputs);
        }
        if batch.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(compute_error("DeepSeek-V4 DSpark batch session 必须严格递增"));
        }
        for &(session, _, _) in batch {
            self.open_session(session)?;
        }
        let profile = self.profile;
        for stage in &mut self.stages {
            stage.context.activate().map_err(compute_error)?;
            if profile {
                stage.context.profile_scope_begin("dspark_cache")?;
            }
            for &(session, _, _) in batch {
                let state = stage.sessions.get_mut(session).and_then(Option::as_mut).ok_or_else(|| compute_error(format!("DSpark batch session={session} 未打开")))?;
                state.block_cache.copy_recent_from(&state.target_cache)?;
            }
            if profile {
                stage.context.profile_scope_end()?;
            }
        }

        let block_size = self.draft_tokens;
        let mut tokens = vec![self.checkpoint.config.dspark_noise_token_id; batch.len() * block_size];
        let positions = batch
            .iter()
            .enumerate()
            .map(|(index, &(_, anchor, position))| {
                tokens[index * block_size] = anchor;
                (position..position + block_size).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.output_context.activate().map_err(compute_error)?;
        if profile {
            self.output_context.profile_scope_begin("dspark_embed")?;
        }
        let embedding = self.checkpoint.target_weights().embedding_rows_bf16(&tokens).map_err(compute_error)?;
        let embedding = embedding.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect();
        let hidden = self.output_context.tensor_from_bf16_bits(embedding, tokens.len(), self.config.hidden_size)?;
        let mut hidden = self.output_context.hyper_connection_expand(&hidden, self.config.hyper_connection_copies)?;
        if profile {
            self.output_context.profile_scope_end()?;
        }
        for layer in 0..self.stages.len() {
            let spec = self.model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?;
            let stage = &mut self.stages[layer];
            stage.context.activate().map_err(compute_error)?;
            if profile {
                stage.context.profile_scope_begin("dspark_layer")?;
            }
            hidden = stage.context.tensor_on_device(hidden)?;
            let mut indexed = stage.sessions.iter_mut().enumerate().filter_map(|(session, state)| batch.binary_search_by_key(&session, |&(session, _, _)| session).ok().map(|index| (index, state))).collect::<Vec<_>>();
            indexed.sort_unstable_by_key(|&(index, _)| index);
            if indexed.len() != batch.len() {
                return Err(compute_error(format!("DSpark batch L{layer} sessions={} 期望={}", indexed.len(), batch.len())));
            }
            let rope = self.rope.layer(spec);
            let mut segments = indexed
                .into_iter()
                .map(|(index, state)| {
                    let state = state.as_mut().expect("DSpark batch session 已打开");
                    DeepSeekV4PrefillSegment { cache: &mut state.block_cache, rope, positions: &positions[index], token_ids: &tokens[index * block_size..][..block_size], causal_batch: false }
                })
                .collect::<Vec<_>>();
            hidden = deepseek_v4_prefill_layer_segmented(&stage.context, &self.config, spec, &stage.layer, &mut stage.experts, layer, &hidden, &mut segments, &spec.hyper_connection)?;
            hidden = stage.context.tensor_as_bf16(hidden)?;
            if profile {
                stage.context.profile_scope_end()?;
            }
        }

        self.output_context.activate().map_err(compute_error)?;
        if profile {
            self.output_context.profile_scope_begin("dspark_head_move")?;
        }
        let hidden = self.output_context.tensor_on_device(hidden)?;
        if profile {
            self.output_context.profile_scope_end()?;
            self.output_context.profile_scope_begin("dspark_head_hidden")?;
        }
        let output_hidden = deepseek_v4_output_hidden(&self.output_context, &self.config, &self.head, &hidden)?;
        if profile {
            self.output_context.profile_scope_end()?;
            self.output_context.profile_scope_begin("dspark_head_linear")?;
        }
        let (_, logits) = crate::runtime::output::norm_and_lm_head(
            &self.output_context,
            &self.head.output,
            &output_hidden,
            &crate::runtime::output::OutputPlan { eps: self.config.rms_eps, norm: crate::runtime::output::OutputNorm::Rms, excluded_tokens: Vec::new() },
        )?;
        if profile {
            self.output_context.profile_scope_end()?;
        }
        let mut previous = batch.iter().map(|&(_, anchor, _)| anchor).collect::<Vec<_>>();
        let mut drafts = vec![Vec::with_capacity(block_size); batch.len()];
        if profile {
            self.output_context.profile_scope_begin("dspark_markov")?;
        }
        for depth in 0..block_size {
            let markov = self.markov_rows(&previous)?;
            let bias = self.output_context.linear(&markov, &self.markov_projection)?;
            let rows = (0..batch.len()).map(|session| u32::try_from(session * block_size + depth).map_err(|_| compute_error("DSpark batch logits row 超过 u32"))).collect::<Result<Vec<_>, _>>()?;
            let token_fences = fences.iter().map(|fence| fence.as_deref().map(|fence| fence.fence()).unwrap_or_default()).collect::<Vec<_>>();
            let mut proposed = argmax_add_rows_fenced(&self.output_context, &logits, &rows, &bias, &token_fences)?;
            if proposed.len() != batch.len() {
                return Err(compute_error(format!("DSpark batch argmax rows={} 期望={}", proposed.len(), batch.len())));
            }
            for index in 0..batch.len() {
                let token = proposed[index];
                proposed[index] = token;
                if let Some(fence) = fences[index].as_deref_mut() {
                    fence.advance(token);
                }
                let drafts = &mut drafts[index];
                drafts.push(token);
            }
            previous = proposed;
        }
        if profile {
            self.output_context.profile_scope_end()?;
        }
        batch.iter().zip(drafts).map(|(&(_, anchor, _), drafts)| SpeculativeBlock::new(anchor, drafts, Vec::new())).collect()
    }

    fn markov_row(&self, token: u32) -> Result<RocmTensor, BackendError> {
        self.markov_rows(&[token])
    }

    fn markov_rows(&self, tokens: &[u32]) -> Result<RocmTensor, BackendError> {
        let rank = self.checkpoint.config.dspark_markov_rank;
        let mut rows = Vec::with_capacity(tokens.len().saturating_mul(rank));
        for &token in tokens {
            let start = (token as usize).checked_mul(rank).ok_or_else(|| compute_error("DeepSeek-V4 DSpark Markov row 溢出"))?;
            rows.extend_from_slice(self.markov_embedding.get(start..start + rank).ok_or_else(|| compute_error(format!("DeepSeek-V4 DSpark Markov token {token} 越界")))?);
        }
        let rows = self.output_context.tensor_from_bf16_bits(rows, tokens.len(), rank)?;
        self.output_context.tensor_as_f32(rows)
    }
}

/// DSpark Markov bias 与主模型 logits 共同决定草稿 token，因此语法围栏必须在两者
/// 相加后应用。无约束时保留原融合 kernel；只有结构化输出才物化选中行。
fn argmax_add_rows_fenced(context: &RocmContext, logits: &RocmTensor, rows: &[u32], bias: &RocmTensor, fences: &[TokenFence]) -> Result<Vec<u32>, BackendError> {
    if rows.len() != fences.len() {
        return Err(compute_error(format!("DSpark fenced argmax rows={} fences={} 不一致", rows.len(), fences.len())));
    }
    if fences.iter().all(TokenFence::is_open) {
        return context.argmax_add_rows(logits, rows, bias);
    }
    let selected = context.select_rows(logits, rows)?;
    let combined = context.add(&selected, bias)?;
    context.argmax_rows_fenced(&combined, fences)
}
