//! GLM 的 DFlash2 ROCm 组合：共享 target embedding/head，aux/cache 沿用块验证生命周期。
use super::{
    Glm52Config,
    dspark_rocm::{RocmDsparkDraftBatch, RocmDsparkProjection},
    rocm::gather_embedding_rows,
    stage::Glm52StageState,
};
use crate::{
    attention::rope::RopeTable,
    backend::{
        Backend, BackendError, SegmentedTensorBackend,
        rocm::{RocmContext, RocmTensor, RocmWeight},
    },
    runtime::{dflash2::Dflash2, dspark::DsparkTargetCache},
    weight::{ResidentWeightQuantization, model::dflash2::Dflash2Checkpoint},
};
use std::{path::Path, sync::Arc};

pub struct RocmDflash2Runtime {
    pub model: Dflash2<RocmWeight>,
    pub rope: RopeTable,
    draft_tokens: usize,
    max_seq_len: usize,
}

impl RocmDflash2Runtime {
    pub fn load(backend: &RocmContext, root: &Path, target: &Glm52Config, max_seq_len: usize, draft_tokens: usize) -> Result<Self, BackendError> {
        let checkpoint = Dflash2Checkpoint::open(root).map_err(compute)?;
        let config = checkpoint.config();
        if draft_tokens == 0 || draft_tokens >= config.block_size || max_seq_len > config.max_position_embeddings {
            return Err(compute("DFlash2 draft/context 长度超出 checkpoint"));
        }
        let model = Dflash2::load_drafter(backend, &checkpoint)?;
        model.capture_plan(target.layer_count, target.hidden_size, target.vocab_size)?;
        let rope = RopeTable::precompute(max_seq_len, config.head_dim, config.rope_theta);
        Ok(Self { model, rope, draft_tokens, max_seq_len })
    }

    pub fn warm_target_cache(&self, backend: &RocmContext, cache: &mut DsparkTargetCache<RocmTensor>, target: &RocmTensor, position: usize) -> Result<(), BackendError> {
        self.model.warm_target_cache(backend, cache, target, position, &self.rope)
    }

    pub fn draft_batch(&self, backend: &RocmContext, batch: &mut [RocmDsparkDraftBatch<'_>], embedding: &Arc<crate::kernel::rocm::hip::DeviceBuffer>, lm_head: &RocmWeight) -> Result<(), BackendError> {
        let config = self.model.config();
        if lm_head.rows() != config.vocab_size || lm_head.cols() != config.hidden_size {
            return Err(compute("DFlash2 target LM head shape 不兼容"));
        }
        for item in batch {
            // 固定非因果块不足剩余上下文时退回普通 decode，不越过 RoPE/KV 上限。
            if item.block_position.checked_add(config.block_size).is_none_or(|end| end > self.max_seq_len) {
                item.drafts.clear();
                continue;
            }
            let ids = self.model.block_token_ids(item.anchor)?;
            let noise = gather_embedding_rows(backend, embedding, &ids, config.hidden_size).map_err(compute)?;
            let hidden = self.model.forward_cached(backend, item.cache, noise, item.target_hidden, item.target_position, item.block_position, &self.rope)?;
            let hidden = backend.slice_token_rows(&hidden, 1, self.draft_tokens)?;
            let logits = backend.linear(&hidden, lm_head)?;
            let (projection, predecessor, successor) = self.model.selector_weights();
            let gate = backend.linear(&hidden, projection)?;
            item.drafts = backend.candidate_greedy(&logits, &gate, predecessor, successor, item.anchor, config.selector_top_k)?;
        }
        Ok(())
    }
}

pub fn attach_dflash2_projections(states: &mut [Glm52StageState<RocmContext>], reference: &[Glm52StageState<RocmContext>], root: &Path, target: &Glm52Config) -> Result<(), BackendError> {
    let checkpoint = Dflash2Checkpoint::open(root).map_err(compute)?;
    let config = checkpoint.config();
    if (target.layer_count, target.hidden_size, target.vocab_size) != (config.target_layer_count, config.hidden_size, config.vocab_size) {
        return Err(compute("DFlash2 stage target 配置不兼容"));
    }
    for (index, layer) in config.target_layer_ids.iter().enumerate() {
        let boundary = layer + 1;
        let Some(state) = states.iter_mut().find(|state| boundary > state.layer_start && boundary <= state.layer_start + state.layers.len()) else { continue };
        if let Some(projector) = reference.iter().find(|r| r.backend.device_id() == state.backend.device_id()).and_then(|r| r.hidden_projectors.iter().find(|p| p.boundary() == boundary)) {
            state.hidden_projectors.push(projector.clone());
        } else {
            let weight = super::dspark::prepare_dspark_matrix(&state.backend, checkpoint.capture_projection(index).map_err(compute)?, ResidentWeightQuantization::Native)?;
            state.hidden_projectors.push(Arc::new(RocmDsparkProjection { boundary, weight }));
        }
    }
    Ok(())
}

fn compute(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{BackendResources, cpu::CpuContext},
        kernel::cpu::CpuTensor,
        weight::model::dflash2::tests::{fixture, tiny_config},
    };

    #[test]
    fn dflash2_rocm_service_draft_and_cache_match_cpu() {
        let Ok(gpu) = RocmContext::new(0) else {
            eprintln!("ROCm 不可用，跳过 GPU oracle");
            return;
        };
        eprintln!("执行 ROCm DFlash2 service oracle，device=0");
        let mut config = tiny_config();
        // 使用设备 GEMV 支持的列宽；CPU 同权重作为 oracle。
        config.intermediate_size = 8;
        let files = fixture(&config);
        let checkpoint = Dflash2Checkpoint::open(&files.0).unwrap();
        let cpu = CpuContext;
        let reference = Dflash2::load(&cpu, &checkpoint).unwrap();
        let model = Dflash2::load(&gpu, &checkpoint).unwrap();
        let rope = RopeTable::precompute(16, 2, config.rope_theta);
        let captures: Vec<_> = (0..2).map(|j| CpuTensor { data: (0..12).map(|i| (((i * 3 + j * 5) % 13) as f32 - 6.) / 8.).collect(), rows: 3, cols: 4 }).collect();
        let projected = reference.project_target(&cpu, &[&captures[0], &captures[1]]).unwrap();
        let device_captures: Vec<_> = captures.iter().map(|x| gpu.tensor_from_f32(x.data.clone(), 3, 4).unwrap()).collect();
        let target = model.project_target(&gpu, &[&device_captures[0], &device_captures[1]]).unwrap();
        let close = |actual: Vec<f32>, expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                assert!((a - b).abs() <= 0.01 + 0.01 * b.abs(), "element {i}: {a} != {b}");
            }
        };
        close(gpu.tensor_to_f32(&target).unwrap(), &projected.data);
        let embedding_values: Vec<_> = (0..20).map(|i| ((i * 3 % 11) as f32 - 5.) / 8.).collect();
        let embedding_bytes: Vec<_> = embedding_values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let embedding = Arc::new(crate::kernel::rocm::hip::DeviceBuffer::upload(0, &embedding_bytes).unwrap());
        let noise = gather_embedding_rows(&gpu, &embedding, &[1, 4, 4], 4).unwrap();
        let cpu_noise = CpuTensor { data: [1, 4, 4].iter().flat_map(|&id| embedding_values[id * 4..id * 4 + 4].iter().copied()).collect(), rows: 3, cols: 4 };
        let expected = reference.forward_hidden(&cpu, cpu_noise.clone(), &projected, 1, 4, &rope).unwrap();
        let mut cache = DsparkTargetCache::new();
        let prefix = gpu.slice_token_rows(&target, 0, 2).unwrap();
        model.warm_target_cache(&gpu, &mut cache, &prefix, 1, &rope).unwrap();
        let hidden = model.forward_cached(&gpu, &mut cache, noise.clone(), &target, 1, 4, &rope).unwrap();
        close(gpu.tensor_to_f32(&hidden).unwrap(), &expected.data);
        assert!(cache.truncate_end(&gpu, 3).unwrap());
        // 拒绝后追加重新计算 suffix；持久化恢复不能带回被拒绝的行。
        let snapshot = cache.try_snapshot(|t| Ok::<_, BackendError>(t.clone())).unwrap().unwrap();
        let mut cache = DsparkTargetCache::from_snapshot(snapshot);
        let hidden = model.forward_cached(&gpu, &mut cache, noise, &target, 1, 4, &rope).unwrap();
        close(gpu.tensor_to_f32(&hidden).unwrap(), &expected.data);
        let head_values: Vec<_> = (0..20).map(|i| ((i * 5 % 13) as f32 - 6.) / 8.).collect();
        let head = gpu.prepare_f32(&head_values, 5, 4).unwrap();
        let runtime = RocmDflash2Runtime { model, rope, draft_tokens: 2, max_seq_len: 16 };
        let mut batch = [RocmDsparkDraftBatch { cache: &mut cache, anchor: 1, target_hidden: &target, target_position: 1, block_position: 4, minimum_drafts: 1, drafts: Vec::new() }];
        runtime.draft_batch(&gpu, &mut batch, &embedding, &head).unwrap();
        let cpu_head = cpu.prepare_f32(&head_values, 5, 4).unwrap();
        let expected_draft = reference.draft_greedy(&cpu, cpu_noise, &projected, 1, 1, 4, &cpu_head, 2, &runtime.rope).unwrap();
        assert_eq!(batch[0].drafts, expected_draft.drafts);
        batch[0].block_position = 14;
        runtime.draft_batch(&gpu, &mut batch, &embedding, &head).unwrap();
        assert!(batch[0].drafts.is_empty());
    }
}
