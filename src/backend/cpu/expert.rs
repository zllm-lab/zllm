//! CPU expert 权重加载与批处理资源实现。

use std::{path::Path, sync::Arc};

use rayon::prelude::*;

use crate::{
    backend::{BackendError, ExpertPrefillBackend},
    kernel::cpu::CpuTensor,
    moe::topk_moe::TopkMoeSpec,
    weight::{
        expert_source::{GgufExpertSource, GgufExpertWeights},
        format::compressed_tensors_hybrid::CompressedTensorsSource,
        format::nvfp4::{Nvfp4ExpertWeights, NvidiaNvfp4Experts},
        format::official_fp8::{ExpertF32, OfficialExpertArchive},
    },
};

use super::CpuContext;

pub(crate) enum CpuExpertArchive {
    Fp8(OfficialExpertArchive),
    Nvfp4(NvidiaNvfp4Experts),
    Gguf(Arc<dyn GgufExpertSource>),
    Ct(CompressedTensorsSource),
}

pub struct CpuPrefillExperts {
    pub(crate) archive: CpuExpertArchive,
}

impl CpuPrefillExperts {
    pub fn fp8(root: &Path, intermediate: usize, hidden: usize, expert_count: usize) -> Result<Self, String> {
        Ok(Self { archive: CpuExpertArchive::Fp8(OfficialExpertArchive::open(root, intermediate, hidden, expert_count)?) })
    }

    pub fn nvfp4(source: NvidiaNvfp4Experts) -> Self {
        Self { archive: CpuExpertArchive::Nvfp4(source) }
    }

    pub fn gguf(source: Arc<dyn GgufExpertSource>) -> Self {
        Self { archive: CpuExpertArchive::Gguf(source) }
    }

    pub fn ct(source: CompressedTensorsSource) -> Self {
        Self { archive: CpuExpertArchive::Ct(source) }
    }
}

pub(crate) enum CpuPrefillExpert {
    F32(ExpertF32),
    Nvfp4(Nvfp4ExpertWeights),
    Gguf(GgufExpertWeights),
}

impl ExpertPrefillBackend for CpuContext {
    type PrefillExperts = CpuPrefillExperts;

    fn prefill_expert_batch(&self, spec: &TopkMoeSpec, layer: usize, experts: &mut CpuPrefillExperts, batch: Vec<crate::backend::ExpertPrefillBatch<CpuTensor>>) -> Result<Vec<CpuTensor>, BackendError> {
        let loaded = batch
            .iter()
            .map(|item| match &mut experts.archive {
                CpuExpertArchive::Fp8(source) => source.load_expert(layer, item.expert).map(CpuPrefillExpert::F32).map_err(BackendError::ExpertLoad),
                CpuExpertArchive::Nvfp4(source) => source.load_expert(layer, item.expert).map(CpuPrefillExpert::Nvfp4).map_err(BackendError::ExpertLoad),
                CpuExpertArchive::Gguf(source) => source.load_expert_gguf(layer, item.expert).map(CpuPrefillExpert::Gguf).map_err(BackendError::ExpertLoad),
                CpuExpertArchive::Ct(source) => {
                    let e = item.expert;
                    let gate = source.load_matrix(&format!("model.layers.{layer}.mlp.experts.{e}.gate_proj.weight")).and_then(|m| m.decode()).map_err(BackendError::ExpertLoad)?;
                    let up = source.load_matrix(&format!("model.layers.{layer}.mlp.experts.{e}.up_proj.weight")).and_then(|m| m.decode()).map_err(BackendError::ExpertLoad)?;
                    let down = source.load_matrix(&format!("model.layers.{layer}.mlp.experts.{e}.down_proj.weight")).and_then(|m| m.decode()).map_err(BackendError::ExpertLoad)?;
                    Ok(CpuPrefillExpert::F32(ExpertF32 { gate, up, down }))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = batch
            .into_par_iter()
            .zip(loaded.into_par_iter())
            .map(|(item, expert)| match expert {
                CpuPrefillExpert::F32(expert) => Ok(crate::kernel::cpu::moe::f32_expert_batch(&item.input, &expert.gate, &expert.up, &expert.down, spec.intermediate_size, &spec.activation)),
                CpuPrefillExpert::Nvfp4(weights) => Ok(crate::kernel::cpu::moe::nvfp4_expert_batch(&item.input, &weights, &spec.activation)),
                CpuPrefillExpert::Gguf(weights) => crate::kernel::cpu::moe::gguf_expert_batch(&item.input, &weights, &spec.activation),
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(BackendError::ExpertLoad)?;
        Ok(outputs)
    }
}
