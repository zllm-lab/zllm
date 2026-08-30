//! Metal 后端资源、权重驻留和执行调度。

pub(crate) mod api;
mod attention;
mod attn_res;
mod compressed_sparse;
pub mod context;
pub mod dsa;
mod expert_decode;
mod gated_delta_net;
mod hyper_connection;
mod kda;
pub mod kv_cache;
pub mod replay;
pub mod resident;
mod resource;
mod tensor;
mod vae;
mod vision;

use crate::backend::BackendError;
use crate::backend::metal::api::Buffer;

pub use compressed_sparse::MetalCompressedKvStorage;
pub use context::{MetalContext, MetalGpuProfile, MetalTensor, MetalTensorDType};
pub use expert_decode::{MetalMoeDecodeState, MetalMoeDecodeStats, MetalMxfp4ActivationMode, MetalPrefillExperts};
pub use gated_delta_net::MetalGatedDeltaNetStorage;
pub use kda::MetalKdaStorage;
pub use kv_cache::{MetalKvCache, MetalKvCacheFormat};
pub use resident::MetalWeight;
pub use resource::{available_residency_bytes, gpu_compute_units, merge_gpu_profiles, report_metal_resource_plan, system_memory_bytes};
pub use tensor::MetalLayerScope;

pub(crate) fn expect_resident_f16<'a>(weight: &'a MetalWeight, msg: &'static str) -> Result<&'a MetalTensor, BackendError> {
    match weight {
        MetalWeight::F16(tensor) => Ok(tensor),
        _ => Err(BackendError::Compute { msg: msg.to_owned() }),
    }
}

pub(crate) fn expect_resident_f16_opt<'a>(weight: Option<&'a MetalWeight>, msg: &'static str) -> Result<Option<&'a MetalTensor>, BackendError> {
    match weight {
        Some(MetalWeight::F16(tensor)) => Ok(Some(tensor)),
        Some(_) => Err(BackendError::Compute { msg: msg.to_owned() }),
        None => Ok(None),
    }
}

pub(crate) fn expect_resident_f32<'a>(weight: &'a MetalWeight, msg: &'static str) -> Result<(&'a Buffer, usize), BackendError> {
    match weight {
        MetalWeight::F32 { buffer, len } => Ok((buffer, *len)),
        _ => Err(BackendError::Compute { msg: msg.to_owned() }),
    }
}
