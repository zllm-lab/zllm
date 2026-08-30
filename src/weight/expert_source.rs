//! Expert 权重来源边界；统一来源身份，不统一不同编码的加载方式。

use std::sync::Arc;

use crate::weight::{
    container::gguf::GgufMatrix,
    format::compressed_tensors_hybrid::{CompressedTensorsSource, CtMatrix},
    format::mxfp8::Mxfp8MatrixBufferMut,
    format::nvfp4::{Nvfp4ArchiveLayout, Nvfp4ExpertBufferMut, Nvfp4ExpertWeights, Nvfp4MatrixBufferMut, NvidiaNvfp4Experts},
    format::official_fp8::{Fp8ExpertWeights, OfficialExpertArchive},
};

pub struct W4A16ExpertWeights {
    pub gate: crate::weight::format::quantization::W4A16Matrix,
    pub up: crate::weight::format::quantization::W4A16Matrix,
    pub down: crate::weight::format::quantization::W4A16Matrix,
}

pub struct CtExpertWeights {
    pub gate: CtMatrix,
    pub up: CtMatrix,
    pub down: CtMatrix,
}

impl CompressedTensorsSource {
    /// CT checkpoint 内同一 expert 的三块矩阵保持实际 W4/W8 编码。
    pub fn load_expert_ct(&self, layer: usize, expert: usize) -> Result<CtExpertWeights, String> {
        let load = |projection: &str| self.load_matrix(&format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"));
        std::thread::scope(|scope| {
            let gate = scope.spawn(|| load("gate_proj"));
            let up = scope.spawn(|| load("up_proj"));
            let down = load("down_proj")?;
            Ok(CtExpertWeights { gate: gate.join().map_err(|_| format!("CT L{layer} E{expert} gate 读取线程 panic"))??, up: up.join().map_err(|_| format!("CT L{layer} E{expert} up 读取线程 panic"))??, down })
        })
    }
}

pub trait W4A16ExpertSource: Send + Sync {
    fn load_expert_w4a16(&self, layer: usize, expert: usize) -> Result<W4A16ExpertWeights, String>;
}

impl W4A16ExpertSource for CompressedTensorsSource {
    fn load_expert_w4a16(&self, layer: usize, expert: usize) -> Result<W4A16ExpertWeights, String> {
        let require_w4 = |projection: &str, matrix| match matrix {
            CtMatrix::W4(matrix) => Ok(matrix),
            CtMatrix::W8(_) => Err(format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight 应为 W4A16 expert 权重，实际为 W8A16")),
        };
        let weights = self.load_expert_ct(layer, expert)?;
        Ok(W4A16ExpertWeights { gate: require_w4("gate_proj", weights.gate)?, up: require_w4("up_proj", weights.up)?, down: require_w4("down_proj", weights.down)? })
    }
}

#[derive(Debug, Clone)]
pub struct GgufExpertWeights {
    pub gate: GgufMatrix,
    pub up: GgufMatrix,
    pub down: GgufMatrix,
}

#[derive(Clone, Debug)]
pub struct Mxfp4ExpertWeights {
    pub gate: crate::weight::format::mxfp4::Mxfp4Matrix,
    pub up: crate::weight::format::mxfp4::Mxfp4Matrix,
    pub down: crate::weight::format::mxfp4::Mxfp4Matrix,
}

pub trait Mxfp4ExpertSource: Send + Sync {
    fn intermediate(&self) -> usize;
    fn hidden(&self) -> usize;
    fn load_expert_mxfp4(&self, layer: usize, expert: usize) -> Result<Mxfp4ExpertWeights, String>;
}

pub trait GgufExpertSource: Sync + Send {
    fn intermediate(&self) -> usize;
    fn hidden(&self) -> usize;
    fn load_expert_gguf(&self, layer: usize, expert: usize) -> Result<GgufExpertWeights, String>;
}

pub trait Fp8ExpertSource {
    fn intermediate(&self) -> usize;
    fn hidden(&self) -> usize;
    fn load_expert_fp8(&self, layer: usize, expert: usize) -> Result<Fp8ExpertWeights, String>;
}

pub trait Mxfp8ExpertSource {
    fn load_expert_into(&self, layer: usize, expert: usize, gate: Mxfp8MatrixBufferMut<'_>, up: Mxfp8MatrixBufferMut<'_>, down: Mxfp8MatrixBufferMut<'_>) -> Result<(), String>;
}

pub trait Nvfp4ExpertSource: Sync {
    fn intermediate(&self) -> usize;
    fn hidden(&self) -> usize;
    fn load_expert_nvfp4(&self, layer: usize, expert: usize) -> Result<Nvfp4ExpertWeights, String>;

    fn load_expert_nvfp4_into(&self, layer: usize, expert: usize, mut gate: Nvfp4MatrixBufferMut<'_>, mut up: Nvfp4MatrixBufferMut<'_>, mut down: Nvfp4MatrixBufferMut<'_>) -> Result<(), String> {
        let weights = self.load_expert_nvfp4(layer, expert)?;
        gate.copy_from(&weights.gate)?;
        up.copy_from(&weights.up)?;
        down.copy_from(&weights.down)
    }

    fn load_experts_nvfp4_into(&self, layer: usize, experts: Vec<Nvfp4ExpertBufferMut<'_>>) -> Result<(), String> {
        for expert in experts {
            self.load_expert_nvfp4_into(layer, expert.expert, expert.gate, expert.up, expert.down)?;
        }
        Ok(())
    }

    fn layer_archive_layout(&self, _layer: usize, _expert_count: usize) -> Result<Option<Nvfp4ArchiveLayout>, String> {
        Ok(None)
    }

    fn read_layer_archive_range(&self, layer: usize, _offset: usize, _destination: &mut [u8]) -> Result<(), String> {
        Err(format!("L{layer} source 不支持 NVFP4 layer archive range read"))
    }
}

impl Nvfp4ExpertSource for NvidiaNvfp4Experts {
    fn intermediate(&self) -> usize {
        NvidiaNvfp4Experts::intermediate(self)
    }

    fn hidden(&self) -> usize {
        NvidiaNvfp4Experts::hidden(self)
    }

    fn load_expert_nvfp4(&self, layer: usize, expert: usize) -> Result<Nvfp4ExpertWeights, String> {
        self.load_expert(layer, expert)
    }

    fn load_expert_nvfp4_into(&self, layer: usize, expert: usize, gate: Nvfp4MatrixBufferMut<'_>, up: Nvfp4MatrixBufferMut<'_>, down: Nvfp4MatrixBufferMut<'_>) -> Result<(), String> {
        self.load_expert_into(layer, expert, gate, up, down)
    }

    fn load_experts_nvfp4_into(&self, layer: usize, experts: Vec<Nvfp4ExpertBufferMut<'_>>) -> Result<(), String> {
        self.load_experts_into(layer, experts)
    }

    fn layer_archive_layout(&self, layer: usize, expert_count: usize) -> Result<Option<Nvfp4ArchiveLayout>, String> {
        self.layer_archive_layout(layer, expert_count)
    }

    fn read_layer_archive_range(&self, layer: usize, offset: usize, destination: &mut [u8]) -> Result<(), String> {
        self.read_layer_archive_range(layer, offset, destination)
    }
}

impl Fp8ExpertSource for OfficialExpertArchive {
    fn intermediate(&self) -> usize {
        OfficialExpertArchive::intermediate(self)
    }

    fn hidden(&self) -> usize {
        OfficialExpertArchive::hidden(self)
    }

    fn load_expert_fp8(&self, layer: usize, expert: usize) -> Result<Fp8ExpertWeights, String> {
        OfficialExpertArchive::load_expert_fp8(self, layer, expert)
    }
}

#[derive(Clone, Copy)]
pub enum ExpertSource<'a> {
    Fp8(&'a dyn Fp8ExpertSource),
    Mxfp8(&'a dyn Mxfp8ExpertSource),
    Mxfp4(&'a dyn Mxfp4ExpertSource),
    W4A16(&'a dyn W4A16ExpertSource),
    Nvfp4(&'a dyn Nvfp4ExpertSource),
    Gguf(&'a dyn GgufExpertSource),
}

impl<'a> ExpertSource<'a> {
    pub fn require_gguf(self) -> Result<&'a dyn GgufExpertSource, String> {
        match self {
            Self::Gguf(source) => Ok(source),
            Self::Fp8(_) => Err("当前 expert backend 不支持 FP8 source".to_owned()),
            Self::Mxfp8(_) => Err("当前 expert backend 不支持 MXFP8 source".to_owned()),
            Self::Mxfp4(_) => Err("当前 expert backend 不支持 MXFP4 source".to_owned()),
            Self::W4A16(_) => Err("当前 expert backend 不支持 W4A16 source".to_owned()),
            Self::Nvfp4(_) => Err("当前 expert backend 不支持 NVFP4 source".to_owned()),
        }
    }
}

pub trait ExpertSourceProvider {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String>;
}

impl ExpertSourceProvider for OfficialExpertArchive {
    fn source(&self, _layer: usize) -> Result<ExpertSource<'_>, String> {
        Ok(ExpertSource::Fp8(self))
    }
}

impl ExpertSourceProvider for Arc<dyn GgufExpertSource> {
    fn source(&self, _layer: usize) -> Result<ExpertSource<'_>, String> {
        Ok(ExpertSource::Gguf(self.as_ref()))
    }
}

pub enum Glm52ExpertSources {
    Fp8(OfficialExpertArchive),
    W4A16(CompressedTensorsSource),
    Nvfp4(NvidiaNvfp4Experts),
    Gguf(std::sync::Arc<dyn GgufExpertSource>),
}

impl ExpertSourceProvider for Glm52ExpertSources {
    fn source(&self, _layer: usize) -> Result<ExpertSource<'_>, String> {
        Ok(match self {
            Self::Fp8(source) => ExpertSource::Fp8(source),
            Self::W4A16(source) => ExpertSource::W4A16(source),
            Self::Nvfp4(source) => ExpertSource::Nvfp4(source),
            Self::Gguf(source) => ExpertSource::Gguf(source.as_ref()),
        })
    }
}
