//! 官方 FP8 expert 权重读取。

use std::path::Path;

use crate::weight::{Fp8Matrix, container::safetensor::SafetensorStore};

pub struct ExpertF32 {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

pub struct Fp8ExpertWeights {
    pub gate: Fp8Matrix,
    pub up: Fp8Matrix,
    pub down: Fp8Matrix,
}

pub struct OfficialExpertArchive {
    source: SafetensorStore,
    intermediate: usize,
    hidden: usize,
    expert_count: usize,
}

impl OfficialExpertArchive {
    pub fn open(root: &Path, intermediate: usize, hidden: usize, expert_count: usize) -> Result<Self, String> {
        if expert_count == 0 {
            return Err("官方 FP8 experts expert_count 必须大于 0".to_owned());
        }
        let source = SafetensorStore::open(root).map_err(|error| format!("打开官方权重 root {}: {error}", root.display()))?;
        Ok(Self { source, intermediate, hidden, expert_count })
    }

    pub fn intermediate(&self) -> usize {
        self.intermediate
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn load_expert(&self, layer: usize, expert: usize) -> Result<ExpertF32, String> {
        let expert = self.load_expert_fp8(layer, expert)?;
        Ok(ExpertF32 { gate: expert.gate.decode(), up: expert.up.decode(), down: expert.down.decode() })
    }

    pub fn load_expert_fp8(&self, layer: usize, expert: usize) -> Result<Fp8ExpertWeights, String> {
        if expert >= self.expert_count {
            return Err(format!("expert 索引越界: {expert} >= {}", self.expert_count));
        }
        Ok(Fp8ExpertWeights { gate: self.load_matrix_fp8(layer, expert, "gate")?, up: self.load_matrix_fp8(layer, expert, "up")?, down: self.load_matrix_fp8(layer, expert, "down")? })
    }

    fn load_matrix_fp8(&self, layer: usize, expert: usize, kind: &str) -> Result<Fp8Matrix, String> {
        let name = format!("model.layers.{layer}.mlp.experts.{expert}.{kind}_proj.weight");
        let tensor = self.source.load(&name).map_err(|error| format!("读取官方专家权重 {name}: {error}"))?;
        let expected_shape = match kind {
            "gate" | "up" => vec![self.intermediate, self.hidden],
            "down" => vec![self.hidden, self.intermediate],
            _ => return Err(format!("不支持的专家矩阵名: {kind}")),
        };
        if tensor.dtype != "F8_E4M3" || tensor.shape != expected_shape {
            return Err(format!("{name} dtype={} shape={:?}，期望 F8_E4M3/{expected_shape:?}", tensor.dtype, tensor.shape));
        }
        let scale_name = format!("{name}_scale_inv");
        let scale = self.source.load(&scale_name).map_err(|error| format!("读取官方专家 scale {scale_name}: {error}"))?;
        let expected_scale = [expected_shape[0].div_ceil(128), expected_shape[1].div_ceil(128)];
        if scale.dtype != "F32" || scale.shape.as_slice() != expected_scale {
            return Err(format!("{scale_name} dtype={} shape={:?}，期望 F32/{expected_scale:?}", scale.dtype, scale.shape));
        }
        Fp8Matrix::new(tensor.data, scale.data, expected_shape[0], expected_shape[1]).map_err(|error| format!("初始化 {name}: {error}"))
    }
}
