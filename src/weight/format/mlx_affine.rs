//! MLX affine safetensors 权重源。

use std::{collections::HashMap, path::Path};

use serde_json::Value;

use crate::weight::{
    container::safetensor::{SafetensorStore, TensorData},
    format::quantization::{MlxAffineMatrix, ScaleDType},
};

#[derive(Clone, Copy)]
struct QuantSpec {
    bits: usize,
    group_size: usize,
}

#[derive(Clone)]
pub struct MlxAffineSource {
    store: SafetensorStore,
    default: QuantSpec,
    overrides: HashMap<String, QuantSpec>,
}

impl MlxAffineSource {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref();
        let config_path = root.join("config.json");
        let config: Value = serde_json::from_slice(&std::fs::read(&config_path).map_err(|error| format!("读取 {} 失败: {error}", config_path.display()))?).map_err(|error| format!("解析 {} 失败: {error}", config_path.display()))?;
        let quantization = config.get("quantization").and_then(Value::as_object).ok_or("config.json 缺少 MLX quantization")?;
        if quantization.get("mode").and_then(Value::as_str) != Some("affine") {
            return Err("当前只支持 MLX quantization.mode=affine".into());
        }
        let parse = |value: &serde_json::Map<String, Value>, fallback: Option<QuantSpec>| -> Result<QuantSpec, String> {
            let bits = value.get("bits").and_then(Value::as_u64).map(|value| value as usize).or(fallback.map(|spec| spec.bits)).ok_or("MLX affine 缺少 bits")?;
            let group_size = value.get("group_size").and_then(Value::as_u64).map(|value| value as usize).or(fallback.map(|spec| spec.group_size)).ok_or("MLX affine 缺少 group_size")?;
            if !matches!(bits, 4 | 8) || group_size == 0 {
                return Err(format!("MLX affine bits={bits} group_size={group_size} 不受支持"));
            }
            Ok(QuantSpec { bits, group_size })
        };
        let default = parse(quantization, None)?;
        let mut overrides = HashMap::new();
        for (name, value) in quantization {
            if let Some(object) = value.as_object() {
                overrides.insert(name.clone(), parse(object, Some(default))?);
            }
        }
        Ok(Self { store: SafetensorStore::open(root)?, default, overrides })
    }

    pub fn has_matrix(&self, base: &str) -> bool {
        ["weight", "scales", "biases"].iter().all(|suffix| self.store.has(&format!("{base}.{suffix}")))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.store.has(name)
    }

    pub fn load_tensor(&self, name: &str) -> Result<TensorData, String> {
        self.store.load(name)
    }

    pub fn load_matrix(&self, base: &str) -> Result<MlxAffineMatrix, String> {
        self.build_matrix(base, self.store.load(&format!("{base}.weight"))?, self.store.load(&format!("{base}.scales"))?, self.store.load(&format!("{base}.biases"))?)
    }

    pub fn load_matrix_rows(&self, base: &str, rows: &[usize]) -> Result<MlxAffineMatrix, String> {
        self.build_matrix(base, self.store.load_rows(&format!("{base}.weight"), rows)?, self.store.load_rows(&format!("{base}.scales"), rows)?, self.store.load_rows(&format!("{base}.biases"), rows)?)
    }

    fn build_matrix(&self, base: &str, packed: TensorData, scales: TensorData, biases: TensorData) -> Result<MlxAffineMatrix, String> {
        let spec = self.overrides.get(base).copied().unwrap_or(self.default);
        if packed.dtype != "U32" || packed.shape.len() != 2 || scales.shape.len() != 2 || scales.shape != biases.shape || scales.dtype != biases.dtype {
            return Err(format!("{base} MLX affine tensor 元数据不一致"));
        }
        let scale_dtype = match scales.dtype.as_str() {
            "BF16" => ScaleDType::Bf16,
            "F16" => ScaleDType::F16,
            "F32" => ScaleDType::F32,
            dtype => return Err(format!("{base} scales dtype={dtype} 不受支持")),
        };
        let rows = packed.shape[0];
        let cols = scales.shape[1].checked_mul(spec.group_size).ok_or("MLX affine columns 溢出")?;
        if scales.shape[0] != rows || packed.shape[1] != cols.div_ceil(32 / spec.bits) {
            return Err(format!("{base} packed/scales shape 不兼容: {:?}/{:?}", packed.shape, scales.shape));
        }
        MlxAffineMatrix::new(packed.data, scales.data, biases.data, scale_dtype, spec.bits, spec.group_size, rows, cols)
    }
}
