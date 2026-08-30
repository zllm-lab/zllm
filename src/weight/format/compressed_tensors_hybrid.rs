//! compressed-tensors 混合精度权重源(W4A16 + W8A16)。
//!
//! 与 [`super::compressed_tensors::W4A16CtSource`](纯 W4A16,服务 Gemma)平行,
//! 本模块支持 Int4-Int8Mix 这类按 tensor 名分流的混合精度:MoE expert 用 W4A16,
//! attention/shared 线性层用 W8A16,norm/embed/router 保持 BF16/FP32。
//!
//! bit 判别以 logical/packed shape 为准；config_groups 的实际规则是:
//! - 普通层 `mlp.experts.{N}.{gate,up,down}_proj` → W4A16
//! - MTP 层（含 experts）→ W8A16 channel-wise
//! - `self_attn.*` / `shared_experts.*` / `gate_up_proj` → W8A16
//! - `layers.0.*` / `lm_head` / `embed` / `*norm*` / `mlp.gate`(router) → 不量化(BF16/FP32)
//!
//! 暂用字符串匹配代替正则解析(避免引入 regex 依赖);GLM-5.2 的命名规律足够规律。

use std::path::Path;

use crate::weight::{
    container::safetensor::{SafetensorStore, TensorData},
    format::quantization::{ScaleDType, W4A16Matrix, W8A16Matrix},
};

/// compressed-tensors 混合精度矩阵(W4 或 W8)。
pub enum CtMatrix {
    W4(W4A16Matrix),
    W8(W8A16Matrix),
}

impl CtMatrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::W4(m) => m.rows,
            Self::W8(m) => m.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::W4(m) => m.cols,
            Self::W8(m) => m.cols,
        }
    }

    /// 解码为 f32(CPU reference 用)。
    pub fn decode(&self) -> Result<Vec<f32>, String> {
        match self {
            Self::W4(m) => m.decode(),
            Self::W8(m) => m.decode(),
        }
    }
}

/// compressed-tensors 混合精度权重源。读取 safetensors + 按 tensor 名判别 bit。
#[derive(Clone)]
pub struct CompressedTensorsSource {
    store: SafetensorStore,
    group_size: usize,
}

impl CompressedTensorsSource {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref();
        // 校验是 compressed-tensors 格式(group_size 从 config 读取,所有 group 应一致)。
        let group_size = read_group_size(root)?;
        Ok(Self { store: SafetensorStore::open(root)?, group_size })
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn store(&self) -> &SafetensorStore {
        &self.store
    }

    /// 该 tensor 是否是量化矩阵(有 `.weight_packed`,而非 BF16/FP32 原始权重)。
    pub fn is_quantized(&self, base: &str) -> bool {
        let base = base.strip_suffix(".weight").unwrap_or(base);
        self.store.has(&format!("{base}.weight_packed"))
    }

    /// 加载一个量化矩阵；位宽由 logical shape 与 packed shape 判定，不能按
    /// tensor 名猜测，因为普通 MoE expert 是 W4，而 MTP expert 是 W8。
    pub fn load_matrix(&self, base: &str) -> Result<CtMatrix, String> {
        let base = base.strip_suffix(".weight").unwrap_or(base);
        let (packed, scales) = self.store.load_pair(&format!("{base}.weight_packed"), &format!("{base}.weight_scale"))?;
        let shape = self.store.load(&format!("{base}.weight_shape"))?;
        let [rows, cols] = super::compressed_tensors::parse_matrix_shape(&shape)?;
        if rows == 0 {
            return Err(format!("{base} rows 不能为 0"));
        }
        let scale_count = scales.shape.iter().product::<usize>();
        if scale_count % rows != 0 {
            return Err(format!("{} scale shape {:?} 不能按 {rows} 行分组", scales.name, scales.shape));
        }
        let groups = scale_count / rows;
        let group_size = if groups == 1 { cols } else { self.group_size };
        let expected_groups = cols.div_ceil(group_size);
        if groups != expected_groups {
            return Err(format!("{} scale shape {:?}，实际每行 {groups} 组，期望 {expected_groups} 组", scales.name, scales.shape,));
        }
        let scale_dtype = match scales.dtype.as_str() {
            "BF16" => ScaleDType::Bf16,
            "F16" => ScaleDType::F16,
            "F32" => ScaleDType::F32,
            dtype => return Err(format!("{} scale dtype {dtype} 不受支持", scales.name)),
        };
        match packed_bits(&packed.name, &packed.dtype, &packed.shape, rows, cols)? {
            4 => W4A16Matrix::new(packed.data, scales.data, scale_dtype, group_size, rows, cols).map(CtMatrix::W4).map_err(|e| format!("{base}: {e}")),
            8 => W8A16Matrix::new(packed.data, scales.data, scale_dtype, group_size, rows, cols).map(CtMatrix::W8).map_err(|e| format!("{base}: {e}")),
            _ => unreachable!("packed_bits 只返回 4/8"),
        }
    }

    /// 加载未量化的原始 tensor(BF16/FP32 norm/embed/router 等)。
    pub fn load_tensor(&self, name: &str) -> Result<TensorData, String> {
        self.store.load(name)
    }

    pub fn load_bf16_rows(&self, name: &str, rows: &[usize]) -> Result<TensorData, String> {
        self.store.load_bf16_rows(name, rows)
    }
}

fn packed_bits(name: &str, dtype: &str, shape: &[usize], rows: usize, cols: usize) -> Result<usize, String> {
    if dtype != "I32" || shape.len() != 2 || shape[0] != rows {
        return Err(format!("{name} packed tensor 非法: dtype={dtype} shape={shape:?} logical=[{rows},{cols}]"));
    }
    let w4_cols = cols.div_ceil(8);
    let w8_cols = cols.div_ceil(4);
    match (shape[1] == w4_cols, shape[1] == w8_cols) {
        (true, false) => Ok(4),
        (false, true) => Ok(8),
        _ => Err(format!("{name} packed shape={shape:?} 无法判定 W4/W8: logical=[{rows},{cols}] W4=[{rows},{w4_cols}] W8=[{rows},{w8_cols}]")),
    }
}

/// 从 config.json 读 quantization_config,取 group_size(校验所有 group 一致)。
fn read_group_size(root: &Path) -> Result<usize, String> {
    #[derive(serde::Deserialize)]
    struct Cfg {
        quantization_config: QuantCfg,
    }
    #[derive(serde::Deserialize)]
    struct QuantCfg {
        quant_method: String,
        config_groups: std::collections::HashMap<String, Group>,
    }
    #[derive(serde::Deserialize)]
    struct Group {
        weights: W,
    }
    #[derive(serde::Deserialize)]
    struct W {
        group_size: i64,
    }

    let config_path = root.join("config.json");
    let cfg: Cfg = serde_json::from_slice(&std::fs::read(&config_path).map_err(|e| format!("读取 {}: {e}", config_path.display()))?).map_err(|e| format!("解析 {}: {e}", config_path.display()))?;
    if cfg.quantization_config.quant_method != "compressed-tensors" {
        return Err(format!("不是 compressed-tensors: {}", cfg.quantization_config.quant_method));
    }
    // group_size 一致性规则与 compressed_tensors.rs 统一:channel-wise(group_size<=0)不参与校验。
    super::compressed_tensors::consistent_group_size(cfg.quantization_config.config_groups.iter().map(|(name, group)| (name.as_str(), group.weights.group_size)))
}

/// 线性权重保持 checkpoint 原始存储，具体计算精度由 backend capability 决定。
pub enum CtLinearWeight {
    Quantized(CtMatrix),
    Bf16(Vec<u8>),
    F32(Vec<f32>),
}

impl CompressedTensorsSource {
    /// 加载一个线性层：量化权重和 BF16 均保持原始存储，避免无效展开。
    pub(crate) fn load_linear(&self, name: &str) -> Result<CtLinearWeight, String> {
        if self.is_quantized(name) {
            Ok(CtLinearWeight::Quantized(self.load_matrix(name)?))
        } else {
            let t = self.store.load(name)?;
            if t.dtype == "BF16" { Ok(CtLinearWeight::Bf16(t.data)) } else { Ok(CtLinearWeight::F32(t.to_f32()?)) }
        }
    }

    /// 加载 norm/embed 等 BF16/FP32 向量。
    pub fn load_norm(&self, name: &str) -> Result<Vec<f32>, String> {
        self.store.load(name)?.to_f32()
    }

    pub(crate) fn load_f32_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        self.store.load(name)?.to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_shape_classifies_regular_and_mtp_experts() {
        assert_eq!(packed_bits("regular expert", "I32", &[6144, 256], 6144, 2048).unwrap(), 4);
        assert_eq!(packed_bits("MTP expert", "I32", &[6144, 512], 6144, 2048).unwrap(), 8);
        assert!(packed_bits("broken", "I32", &[6144, 1024], 6144, 2048).is_err());
    }
}
