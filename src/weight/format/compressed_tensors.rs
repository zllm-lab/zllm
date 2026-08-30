//! compressed-tensors 权重源。外部文件命名与配置在这里终止。

use std::{collections::HashMap, path::Path};

use serde::Deserialize;

use crate::weight::{
    container::safetensor::{SafetensorStore, TensorData},
    format::quantization::{ScaleDType, W4A16Matrix},
};

#[derive(Deserialize)]
struct ModelConfig {
    quantization_config: QuantizationConfig,
}

#[derive(Deserialize)]
struct QuantizationConfig {
    quant_method: String,
    quantization_status: String,
    config_groups: HashMap<String, QuantizationGroup>,
}

#[derive(Deserialize)]
struct QuantizationGroup {
    format: String,
    input_activations: Option<serde_json::Value>,
    output_activations: Option<serde_json::Value>,
    weights: QuantizationWeights,
}

#[derive(Deserialize)]
struct QuantizationWeights {
    num_bits: usize,
    #[serde(rename = "type")]
    kind: String,
    strategy: String,
    // i64: channel-wise 量化在 config 里写作 group_size=-1,不能因为反序列化失败而拒读。
    group_size: i64,
    symmetric: bool,
    dynamic: bool,
    actorder: Option<serde_json::Value>,
    block_structure: Option<serde_json::Value>,
}

/// 校验所有 group-wise 量化 group 的 group_size 一致并返回。
/// group_size<=0 是 channel-wise 量化(整条通道一个 scale),与 group-wise 是两种语义,不参与一致性校验
/// ——GLM CT checkpoint 的 MTP 层就是 channel-wise,混在一起只会误报"混合 group_size"。
pub(crate) fn consistent_group_size<'a>(groups: impl Iterator<Item = (&'a str, i64)>) -> Result<usize, String> {
    let mut group_size = None;
    for (name, gs) in groups {
        if gs <= 0 {
            continue;
        }
        let gs = gs as usize;
        match group_size {
            None => group_size = Some(gs),
            Some(existing) if existing == gs => {}
            Some(existing) => return Err(format!("compressed-tensors 混合 group_size: {existing} 与 {gs}(group {name})")),
        }
    }
    group_size.ok_or_else(|| "compressed-tensors 缺少 group-wise quantization config group".to_owned())
}

/// 当前只接受 Gemma 4 使用的静态、对称、group-wise W4A16 pack-quantized 子集。
#[derive(Clone)]
pub struct W4A16CtSource {
    store: SafetensorStore,
    group_size: usize,
}

impl W4A16CtSource {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref();
        let config_path = root.join("config.json");
        let config: ModelConfig = serde_json::from_slice(&std::fs::read(&config_path).map_err(|error| format!("读取 {}: {error}", config_path.display()))?).map_err(|error| format!("解析 {}: {error}", config_path.display()))?;
        let quantization = config.quantization_config;
        if quantization.quant_method != "compressed-tensors" || quantization.quantization_status != "compressed" {
            return Err(format!("{} 不是已压缩 compressed-tensors: method={} status={}", config_path.display(), quantization.quant_method, quantization.quantization_status,));
        }
        for (name, group) in &quantization.config_groups {
            let weights = &group.weights;
            let supported = group.format == "pack-quantized"
                && group.input_activations.is_none()
                && group.output_activations.is_none()
                && weights.num_bits == 4
                && weights.kind == "int"
                && weights.strategy == "group"
                && weights.symmetric
                && !weights.dynamic
                && weights.actorder.is_none()
                && weights.block_structure.is_none()
                && weights.group_size > 0;
            if !supported {
                return Err(format!("compressed-tensors config group {name} 不是 zLLM 支持的 W4A16 group-wise symmetric 格式"));
            }
        }
        let group_size = consistent_group_size(quantization.config_groups.iter().map(|(name, group)| (name.as_str(), group.weights.group_size)))?;
        Ok(Self { store: SafetensorStore::open(root)?, group_size })
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn has(&self, base: &str) -> bool {
        let base = base.strip_suffix(".weight").unwrap_or(base);
        self.store.has(&format!("{base}.weight_packed"))
    }

    pub fn load_matrix(&self, base: &str) -> Result<W4A16Matrix, String> {
        let base = base.strip_suffix(".weight").unwrap_or(base);
        let packed = self.store.load(&format!("{base}.weight_packed"))?;
        let scales = self.store.load(&format!("{base}.weight_scale"))?;
        let shape = self.store.load(&format!("{base}.weight_shape"))?;
        let [rows, cols] = parse_matrix_shape(&shape)?;
        let packed_cols = cols.div_ceil(8);
        if packed.dtype != "I32" || packed.shape != [rows, packed_cols] {
            return Err(format!("{} 需要 I32 [{rows},{packed_cols}]，实际 dtype={} shape={:?}", packed.name, packed.dtype, packed.shape,));
        }
        let scale_dtype = match scales.dtype.as_str() {
            "BF16" => ScaleDType::Bf16,
            "F16" => ScaleDType::F16,
            "F32" => ScaleDType::F32,
            dtype => return Err(format!("{} scale dtype {dtype} 不受支持", scales.name)),
        };
        let scale_count = rows.checked_mul(cols.div_ceil(self.group_size)).ok_or_else(|| format!("{base} scale 数量溢出"))?;
        if scales.shape.iter().product::<usize>() != scale_count {
            return Err(format!("{} scale shape {:?}，期望 {scale_count} 个元素", scales.name, scales.shape));
        }
        W4A16Matrix::new(packed.data, scales.data, scale_dtype, self.group_size, rows, cols).map_err(|error| format!("{base}: {error}"))
    }

    pub fn load_tensor(&self, name: &str) -> Result<TensorData, String> {
        self.store.load(name)
    }

    pub fn load_bf16_rows(&self, name: &str, rows: &[usize]) -> Result<TensorData, String> {
        self.store.load_bf16_rows(name, rows)
    }
}

pub(crate) fn parse_matrix_shape(tensor: &TensorData) -> Result<[usize; 2], String> {
    if tensor.shape != [2] {
        return Err(format!("{} shape tensor 需要 [2]，实际 {:?}", tensor.name, tensor.shape));
    }
    let values: Vec<u64> = match tensor.dtype.as_str() {
        "I64" => tensor.data.chunks_exact(8).map(|bytes| i64::from_le_bytes(bytes.try_into().expect("I64 chunk"))).map(|value| u64::try_from(value).map_err(|_| format!("{} 含负 shape {value}", tensor.name))).collect::<Result<_, _>>()?,
        "U64" => tensor.data.chunks_exact(8).map(|bytes| u64::from_le_bytes(bytes.try_into().expect("U64 chunk"))).collect(),
        "I32" => tensor.data.chunks_exact(4).map(|bytes| i32::from_le_bytes(bytes.try_into().expect("I32 chunk"))).map(|value| u64::try_from(value).map_err(|_| format!("{} 含负 shape {value}", tensor.name))).collect::<Result<_, _>>()?,
        "U32" => tensor.data.chunks_exact(4).map(|bytes| u32::from_le_bytes(bytes.try_into().expect("U32 chunk")) as u64).collect(),
        dtype => return Err(format!("{} shape dtype {dtype} 不受支持", tensor.name)),
    };
    if values.len() != 2 {
        return Err(format!("{} shape 数据长度错误: {}", tensor.name, values.len()));
    }
    Ok([usize::try_from(values[0]).map_err(|_| format!("{} rows 超出 usize", tensor.name))?, usize::try_from(values[1]).map_err(|_| format!("{} cols 超出 usize", tensor.name))?])
}
