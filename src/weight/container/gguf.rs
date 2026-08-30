//! 轻量 GGUF v3 reader：解析 metadata/tensor directory，并按需读取单个 tensor。

use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, Read, Seek},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use super::file_ext::FileExt;
use crate::tokenizer::{Detokenizer, Tokenizer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = String;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return Err(format!("未知 GGUF metadata type {value}")),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array { element_type: GgufValueType, len: usize, values: Vec<GgufValue>, truncated: bool },
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        let Self::Array { values, truncated: false, .. } = self else {
            return None;
        };
        values.iter().map(Self::as_str).collect()
    }

    pub fn as_i64_array(&self) -> Option<Vec<i64>> {
        let Self::Array { values, truncated: false, .. } = self else {
            return None;
        };
        values
            .iter()
            .map(|value| match value {
                Self::Signed(value) => Some(*value),
                Self::Unsigned(value) => i64::try_from(*value).ok(),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GgmlType(pub u32);

impl GgmlType {
    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            39 => "MXFP4",
            40 => "NVFP4",
            41 => "Q1_0",
            42 => "Q2_0",
            _ => "UNKNOWN",
        }
    }

    pub fn storage_bytes(self, elements: usize) -> Result<usize, String> {
        let (block_elements, block_bytes) = crate::weight::codec::ggml::block_layout(self.0)?;
        if !elements.is_multiple_of(block_elements) {
            return Err(format!("{} 元素数 {elements} 未按 block {block_elements} 对齐", self.name()));
        }
        elements.checked_div(block_elements).and_then(|blocks| blocks.checked_mul(block_bytes)).ok_or_else(|| "GGUF tensor 字节数溢出".to_owned())
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// GGUF 原始 dimension 顺序；二维矩阵通常是 `[cols, rows]`。
    pub dims: Vec<usize>,
    pub tensor_type: GgmlType,
    pub offset: u64,
    pub bytes: usize,
    shard: usize,
}

#[derive(Debug, Clone)]
pub struct GgufMatrix {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub tensor_type: GgmlType,
    file: Arc<File>,
    offset: u64,
    bytes_len: usize,
    bytes: Arc<OnceLock<Vec<u8>>>,
}

impl GgufMatrix {
    pub fn storage_len(&self) -> usize {
        self.bytes_len
    }

    #[cfg(target_env = "ohos")]
    pub(crate) fn file_range(&self) -> (std::os::fd::RawFd, u64) {
        use std::os::fd::AsRawFd;
        (self.file.as_raw_fd(), self.offset)
    }

    pub fn bytes(&self) -> Result<&[u8], String> {
        if self.bytes.get().is_none() {
            let mut bytes = vec![0u8; self.bytes_len];
            FileExt::read_exact_at(&self.file, &mut bytes, self.offset).map_err(|error| format!("读取 GGUF matrix {}: {error}", self.name))?;
            let _ = self.bytes.set(bytes);
        }
        Ok(self.bytes.get().expect("GGUF matrix bytes 已加载"))
    }

    pub fn read_into(&self, output: &mut [u8]) -> Result<(), String> {
        if output.len() != self.bytes_len {
            return Err(format!("GGUF matrix {} destination={}，期望 {}", self.name, output.len(), self.bytes_len));
        }
        FileExt::read_exact_at(&self.file, output, self.offset).map_err(|error| format!("读取 GGUF matrix {}: {error}", self.name))
    }

    /// 按给定顺序读取任意矩阵行，供 draft vocabulary 保持 GGUF 量化布局裁剪 LM head。
    pub fn read_rows_into(&self, rows: &[u32], output: &mut [u8]) -> Result<(), String> {
        let row_bytes = self.tensor_type.storage_bytes(self.columns)?;
        let expected = rows.len().checked_mul(row_bytes).ok_or_else(|| format!("GGUF matrix {} selected rows 大小溢出", self.name))?;
        if rows.is_empty() || output.len() != expected {
            return Err(format!("GGUF matrix {} selected destination={}，期望 {expected}", self.name, output.len()));
        }
        for (&row, destination) in rows.iter().zip(output.chunks_exact_mut(row_bytes)) {
            if row as usize >= self.rows {
                return Err(format!("GGUF matrix {} selected row={row} 超出 {}", self.name, self.rows));
            }
            let offset = self.offset.checked_add(row as u64 * row_bytes as u64).ok_or_else(|| format!("GGUF matrix {} selected row offset 溢出", self.name))?;
            FileExt::read_exact_at(&self.file, destination, offset).map_err(|error| format!("读取 GGUF matrix {} row={row}: {error}", self.name))?;
        }
        Ok(())
    }

    /// 流式执行时只在当前算子生命周期内持有 packed 权重，避免把整个 GGUF 累积进内存。
    pub fn read_bytes(&self) -> Result<Vec<u8>, String> {
        let mut output = vec![0u8; self.bytes_len];
        self.read_into(&mut output)?;
        Ok(output)
    }

    pub fn decode(&self) -> Result<Vec<f32>, String> {
        crate::weight::codec::ggml::dequantize(self.tensor_type.0, self.bytes()?, self.rows * self.columns).map_err(|error| format!("解码 GGUF matrix {}: {error}", self.name))
    }

    pub fn row_slice(&self, start: usize, rows: usize) -> Result<Self, String> {
        let end = start.checked_add(rows).ok_or_else(|| format!("GGUF matrix {} 行范围溢出", self.name))?;
        if rows == 0 || end > self.rows {
            return Err(format!("GGUF matrix {} 行范围 {start}..{end} 超出 {}", self.name, self.rows));
        }
        let row_bytes = self.tensor_type.storage_bytes(self.columns)?;
        let offset = self.offset.checked_add((start * row_bytes) as u64).ok_or_else(|| format!("GGUF matrix {} 文件 offset 溢出", self.name))?;
        Ok(Self { name: format!("{}[{start}..{end}]", self.name), rows, columns: self.columns, tensor_type: self.tensor_type, file: Arc::clone(&self.file), offset, bytes_len: rows * row_bytes, bytes: Arc::new(OnceLock::new()) })
    }
}

impl GgufTensorInfo {
    pub fn elements(&self) -> usize {
        self.dims.iter().product()
    }
}

pub struct GgufReader {
    shards: Vec<GgufShard>,
    file_len: u64,
    version: u32,
    data_offset: u64,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
    tensor_index: HashMap<String, usize>,
}

struct GgufShard {
    file: Arc<File>,
    data_offset: u64,
}

struct ParsedShard {
    file: File,
    file_len: u64,
    version: u32,
    data_offset: u64,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
}

impl GgufReader {
    /// 从文件或一组标准 GGUF split 所在目录定位首分片；模型专属 wrapper 复用这一入口。
    pub fn locate(path: &Path) -> Result<PathBuf, String> {
        if path.is_file() {
            return Ok(path.to_owned());
        }
        let mut models = std::fs::read_dir(path)
            .map_err(|error| format!("读取 GGUF 模型目录 {}: {error}", path.display()))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension().and_then(|value| value.to_str()).is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
                    && !path.file_name().and_then(|value| value.to_str()).is_some_and(|value| value.starts_with("mmproj"))
                    && !path.file_name().and_then(|value| value.to_str()).is_some_and(|value| value.starts_with("._"))
            })
            .collect::<Vec<_>>();
        models.sort();
        let split_heads = models.iter().filter(|model| split_name(model).is_some_and(|(_, index, _)| index == 1)).cloned().collect::<Vec<_>>();
        if let [head] = split_heads.as_slice() {
            let (_, _, count) = split_name(head).expect("已筛选 split 首分片");
            let siblings = split_paths(head, count)?;
            if siblings.iter().all(|shard| shard.is_file()) {
                return Ok(head.clone());
            }
        }
        match models.as_slice() {
            [model] => Ok(model.clone()),
            [] => Err(format!("{} 中没有主 GGUF", path.display())),
            _ => Err(format!("{} 中有多个主 GGUF，请直接指定文件路径", path.display())),
        }
    }

    pub fn open(path: &Path) -> Result<Self, String> {
        let paths = match split_name(path) {
            Some((_, _, count)) => split_paths(path, count)?,
            None => vec![path.to_owned()],
        };
        let mut parsed = paths.iter().map(|path| parse_shard(path)).collect::<Result<Vec<_>, _>>()?;
        let first = parsed.first().ok_or_else(|| "GGUF 没有文件分片".to_owned())?;
        let version = first.version;
        let data_offset = first.data_offset;
        let metadata = first.metadata.clone();
        let expected_shards = metadata.get("split.count").and_then(GgufValue::as_u64).unwrap_or(1);
        if expected_shards != parsed.len() as u64 {
            return Err(format!("GGUF split.count={expected_shards}，实际打开 {} 个分片", parsed.len()));
        }
        let expected_tensors = metadata.get("split.tensors.count").and_then(GgufValue::as_u64);
        let file_len = parsed.iter().try_fold(0u64, |total, shard| total.checked_add(shard.file_len).ok_or("GGUF split 总大小溢出"))?;
        let tensor_count = parsed.iter().map(|shard| shard.tensors.len()).sum();
        if let Some(expected) = expected_tensors
            && expected != tensor_count as u64
        {
            return Err(format!("GGUF split.tensors.count={expected}，实际解析 {tensor_count} 个 tensor"));
        }
        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_index = HashMap::with_capacity(tensor_count);
        for (shard_index, shard) in parsed.iter_mut().enumerate() {
            if shard.version != version {
                return Err(format!("GGUF split {} version={}，首分片为 v{version}", shard_index + 1, shard.version));
            }
            if let Some(actual) = shard.metadata.get("split.no").and_then(GgufValue::as_u64)
                && actual != shard_index as u64
            {
                return Err(format!("GGUF split {} 的 split.no={actual}，期望 {shard_index}", shard_index + 1));
            }
            for mut tensor in std::mem::take(&mut shard.tensors) {
                tensor.shard = shard_index;
                let index = tensors.len();
                if tensor_index.insert(tensor.name.clone(), index).is_some() {
                    return Err(format!("GGUF split tensor 重复: {}", tensor.name));
                }
                tensors.push(tensor);
            }
        }
        let shards = parsed.into_iter().map(|shard| GgufShard { file: Arc::new(shard.file), data_offset: shard.data_offset }).collect();
        Ok(Self { shards, file_len, version, data_offset, metadata, tensors, tensor_index })
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    pub fn metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_index.get(name).map(|index| &self.tensors[*index])
    }

    pub fn expect_tensor(&self, name: &str, dims: &[usize]) -> Result<&GgufTensorInfo, String> {
        let tensor = self.tensor(name).ok_or_else(|| format!("GGUF 缺少 tensor {name}"))?;
        if tensor.dims != dims {
            return Err(format!("GGUF tensor {name} shape={:?}，期望 {dims:?}", tensor.dims));
        }
        Ok(tensor)
    }

    pub fn metadata_u64(&self, key: &str) -> Result<u64, String> {
        self.metadata(key).and_then(GgufValue::as_u64).ok_or_else(|| format!("GGUF metadata {key} 缺失或类型错误"))
    }

    pub fn expect_metadata_u64(&self, key: &str, expected: u64) -> Result<(), String> {
        let actual = self.metadata_u64(key)?;
        if actual != expected {
            return Err(format!("GGUF metadata {key}={actual}，期望 {expected}"));
        }
        Ok(())
    }

    pub fn expect_metadata_str(&self, key: &str, expected: &str) -> Result<(), String> {
        let actual = self.metadata(key).and_then(GgufValue::as_str).ok_or_else(|| format!("GGUF metadata {key} 缺失或类型错误"))?;
        if actual != expected {
            return Err(format!("GGUF metadata {key}={actual}，期望 {expected}"));
        }
        Ok(())
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, String> {
        let tensor = self.tensor(name).ok_or_else(|| format!("GGUF 无 tensor {name}"))?;
        self.read_tensor_range(name, 0, tensor.bytes)
    }

    pub fn read_tensor_range(&self, name: &str, offset: usize, len: usize) -> Result<Vec<u8>, String> {
        let (file, absolute) = self.tensor_storage(name, offset, len)?;
        let mut output = vec![0u8; len];
        FileExt::read_exact_at(&file, &mut output, absolute).map_err(io_error)?;
        Ok(output)
    }

    fn tensor_storage(&self, name: &str, offset: usize, len: usize) -> Result<(Arc<File>, u64), String> {
        let tensor = self.tensor(name).ok_or_else(|| format!("GGUF 无 tensor {name}"))?;
        let end = offset.checked_add(len).ok_or_else(|| format!("GGUF tensor {name} range 溢出"))?;
        if end > tensor.bytes {
            return Err(format!("GGUF tensor {name} range {offset}..{end} 超出 {}", tensor.bytes));
        }
        let shard = self.shards.get(tensor.shard).ok_or_else(|| format!("GGUF tensor {name} 指向无效分片 {}", tensor.shard))?;
        let absolute = shard.data_offset.checked_add(tensor.offset).and_then(|value| value.checked_add(offset as u64)).ok_or_else(|| format!("GGUF tensor {name} 文件 offset 溢出"))?;
        Ok((Arc::clone(&shard.file), absolute))
    }

    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let tensor = self.tensor(name).ok_or_else(|| format!("GGUF 无 tensor {name}"))?;
        let tensor_type = tensor.tensor_type.0;
        let elements = tensor.elements();
        let bytes = self.read_tensor(name)?;
        crate::weight::codec::ggml::dequantize(tensor_type, &bytes, elements).map_err(|error| format!("解码 GGUF tensor {name}: {error}"))
    }

    pub fn read_tensor_i32(&self, name: &str) -> Result<Vec<i32>, String> {
        let tensor = self.tensor(name).ok_or_else(|| format!("GGUF 无 tensor {name}"))?;
        if tensor.tensor_type.0 != 26 {
            return Err(format!("GGUF tensor {name} type={}，期望 I32", tensor.tensor_type.name()));
        }
        let bytes = self.read_tensor(name)?;
        Ok(bytes.chunks_exact(4).map(|value| i32::from_le_bytes(value.try_into().expect("I32 block"))).collect())
    }

    pub fn read_matrix(&self, name: &str) -> Result<GgufMatrix, String> {
        let tensor = self.tensor(name).ok_or_else(|| format!("GGUF 无 tensor {name}"))?;
        if tensor.dims.len() != 2 {
            return Err(format!("GGUF tensor {name} shape={:?}，不是矩阵", tensor.dims));
        }
        let (columns, rows, tensor_type) = (tensor.dims[0], tensor.dims[1], tensor.tensor_type);
        let (file, offset) = self.tensor_storage(name, 0, tensor.bytes)?;
        Ok(GgufMatrix { name: name.to_owned(), rows, columns, tensor_type, file, offset, bytes_len: tensor.bytes, bytes: Arc::new(OnceLock::new()) })
    }

    pub fn read_matrix_slice(&self, name: &str, index: usize) -> Result<GgufMatrix, String> {
        let tensor = self.tensor(name).ok_or_else(|| format!("GGUF 无 tensor {name}"))?;
        if tensor.dims.len() != 3 {
            return Err(format!("GGUF tensor {name} shape={:?}，不是矩阵数组", tensor.dims));
        }
        let (columns, rows, count) = (tensor.dims[0], tensor.dims[1], tensor.dims[2]);
        if index >= count {
            return Err(format!("GGUF tensor {name} matrix index={index} 超出 {count}"));
        }
        let elements = columns.checked_mul(rows).ok_or_else(|| format!("GGUF tensor {name} slice 元素数溢出"))?;
        let slice_bytes = tensor.tensor_type.storage_bytes(elements)?;
        if slice_bytes.checked_mul(count) != Some(tensor.bytes) {
            return Err(format!("GGUF tensor {name} 第三维不是独立 block 对齐矩阵"));
        }
        let tensor_type = tensor.tensor_type;
        let offset = index * slice_bytes;
        let (file, offset) = self.tensor_storage(name, offset, slice_bytes)?;
        Ok(GgufMatrix { name: format!("{name}[{index}]"), rows, columns, tensor_type, file, offset, bytes_len: slice_bytes, bytes: Arc::new(OnceLock::new()) })
    }

    pub fn read_matrix_row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>, String> {
        let tensor = self.tensor(name).ok_or_else(|| format!("GGUF 无 tensor {name}"))?;
        if tensor.dims.len() != 2 {
            return Err(format!("GGUF tensor {name} shape={:?}，不是矩阵", tensor.dims));
        }
        let (columns, rows, tensor_type) = (tensor.dims[0], tensor.dims[1], tensor.tensor_type);
        if row >= rows {
            return Err(format!("GGUF tensor {name} row={row} 超出 {rows}"));
        }
        let row_bytes = tensor_type.storage_bytes(columns)?;
        let bytes = self.read_tensor_range(name, row * row_bytes, row_bytes)?;
        crate::weight::codec::ggml::dequantize(tensor_type.0, &bytes, columns).map_err(|error| format!("解码 GGUF tensor {name} row {row}: {error}"))
    }

    pub fn bpe_tokenizer(&self) -> Result<Tokenizer, String> {
        let (tokens, merges, special) = self.bpe_tokenizer_parts()?;
        if self.is_gemma4_tokenizer() {
            return Tokenizer::from_gguf_gemma4_tokens(&tokens, &merges, &special).map_err(|error| format!("构造 GGUF gemma4 tokenizer: {error}"));
        }
        let preset = self.metadata("tokenizer.ggml.pre").and_then(GgufValue::as_str);
        Tokenizer::from_gguf_bpe_tokens(&tokens, &merges, &special, preset).map_err(|error| format!("构造 GGUF tokenizer: {error}"))
    }

    pub fn bpe_detokenizer(&self) -> Result<Detokenizer, String> {
        let (tokens, _, special) = self.bpe_tokenizer_parts()?;
        if self.is_gemma4_tokenizer() {
            return Detokenizer::from_gguf_gemma4_tokens(&tokens, &special).map_err(|error| format!("构造 GGUF gemma4 detokenizer: {error}"));
        }
        Detokenizer::from_bpe_tokens(&tokens, &special).map_err(|error| format!("构造 GGUF detokenizer: {error}"))
    }

    // gemma4 的 tokens 是字面文本(▁ 表空格、<0xXX> 表 byte fallback),与 ByteLevel 词表不同构,需要独立构造路径。
    fn is_gemma4_tokenizer(&self) -> bool {
        self.metadata("tokenizer.ggml.model").and_then(GgufValue::as_str) == Some("gemma4")
    }

    // -------- 共享的模型语义转换方法(所有 GGUF 模型 wrapper 复用) --------

    /// GGUF 转换器把 Gemma 零中心 norm 加一存储;在权重边界还原架构语义。
    pub fn gemma_norm_vector(&self, name: &str) -> Result<Vec<f32>, String> {
        let mut values = self.read_tensor_f32(name)?;
        for value in &mut values {
            *value -= 1.0;
        }
        Ok(values)
    }

    /// GGUF `ssm_a` 保存为 `-exp(A_log)`,设备 capability 接收原始 A_log。
    pub fn a_log_vector(&self, name: &str) -> Result<Vec<f32>, String> {
        let mut values = self.read_tensor_f32(name)?;
        for value in &mut values {
            match decode_ssm_a(*value) {
                Ok(decoded) => *value = decoded,
                // 诊断模式：值域不符时带文件偏移与原始值，便于定位上游存储约定差异。
                Err(error) => {
                    let tensor = self.tensor(name);
                    return Err(format!("GGUF {name} {error}（raw[:4]={:?}, offset={:?}, type={:?}, dims={:?}）", &values[..values.len().min(4)], tensor.map(|t| t.offset), tensor.map(|t| t.tensor_type.0), tensor.map(|t| t.dims.clone())));
                }
            }
        }
        Ok(values)
    }

    /// 按 token id 读取 embedding 矩阵的指定行,解量化为 F32。
    pub fn embedding_rows(&self, name: &str, token_ids: &[u32], hidden_size: usize, vocab_size: usize) -> Result<Vec<f32>, String> {
        let mut output = Vec::with_capacity(token_ids.len() * hidden_size);
        for &token in token_ids {
            let token = token as usize;
            if token >= vocab_size {
                return Err(format!("GGUF token id {token} 超出 vocab {vocab_size}"));
            }
            output.extend(self.read_matrix_row_f32(name, token)?);
        }
        Ok(output)
    }

    /// 是否含 MTP 层(检查 blk.{layer_count}.nextn.eh_proj.weight 是否存在)。
    pub fn has_mtp(&self, layer_count: usize) -> bool {
        self.tensor(&format!("blk.{layer_count}.nextn.eh_proj.weight")).is_some()
    }

    #[allow(clippy::type_complexity)]
    fn bpe_tokenizer_parts(&self) -> Result<(Vec<String>, Vec<String>, Vec<bool>), String> {
        let tokens = self.metadata("tokenizer.ggml.tokens").and_then(GgufValue::as_string_array).ok_or_else(|| "GGUF 缺少完整 tokenizer.ggml.tokens".to_owned())?.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let merges = self.metadata("tokenizer.ggml.merges").and_then(GgufValue::as_string_array).ok_or_else(|| "GGUF 缺少完整 tokenizer.ggml.merges".to_owned())?.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let token_types = self.metadata("tokenizer.ggml.token_type").and_then(GgufValue::as_i64_array).ok_or_else(|| "GGUF 缺少完整 tokenizer.ggml.token_type".to_owned())?;
        if token_types.len() != tokens.len() {
            return Err(format!("GGUF tokenizer token_type={}，tokens={}", token_types.len(), tokens.len()));
        }
        let special = token_types.into_iter().map(|kind| matches!(kind, 2..=4)).collect();
        Ok((tokens, merges, special))
    }
}

fn decode_ssm_a(value: f32) -> Result<f32, String> {
    // Ornith-1.5-397B 上游自定义量化管线在个别层留下 ≈0 的 denormal 噪声
    // （正负混杂的 1e-5~1e-35）。绝对值不大于 1e-4 一律按零衰减率处理：
    // 这些 channel 的 -exp(A_log) 本来就 ≈0，钳到 0 不改变语义。
    if value.is_finite() && value.abs() <= 1e-4 {
        return Ok(f32::NEG_INFINITY);
    }
    if !value.is_finite() || value > 0.0 {
        return Err(format!("期望负的 -exp(A_log),实际 {value}"));
    }
    // GGUF 可把精确的零衰减率保存为 -exp(A_log)=0；逆变换是 -∞，
    // 下游 exp(A_log) 会恢复为 0，不能把整层零值误判成损坏权重。
    Ok(if value == 0.0 { f32::NEG_INFINITY } else { (-value).ln() })
}

fn parse_shard(path: &Path) -> Result<ParsedShard, String> {
    let file = File::open(path).map_err(|error| format!("打开 GGUF {}: {error}", path.display()))?;
    let file_len = file.metadata().map_err(|error| format!("读取 GGUF metadata: {error}"))?.len();
    let mut reader = BufReader::new(file.try_clone().map_err(|error| format!("clone GGUF file: {error}"))?);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(io_error)?;
    if &magic != b"GGUF" {
        return Err(format!("{} 不是 GGUF 文件", path.display()));
    }
    let version = read_u32(&mut reader)?;
    if version != 3 {
        return Err(format!("仅支持 GGUF v3，实际 v{version}"));
    }
    let tensor_count = usize::try_from(read_u64(&mut reader)?).map_err(|_| "GGUF tensor_count 超出 usize".to_owned())?;
    let metadata_count = usize::try_from(read_u64(&mut reader)?).map_err(|_| "GGUF metadata_count 超出 usize".to_owned())?;
    let mut metadata = HashMap::with_capacity(metadata_count);
    for _ in 0..metadata_count {
        let key = read_string(&mut reader)?;
        let value_type = GgufValueType::try_from(read_u32(&mut reader)?)?;
        // 只有 tokenizer.* 大数组允许截断(词表量级条目,探测场景不需要全量);
        // 其它 metadata(如 deepseek4.attention.compress_ratios,每层一个元素)必须保留完整数组,否则 >64 层会被误报缺失。
        let truncate_large = key.starts_with("tokenizer.") && !matches!(key.as_str(), "tokenizer.ggml.tokens" | "tokenizer.ggml.merges" | "tokenizer.ggml.token_type");
        let value = read_value(&mut reader, value_type, truncate_large)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(format!("GGUF metadata key 重复: {key}"));
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut tensor_index = HashMap::with_capacity(tensor_count);
    for index in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let dimensions = usize::try_from(read_u32(&mut reader)?).map_err(|_| "GGUF dimension_count 超出 usize".to_owned())?;
        if dimensions == 0 || dimensions > 4 {
            return Err(format!("GGUF tensor {name} dimension_count={dimensions} 非法"));
        }
        let mut dims = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            dims.push(usize::try_from(read_u64(&mut reader)?).map_err(|_| format!("GGUF tensor {name} dimension 超出 usize"))?);
        }
        let tensor_type = GgmlType(read_u32(&mut reader)?);
        let offset = read_u64(&mut reader)?;
        let elements = dims.iter().try_fold(1usize, |total, dimension| total.checked_mul(*dimension).ok_or("GGUF tensor 元素数溢出"))?;
        let bytes = tensor_type.storage_bytes(elements)?;
        if tensor_index.insert(name.clone(), index).is_some() {
            return Err(format!("GGUF tensor 重复: {name}"));
        }
        tensors.push(GgufTensorInfo { name, dims, tensor_type, offset, bytes, shard: 0 });
    }
    let alignment = metadata.get("general.alignment").and_then(GgufValue::as_u64).unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!("GGUF alignment={alignment} 非法"));
    }
    let directory_end = reader.stream_position().map_err(io_error)?;
    let data_offset = directory_end.div_ceil(alignment) * alignment;
    for tensor in &tensors {
        let end = data_offset.checked_add(tensor.offset).and_then(|offset| offset.checked_add(tensor.bytes as u64)).ok_or_else(|| format!("GGUF tensor {} offset 溢出", tensor.name))?;
        if end > file_len {
            return Err(format!("GGUF tensor {} 超出文件: end={end}, file={file_len}", tensor.name));
        }
    }
    Ok(ParsedShard { file, file_len, version, data_offset, metadata, tensors })
}

fn split_name(path: &Path) -> Option<(String, usize, usize)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (indexed_prefix, count) = stem.rsplit_once("-of-")?;
    let (prefix, index) = indexed_prefix.rsplit_once('-')?;
    let index = index.parse().ok()?;
    let count = count.parse().ok()?;
    (index > 0 && count > 1 && index <= count).then(|| (prefix.to_owned(), index, count))
}

fn split_paths(path: &Path, count: usize) -> Result<Vec<PathBuf>, String> {
    let (prefix, _, parsed_count) = split_name(path).ok_or_else(|| format!("{} 不是标准 GGUF split 文件名", path.display()))?;
    if parsed_count != count {
        return Err(format!("GGUF split 文件名 count={parsed_count}/{count} 不一致"));
    }
    let width = count.to_string().len().max(5);
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    Ok((1..=count).map(|index| directory.join(format!("{prefix}-{index:0width$}-of-{count:0width$}.gguf"))).collect())
}

// 文件内声明的长度直接决定分配大小;恶意/损坏文件可借超大值触发 OOM abort。
// 上限远超真实模型(GGUF metadata 字符串很短、数组最大为词表量级数十万),超限即判文件损坏。
const MAX_METADATA_STRING_BYTES: usize = 1 << 28;
const MAX_METADATA_ARRAY_LEN: usize = 1 << 24;

fn read_value<R: Read + Seek>(reader: &mut R, value_type: GgufValueType, truncate_large: bool) -> Result<GgufValue, String> {
    Ok(match value_type {
        GgufValueType::U8 => GgufValue::Unsigned(read_u8(reader)? as u64),
        GgufValueType::I8 => GgufValue::Signed(read_i8(reader)? as i64),
        GgufValueType::U16 => GgufValue::Unsigned(read_u16(reader)? as u64),
        GgufValueType::I16 => GgufValue::Signed(read_i16(reader)? as i64),
        GgufValueType::U32 => GgufValue::Unsigned(read_u32(reader)? as u64),
        GgufValueType::I32 => GgufValue::Signed(read_i32(reader)? as i64),
        GgufValueType::F32 => GgufValue::Float(read_f32(reader)? as f64),
        GgufValueType::Bool => GgufValue::Bool(read_u8(reader)? != 0),
        GgufValueType::String => GgufValue::String(read_string(reader)?),
        GgufValueType::U64 => GgufValue::Unsigned(read_u64(reader)?),
        GgufValueType::I64 => GgufValue::Signed(read_i64(reader)?),
        GgufValueType::F64 => GgufValue::Float(read_f64(reader)?),
        GgufValueType::Array => {
            let element_type = GgufValueType::try_from(read_u32(reader)?)?;
            if element_type == GgufValueType::Array {
                return Err("GGUF 不允许嵌套 array".to_owned());
            }
            let len = usize::try_from(read_u64(reader)?).map_err(|_| "GGUF array len 超出 usize".to_owned())?;
            if len > MAX_METADATA_ARRAY_LEN {
                return Err(format!("GGUF array len={len} 超过合理性上限 {MAX_METADATA_ARRAY_LEN}，文件可能损坏"));
            }
            const KEEP_VALUES: usize = 64;
            let keep = !truncate_large || len <= KEEP_VALUES;
            let mut values = Vec::with_capacity(if keep { len } else { 0 });
            for _ in 0..len {
                let value = read_value(reader, element_type, truncate_large)?;
                if keep {
                    values.push(value);
                }
            }
            GgufValue::Array { element_type, len, values, truncated: !keep }
        }
    })
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, String> {
    let len = usize::try_from(read_u64(reader)?).map_err(|_| "GGUF string len 超出 usize".to_owned())?;
    if len > MAX_METADATA_STRING_BYTES {
        return Err(format!("GGUF string len={len} 超过合理性上限 {MAX_METADATA_STRING_BYTES}，文件可能损坏"));
    }
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    String::from_utf8(bytes).map_err(|error| format!("GGUF string 不是 UTF-8: {error}"))
}

fn io_error(error: io::Error) -> String {
    format!("读取 GGUF: {error}")
}

macro_rules! read_number {
    ($name:ident, $type:ty) => {
        fn $name<R: Read>(reader: &mut R) -> Result<$type, String> {
            let mut bytes = [0u8; std::mem::size_of::<$type>()];
            reader.read_exact(&mut bytes).map_err(io_error)?;
            Ok(<$type>::from_le_bytes(bytes))
        }
    };
}

read_number!(read_u8, u8);
read_number!(read_i8, i8);
read_number!(read_u16, u16);
read_number!(read_i16, i16);
read_number!(read_u32, u32);
read_number!(read_i32, i32);
read_number!(read_u64, u64);
read_number!(read_i64, i64);
read_number!(read_f32, f32);
read_number!(read_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    /// MTP 头结构诊断:张量名与形状。
    /// `ZLLM_GEMMA4_MTP=/path/to/mtp.gguf cargo test --lib mtp_structure -- --nocapture`
    #[test]
    fn mtp_structure() {
        let Some(path) = std::env::var_os("ZLLM_GEMMA4_MTP").map(std::path::PathBuf::from) else { return };
        let reader = GgufReader::open(&path).expect("open mtp");
        let mut names: Vec<&str> = reader.tensors().iter().map(|tensor| tensor.name.as_str()).collect();
        names.sort();
        println!("[mtp] 张量数={}", names.len());
        for name in names {
            let info = reader.tensor(name).expect("tensor");
            println!("[mtp-tensor] {name} dims={:?} type={}", info.dims, info.tensor_type.name());
        }
    }

    #[test]
    fn standard_storage_sizes_match_ggml_blocks() {
        assert_eq!(GgmlType(11).storage_bytes(256).unwrap(), 110);
        assert_eq!(GgmlType(12).storage_bytes(256).unwrap(), 144);
        assert_eq!(GgmlType(13).storage_bytes(256).unwrap(), 176);
        assert_eq!(GgmlType(14).storage_bytes(256).unwrap(), 210);
        assert_eq!(GgmlType(16).storage_bytes(256).unwrap(), 66);
        assert_eq!(GgmlType(17).storage_bytes(256).unwrap(), 74);
        assert_eq!(GgmlType(18).storage_bytes(256).unwrap(), 98);
        assert_eq!(GgmlType(22).storage_bytes(256).unwrap(), 82);
        assert_eq!(GgmlType(23).storage_bytes(256).unwrap(), 136);
    }

    #[test]
    fn zero_ssm_a_roundtrips_to_zero_decay_rate() {
        let a_log = decode_ssm_a(0.0).unwrap();
        assert_eq!(a_log, f32::NEG_INFINITY);
        assert_eq!(a_log.exp(), 0.0);
        assert!(decode_ssm_a(0.1).is_err());
    }

    #[test]
    fn matrix_location_reads_directly_and_slices_without_copy() {
        let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("zllm-gguf-view-{}-{unique}.bin", std::process::id()));
        let payload = (0..32u8).collect::<Vec<_>>();
        let mut file_bytes = vec![91, 92, 93];
        file_bytes.extend_from_slice(&payload);
        std::fs::write(&path, &file_bytes).unwrap();
        let matrix = GgufMatrix { name: "test.weight".to_owned(), rows: 2, columns: 4, tensor_type: GgmlType(0), file: Arc::new(File::open(&path).unwrap()), offset: 3, bytes_len: payload.len(), bytes: Arc::new(OnceLock::new()) };
        let mut direct = vec![0u8; payload.len()];
        matrix.read_into(&mut direct).unwrap();
        assert_eq!(direct, payload);
        assert_eq!(matrix.read_bytes().unwrap(), payload);
        assert!(matrix.bytes.get().is_none());
        assert_eq!(matrix.bytes().unwrap(), payload);
        assert_eq!(matrix.row_slice(1, 1).unwrap().bytes().unwrap(), &payload[16..]);
        let mut selected = vec![0u8; payload.len()];
        matrix.read_rows_into(&[1, 0], &mut selected).unwrap();
        assert_eq!(selected, [&payload[16..], &payload[..16]].concat());
        std::fs::remove_file(path).unwrap();
    }
}
