//! 轻量 safetensors 索引读取器。
//!
//! 支持按 tensor 名字打开数据，以及按行读取连续 rank-2 tensor。

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

use super::file_ext::FileExt;
use std::sync::{Arc, Mutex};

use half::{bf16, f16};

const INDEX_FILE: &str = "model.safetensors.index.json";
const SINGLE_FILE: &str = "model.safetensors";

#[derive(Clone, Debug)]
pub struct TensorData {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorData {
    /// 将 BF16 / F16 / F32 字节解码为 `Vec<f32>`,并校验 `data.len() == elements * elem_size`。
    /// `tensor.name` 出现在错误信息中,便于追溯出错的具体权重。
    pub fn to_f32(&self) -> Result<Vec<f32>, String> {
        let elements = self.shape.iter().try_fold(1usize, |elements, &dimension| elements.checked_mul(dimension)).ok_or_else(|| format!("{} shape {:?} 元素数量溢出", self.name, self.shape))?;
        let element_bytes = match self.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            dtype => return Err(format!("{} dtype={dtype} 不支持转换为 F32", self.name)),
        };
        let expected_bytes = elements.checked_mul(element_bytes).ok_or_else(|| format!("{} shape {:?} 字节数溢出", self.name, self.shape))?;
        if self.data.len() != expected_bytes {
            return Err(format!("{} dtype={} bytes={} 与 shape {:?} 不兼容，期望 {expected_bytes}", self.name, self.dtype, self.data.len(), self.shape));
        }
        decode_to_f32(&self.name, &self.dtype, &self.data)
    }

    /// 将 BF16 / F16 / F32 字节解码为 `Vec<f16>`;与 `to_f32` 同一解码路径再收窄,
    /// BF16/F16 数值无损,F32 可能溢出为非有限值(是否拒绝由调用方决定)。
    pub fn to_f16(&self) -> Result<Vec<f16>, String> {
        Ok(self.to_f32()?.into_iter().map(f16::from_f32).collect())
    }

    /// 校验 shape 与期望一致，不一致则返回带张量名的错误。
    pub fn expect_shape(&self, expected: &[usize]) -> Result<(), String> {
        if self.shape != expected {
            return Err(format!("{} shape {:?}，期望 {expected:?}", self.name, self.shape));
        }
        Ok(())
    }
}

/// BF16 / F16 / F32 小端字节解码为 `Vec<f32>`;`name` 只用于错误上下文。
/// 各权重加载路径共用这一份转换;字节数不是元素大小整数倍时判数据损坏,
/// 避免 chunks_exact 静默丢弃尾部字节。
pub fn decode_to_f32(name: &str, dtype: &str, data: &[u8]) -> Result<Vec<f32>, String> {
    let element_bytes = match dtype {
        "BF16" | "F16" => 2,
        "F32" => 4,
        other => return Err(format!("{name} dtype={other} 不支持转换为 F32(期望 BF16/F16/F32)")),
    };
    if !data.len().is_multiple_of(element_bytes) {
        return Err(format!("{name} dtype={dtype} bytes={} 不是元素大小 {element_bytes} 的整数倍", data.len()));
    }
    Ok(match dtype {
        "BF16" => data.chunks_exact(2).map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        "F16" => data.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        "F32" => data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        _ => unreachable!("dtype 已校验"),
    })
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

pub struct TensorDestination<'a> {
    pub name: String,
    pub data: &'a mut [u8],
}

#[derive(Clone, Debug)]
pub struct SafetensorStore {
    root: PathBuf,
    weight_map: HashMap<String, String>,
    shards: Arc<Mutex<HashMap<String, Arc<CachedShard>>>>,
}

#[derive(Deserialize)]
struct SourceIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct CachedShard {
    file: File,
    header: HashMap<String, TensorMetadata>,
    data_offset: u64,
}

type SharedShards = Arc<Mutex<HashMap<String, Arc<CachedShard>>>>;

fn shared_shards(root: &Path) -> Result<SharedShards, String> {
    type SharedShardCaches = HashMap<PathBuf, std::sync::Weak<Mutex<HashMap<String, Arc<CachedShard>>>>>;
    static CACHES: std::sync::OnceLock<Mutex<SharedShardCaches>> = std::sync::OnceLock::new();

    let mut caches = CACHES.get_or_init(|| Mutex::new(HashMap::new())).lock().map_err(|_| "safetensors 全局 shard cache 锁中毒".to_owned())?;
    caches.retain(|_, shards| shards.strong_count() != 0);
    if let Some(shards) = caches.get(root).and_then(std::sync::Weak::upgrade) {
        return Ok(shards);
    }
    let shards = Arc::new(Mutex::new(HashMap::new()));
    caches.insert(root.to_path_buf(), Arc::downgrade(&shards));
    Ok(shards)
}

impl SafetensorStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        let index_path = root.join(INDEX_FILE);
        if index_path.is_file() {
            let index_data = std::fs::read(&index_path).map_err(|error| format!("读取 safetensors index {} 失败: {error}", index_path.display()))?;
            let index: SourceIndex = serde_json::from_slice(&index_data).map_err(|error| format!("解析 {index_path:?} 失败: {error}"))?;
            let shards = shared_shards(&root)?;
            return Ok(Self { root, weight_map: index.weight_map, shards });
        }

        let shard_name = SINGLE_FILE.to_owned();
        let shard_path = root.join(SINGLE_FILE);
        let shards = shared_shards(&root)?;
        let weight_map = {
            let mut cached = shards.lock().map_err(|_| "safetensors shard cache 锁中毒".to_owned())?;
            if !cached.contains_key(&shard_name) {
                let mut file = File::open(&shard_path).map_err(|error| format!("缺少 {}，且打开单文件 {} 失败: {error}", index_path.display(), shard_path.display()))?;
                let (header, data_offset) = header_and_data_offset(&mut file)?;
                cached.insert(shard_name.clone(), Arc::new(CachedShard { file, header, data_offset }));
            }
            cached[&shard_name].header.keys().map(|name| (name.clone(), shard_name.clone())).collect()
        };
        Ok(Self { root, weight_map, shards })
    }

    pub fn has(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    /// 返回所有 tensor 名（排序）。
    pub fn tensor_names(&self) -> Vec<String> {
        self.weight_map.keys().cloned().collect()
    }

    pub fn load(&self, name: &str) -> Result<TensorData, String> {
        let (shard_name, shard) = self.cached_shard(name)?;
        load_from_shard(&shard_name, &shard, name)
    }

    /// 读取 rank-2 tensor 的连续列区间，避免为一个投影分片加载整张矩阵。
    pub fn load_columns(&self, name: &str, columns: Range<usize>) -> Result<TensorData, String> {
        let (shard_name, shard) = self.cached_shard(name)?;
        let metadata = shard.header.get(name).ok_or_else(|| format!("{shard_name} 中缺少 {name:?}"))?;
        if metadata.shape.len() != 2 || columns.is_empty() || columns.end > metadata.shape[1] {
            return Err(format!("{name} shape {:?} 无法读取列 {columns:?}", metadata.shape));
        }
        let element_bytes = match metadata.dtype.as_str() {
            "BF16" | "F16" => 2usize,
            "F32" => 4,
            dtype => return Err(format!("{name} dtype={dtype} 不支持列分片读取")),
        };
        let rows = metadata.shape[0];
        let source_columns = metadata.shape[1];
        let output_columns = columns.len();
        let row_bytes = output_columns.checked_mul(element_bytes).ok_or_else(|| format!("{name} 列分片行字节溢出"))?;
        let mut data = vec![0u8; rows.checked_mul(row_bytes).ok_or_else(|| format!("{name} 列分片字节溢出"))?];
        let tensor_offset = shard.data_offset.checked_add(metadata.data_offsets[0]).ok_or_else(|| format!("{name} 数据偏移溢出"))?;
        for row in 0..rows {
            let element_offset = row.checked_mul(source_columns).and_then(|offset| offset.checked_add(columns.start)).ok_or_else(|| format!("{name} L{row} 列偏移溢出"))?;
            let byte_offset = element_offset.checked_mul(element_bytes).ok_or_else(|| format!("{name} L{row} 字节偏移溢出"))?;
            let file_offset = tensor_offset.checked_add(byte_offset as u64).ok_or_else(|| format!("{name} L{row} 文件偏移溢出"))?;
            shard.file.read_exact_at(&mut data[row * row_bytes..(row + 1) * row_bytes], file_offset).map_err(|error| format!("读取 {shard_name}:{name} L{row} columns={columns:?} 失败: {error}"))?;
        }
        Ok(TensorData { name: name.to_owned(), dtype: metadata.dtype.clone(), shape: vec![rows, output_columns], data })
    }

    /// 只读取 tensor 元数据，不触碰数据区；用于大 checkpoint 启动时校验。
    pub fn tensor_info(&self, name: &str) -> Result<TensorInfo, String> {
        let (shard_name, shard) = self.cached_shard(name)?;
        let metadata = shard.header.get(name).ok_or_else(|| format!("{shard_name} 中缺少 {name:?}"))?;
        Ok(TensorInfo { name: name.to_owned(), dtype: metadata.dtype.clone(), shape: metadata.shape.clone() })
    }

    /// 同 shard 的一对 tensor 共用一次 cache 查找；数据读取顺序与单独 load 保持一致。
    pub fn load_pair(&self, first: &str, second: &str) -> Result<(TensorData, TensorData), String> {
        let (first_shard_name, first_shard) = self.cached_shard(first)?;
        let first_data = load_from_shard(&first_shard_name, &first_shard, first)?;
        let second_shard_name = self.weight_map.get(second).ok_or_else(|| format!("权重 {second:?} 不在当前 safetensors index 中"))?;
        let second_data = if second_shard_name == &first_shard_name { load_from_shard(&first_shard_name, &first_shard, second)? } else { self.load(second)? };
        Ok((first_data, second_data))
    }

    /// 直接把 tensor 文件内容读入调用方提供的存储，避免中间 Vec。
    pub fn load_into(&self, name: &str, destination: &mut [u8]) -> Result<TensorInfo, String> {
        let (shard_name, shard) = self.cached_shard(name)?;
        let metadata = shard.header.get(name).ok_or_else(|| format!("{shard_name} 中缺少 {name:?}"))?;
        let size = metadata.data_offsets[1].checked_sub(metadata.data_offsets[0]).ok_or_else(|| format!("safetensors {shard_name}:{name} data_offsets 无效"))?;
        let size = usize::try_from(size).map_err(|_| "tensor 数据过大".to_owned())?;
        if destination.len() != size {
            return Err(format!("{shard_name}:{name} bytes={size}，destination={}", destination.len()));
        }
        let offset = shard.data_offset.checked_add(metadata.data_offsets[0]).ok_or_else(|| "safetensors 数据偏移溢出".to_owned())?;
        shard.file.read_exact_at(destination, offset).map_err(|error| format!("读取 {shard_name}:{name} 失败: {error}"))?;
        Ok(TensorInfo { name: name.to_owned(), dtype: metadata.dtype.clone(), shape: metadata.shape.clone() })
    }

    /// 批量读取先按 shard/offset 排序，让少量长生命周期 worker 保持顺序 SSD I/O。
    pub fn load_many_into(&self, destinations: Vec<TensorDestination<'_>>) -> Result<Vec<TensorInfo>, String> {
        struct PlannedRead<'a> {
            index: usize,
            shard_name: String,
            shard: Arc<CachedShard>,
            offset: u64,
            data: &'a mut [u8],
        }

        let count = destinations.len();
        let mut infos: Vec<Option<TensorInfo>> = (0..count).map(|_| None).collect();
        let mut opened = HashMap::<String, Arc<CachedShard>>::new();
        let mut reads = Vec::with_capacity(count);
        for (index, destination) in destinations.into_iter().enumerate() {
            let shard_name = self.weight_map.get(&destination.name).ok_or_else(|| format!("权重 {:?} 不在当前 safetensors index 中", destination.name))?.clone();
            let shard = if let Some(shard) = opened.get(&shard_name) {
                shard.clone()
            } else {
                let (_, shard) = self.cached_shard(&destination.name)?;
                opened.insert(shard_name.clone(), shard.clone());
                shard
            };
            let metadata = shard.header.get(&destination.name).ok_or_else(|| format!("{shard_name} 中缺少 {:?}", destination.name))?;
            let size = metadata.data_offsets[1].checked_sub(metadata.data_offsets[0]).and_then(|size| usize::try_from(size).ok()).ok_or_else(|| format!("safetensors {shard_name}:{} 数据大小无效", destination.name))?;
            if destination.data.len() != size {
                return Err(format!("{shard_name}:{} bytes={size}，destination={}", destination.name, destination.data.len()));
            }
            let offset = shard.data_offset.checked_add(metadata.data_offsets[0]).ok_or_else(|| "safetensors 数据偏移溢出".to_owned())?;
            infos[index] = Some(TensorInfo { name: destination.name, dtype: metadata.dtype.clone(), shape: metadata.shape.clone() });
            reads.push(PlannedRead { index, shard_name, shard, offset, data: destination.data });
        }
        reads.sort_unstable_by(|left, right| left.shard_name.cmp(&right.shard_name).then_with(|| left.offset.cmp(&right.offset)));
        for read in reads {
            read.shard.file.read_exact_at(read.data, read.offset).map_err(|error| format!("读取 {}:{} 失败: {error}", read.shard_name, infos[read.index].as_ref().expect("读取计划 info 已设置").name))?;
        }
        Ok(infos.into_iter().map(|info| info.expect("读取计划 info 已设置")).collect())
    }

    pub fn load_bf16_rows(&self, name: &str, rows: &[usize]) -> Result<TensorData, String> {
        let tensor = self.load_rows(name, rows)?;
        if tensor.dtype != "BF16" {
            return Err(format!("{name:?} dtype={}，期望 BF16", tensor.dtype));
        }
        Ok(tensor)
    }

    pub fn load_rows(&self, name: &str, rows: &[usize]) -> Result<TensorData, String> {
        if rows.is_empty() {
            return Err("tensor 行选择不能为空".into());
        }
        let (shard_name, shard) = self.cached_shard(name)?;
        let metadata = shard.header.get(name).ok_or_else(|| format!("{name:?} 在 {shard_name} 中缺失"))?;
        if metadata.shape.len() != 2 {
            return Err(format!("{name:?} 不是 rank-2 tensor，shape={:?}", metadata.shape));
        }
        let element_bytes = match metadata.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" | "U32" | "I32" => 4,
            "U8" | "I8" => 1,
            dtype => return Err(format!("{name:?} dtype={dtype} 暂不支持按行读取")),
        };
        let source_rows = metadata.shape[0];
        let columns = metadata.shape[1];
        let row_bytes = columns.checked_mul(element_bytes).ok_or_else(|| format!("{name:?} 行宽溢出"))?;
        let mut data = vec![0_u8; rows.len().checked_mul(row_bytes).ok_or_else(|| format!("{name:?} 行选择过大"))?];
        let tensor_start = shard.data_offset.checked_add(metadata.data_offsets[0]).ok_or_else(|| "tensor 数据起始偏移溢出".to_owned())?;
        for (dst_row, &source_row) in rows.iter().enumerate() {
            if source_row >= source_rows {
                return Err(format!("row {source_row} 越界于 {name:?} 的 {source_rows} 行"));
            }
            let source_offset = tensor_start.checked_add(source_row.checked_mul(row_bytes).ok_or_else(|| "tensor 行偏移溢出".to_owned())? as u64).ok_or_else(|| "tensor 行偏移溢出".to_owned())?;
            shard.file.read_exact_at(&mut data[dst_row * row_bytes..(dst_row + 1) * row_bytes], source_offset).map_err(|error| format!("读取 row {name} 失败: {error}"))?;
        }
        Ok(TensorData { name: name.to_owned(), dtype: metadata.dtype.clone(), shape: vec![rows.len(), columns], data })
    }

    fn cached_shard(&self, name: &str) -> Result<(String, Arc<CachedShard>), String> {
        let shard_name = self.weight_map.get(name).ok_or_else(|| format!("权重 {name:?} 不在当前 safetensors index 中"))?.clone();
        if let Some(shard) = self.shards.lock().map_err(|_| "safetensors shard cache 锁中毒".to_owned())?.get(&shard_name).cloned() {
            return Ok((shard_name, shard));
        }

        let mut file = File::open(self.root.join(&shard_name)).map_err(|error| format!("打开 shard {shard_name} 失败: {error}"))?;
        let (header, data_offset) = header_and_data_offset(&mut file)?;
        let loaded = Arc::new(CachedShard { file, header, data_offset });
        let shard = self.shards.lock().map_err(|_| "safetensors shard cache 锁中毒".to_owned())?.entry(shard_name.clone()).or_insert(loaded).clone();
        Ok((shard_name, shard))
    }
}

fn load_from_shard(shard_name: &str, shard: &CachedShard, name: &str) -> Result<TensorData, String> {
    let metadata = shard.header.get(name).ok_or_else(|| format!("{shard_name} 中缺少 {name:?}"))?;
    let start = metadata.data_offsets[0];
    let end = metadata.data_offsets[1];
    let size = end.checked_sub(start).ok_or_else(|| format!("safetensors {shard_name}:{name} data_offsets 无效"))?;
    let size = usize::try_from(size).map_err(|_| "tensor 数据过大".to_owned())?;
    let mut data = vec![0_u8; size];
    let offset = shard.data_offset.checked_add(start).ok_or_else(|| "safetensors 数据偏移溢出".to_owned())?;
    shard.file.read_exact_at(&mut data, offset).map_err(|error| format!("读取 {shard_name}:{name} 失败: {error}"))?;
    Ok(TensorData { name: name.to_owned(), dtype: metadata.dtype.clone(), shape: metadata.shape.clone(), data })
}

fn header_and_data_offset(file: &mut File) -> Result<(HashMap<String, TensorMetadata>, u64), String> {
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix).map_err(|error| format!("读取 safetensors shard 头失败: {error}"))?;
    let header_len = usize::try_from(u64::from_le_bytes(prefix)).map_err(|_| "safetensors header 超长".to_owned())?;
    // header_len 来自文件并直接决定分配大小;真实 header 为 JSON(通常 MB 级),超限判文件损坏,避免 OOM abort。
    const MAX_HEADER_BYTES: usize = 1 << 30;
    if header_len > MAX_HEADER_BYTES {
        return Err(format!("safetensors header len={header_len} 超过合理性上限 {MAX_HEADER_BYTES}，文件可能损坏"));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header).map_err(|error| format!("读取 safetensors shard header 失败: {error}"))?;
    let entries: HashMap<String, serde_json::Value> = serde_json::from_slice(trim_spaces(&header)).map_err(|error| "解析 safetensors header 失败".to_owned() + &format!(": {error}"))?;
    let mut tensors = HashMap::with_capacity(entries.len());
    for (name, value) in entries {
        if name == "__metadata__" {
            continue;
        }
        let metadata = serde_json::from_value(value).map_err(|error| format!("解析 safetensors tensor {name:?} 失败: {error}"))?;
        tensors.insert(name, metadata);
    }
    let data_offset = 8_u64.checked_add(header_len as u64).ok_or_else(|| "safetensors 数据偏移溢出".to_owned())?;
    Ok((tensors, data_offset))
}

fn trim_spaces(data: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < data.len() && data[start].is_ascii_whitespace() {
        start += 1;
    }
    let mut end = data.len();
    while end > start && data[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &data[start..end]
}

#[cfg(test)]
mod tests {
    use super::{SafetensorStore, TensorData};

    #[test]
    fn tensor_to_f32_rejects_invalid_size_and_dtype() {
        let truncated = TensorData { name: "weight".to_owned(), dtype: "F32".to_owned(), shape: vec![2], data: vec![0; 4] };
        assert!(truncated.to_f32().unwrap_err().contains("期望 8"));
        let unsupported = TensorData { name: "weight".to_owned(), dtype: "I8".to_owned(), shape: vec![1], data: vec![0] };
        assert!(unsupported.to_f32().unwrap_err().contains("不支持转换"));
    }

    #[test]
    fn tensor_to_f32_rejects_shape_overflow() {
        let tensor = TensorData { name: "weight".to_owned(), dtype: "F16".to_owned(), shape: vec![usize::MAX, 2], data: Vec::new() };
        assert!(tensor.to_f32().unwrap_err().contains("元素数量溢出"));
    }

    #[test]
    fn column_slice_keeps_source_row_order() {
        let root = std::env::temp_dir().join(format!("zllm-safetensor-columns-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap();
        let header = br#"{"weight":{"dtype":"BF16","shape":[3,4],"data_offsets":[0,24]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        for value in 1u16..=12 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(root.join("model.safetensors"), bytes).unwrap();
        let tensor = SafetensorStore::open(&root).unwrap().load_columns("weight", 1..3).unwrap();
        let values = tensor.data.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
        assert_eq!(tensor.shape, [3, 2]);
        assert_eq!(values, [2, 3, 6, 7, 10, 11]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
