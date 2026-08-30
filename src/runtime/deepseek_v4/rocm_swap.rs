//! DeepSeek-V4 MP resident block graph 的分块 SSD 快照。

use std::{
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    backend::rocm::{RocmCompressedKvSerde, RocmGatedPoolSerde},
    kv_cache::{
        FjallBlob, FjallCacheStore, FjallChunkKind,
        terminal_cache::{SnapshotReader, SnapshotWriter},
    },
    runtime::deepseek_v4::{dspark_rocm::RocmDeepSeekV4DsparkCache, rocm_stage::DeepSeekV4StageCache},
};

const VERSION: u32 = 2;
const INFO_MAGIC: [u8; 8] = *b"ZDSV4I01";
const MANIFEST_MAGIC: [u8; 8] = *b"ZDSV4M01";
static GENERATION: AtomicU64 = AtomicU64::new(1);

pub struct DeepSeekV4CacheSnapshot {
    pub cache_id: String,
    pub tokens: Vec<u32>,
    pub cache_namespace: Option<String>,
    pub round: usize,
    pub head: bool,
    pub stages: Vec<DeepSeekV4StageCache>,
    pub dspark: Option<RocmDeepSeekV4DsparkCache>,
    pub resident_bytes: u64,
    pub modified_unix: u64,
}

#[derive(Clone)]
pub struct DeepSeekV4SwapInfo {
    pub cache_id: String,
    pub token_count: usize,
    pub resident_bytes: u64,
    pub file_bytes: u64,
    pub modified_unix: u64,
}

struct Manifest {
    generation: u64,
    cache_id: String,
    tokens: Vec<u32>,
    cache_namespace: Option<String>,
    round: usize,
    head: bool,
    resident_bytes: u64,
    modified_unix: u64,
    blob: FjallBlob,
}

pub struct DeepSeekV4SwapStore {
    store: FjallCacheStore,
}

impl DeepSeekV4SwapStore {
    pub fn open(path: impl AsRef<Path>, metadata: &[(String, String)]) -> Result<Self, String> {
        let store = FjallCacheStore::open(path.as_ref().join("fjall"))?;
        store.bind_metadata(metadata)?;
        Ok(Self { store })
    }

    pub fn put(&self, snapshot: &DeepSeekV4CacheSnapshot) -> Result<DeepSeekV4SwapInfo, String> {
        if snapshot.cache_id.is_empty() || snapshot.tokens.is_empty() || snapshot.stages.is_empty() {
            return Err("DeepSeek-V4 cache snapshot 缺少 cache_id/token/stage".to_owned());
        }
        let previous = self.manifest(&snapshot.cache_id)?.map(|manifest| manifest.generation);
        let generation = new_generation();
        let mut writer = self.store.writer(&snapshot.cache_id, generation, 0, 0, FjallChunkKind::Kv);
        if let Err(error) = write_session(&mut writer, &snapshot.stages, snapshot.dspark.as_ref()) {
            let _ = self.store.remove_generation(&snapshot.cache_id, generation);
            return Err(error);
        }
        let blob = writer.finish()?;
        let info = DeepSeekV4SwapInfo { cache_id: snapshot.cache_id.clone(), token_count: snapshot.tokens.len(), resident_bytes: snapshot.resident_bytes, file_bytes: blob.bytes, modified_unix: snapshot.modified_unix };
        let manifest = Manifest {
            generation,
            cache_id: snapshot.cache_id.clone(),
            tokens: snapshot.tokens.clone(),
            cache_namespace: snapshot.cache_namespace.clone(),
            round: snapshot.round,
            head: snapshot.head,
            resident_bytes: snapshot.resident_bytes,
            modified_unix: snapshot.modified_unix,
            blob,
        };
        self.store.commit(&snapshot.cache_id, &encode_info(&info)?, &encode_manifest(&manifest)?)?;
        if let Some(previous) = previous.filter(|previous| *previous != generation) {
            let _ = self.store.remove_generation(&snapshot.cache_id, previous);
        }
        Ok(info)
    }

    pub fn get(&self, cache_id: &str) -> Result<Option<DeepSeekV4CacheSnapshot>, String> {
        let Some(manifest) = self.manifest(cache_id)? else { return Ok(None) };
        let reader = self.store.reader(cache_id, manifest.generation, 0, 0, FjallChunkKind::Kv, manifest.blob);
        let (stages, dspark) = read_session(reader)?;
        Ok(Some(DeepSeekV4CacheSnapshot {
            cache_id: manifest.cache_id,
            tokens: manifest.tokens,
            cache_namespace: manifest.cache_namespace,
            round: manifest.round,
            head: manifest.head,
            stages,
            dspark,
            resident_bytes: manifest.resident_bytes,
            modified_unix: manifest.modified_unix,
        }))
    }

    pub fn longest_prefix_cache_id(&self, tokens: &[u32]) -> Result<Option<String>, String> {
        let mut best = None::<(usize, String)>;
        let mut near_miss = None::<(usize, usize, String)>;
        for bytes in self.store.manifest_values()? {
            let Ok(manifest) = decode_manifest(&bytes) else { continue };
            if manifest.head || manifest.tokens.is_empty() {
                continue;
            }
            if tokens.starts_with(&manifest.tokens) {
                if best.as_ref().is_none_or(|(length, _)| manifest.tokens.len() > *length) {
                    best = Some((manifest.tokens.len(), manifest.cache_id.clone()));
                }
            } else {
                // 与内存侧同款前缀链诊断:近失配快照打印公共前缀长度,定位回显分叉。
                let common = manifest.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count();
                if common >= 1024 && common * 10 >= manifest.tokens.len() * 9 && near_miss.as_ref().is_none_or(|(length, _, _)| common > *length) {
                    near_miss = Some((common, manifest.tokens.len(), manifest.cache_id.clone()));
                }
            }
        }
        if let Some((common, snapshot_len, cache_id)) = near_miss
            && best.as_ref().is_none_or(|(_, best_id)| *best_id != cache_id)
        {
            eprintln!("[deepseek-v4-cache] SSD 前缀近失配 cache_id={cache_id} 公共前缀={common} 快照长度={snapshot_len} prompt 长度={}", tokens.len());
        }
        Ok(best.map(|(_, cache_id)| cache_id))
    }

    pub fn delete(&self, cache_id: &str) -> Result<(), String> {
        let generation = self.manifest(cache_id)?.map(|manifest| manifest.generation);
        self.store.remove_entry(cache_id)?;
        if let Some(generation) = generation {
            self.store.remove_generation(cache_id, generation)?;
        }
        Ok(())
    }

    pub fn set_head(&self, cache_id: &str, head: bool) -> Result<(), String> {
        let Some(mut manifest) = self.manifest(cache_id)? else { return Ok(()) };
        if manifest.head == head {
            return Ok(());
        }
        manifest.head = head;
        let info = DeepSeekV4SwapInfo { cache_id: manifest.cache_id.clone(), token_count: manifest.tokens.len(), resident_bytes: manifest.resident_bytes, file_bytes: manifest.blob.bytes, modified_unix: manifest.modified_unix };
        self.store.commit(cache_id, &encode_info(&info)?, &encode_manifest(&manifest)?)
    }

    pub fn infos(&self) -> Vec<DeepSeekV4SwapInfo> {
        let Ok(values) = self.store.info_values() else { return Vec::new() };
        values.into_iter().filter_map(|bytes| decode_info(&bytes).ok()).collect()
    }

    fn manifest(&self, cache_id: &str) -> Result<Option<Manifest>, String> {
        self.store.manifest(cache_id)?.map(|bytes| decode_manifest(&bytes)).transpose()
    }
}

fn encode_info(info: &DeepSeekV4SwapInfo) -> Result<Vec<u8>, String> {
    let mut writer = SnapshotWriter::new();
    writer.bytes(&INFO_MAGIC);
    writer.u32(VERSION);
    writer.string(&info.cache_id)?;
    writer.usize(info.token_count)?;
    writer.u64(info.resident_bytes);
    writer.u64(info.file_bytes);
    writer.u64(info.modified_unix);
    Ok(writer.into_inner())
}

fn decode_info(bytes: &[u8]) -> Result<DeepSeekV4SwapInfo, String> {
    let mut reader = SnapshotReader::new(bytes);
    require_header(&mut reader, INFO_MAGIC, "DeepSeek cache info")?;
    let info = DeepSeekV4SwapInfo {
        cache_id: reader.string("DeepSeek info cache_id")?,
        token_count: reader.usize("DeepSeek info token_count")?,
        resident_bytes: reader.u64("DeepSeek info resident_bytes")?,
        file_bytes: reader.u64("DeepSeek info file_bytes")?,
        modified_unix: reader.u64("DeepSeek info modified_unix")?,
    };
    reader.finish()?;
    Ok(info)
}

fn encode_manifest(manifest: &Manifest) -> Result<Vec<u8>, String> {
    let mut writer = SnapshotWriter::new();
    writer.bytes(&MANIFEST_MAGIC);
    writer.u32(VERSION);
    writer.u64(manifest.generation);
    writer.string(&manifest.cache_id)?;
    writer.u32s(&manifest.tokens)?;
    writer.optional_string(manifest.cache_namespace.as_deref())?;
    writer.usize(manifest.round)?;
    writer.flag(manifest.head);
    writer.u64(manifest.resident_bytes);
    writer.u64(manifest.modified_unix);
    write_blob(&mut writer, manifest.blob);
    Ok(writer.into_inner())
}

fn decode_manifest(bytes: &[u8]) -> Result<Manifest, String> {
    let mut reader = SnapshotReader::new(bytes);
    require_header(&mut reader, MANIFEST_MAGIC, "DeepSeek cache manifest")?;
    let manifest = Manifest {
        generation: reader.u64("DeepSeek generation")?,
        cache_id: reader.string("DeepSeek cache_id")?,
        tokens: reader.u32s("DeepSeek tokens")?,
        cache_namespace: reader.optional_string("DeepSeek namespace")?,
        round: reader.usize("DeepSeek round")?,
        head: reader.flag("DeepSeek head")?,
        resident_bytes: reader.u64("DeepSeek resident_bytes")?,
        modified_unix: reader.u64("DeepSeek modified_unix")?,
        blob: read_blob(&mut reader, "DeepSeek session blob")?,
    };
    reader.finish()?;
    Ok(manifest)
}

fn require_header(reader: &mut SnapshotReader<'_>, magic: [u8; 8], what: &str) -> Result<(), String> {
    if reader.take(8, what)? != magic {
        return Err(format!("{what} magic 不匹配"));
    }
    let version = reader.u32(what)?;
    if version != VERSION {
        return Err(format!("{what} version={version}，当前支持 {VERSION}"));
    }
    Ok(())
}

fn write_blob(writer: &mut SnapshotWriter, blob: FjallBlob) {
    writer.u32(blob.chunks);
    writer.u64(blob.bytes);
}

fn read_blob(reader: &mut SnapshotReader<'_>, what: &str) -> Result<FjallBlob, String> {
    Ok(FjallBlob { chunks: reader.u32(what)?, bytes: reader.u64(what)? })
}

fn new_generation() -> u64 {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
    now ^ GENERATION.fetch_add(1, Ordering::Relaxed)
}

fn write_session(output: &mut impl Write, stages: &[DeepSeekV4StageCache], dspark: Option<&RocmDeepSeekV4DsparkCache>) -> Result<(), String> {
    put_usize(output, stages.len())?;
    for stage in stages {
        put_usize(output, stage.layer_start)?;
        put_usize(output, stage.layers.len())?;
        for layer in &stage.layers {
            write_layer(output, layer)?;
        }
    }
    put_bool(output, dspark.is_some())?;
    if let Some(dspark) = dspark {
        put_usize(output, dspark.stages.len())?;
        for (target, block) in &dspark.stages {
            write_layer(output, target)?;
            write_layer(output, block)?;
        }
    }
    Ok(())
}

fn read_session(mut input: impl Read) -> Result<(Vec<DeepSeekV4StageCache>, Option<RocmDeepSeekV4DsparkCache>), String> {
    let stage_count = get_usize(&mut input)?;
    let mut stages = Vec::with_capacity(stage_count);
    for _ in 0..stage_count {
        let layer_start = get_usize(&mut input)?;
        let layer_count = get_usize(&mut input)?;
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            layers.push(read_layer(&mut input)?);
        }
        stages.push(DeepSeekV4StageCache { layer_start, layers });
    }
    let dspark = if get_bool(&mut input)? {
        let stage_count = get_usize(&mut input)?;
        let mut stages = Vec::with_capacity(stage_count);
        for _ in 0..stage_count {
            stages.push((read_layer(&mut input)?, read_layer(&mut input)?));
        }
        Some(RocmDeepSeekV4DsparkCache { stages })
    } else {
        None
    };
    Ok((stages, dspark))
}

fn write_layer(output: &mut impl Write, layer: &RocmCompressedKvSerde) -> Result<(), String> {
    for value in [layer.window_size, layer.kv_width, layer.q8_group_size, layer.recent_capacity, layer.recent_start, layer.recent_len, layer.recent_first_position] {
        put_usize(output, value)?;
    }
    put_option_usize(output, layer.next_recent_position)?;
    put_bool(output, layer.recent_shared)?;
    for bytes in [&layer.recent_key, &layer.recent_key_scales, &layer.recent_value, &layer.recent_value_scales] {
        put_bytes(output, bytes)?;
    }
    put_usizes(output, &layer.compressed_positions)?;
    put_bool(output, layer.compressed_shared)?;
    for bytes in [&layer.compressed_key, &layer.compressed_key_scales, &layer.compressed_value, &layer.compressed_value_scales] {
        put_bytes(output, bytes)?;
    }
    put_usize(output, layer.compressed_index_width)?;
    put_option_bytes(output, layer.compressed_index_key.as_deref())?;
    write_pool(output, &layer.compressor)?;
    write_pool(output, &layer.indexer)
}

fn read_layer(input: &mut impl Read) -> Result<RocmCompressedKvSerde, String> {
    let window_size = get_usize(input)?;
    let kv_width = get_usize(input)?;
    let q8_group_size = get_usize(input)?;
    let recent_capacity = get_usize(input)?;
    let recent_start = get_usize(input)?;
    let recent_len = get_usize(input)?;
    let recent_first_position = get_usize(input)?;
    let next_recent_position = get_option_usize(input)?;
    let recent_shared = get_bool(input)?;
    let recent_key = get_bytes(input)?;
    let recent_key_scales = get_bytes(input)?;
    let recent_value = get_bytes(input)?;
    let recent_value_scales = get_bytes(input)?;
    let compressed_positions = get_usizes(input)?;
    let compressed_shared = get_bool(input)?;
    let compressed_key = get_bytes(input)?;
    let compressed_key_scales = get_bytes(input)?;
    let compressed_value = get_bytes(input)?;
    let compressed_value_scales = get_bytes(input)?;
    let compressed_index_width = get_usize(input)?;
    let compressed_index_key = get_option_bytes(input)?;
    Ok(RocmCompressedKvSerde {
        window_size,
        kv_width,
        q8_group_size,
        recent_capacity,
        recent_start,
        recent_len,
        recent_first_position,
        next_recent_position,
        recent_shared,
        recent_key,
        recent_key_scales,
        recent_value,
        recent_value_scales,
        compressed_positions,
        compressed_shared,
        compressed_key,
        compressed_key_scales,
        compressed_value,
        compressed_value_scales,
        compressed_index_width,
        compressed_index_key,
        compressor: read_pool(input)?,
        indexer: read_pool(input)?,
    })
}

fn write_pool(output: &mut impl Write, pool: &RocmGatedPoolSerde) -> Result<(), String> {
    for value in [pool.next_position, pool.entry_count, pool.pending_rows, pool.ratio, pool.width, pool.channels] {
        put_usize(output, value)?;
    }
    put_bool(output, pool.overlap)?;
    for bytes in [&pool.pending_key, &pool.pending_gate, &pool.overlap_key, &pool.overlap_gate] {
        put_option_bytes(output, bytes.as_deref())?;
    }
    Ok(())
}

fn read_pool(input: &mut impl Read) -> Result<RocmGatedPoolSerde, String> {
    Ok(RocmGatedPoolSerde {
        next_position: get_usize(input)?,
        entry_count: get_usize(input)?,
        pending_rows: get_usize(input)?,
        ratio: get_usize(input)?,
        width: get_usize(input)?,
        channels: get_usize(input)?,
        overlap: get_bool(input)?,
        pending_key: get_option_bytes(input)?,
        pending_gate: get_option_bytes(input)?,
        overlap_key: get_option_bytes(input)?,
        overlap_gate: get_option_bytes(input)?,
    })
}

fn put_usize(output: &mut impl Write, value: usize) -> Result<(), String> {
    output.write_all(&(value as u64).to_le_bytes()).map_err(|error| error.to_string())
}
fn get_usize(input: &mut impl Read) -> Result<usize, String> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes).map_err(|error| error.to_string())?;
    usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| "DeepSeek cache usize 溢出".to_owned())
}
fn put_bool(output: &mut impl Write, value: bool) -> Result<(), String> {
    output.write_all(&[u8::from(value)]).map_err(|error| error.to_string())
}
fn get_bool(input: &mut impl Read) -> Result<bool, String> {
    let mut byte = [0];
    input.read_exact(&mut byte).map_err(|error| error.to_string())?;
    match byte[0] {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(format!("DeepSeek cache bool={value} 非法")),
    }
}
fn put_option_usize(output: &mut impl Write, value: Option<usize>) -> Result<(), String> {
    put_bool(output, value.is_some())?;
    if let Some(value) = value {
        put_usize(output, value)?;
    }
    Ok(())
}
fn get_option_usize(input: &mut impl Read) -> Result<Option<usize>, String> {
    if get_bool(input)? { get_usize(input).map(Some) } else { Ok(None) }
}
fn put_bytes(output: &mut impl Write, bytes: &[u8]) -> Result<(), String> {
    put_usize(output, bytes.len())?;
    output.write_all(bytes).map_err(|error| error.to_string())
}
fn get_bytes(input: &mut impl Read) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0; get_usize(input)?];
    input.read_exact(&mut bytes).map_err(|error| error.to_string())?;
    Ok(bytes)
}
fn put_option_bytes(output: &mut impl Write, bytes: Option<&[u8]>) -> Result<(), String> {
    put_bool(output, bytes.is_some())?;
    if let Some(bytes) = bytes {
        put_bytes(output, bytes)?;
    }
    Ok(())
}
fn get_option_bytes(input: &mut impl Read) -> Result<Option<Vec<u8>>, String> {
    if get_bool(input)? { get_bytes(input).map(Some) } else { Ok(None) }
}
fn put_usizes(output: &mut impl Write, values: &[usize]) -> Result<(), String> {
    put_usize(output, values.len())?;
    for &value in values {
        put_usize(output, value)?;
    }
    Ok(())
}
fn get_usizes(input: &mut impl Read) -> Result<Vec<usize>, String> {
    let count = get_usize(input)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(get_usize(input)?);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> RocmGatedPoolSerde {
        RocmGatedPoolSerde { next_position: 12, entry_count: 3, pending_rows: 1, ratio: 4, width: 2, channels: 4, overlap: true, pending_key: Some(vec![1, 2]), pending_gate: Some(vec![3]), overlap_key: None, overlap_gate: Some(vec![4, 5]) }
    }

    fn layer() -> RocmCompressedKvSerde {
        RocmCompressedKvSerde {
            window_size: 2,
            kv_width: 2,
            q8_group_size: 1,
            recent_capacity: 2,
            recent_start: 1,
            recent_len: 2,
            recent_first_position: 8,
            next_recent_position: Some(10),
            recent_shared: false,
            recent_key: vec![1; 4],
            recent_key_scales: vec![2; 8],
            recent_value: vec![3; 4],
            recent_value_scales: vec![4; 8],
            compressed_positions: vec![3, 7],
            compressed_shared: true,
            compressed_key: vec![5; 4],
            compressed_key_scales: vec![6; 8],
            compressed_value: Vec::new(),
            compressed_value_scales: Vec::new(),
            compressed_index_width: 1,
            compressed_index_key: Some(vec![7; 8]),
            compressor: pool(),
            indexer: pool(),
        }
    }

    fn snapshot(cache_id: &str, tokens: Vec<u32>, namespace: &str, round: usize, head: bool) -> DeepSeekV4CacheSnapshot {
        DeepSeekV4CacheSnapshot {
            cache_id: cache_id.to_owned(),
            tokens,
            cache_namespace: Some(namespace.to_owned()),
            round,
            head,
            stages: vec![DeepSeekV4StageCache { layer_start: 7, layers: vec![layer()] }],
            dspark: Some(RocmDeepSeekV4DsparkCache { stages: vec![(layer(), layer())] }),
            resident_bytes: 4096,
            modified_unix: 123,
        }
    }

    #[test]
    fn session_binary_roundtrip_preserves_mp_and_dspark_layers() {
        let stages = vec![DeepSeekV4StageCache { layer_start: 7, layers: vec![layer()] }];
        let dspark = RocmDeepSeekV4DsparkCache { stages: vec![(layer(), layer())] };
        let mut bytes = Vec::new();
        write_session(&mut bytes, &stages, Some(&dspark)).unwrap();
        let (restored, restored_dspark) = read_session(bytes.as_slice()).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].layer_start, 7);
        assert_eq!(restored[0].layers[0].compressed_positions, vec![3, 7]);
        assert_eq!(restored[0].layers[0].compressor.next_position, 12);
        assert_eq!(restored_dspark.unwrap().stages.len(), 1);
    }

    #[test]
    fn info和manifest使用版本化二进制() {
        let info = DeepSeekV4SwapInfo { cache_id: "cache".into(), token_count: 3, resident_bytes: 4, file_bytes: 5, modified_unix: 6 };
        let info_bytes = encode_info(&info).unwrap();
        assert_eq!(&info_bytes[..8], &INFO_MAGIC);
        assert_eq!(decode_info(&info_bytes).unwrap().cache_id, "cache");

        let manifest =
            Manifest { generation: 7, cache_id: "cache".into(), tokens: vec![1, 2, 3], cache_namespace: Some("namespace".into()), round: 2, head: false, resident_bytes: 4, modified_unix: 6, blob: FjallBlob { chunks: 1, bytes: 10 } };
        let manifest_bytes = encode_manifest(&manifest).unwrap();
        assert_eq!(&manifest_bytes[..8], &MANIFEST_MAGIC);
        assert_eq!(decode_manifest(&manifest_bytes).unwrap().tokens, vec![1, 2, 3]);
    }

    #[test]
    fn store_reopens_and_restores_longest_global_prefix() {
        let directory = std::env::temp_dir().join(format!("zllm-deepseek-v4-swap-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let metadata = vec![("architecture".to_owned(), "deepseek-v4-flash".to_owned())];
        {
            let store = DeepSeekV4SwapStore::open(&directory, &metadata).unwrap();
            store.put(&snapshot("a", vec![1, 2], "org-a", 2, false)).unwrap();
            store.put(&snapshot("b", vec![1, 2, 3], "org-a", 3, false)).unwrap();
            store.put(&snapshot("private", vec![1, 2, 3, 4], "org-b", 4, false)).unwrap();
            store.put(&snapshot("head", vec![1, 2, 3, 4], "org-a", 4, true)).unwrap();
        }

        let store = DeepSeekV4SwapStore::open(&directory, &metadata).unwrap();
        let restored = store.get("b").unwrap().unwrap();
        assert_eq!(restored.tokens, vec![1, 2, 3]);
        assert_eq!(restored.round, 3);
        assert_eq!(restored.stages[0].layers[0].compressed_positions, vec![3, 7]);
        assert_eq!(restored.dspark.unwrap().stages.len(), 1);
        assert_eq!(store.longest_prefix_cache_id(&[1, 2, 3, 4, 9]).unwrap().as_deref(), Some("private"));
        assert_eq!(store.infos().len(), 4);

        store.set_head("b", true).unwrap();
        assert_eq!(store.longest_prefix_cache_id(&[1, 2, 3, 9]).unwrap().as_deref(), Some("a"));
        store.delete("a").unwrap();
        assert!(store.get("a").unwrap().is_none());
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
