//! KV cache 的 fjall 分块持久化。
//!
//! 小型索引和 manifest 原子切换 generation，大型量化数据按层写入 value-log。

use std::{
    io::{self, Cursor, Read, Write},
    path::Path,
};

use fjall::{Database, Keyspace, KeyspaceCreateOptions, KvSeparationOptions, PersistMode};
use serde::{Deserialize, Serialize};

const CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FjallBlob {
    pub chunks: u32,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FjallChunkKind {
    Kv,
    Dsa,
    Aux,
}

impl FjallChunkKind {
    const fn code(self) -> u8 {
        match self {
            Self::Kv => 0,
            Self::Dsa => 1,
            Self::Aux => 2,
        }
    }
}

pub struct FjallCacheStore {
    database: Database,
    metadata: Keyspace,
    infos: Keyspace,
    manifests: Keyspace,
    chunks: Keyspace,
}

impl FjallCacheStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let database = Database::builder(path).open().map_err(|error| format!("打开 fjall cache {}: {error}", path.display()))?;
        let metadata = database.keyspace("cache_metadata", KeyspaceCreateOptions::default).map_err(fjall_error)?;
        let infos = database.keyspace("cache_info", KeyspaceCreateOptions::default).map_err(fjall_error)?;
        let manifests = database.keyspace("cache_manifest", KeyspaceCreateOptions::default).map_err(fjall_error)?;
        let chunks = database.keyspace("cache_chunks", || KeyspaceCreateOptions::default().with_kv_separation(Some(KvSeparationOptions::default()))).map_err(fjall_error)?;
        Ok(Self { database, metadata, infos, manifests, chunks })
    }

    /// 首次使用时把缓存身份写入 fjall；后续打开必须逐项完全匹配。
    pub fn bind_metadata(&self, expected: &[(String, String)]) -> Result<(), String> {
        if expected.is_empty() {
            return Err("fjall cache metadata 不能为空".to_owned());
        }
        let metadata_exists = keyspace_has_entries(&self.metadata)?;
        if !metadata_exists {
            if keyspace_has_entries(&self.infos)? || keyspace_has_entries(&self.manifests)? || keyspace_has_entries(&self.chunks)? {
                return Err("fjall cache 已有数据但缺少 metadata，拒绝自动绑定；请改用空缓存目录".to_owned());
            }
            let mut batch = self.database.batch();
            for (key, value) in expected {
                batch.insert(&self.metadata, key.as_bytes(), value.as_bytes());
            }
            batch.commit().map_err(fjall_error)?;
            return self.database.persist(PersistMode::SyncAll).map_err(fjall_error);
        }
        for (key, value) in expected {
            let actual = self.metadata.get(key.as_bytes()).map_err(fjall_error)?;
            let Some(actual) = actual else {
                return Err(format!("fjall cache metadata 缺少 {key}，拒绝复用缓存"));
            };
            if actual.as_ref() != value.as_bytes() {
                return Err(format!("fjall cache metadata 不匹配: {key}，缓存={}，当前={value}，拒绝复用缓存", String::from_utf8_lossy(actual.as_ref())));
            }
        }
        Ok(())
    }

    pub fn info_values(&self) -> Result<Vec<Vec<u8>>, String> {
        let mut values = Vec::new();
        for entry in self.infos.iter() {
            values.push(entry.value().map_err(fjall_error)?.to_vec());
        }
        Ok(values)
    }

    pub fn manifest_values(&self) -> Result<Vec<Vec<u8>>, String> {
        let mut values = Vec::new();
        for entry in self.manifests.iter() {
            values.push(entry.value().map_err(fjall_error)?.to_vec());
        }
        Ok(values)
    }

    pub fn manifest(&self, cache_id: &str) -> Result<Option<Vec<u8>>, String> {
        self.manifests.get(cache_id.as_bytes()).map_err(fjall_error).map(|value| value.map(|value| value.to_vec()))
    }

    pub fn info(&self, cache_id: &str) -> Result<Option<Vec<u8>>, String> {
        self.infos.get(cache_id.as_bytes()).map_err(fjall_error).map(|value| value.map(|value| value.to_vec()))
    }

    pub fn contains(&self, cache_id: &str) -> Result<bool, String> {
        self.manifests.contains_key(cache_id.as_bytes()).map_err(fjall_error)
    }

    pub fn commit(&self, cache_id: &str, info: &[u8], manifest: &[u8]) -> Result<(), String> {
        // 先确保 value-log 中的大块落盘，再原子切换可见 generation。
        self.database.persist(PersistMode::SyncData).map_err(fjall_error)?;
        let mut batch = self.database.batch();
        batch.insert(&self.infos, cache_id.as_bytes(), info);
        batch.insert(&self.manifests, cache_id.as_bytes(), manifest);
        batch.commit().map_err(fjall_error)?;
        self.database.persist(PersistMode::SyncAll).map_err(fjall_error)
    }

    pub fn writer(&self, cache_id: &str, generation: u64, stage: usize, layer: usize, kind: FjallChunkKind) -> FjallChunkWriter<'_> {
        FjallChunkWriter { store: self, cache_id: cache_id.to_owned(), generation, stage, layer, kind, buffer: Vec::with_capacity(CHUNK_BYTES), chunks: 0, bytes: 0 }
    }

    pub fn reader(&self, cache_id: &str, generation: u64, stage: usize, layer: usize, kind: FjallChunkKind, blob: FjallBlob) -> FjallChunkReader<'_> {
        FjallChunkReader { store: self, cache_id: cache_id.to_owned(), generation, stage, layer, kind, blob, next_chunk: 0, remaining: blob.bytes, current: Cursor::new(Vec::new()) }
    }

    pub fn remove_entry(&self, cache_id: &str) -> Result<(), String> {
        let mut batch = self.database.batch();
        batch.remove(&self.infos, cache_id.as_bytes());
        batch.remove(&self.manifests, cache_id.as_bytes());
        batch.commit().map_err(fjall_error)?;
        self.database.persist(PersistMode::SyncAll).map_err(fjall_error)
    }

    pub fn remove_generation(&self, cache_id: &str, generation: u64) -> Result<(), String> {
        let prefix = chunk_prefix(cache_id, generation)?;
        let mut keys = Vec::new();
        for entry in self.chunks.prefix(prefix) {
            keys.push(entry.key().map_err(fjall_error)?.to_vec());
        }
        if keys.is_empty() {
            return Ok(());
        }
        let mut batch = self.database.batch();
        for key in keys {
            batch.remove(&self.chunks, key);
        }
        batch.commit().map_err(fjall_error)?;
        self.database.persist(PersistMode::SyncAll).map_err(fjall_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn put_chunk(&self, cache_id: &str, generation: u64, stage: usize, layer: usize, kind: FjallChunkKind, chunk: u32, bytes: &[u8]) -> Result<(), String> {
        let key = chunk_key(cache_id, generation, stage, layer, kind, chunk)?;
        self.chunks.insert(key, bytes).map_err(fjall_error)
    }

    fn get_chunk(&self, cache_id: &str, generation: u64, stage: usize, layer: usize, kind: FjallChunkKind, chunk: u32) -> Result<Option<Vec<u8>>, String> {
        let key = chunk_key(cache_id, generation, stage, layer, kind, chunk)?;
        self.chunks.get(key).map_err(fjall_error).map(|value| value.map(|value| value.to_vec()))
    }
}

fn keyspace_has_entries(keyspace: &Keyspace) -> Result<bool, String> {
    let Some(entry) = keyspace.iter().next() else { return Ok(false) };
    entry.key().map_err(fjall_error)?;
    Ok(true)
}

/// 小型持久化状态表。大体积 KV 数据使用 FjallCacheStore，任务 journal 等小值使用此表。
pub struct FjallValueStore {
    database: Database,
    entries: Keyspace,
}

impl FjallValueStore {
    const IDENTITY_KEY: &'static str = "@store_identity";

    pub fn open(path: impl AsRef<Path>, keyspace: &str) -> Result<Self, String> {
        let path = path.as_ref();
        let database = Database::builder(path).open().map_err(|error| format!("打开 fjall value store {}: {error}", path.display()))?;
        let entries = database.keyspace(keyspace, KeyspaceCreateOptions::default).map_err(fjall_error)?;
        Ok(Self { database, entries })
    }

    pub fn put(&self, key: &str, value: &[u8]) -> Result<(), String> {
        self.entries.insert(key.as_bytes(), value).map_err(fjall_error)?;
        self.database.persist(PersistMode::SyncAll).map_err(fjall_error)
    }

    /// 把 value store 绑定到具体模型/布局。已有数据但没有 identity 的旧 store
    /// 不可安全判断兼容性，必须拒绝，而不是尝试恢复形状碰巧相同的 KV。
    pub fn bind_identity(&self, expected: &str) -> Result<(), String> {
        match self.get(Self::IDENTITY_KEY)? {
            Some(actual) if actual == expected.as_bytes() => Ok(()),
            Some(actual) => Err(format!("fjall value store identity 不匹配: cache={} current={expected}", String::from_utf8_lossy(&actual))),
            None if self.entries.iter().next().is_some() => Err("fjall value store 已有数据但缺少 identity；请清空或迁移旧 cache_directory".to_owned()),
            None => self.put(Self::IDENTITY_KEY, expected.as_bytes()),
        }
    }

    pub fn put_pair(&self, first_key: &str, first_value: &[u8], second_key: &str, second_value: &[u8]) -> Result<(), String> {
        let mut batch = self.database.batch();
        batch.insert(&self.entries, first_key.as_bytes(), first_value);
        batch.insert(&self.entries, second_key.as_bytes(), second_value);
        batch.commit().map_err(fjall_error)?;
        self.database.persist(PersistMode::SyncAll).map_err(fjall_error)
    }

    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        self.entries.get(key.as_bytes()).map_err(fjall_error).map(|value| value.map(|value| value.to_vec()))
    }

    pub fn remove(&self, key: &str) -> Result<(), String> {
        self.entries.remove(key.as_bytes()).map_err(fjall_error)?;
        self.database.persist(PersistMode::SyncAll).map_err(fjall_error)
    }

    pub fn remove_pair(&self, first_key: &str, second_key: &str) -> Result<(), String> {
        let mut batch = self.database.batch();
        batch.remove(&self.entries, first_key.as_bytes());
        batch.remove(&self.entries, second_key.as_bytes());
        batch.commit().map_err(fjall_error)?;
        self.database.persist(PersistMode::SyncAll).map_err(fjall_error)
    }

    pub fn values(&self) -> Result<Vec<Vec<u8>>, String> {
        let mut values = Vec::new();
        for entry in self.entries.iter() {
            values.push(entry.value().map_err(fjall_error)?.to_vec());
        }
        Ok(values)
    }

    /// 列出全部 key;供节点启动时扫 swap 上持久化的 cache_id / metadata。
    /// 顺序无定义,只用作启动时视图重建。
    pub fn keys(&self) -> Result<Vec<String>, String> {
        let mut keys = Vec::new();
        for entry in self.entries.iter() {
            let key = entry.key().map_err(fjall_error)?;
            keys.push(String::from_utf8(key.to_vec()).map_err(|error| format!("fjall key 不是 UTF-8: {error}"))?);
        }
        Ok(keys)
    }
}

pub struct FjallChunkWriter<'a> {
    store: &'a FjallCacheStore,
    cache_id: String,
    generation: u64,
    stage: usize,
    layer: usize,
    kind: FjallChunkKind,
    buffer: Vec<u8>,
    chunks: u32,
    bytes: u64,
}

impl FjallChunkWriter<'_> {
    pub fn finish(mut self) -> Result<FjallBlob, String> {
        self.flush_chunk()?;
        Ok(FjallBlob { chunks: self.chunks, bytes: self.bytes })
    }

    fn flush_chunk(&mut self) -> Result<(), String> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.store.put_chunk(&self.cache_id, self.generation, self.stage, self.layer, self.kind, self.chunks, &self.buffer)?;
        self.chunks = self.chunks.checked_add(1).ok_or("fjall cache 分块数量溢出")?;
        self.buffer.clear();
        Ok(())
    }
}

impl Write for FjallChunkWriter<'_> {
    fn write(&mut self, mut input: &[u8]) -> io::Result<usize> {
        let input_len = input.len();
        while !input.is_empty() {
            let available = CHUNK_BYTES - self.buffer.len();
            let count = available.min(input.len());
            self.buffer.extend_from_slice(&input[..count]);
            self.bytes = self.bytes.checked_add(count as u64).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "fjall cache 字节数溢出"))?;
            input = &input[count..];
            if self.buffer.len() == CHUNK_BYTES {
                self.flush_chunk().map_err(fjall_io_error)?;
            }
        }
        Ok(input_len)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_chunk().map_err(fjall_io_error)
    }
}

pub struct FjallChunkReader<'a> {
    store: &'a FjallCacheStore,
    cache_id: String,
    generation: u64,
    stage: usize,
    layer: usize,
    kind: FjallChunkKind,
    blob: FjallBlob,
    next_chunk: u32,
    remaining: u64,
    current: Cursor<Vec<u8>>,
}

impl Read for FjallChunkReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() || self.remaining == 0 {
            return Ok(0);
        }
        while self.current.position() == self.current.get_ref().len() as u64 {
            if self.next_chunk >= self.blob.chunks {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("fjall cache 剩余 {} bytes，但没有更多分块", self.remaining)));
            }
            let chunk = self
                .store
                .get_chunk(&self.cache_id, self.generation, self.stage, self.layer, self.kind, self.next_chunk)
                .map_err(fjall_io_error)?
                .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, format!("fjall cache 缺少 chunk {}", self.next_chunk)))?;
            self.next_chunk += 1;
            self.current = Cursor::new(chunk);
        }
        let limit = output.len().min(usize::try_from(self.remaining).unwrap_or(usize::MAX));
        let count = self.current.read(&mut output[..limit])?;
        self.remaining -= count as u64;
        Ok(count)
    }
}

fn chunk_prefix(cache_id: &str, generation: u64) -> Result<Vec<u8>, String> {
    let id_len = u32::try_from(cache_id.len()).map_err(|_| "fjall cache_id 超过 u32".to_owned())?;
    let mut key = Vec::with_capacity(4 + cache_id.len() + 8);
    key.extend_from_slice(&id_len.to_be_bytes());
    key.extend_from_slice(cache_id.as_bytes());
    key.extend_from_slice(&generation.to_be_bytes());
    Ok(key)
}

fn chunk_key(cache_id: &str, generation: u64, stage: usize, layer: usize, kind: FjallChunkKind, chunk: u32) -> Result<Vec<u8>, String> {
    let stage = u32::try_from(stage).map_err(|_| "fjall cache stage 超过 u32".to_owned())?;
    let layer = u32::try_from(layer).map_err(|_| "fjall cache layer 超过 u32".to_owned())?;
    let mut key = chunk_prefix(cache_id, generation)?;
    key.extend_from_slice(&stage.to_be_bytes());
    key.extend_from_slice(&layer.to_be_bytes());
    key.push(kind.code());
    key.extend_from_slice(&chunk.to_be_bytes());
    Ok(key)
}

fn fjall_error(error: impl std::fmt::Display) -> String {
    format!("fjall cache: {error}")
}

fn fjall_io_error(error: String) -> io::Error {
    io::Error::other(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(format: &str) -> Vec<(String, String)> {
        vec![("schema_version".to_owned(), "1".to_owned()), ("quantization".to_owned(), format.to_owned())]
    }

    #[test]
    fn metadata首次绑定并校验重开() {
        let root = std::env::temp_dir().join(format!("zllm-fjall-metadata-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let store = FjallCacheStore::open(&root).unwrap();
        store.bind_metadata(&metadata("gguf:q4_k")).unwrap();
        drop(store);

        let store = FjallCacheStore::open(&root).unwrap();
        store.bind_metadata(&metadata("gguf:q4_k")).unwrap();
        let error = store.bind_metadata(&metadata("compressed-tensors:w4+w8")).unwrap_err();
        assert!(error.contains("quantization"), "{error}");
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy数据不能自动绑定metadata() {
        let root = std::env::temp_dir().join(format!("zllm-fjall-legacy-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let store = FjallCacheStore::open(&root).unwrap();
        store.commit("session-a", b"info", b"manifest").unwrap();
        let error = store.bind_metadata(&metadata("gguf:q4_k")).unwrap_err();
        assert!(error.contains("已有数据但缺少 metadata"), "{error}");
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn value_store_identity首次绑定且拒绝错模型() {
        let root = std::env::temp_dir().join(format!("zllm-fjall-value-identity-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let store = FjallValueStore::open(&root, "terminal").unwrap();
        store.bind_identity("gemma4:model-a:layout-1").unwrap();
        store.bind_identity("gemma4:model-a:layout-1").unwrap();
        let error = store.bind_identity("qwen36:model-b:layout-1").unwrap_err();
        assert!(error.contains("identity 不匹配"), "{error}");
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn value_store旧数据不能自动绑定identity() {
        let root = std::env::temp_dir().join(format!("zllm-fjall-value-legacy-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let store = FjallValueStore::open(&root, "terminal").unwrap();
        store.put("session-a", b"snapshot").unwrap();
        let error = store.bind_identity("gemma4:model-a:layout-1").unwrap_err();
        assert!(error.contains("已有数据但缺少 identity"), "{error}");
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn streams_and_removes_generation() {
        let root = std::env::temp_dir().join(format!("zllm-fjall-cache-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let store = FjallCacheStore::open(&root).unwrap();
        let payload = (0..CHUNK_BYTES + 17).map(|index| index as u8).collect::<Vec<_>>();
        let mut writer = store.writer("session-a", 7, 0, 40, FjallChunkKind::Kv);
        writer.write_all(&payload).unwrap();
        let blob = writer.finish().unwrap();
        assert_eq!(blob, FjallBlob { chunks: 2, bytes: payload.len() as u64 });
        store.commit("session-a", b"info", b"manifest").unwrap();
        drop(store);

        let store = FjallCacheStore::open(&root).unwrap();
        assert_eq!(store.info_values().unwrap(), vec![b"info".to_vec()]);
        assert_eq!(store.manifest("session-a").unwrap(), Some(b"manifest".to_vec()));
        let mut reader = store.reader("session-a", 7, 0, 40, FjallChunkKind::Kv, blob);
        let mut restored = Vec::new();
        reader.read_to_end(&mut restored).unwrap();
        assert_eq!(restored, payload);
        store.remove_entry("session-a").unwrap();
        store.remove_generation("session-a", 7).unwrap();
        assert!(!store.contains("session-a").unwrap());
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn uncommitted_generation_does_not_replace_manifest() {
        let root = std::env::temp_dir().join(format!("zllm-fjall-generation-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let store = FjallCacheStore::open(&root).unwrap();
        let mut committed = store.writer("session-a", 7, 0, 40, FjallChunkKind::Kv);
        committed.write_all(b"committed-kv").unwrap();
        let committed_blob = committed.finish().unwrap();
        store.commit("session-a", b"info-7", b"manifest-7").unwrap();

        let mut interrupted = store.writer("session-a", 8, 0, 40, FjallChunkKind::Kv);
        interrupted.write_all(b"incomplete-kv").unwrap();
        interrupted.finish().unwrap();
        drop(store);

        let store = FjallCacheStore::open(&root).unwrap();
        assert_eq!(store.manifest("session-a").unwrap(), Some(b"manifest-7".to_vec()));
        let mut reader = store.reader("session-a", 7, 0, 40, FjallChunkKind::Kv, committed_blob);
        let mut restored = Vec::new();
        reader.read_to_end(&mut restored).unwrap();
        assert_eq!(restored, b"committed-kv");
        store.remove_entry("session-a").unwrap();
        store.remove_generation("session-a", 7).unwrap();
        store.remove_generation("session-a", 8).unwrap();
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }
}
