//! GLM-5.2 ROCm 组合执行状态的本机 SSD 存储。
//!
//! 持久化本 stage 拥有的 INT8 KV/DSA、最后一行 hidden、MTP 与 DSpark resident state；
//! resident 权重仍由运行时持有。

#![cfg(target_os = "linux")]

use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    config::KvCacheFormat,
    kv_cache::{
        FjallBlob, FjallCacheStore, FjallChunkKind,
        terminal_cache::{SnapshotReader, SnapshotWriter},
    },
    runtime::glm52::{Glm52Config, stage::Glm52StageState},
    weight::Glm52Weights,
};

use crate::backend::rocm::{DsaLayerSerde, MlaLayerSerde, RocmContext};

const VERSION: u32 = 8;
const INFO_MAGIC: [u8; 8] = *b"ZGLM5I01";
const MANIFEST_MAGIC: [u8; 8] = *b"ZGLM5M01";

/// 单个 ROCm stage 的可持久化会话态；resident 权重和 expert source 不进入 SSD。
pub struct Glm52StageCache {
    pub layer_start: usize,
    pub kv: Vec<Option<MlaLayerSerde>>,
    pub dsa: Vec<Option<DsaLayerSerde>>,
}

/// MTP L78 跨请求继续 draft 所需的稳定状态；本轮临时参数不落盘。
pub struct Glm52MtpCache {
    pub position: usize,
    pub pending_hidden: Vec<u16>,
    pub prompt_tokens: Vec<u32>,
    pub kv: Vec<Option<MlaLayerSerde>>,
    pub dsa: Vec<Option<DsaLayerSerde>>,
}

/// DSpark 恢复 draft 投影所需的最近一段归一化 target aux hidden。
pub struct Glm52DsparkAuxCache {
    pub start_position: usize,
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<u16>,
}

/// DSpark target K/V 必须保持运行时 F32，避免 SSD 恢复改变 draft acceptance。
pub struct Glm52DsparkTargetTensor {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
}

pub struct Glm52DsparkTargetLayerCache {
    pub start_position: usize,
    pub key: Glm52DsparkTargetTensor,
    pub value: Glm52DsparkTargetTensor,
}

pub struct Glm52DsparkTargetCache {
    pub layers: Vec<Glm52DsparkTargetLayerCache>,
}

pub fn download_glm52_session(states: &[Glm52StageState<RocmContext>]) -> Result<Vec<Glm52StageCache>, String> {
    states
        .iter()
        .map(|state| {
            state.backend.activate().map_err(|error| format!("激活 ROCm device {}: {error}", state.backend.device_id()))?;
            Ok(Glm52StageCache {
                layer_start: state.layer_start,
                kv: state.cache.download_layers().map_err(|error| format!("下载 L{} KV: {error:?}", state.layer_start))?,
                dsa: state.dsa.download_layers().map_err(|error| format!("下载 L{} DSA: {error:?}", state.layer_start))?,
            })
        })
        .collect()
}

pub fn upload_glm52_session(states: &mut [Glm52StageState<RocmContext>], snapshot: &[Glm52StageCache], cfg: &Glm52Config, max_seq_len: usize, reserved_rows: usize) -> Result<(), String> {
    if states.len() != snapshot.len() {
        return Err(format!("GLM stage cache 数量={}，当前 device stage={}", snapshot.len(), states.len()));
    }
    for (state, cached) in states.iter_mut().zip(snapshot) {
        if cached.layer_start != state.layer_start {
            return Err(format!("GLM stage cache layer_start={}，当前 stage={}", cached.layer_start, state.layer_start));
        }
        state.backend.activate().map_err(|error| format!("激活 ROCm device {}: {error}", state.backend.device_id()))?;
        state.reset_session(cfg, max_seq_len).map_err(|error| format!("重置 L{} session: {error:?}", state.layer_start))?;
        state.cache.upload_layers(&state.backend, &cached.kv, reserved_rows).map_err(|error| format!("恢复 L{} KV: {error:?}", state.layer_start))?;
        state.dsa.upload_layers(&state.backend, &cached.dsa, reserved_rows).map_err(|error| format!("恢复 L{} DSA: {error:?}", state.layer_start))?;
    }
    Ok(())
}

pub struct Glm52CacheSnapshot {
    pub cache_id: String,
    /// API key 的单向 namespace；只用于 prefix cache 隔离，不进入模型状态。
    pub cache_namespace: Option<String>,
    /// 链头保存完整 token；后继只保存数量，tokens 为空。
    pub tokens: Vec<u32>,
    pub token_count: usize,
    /// 已采样但尚未进入完整模型 KV 的尾 token。None 表示旧快照没有该元数据。
    pub pending_tokens: Option<Vec<u32>>,
    pub last_hidden: Vec<u16>,
    pub stages: Vec<Glm52StageCache>,
    pub mtp: Option<Glm52MtpCache>,
    pub dspark_aux: Option<Glm52DsparkAuxCache>,
    pub dspark_target: Option<Glm52DsparkTargetCache>,
}

impl Glm52CacheSnapshot {
    pub fn resident_bytes(&self) -> u64 {
        let hidden = self.last_hidden.len().saturating_mul(2) as u64;
        let stages = self.stages.iter().fold(hidden, |total, stage| {
            let kv = stage
                .kv
                .iter()
                .flatten()
                .fold(0_u64, |bytes, layer| bytes.saturating_add(layer.latent.len() as u64).saturating_add(layer.latent_scales.as_ref().map_or(0, |scales| scales.len()) as u64).saturating_add(layer.rope.len() as u64));
            let dsa = stage.dsa.iter().flatten().fold(0_u64, |bytes, layer| bytes.saturating_add(layer.keys.len() as u64).saturating_add(layer.scales.len() as u64));
            total.saturating_add(kv).saturating_add(dsa)
        });
        let stages = self.mtp.as_ref().map_or(stages, |mtp| {
            let kv = mtp
                .kv
                .iter()
                .flatten()
                .fold(0_u64, |bytes, layer| bytes.saturating_add(layer.latent.len() as u64).saturating_add(layer.latent_scales.as_ref().map_or(0, |scales| scales.len()) as u64).saturating_add(layer.rope.len() as u64));
            let dsa = mtp.dsa.iter().flatten().fold(0_u64, |bytes, layer| bytes.saturating_add(layer.keys.len() as u64).saturating_add(layer.scales.len() as u64));
            stages.saturating_add(mtp.pending_hidden.len().saturating_mul(2) as u64).saturating_add(kv).saturating_add(dsa)
        });
        let stages = self.dspark_aux.as_ref().map_or(stages, |aux| stages.saturating_add(aux.values.len().saturating_mul(2) as u64));
        self.dspark_target.as_ref().map_or(stages, |target| {
            target
                .layers
                .iter()
                .fold(stages, |bytes, layer| bytes.saturating_add(layer.key.values.len().saturating_mul(std::mem::size_of::<f32>()) as u64).saturating_add(layer.value.values.len().saturating_mul(std::mem::size_of::<f32>()) as u64))
        })
    }
}

#[derive(Clone, Debug)]
pub struct Glm52SwapInfo {
    pub cache_id: String,
    pub token_count: usize,
    pub resident_bytes: u64,
    pub file_bytes: u64,
    pub modified_unix: u64,
}

struct Glm52Manifest {
    generation: u64,
    cache_id: String,
    cache_namespace: Option<String>,
    tokens: Vec<u32>,
    token_count: usize,
    pending_tokens: Option<Vec<u32>>,
    last_hidden: Vec<u16>,
    stages: Vec<Glm52StageManifest>,
    mtp: Option<Glm52MtpManifest>,
    dspark_aux: Option<Glm52DsparkAuxManifest>,
    dspark_target: Option<Glm52DsparkTargetManifest>,
}

struct Glm52StageManifest {
    layer_start: usize,
    kv: Vec<Option<FjallBlob>>,
    dsa: Vec<Option<FjallBlob>>,
}

struct Glm52MtpManifest {
    position: usize,
    pending_hidden: Vec<u16>,
    prompt_tokens: Vec<u32>,
    kv: Vec<Option<FjallBlob>>,
    dsa: Vec<Option<FjallBlob>>,
}

struct Glm52DsparkAuxManifest {
    start_position: usize,
    rows: usize,
    columns: usize,
    blob: FjallBlob,
}

struct Glm52DsparkTargetManifest {
    layers: Vec<Glm52DsparkTargetLayerManifest>,
}

struct Glm52DsparkTargetLayerManifest {
    start_position: usize,
    rows: usize,
    key_columns: usize,
    value_columns: usize,
    key: FjallBlob,
    value: FjallBlob,
}

fn encode_info(info: &Glm52SwapInfo) -> Result<Vec<u8>, String> {
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

fn decode_info(bytes: &[u8]) -> Result<Glm52SwapInfo, String> {
    let mut reader = SnapshotReader::new(bytes);
    require_header(&mut reader, INFO_MAGIC, "GLM cache info")?;
    let info = Glm52SwapInfo {
        cache_id: reader.string("GLM info cache_id")?,
        token_count: reader.usize("GLM info token_count")?,
        resident_bytes: reader.u64("GLM info resident_bytes")?,
        file_bytes: reader.u64("GLM info file_bytes")?,
        modified_unix: reader.u64("GLM info modified_unix")?,
    };
    reader.finish()?;
    Ok(info)
}

fn encode_manifest(manifest: &Glm52Manifest) -> Result<Vec<u8>, String> {
    let mut writer = SnapshotWriter::new();
    writer.bytes(&MANIFEST_MAGIC);
    writer.u32(VERSION);
    writer.u64(manifest.generation);
    writer.string(&manifest.cache_id)?;
    writer.optional_string(manifest.cache_namespace.as_deref())?;
    writer.u32s(&manifest.tokens)?;
    writer.usize(manifest.token_count)?;
    writer.flag(manifest.pending_tokens.is_some());
    if let Some(tokens) = &manifest.pending_tokens {
        writer.u32s(tokens)?;
    }
    writer.u16s(&manifest.last_hidden)?;
    writer.usize(manifest.stages.len())?;
    for stage in &manifest.stages {
        write_stage_manifest(&mut writer, stage)?;
    }
    writer.flag(manifest.mtp.is_some());
    if let Some(mtp) = &manifest.mtp {
        writer.usize(mtp.position)?;
        writer.u16s(&mtp.pending_hidden)?;
        writer.u32s(&mtp.prompt_tokens)?;
        write_optional_blobs(&mut writer, &mtp.kv)?;
        write_optional_blobs(&mut writer, &mtp.dsa)?;
    }
    writer.flag(manifest.dspark_aux.is_some());
    if let Some(aux) = &manifest.dspark_aux {
        writer.usize(aux.start_position)?;
        writer.usize(aux.rows)?;
        writer.usize(aux.columns)?;
        write_blob(&mut writer, aux.blob);
    }
    writer.flag(manifest.dspark_target.is_some());
    if let Some(target) = &manifest.dspark_target {
        writer.usize(target.layers.len())?;
        for layer in &target.layers {
            writer.usize(layer.start_position)?;
            writer.usize(layer.rows)?;
            writer.usize(layer.key_columns)?;
            writer.usize(layer.value_columns)?;
            write_blob(&mut writer, layer.key);
            write_blob(&mut writer, layer.value);
        }
    }
    Ok(writer.into_inner())
}

fn decode_manifest(bytes: &[u8]) -> Result<Glm52Manifest, String> {
    let mut reader = SnapshotReader::new(bytes);
    require_header(&mut reader, MANIFEST_MAGIC, "GLM cache manifest")?;
    let generation = reader.u64("GLM generation")?;
    let cache_id = reader.string("GLM cache_id")?;
    let cache_namespace = reader.optional_string("GLM namespace")?;
    let tokens = reader.u32s("GLM tokens")?;
    let token_count = reader.usize("GLM token_count")?;
    let pending_tokens = if reader.flag("GLM pending_tokens")? { Some(reader.u32s("GLM pending_tokens")?) } else { None };
    let last_hidden = reader.u16s("GLM last_hidden")?;
    let stage_count = reader.count("GLM stage_count", 24)?;
    let mut stages = Vec::with_capacity(stage_count);
    for _ in 0..stage_count {
        stages.push(read_stage_manifest(&mut reader)?);
    }
    let mtp = if reader.flag("GLM mtp")? {
        Some(Glm52MtpManifest {
            position: reader.usize("GLM MTP position")?,
            pending_hidden: reader.u16s("GLM MTP pending_hidden")?,
            prompt_tokens: reader.u32s("GLM MTP prompt_tokens")?,
            kv: read_optional_blobs(&mut reader, "GLM MTP KV")?,
            dsa: read_optional_blobs(&mut reader, "GLM MTP DSA")?,
        })
    } else {
        None
    };
    let dspark_aux = if reader.flag("GLM DSpark aux")? {
        Some(Glm52DsparkAuxManifest {
            start_position: reader.usize("GLM DSpark aux start")?,
            rows: reader.usize("GLM DSpark aux rows")?,
            columns: reader.usize("GLM DSpark aux columns")?,
            blob: read_blob(&mut reader, "GLM DSpark aux blob")?,
        })
    } else {
        None
    };
    let dspark_target = if reader.flag("GLM DSpark target")? {
        let count = reader.count("GLM DSpark target layers", 56)?;
        let mut layers = Vec::with_capacity(count);
        for _ in 0..count {
            layers.push(Glm52DsparkTargetLayerManifest {
                start_position: reader.usize("GLM DSpark target start")?,
                rows: reader.usize("GLM DSpark target rows")?,
                key_columns: reader.usize("GLM DSpark target key columns")?,
                value_columns: reader.usize("GLM DSpark target value columns")?,
                key: read_blob(&mut reader, "GLM DSpark target key")?,
                value: read_blob(&mut reader, "GLM DSpark target value")?,
            });
        }
        Some(Glm52DsparkTargetManifest { layers })
    } else {
        None
    };
    reader.finish()?;
    Ok(Glm52Manifest { generation, cache_id, cache_namespace, tokens, token_count, pending_tokens, last_hidden, stages, mtp, dspark_aux, dspark_target })
}

fn write_stage_manifest(writer: &mut SnapshotWriter, stage: &Glm52StageManifest) -> Result<(), String> {
    writer.usize(stage.layer_start)?;
    write_optional_blobs(writer, &stage.kv)?;
    write_optional_blobs(writer, &stage.dsa)
}

fn read_stage_manifest(reader: &mut SnapshotReader<'_>) -> Result<Glm52StageManifest, String> {
    Ok(Glm52StageManifest { layer_start: reader.usize("GLM stage layer_start")?, kv: read_optional_blobs(reader, "GLM stage KV")?, dsa: read_optional_blobs(reader, "GLM stage DSA")? })
}

fn write_optional_blobs(writer: &mut SnapshotWriter, blobs: &[Option<FjallBlob>]) -> Result<(), String> {
    writer.usize(blobs.len())?;
    for blob in blobs {
        writer.flag(blob.is_some());
        if let Some(blob) = blob {
            write_blob(writer, *blob);
        }
    }
    Ok(())
}

fn read_optional_blobs(reader: &mut SnapshotReader<'_>, what: &str) -> Result<Vec<Option<FjallBlob>>, String> {
    let count = reader.count(what, 1)?;
    let mut blobs = Vec::with_capacity(count);
    for _ in 0..count {
        blobs.push(if reader.flag(what)? { Some(read_blob(reader, what)?) } else { None });
    }
    Ok(blobs)
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

pub struct Glm52SwapStore {
    store: FjallCacheStore,
}

pub struct Glm52CacheIdentity {
    metadata: Vec<(String, String)>,
}

impl Glm52CacheIdentity {
    pub fn new(weights: &Glm52Weights, kv_cache_format: KvCacheFormat, layer_start: usize, layer_end: usize, mtp: bool, tail_sampling: bool, max_seq_len: usize, dspark_source: Option<&std::path::Path>) -> Self {
        let cfg = weights.cfg();
        let mut metadata = vec![
            ("schema_version".to_owned(), "2".to_owned()),
            ("architecture".to_owned(), "glm52".to_owned()),
            ("weight_source".to_owned(), weights.cache_source_path().to_string_lossy().into_owned()),
            ("quantization".to_owned(), weights.cache_quantization()),
            (
                "kv_cache_format".to_owned(),
                match kv_cache_format {
                    KvCacheFormat::F16 => "f16",
                    KvCacheFormat::Q8g64 => "q8g64",
                }
                .to_owned(),
            ),
            ("layer_start".to_owned(), layer_start.to_string()),
            ("layer_end".to_owned(), layer_end.to_string()),
            ("mtp".to_owned(), mtp.to_string()),
            ("tail_sampling".to_owned(), tail_sampling.to_string()),
            ("dsa_cache_format".to_owned(), if crate::kernel::rocm::hip::options().dsa_hadamard_i8 { "hadamard_q8" } else { "q8" }.to_owned()),
            ("max_sequence_length".to_owned(), max_seq_len.to_string()),
            ("model_layout".to_owned(), format!("layers={};hidden={};kv_lora={};qk_rope={}", cfg.layer_count, cfg.hidden_size, cfg.kv_lora_rank, cfg.qk_rope_head_dim)),
        ];
        if let Some(source) = dspark_source {
            metadata.push(("dspark_source".to_owned(), source.to_string_lossy().into_owned()));
        }
        Self { metadata }
    }
}

impl Glm52SwapStore {
    pub fn open(dir: impl Into<PathBuf>, identity: &Glm52CacheIdentity) -> Result<Self, String> {
        let dir = dir.into();
        let store = FjallCacheStore::open(dir.join("fjall"))?;
        store.bind_metadata(&identity.metadata)?;
        Ok(Self { store })
    }

    pub fn put(&self, snapshot: &Glm52CacheSnapshot) -> Result<Glm52SwapInfo, String> {
        validate_snapshot(snapshot)?;
        let previous = self.manifest(&snapshot.cache_id)?.map(|manifest| manifest.generation);
        let generation = new_generation();
        let prepared = self.prepare(snapshot, generation);
        let (info, manifest) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                let _ = self.store.remove_generation(&snapshot.cache_id, generation);
                return Err(error);
            }
        };
        let info_bytes = encode_info(&info)?;
        let manifest_bytes = encode_manifest(&manifest)?;
        self.store.commit(&snapshot.cache_id, &info_bytes, &manifest_bytes)?;
        if let Some(previous) = previous.filter(|previous| *previous != generation) {
            let _ = self.store.remove_generation(&snapshot.cache_id, previous);
        }
        Ok(info)
    }

    fn prepare(&self, snapshot: &Glm52CacheSnapshot, generation: u64) -> Result<(Glm52SwapInfo, Glm52Manifest), String> {
        let mut file_bytes = 0_u64;
        let mut stages = Vec::with_capacity(snapshot.stages.len());
        for (stage_index, stage) in snapshot.stages.iter().enumerate() {
            let (bytes, manifest) = self.prepare_stage(&snapshot.cache_id, generation, stage_index, stage.layer_start, &stage.kv, &stage.dsa)?;
            file_bytes = file_bytes.saturating_add(bytes);
            stages.push(manifest);
        }
        let mtp = snapshot
            .mtp
            .as_ref()
            .map(|mtp| {
                let (bytes, manifest) = self.prepare_stage(&snapshot.cache_id, generation, snapshot.stages.len(), 0, &mtp.kv, &mtp.dsa)?;
                file_bytes = file_bytes.saturating_add(bytes);
                Ok::<_, String>(Glm52MtpManifest { position: mtp.position, pending_hidden: mtp.pending_hidden.clone(), prompt_tokens: mtp.prompt_tokens.clone(), kv: manifest.kv, dsa: manifest.dsa })
            })
            .transpose()?;
        let dspark_aux = snapshot
            .dspark_aux
            .as_ref()
            .map(|aux| {
                let mut writer = self.store.writer(&snapshot.cache_id, generation, snapshot.stages.len(), 0, FjallChunkKind::Aux);
                write_u16_values(&mut writer, &aux.values)?;
                let blob = writer.finish()?;
                file_bytes = file_bytes.saturating_add(blob.bytes);
                Ok::<_, String>(Glm52DsparkAuxManifest { start_position: aux.start_position, rows: aux.rows, columns: aux.columns, blob })
            })
            .transpose()?;
        let dspark_target = snapshot
            .dspark_target
            .as_ref()
            .map(|target| {
                let stage_index = snapshot.stages.len();
                let mut layers = Vec::with_capacity(target.layers.len());
                for (layer_index, layer) in target.layers.iter().enumerate() {
                    let key_index = layer_index.checked_mul(2).and_then(|index| index.checked_add(1)).ok_or("GLM DSpark target chunk index 溢出")?;
                    let value_index = key_index.checked_add(1).ok_or("GLM DSpark target chunk index 溢出")?;
                    let mut key_writer = self.store.writer(&snapshot.cache_id, generation, stage_index, key_index, FjallChunkKind::Aux);
                    write_f32_values(&mut key_writer, &layer.key.values)?;
                    let key = key_writer.finish()?;
                    let mut value_writer = self.store.writer(&snapshot.cache_id, generation, stage_index, value_index, FjallChunkKind::Aux);
                    write_f32_values(&mut value_writer, &layer.value.values)?;
                    let value = value_writer.finish()?;
                    file_bytes = file_bytes.saturating_add(key.bytes).saturating_add(value.bytes);
                    layers.push(Glm52DsparkTargetLayerManifest { start_position: layer.start_position, rows: layer.key.rows, key_columns: layer.key.columns, value_columns: layer.value.columns, key, value });
                }
                Ok::<_, String>(Glm52DsparkTargetManifest { layers })
            })
            .transpose()?;
        let info = Glm52SwapInfo {
            cache_id: snapshot.cache_id.clone(),
            token_count: snapshot.token_count,
            resident_bytes: snapshot.resident_bytes(),
            file_bytes,
            modified_unix: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        };
        let manifest = Glm52Manifest {
            generation,
            cache_id: snapshot.cache_id.clone(),
            cache_namespace: snapshot.cache_namespace.clone(),
            tokens: snapshot.tokens.clone(),
            token_count: snapshot.token_count,
            pending_tokens: snapshot.pending_tokens.clone(),
            last_hidden: snapshot.last_hidden.clone(),
            stages,
            mtp,
            dspark_aux,
            dspark_target,
        };
        Ok((info, manifest))
    }

    fn prepare_stage(&self, cache_id: &str, generation: u64, stage_index: usize, layer_start: usize, layers: &[Option<MlaLayerSerde>], dsa_layers: &[Option<DsaLayerSerde>]) -> Result<(u64, Glm52StageManifest), String> {
        let mut file_bytes = 0_u64;
        let mut kv = Vec::with_capacity(layers.len());
        for (layer_index, layer) in layers.iter().enumerate() {
            let blob = match layer {
                None => None,
                Some(layer) => {
                    let logical_layer = layer_start.checked_add(layer_index).ok_or("GLM cache layer 溢出")?;
                    let mut writer = self.store.writer(cache_id, generation, stage_index, logical_layer, FjallChunkKind::Kv);
                    write_mla_layer(&mut writer, layer)?;
                    let blob = writer.finish()?;
                    file_bytes = file_bytes.saturating_add(blob.bytes);
                    Some(blob)
                }
            };
            kv.push(blob);
        }
        let mut dsa = Vec::with_capacity(dsa_layers.len());
        for (layer_index, layer) in dsa_layers.iter().enumerate() {
            let blob = match layer {
                None => None,
                Some(layer) => {
                    let logical_layer = layer_start.checked_add(layer_index).ok_or("GLM cache layer 溢出")?;
                    let mut writer = self.store.writer(cache_id, generation, stage_index, logical_layer, FjallChunkKind::Dsa);
                    write_dsa_layer(&mut writer, layer)?;
                    let blob = writer.finish()?;
                    file_bytes = file_bytes.saturating_add(blob.bytes);
                    Some(blob)
                }
            };
            dsa.push(blob);
        }
        Ok((file_bytes, Glm52StageManifest { layer_start, kv, dsa }))
    }

    pub fn get(&self, cache_id: &str) -> Result<Option<Glm52CacheSnapshot>, String> {
        let Some(manifest) = self.manifest(cache_id)? else { return Ok(None) };
        let Glm52Manifest {
            generation, cache_id, cache_namespace, tokens, token_count, pending_tokens, last_hidden, stages: stage_manifests, mtp: mtp_manifest, dspark_aux: dspark_aux_manifest, dspark_target: dspark_target_manifest, ..
        } = manifest;
        let mtp_stage_index = stage_manifests.len();
        let mut stages = Vec::with_capacity(stage_manifests.len());
        for (stage_index, stage) in stage_manifests.into_iter().enumerate() {
            stages.push(self.restore_stage(&cache_id, generation, stage_index, stage)?);
        }
        let mtp = mtp_manifest
            .map(|mtp| {
                let Glm52MtpManifest { position, pending_hidden, prompt_tokens, kv, dsa } = mtp;
                let stage = self.restore_stage(&cache_id, generation, mtp_stage_index, Glm52StageManifest { layer_start: 0, kv, dsa })?;
                Ok::<_, String>(Glm52MtpCache { position, pending_hidden, prompt_tokens, kv: stage.kv, dsa: stage.dsa })
            })
            .transpose()?;
        let dspark_aux = dspark_aux_manifest.map(|aux| self.restore_dspark_aux(&cache_id, generation, mtp_stage_index, aux)).transpose()?;
        let dspark_target = dspark_target_manifest.map(|target| self.restore_dspark_target(&cache_id, generation, mtp_stage_index, target)).transpose()?;
        let snapshot = Glm52CacheSnapshot { cache_id, cache_namespace, tokens, token_count, pending_tokens, last_hidden, stages, mtp, dspark_aux, dspark_target };
        validate_snapshot(&snapshot)?;
        Ok(Some(snapshot))
    }

    /// SSD miss 路径才扫描 manifest；大块 KV 仍只恢复最终选中的最长前缀。
    /// 旧 manifest 没有 namespace，只允许 exact cache_id 路径恢复，避免跨 API key 复用。
    pub fn longest_prefix_cache_id(&self, cache_namespace: Option<&str>, tokens: &[u32]) -> Result<Option<String>, String> {
        let mut best = None::<(usize, String)>;
        for bytes in self.store.manifest_values()? {
            let Ok(manifest) = decode_manifest(&bytes) else { continue };
            if manifest.cache_namespace.as_deref() != cache_namespace || manifest.tokens.is_empty() || !tokens.starts_with(&manifest.tokens) {
                continue;
            }
            if best.as_ref().is_none_or(|(length, _)| manifest.tokens.len() > *length) {
                best = Some((manifest.tokens.len(), manifest.cache_id));
            }
        }
        Ok(best.map(|(_, cache_id)| cache_id))
    }

    fn restore_stage(&self, cache_id: &str, generation: u64, stage_index: usize, stage: Glm52StageManifest) -> Result<Glm52StageCache, String> {
        let Glm52StageManifest { layer_start, kv: kv_manifests, dsa: dsa_manifests } = stage;
        let mut kv = Vec::with_capacity(kv_manifests.len());
        for (layer_index, blob) in kv_manifests.into_iter().enumerate() {
            let layer = match blob {
                None => None,
                Some(blob) => {
                    let logical_layer = layer_start.checked_add(layer_index).ok_or("GLM cache layer 溢出")?;
                    let reader = self.store.reader(cache_id, generation, stage_index, logical_layer, FjallChunkKind::Kv, blob);
                    let mut reader = CacheReader { inner: reader, remaining: blob.bytes };
                    let layer = read_mla_layer(&mut reader)?;
                    reader.finish("KV layer")?;
                    Some(layer)
                }
            };
            kv.push(layer);
        }
        let mut dsa = Vec::with_capacity(dsa_manifests.len());
        for (layer_index, blob) in dsa_manifests.into_iter().enumerate() {
            let layer = match blob {
                None => None,
                Some(blob) => {
                    let logical_layer = layer_start.checked_add(layer_index).ok_or("GLM cache layer 溢出")?;
                    let reader = self.store.reader(cache_id, generation, stage_index, logical_layer, FjallChunkKind::Dsa, blob);
                    let mut reader = CacheReader { inner: reader, remaining: blob.bytes };
                    let layer = read_dsa_layer(&mut reader)?;
                    reader.finish("DSA layer")?;
                    Some(layer)
                }
            };
            dsa.push(layer);
        }
        Ok(Glm52StageCache { layer_start, kv, dsa })
    }

    fn restore_dspark_aux(&self, cache_id: &str, generation: u64, stage_index: usize, aux: Glm52DsparkAuxManifest) -> Result<Glm52DsparkAuxCache, String> {
        let Glm52DsparkAuxManifest { start_position, rows, columns, blob } = aux;
        let expected_values = rows.checked_mul(columns).ok_or("GLM DSpark aux shape 溢出")?;
        let expected_bytes = expected_values.checked_mul(2).ok_or("GLM DSpark aux bytes 溢出")? as u64;
        if blob.bytes != expected_bytes {
            return Err(format!("GLM DSpark aux blob={} bytes，shape=[{rows},{columns}] 需要 {expected_bytes}", blob.bytes));
        }
        let reader = self.store.reader(cache_id, generation, stage_index, 0, FjallChunkKind::Aux, blob);
        let mut reader = CacheReader { inner: reader, remaining: blob.bytes };
        let values = read_u16_values(&mut reader, expected_values)?;
        reader.finish("DSpark aux")?;
        Ok(Glm52DsparkAuxCache { start_position, rows, columns, values })
    }

    fn restore_dspark_target(&self, cache_id: &str, generation: u64, stage_index: usize, target: Glm52DsparkTargetManifest) -> Result<Glm52DsparkTargetCache, String> {
        let mut layers = Vec::with_capacity(target.layers.len());
        for (layer_index, layer) in target.layers.into_iter().enumerate() {
            let Glm52DsparkTargetLayerManifest { start_position, rows, key_columns, value_columns, key, value } = layer;
            let key_values = rows.checked_mul(key_columns).ok_or("GLM DSpark target key shape 溢出")?;
            let value_values = rows.checked_mul(value_columns).ok_or("GLM DSpark target value shape 溢出")?;
            let key_bytes = key_values.checked_mul(std::mem::size_of::<f32>()).ok_or("GLM DSpark target key bytes 溢出")? as u64;
            let value_bytes = value_values.checked_mul(std::mem::size_of::<f32>()).ok_or("GLM DSpark target value bytes 溢出")? as u64;
            if key.bytes != key_bytes || value.bytes != value_bytes {
                return Err(format!("GLM DSpark target L{layer_index} blob={}/{} bytes，shape=[{rows},{key_columns}]/[{rows},{value_columns}] 需要 {key_bytes}/{value_bytes}", key.bytes, value.bytes));
            }
            let key_index = layer_index.checked_mul(2).and_then(|index| index.checked_add(1)).ok_or("GLM DSpark target chunk index 溢出")?;
            let value_index = key_index.checked_add(1).ok_or("GLM DSpark target chunk index 溢出")?;
            let key_reader = self.store.reader(cache_id, generation, stage_index, key_index, FjallChunkKind::Aux, key);
            let mut key_reader = CacheReader { inner: key_reader, remaining: key.bytes };
            let key = Glm52DsparkTargetTensor { rows, columns: key_columns, values: read_f32_values(&mut key_reader, key_values)? };
            key_reader.finish("DSpark target key")?;
            let value_reader = self.store.reader(cache_id, generation, stage_index, value_index, FjallChunkKind::Aux, value);
            let mut value_reader = CacheReader { inner: value_reader, remaining: value.bytes };
            let value = Glm52DsparkTargetTensor { rows, columns: value_columns, values: read_f32_values(&mut value_reader, value_values)? };
            value_reader.finish("DSpark target value")?;
            layers.push(Glm52DsparkTargetLayerCache { start_position, key, value });
        }
        Ok(Glm52DsparkTargetCache { layers })
    }

    pub fn delete(&self, cache_id: &str) -> Result<(), String> {
        let generation = self.manifest(cache_id)?.map(|manifest| manifest.generation);
        self.store.remove_entry(cache_id)?;
        if let Some(generation) = generation {
            self.store.remove_generation(cache_id, generation)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn contains(&self, cache_id: &str) -> bool {
        self.store.contains(cache_id).unwrap_or(false)
    }

    // 本文件同时被 standalone 与 node 二进制 include，只有 node 需要枚举缓存。
    #[allow(dead_code)]
    pub fn infos(&self) -> Vec<Glm52SwapInfo> {
        let Ok(manifests) = self.store.manifest_values() else { return Vec::new() };
        let current = manifests.into_iter().filter_map(|bytes| decode_manifest(&bytes).ok()).map(|manifest| manifest.cache_id).collect::<std::collections::HashSet<_>>();
        let Ok(values) = self.store.info_values() else { return Vec::new() };
        let mut infos = values.into_iter().filter_map(|bytes| decode_info(&bytes).ok()).filter(|info| current.contains(&info.cache_id)).collect::<Vec<_>>();
        infos.sort_by_key(|info| info.modified_unix);
        infos
    }

    fn manifest(&self, cache_id: &str) -> Result<Option<Glm52Manifest>, String> {
        let Some(bytes) = self.store.manifest(cache_id)? else { return Ok(None) };
        let manifest = decode_manifest(&bytes)?;
        if manifest.cache_id != cache_id {
            return Err(format!("GLM cache_id 不匹配: 请求={cache_id} 存储={}", manifest.cache_id));
        }
        Ok(Some(manifest))
    }
}

fn validate_snapshot(snapshot: &Glm52CacheSnapshot) -> Result<(), String> {
    if snapshot.cache_id.is_empty() || snapshot.token_count == 0 || snapshot.last_hidden.is_empty() || snapshot.stages.is_empty() {
        return Err("GLM cache snapshot 缺少 cache_id/token/hidden/stage".to_owned());
    }
    if !snapshot.tokens.is_empty() && snapshot.tokens.len() != snapshot.token_count {
        return Err(format!("GLM cache tokens={}，token_count={}", snapshot.tokens.len(), snapshot.token_count));
    }
    if snapshot.pending_tokens.as_ref().is_some_and(|pending| pending.len() > 1) {
        return Err(format!("GLM cache pending tokens={}，期望至多 1", snapshot.pending_tokens.as_ref().map_or(0, Vec::len)));
    }
    if let Some(mtp) = &snapshot.mtp {
        let expected = snapshot.token_count.saturating_sub(1);
        if mtp.position != expected || mtp.pending_hidden.len() != snapshot.last_hidden.len() || mtp.prompt_tokens.is_empty() || mtp.prompt_tokens.len() > snapshot.token_count {
            return Err(format!("GLM MTP cache 元数据非法: position={}/{} hidden={}/{} prompt={}/{}", mtp.position, expected, mtp.pending_hidden.len(), snapshot.last_hidden.len(), mtp.prompt_tokens.len(), snapshot.token_count));
        }
        if !snapshot.tokens.is_empty() && !snapshot.tokens.starts_with(&mtp.prompt_tokens) {
            return Err(format!("GLM MTP cache token 前缀不匹配: mtp={} terminal={}", mtp.prompt_tokens.len(), snapshot.tokens.len()));
        }
    }
    match (&snapshot.dspark_aux, &snapshot.dspark_target) {
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => return Err("GLM DSpark aux 与 target cache 必须同时持久化".to_owned()),
        (Some(aux), Some(target)) => {
            let end = aux.start_position.checked_add(aux.rows).ok_or("GLM DSpark aux position 溢出")?;
            let expected_values = aux.rows.checked_mul(aux.columns).ok_or("GLM DSpark aux shape 溢出")?;
            if aux.rows == 0 || aux.columns == 0 || end != snapshot.token_count || aux.values.len() != expected_values || target.layers.is_empty() {
                return Err(format!(
                    "GLM DSpark aux/target 元数据非法: range=[{},{end}) shape=[{},{}] values={} target_layers={} terminal={}",
                    aux.start_position,
                    aux.rows,
                    aux.columns,
                    aux.values.len(),
                    target.layers.len(),
                    snapshot.token_count
                ));
            }
            for (layer_index, layer) in target.layers.iter().enumerate() {
                let key_values = layer.key.rows.checked_mul(layer.key.columns).ok_or("GLM DSpark target key shape 溢出")?;
                let value_values = layer.value.rows.checked_mul(layer.value.columns).ok_or("GLM DSpark target value shape 溢出")?;
                if layer.start_position != aux.start_position
                    || layer.key.rows != aux.rows
                    || layer.value.rows != aux.rows
                    || layer.key.columns == 0
                    || layer.value.columns == 0
                    || layer.key.values.len() != key_values
                    || layer.value.values.len() != value_values
                {
                    return Err(format!(
                        "GLM DSpark target L{layer_index} 元数据非法: start={} key=[{},{}]/{} value=[{},{}]/{}，aux=[{},{}]",
                        layer.start_position,
                        layer.key.rows,
                        layer.key.columns,
                        layer.key.values.len(),
                        layer.value.rows,
                        layer.value.columns,
                        layer.value.values.len(),
                        aux.start_position,
                        aux.rows
                    ));
                }
            }
        }
    }
    Ok(())
}

fn write_u16_values(writer: &mut impl Write, values: &[u16]) -> Result<(), String> {
    let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    writer.write_all(&bytes).map_err(io_error)
}

fn write_f32_values(writer: &mut impl Write, values: &[f32]) -> Result<(), String> {
    const ELEMENTS_PER_CHUNK: usize = 1024 * 1024;
    for values in values.chunks(ELEMENTS_PER_CHUNK) {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        writer.write_all(&bytes).map_err(io_error)?;
    }
    Ok(())
}

fn read_u16_values(reader: &mut CacheReader<impl Read>, len: usize) -> Result<Vec<u16>, String> {
    let mut bytes = vec![0_u8; len.checked_mul(2).ok_or("GLM cache u16 bytes 溢出")?];
    reader.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect())
}

fn read_f32_values(reader: &mut CacheReader<impl Read>, len: usize) -> Result<Vec<f32>, String> {
    const ELEMENTS_PER_CHUNK: usize = 1024 * 1024;
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let count = (len - values.len()).min(ELEMENTS_PER_CHUNK);
        let mut bytes = vec![0_u8; count.checked_mul(std::mem::size_of::<f32>()).ok_or("GLM cache f32 bytes 溢出")?];
        reader.read_exact(&mut bytes)?;
        values.extend(bytes.chunks_exact(4).map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])));
    }
    Ok(values)
}

fn new_generation() -> u64 {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    (nanos as u64) ^ ((nanos >> 64) as u64) ^ ((std::process::id() as u64) << 32)
}

fn write_mla_layer(writer: &mut impl Write, layer: &MlaLayerSerde) -> Result<(), String> {
    write_usize(writer, layer.rows)?;
    write_usize(writer, layer.latent_cols)?;
    write_usize(writer, layer.rope_cols)?;
    write_usize(writer, layer.latent_group_size)?;
    write_bytes(writer, &layer.latent)?;
    match &layer.latent_scales {
        None => writer.write_all(&[0]).map_err(io_error)?,
        Some(scales) => {
            writer.write_all(&[1]).map_err(io_error)?;
            write_bytes(writer, scales)?;
        }
    }
    write_bytes(writer, &layer.rope)
}

fn write_dsa_layer(writer: &mut impl Write, layer: &DsaLayerSerde) -> Result<(), String> {
    write_usize(writer, layer.rows)?;
    write_usize(writer, layer.key_group_size)?;
    writer.write_all(&[u8::from(layer.hadamard)]).map_err(io_error)?;
    write_bytes(writer, &layer.keys)?;
    write_bytes(writer, &layer.scales)
}

fn read_mla_layer(reader: &mut CacheReader<impl Read>) -> Result<MlaLayerSerde, String> {
    let rows = reader.usize()?;
    let latent_cols = reader.usize()?;
    let rope_cols = reader.usize()?;
    let latent_group_size = reader.usize()?;
    let latent = reader.bytes()?;
    let latent_scales = if reader.flag()? { Some(reader.bytes()?) } else { None };
    let rope = reader.bytes()?;
    Ok(MlaLayerSerde { rows, latent_cols, rope_cols, latent_group_size, latent, latent_scales, rope })
}

fn read_dsa_layer(reader: &mut CacheReader<impl Read>) -> Result<DsaLayerSerde, String> {
    Ok(DsaLayerSerde { rows: reader.usize()?, key_group_size: reader.usize()?, hadamard: reader.flag()?, keys: reader.bytes()?, scales: reader.bytes()? })
}

fn write_usize(writer: &mut impl Write, value: usize) -> Result<(), String> {
    let value = u64::try_from(value).map_err(|_| "GLM cache usize 超过 u64".to_owned())?;
    writer.write_all(&value.to_le_bytes()).map_err(io_error)
}

fn write_bytes(writer: &mut impl Write, bytes: &[u8]) -> Result<(), String> {
    write_usize(writer, bytes.len())?;
    writer.write_all(bytes).map_err(io_error)
}

fn io_error(error: std::io::Error) -> String {
    format!("GLM cache I/O: {error}")
}

struct CacheReader<R> {
    inner: R,
    remaining: u64,
}

impl<R: Read> CacheReader<R> {
    fn read_exact(&mut self, output: &mut [u8]) -> Result<(), String> {
        let len = output.len() as u64;
        if len > self.remaining {
            return Err(format!("GLM cache 截断: 需要 {len}，剩余 {}", self.remaining));
        }
        self.inner.read_exact(output).map_err(io_error)?;
        self.remaining -= len;
        Ok(())
    }

    fn usize(&mut self) -> Result<usize, String> {
        let mut bytes = [0_u8; 8];
        self.read_exact(&mut bytes)?;
        usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| "GLM cache 长度超过 usize".to_owned())
    }

    fn flag(&mut self) -> Result<bool, String> {
        let mut byte = [0_u8; 1];
        self.read_exact(&mut byte)?;
        match byte[0] {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(format!("GLM cache flag={value} 非法")),
        }
    }

    fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let len = self.usize()?;
        if len as u64 > self.remaining {
            return Err(format!("GLM cache blob={len}，剩余 {}", self.remaining));
        }
        let mut bytes = vec![0_u8; len];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn finish(self, label: &str) -> Result<(), String> {
        if self.remaining == 0 { Ok(()) } else { Err(format!("GLM cache {label} 末尾多出 {} bytes", self.remaining)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn identity() -> Glm52CacheIdentity {
        Glm52CacheIdentity { metadata: vec![("schema_version".to_owned(), "test".to_owned())] }
    }

    fn snapshot() -> Glm52CacheSnapshot {
        Glm52CacheSnapshot {
            cache_id: "session-a".to_owned(),
            cache_namespace: Some("tenant-a".to_owned()),
            tokens: vec![1, 2, 3],
            token_count: 3,
            pending_tokens: Some(vec![4]),
            last_hidden: vec![10, 20],
            stages: vec![Glm52StageCache {
                layer_start: 40,
                kv: vec![Some(MlaLayerSerde { rows: 3, latent_cols: 2, rope_cols: 1, latent_group_size: 0, latent: vec![1; 12], latent_scales: None, rope: vec![2; 6] })],
                dsa: vec![Some(DsaLayerSerde { rows: 3, key_group_size: 2, hadamard: true, keys: vec![3; 6], scales: vec![4; 6] })],
            }],
            mtp: Some(Glm52MtpCache {
                position: 2,
                pending_hidden: vec![30, 40],
                prompt_tokens: vec![1, 2, 3],
                kv: vec![Some(MlaLayerSerde { rows: 2, latent_cols: 2, rope_cols: 1, latent_group_size: 0, latent: vec![5; 8], latent_scales: None, rope: vec![6; 4] })],
                dsa: vec![Some(DsaLayerSerde { rows: 2, key_group_size: 2, hadamard: true, keys: vec![7; 4], scales: vec![8; 4] })],
            }),
            dspark_aux: Some(Glm52DsparkAuxCache { start_position: 1, rows: 2, columns: 2, values: vec![9, 10, 11, 12] }),
            dspark_target: Some(Glm52DsparkTargetCache {
                layers: vec![Glm52DsparkTargetLayerCache {
                    start_position: 1,
                    key: Glm52DsparkTargetTensor { rows: 2, columns: 2, values: vec![1.25, 2.5, 3.75, 4.0] },
                    value: Glm52DsparkTargetTensor { rows: 2, columns: 1, values: vec![5.25, 6.5] },
                }],
            }),
        }
    }

    #[test]
    fn disk_roundtrip_lists_and_deletes() {
        let root = std::env::temp_dir().join(format!("zllm-glm52-swap-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let store = Glm52SwapStore::open(&root, &identity()).unwrap();
        let info = store.put(&snapshot()).unwrap();
        assert_eq!(info.cache_id, "session-a");
        assert!(info.file_bytes > 0);
        assert!(store.contains("session-a"));
        assert_eq!(store.infos().len(), 1);
        assert_eq!(&store.store.manifest("session-a").unwrap().unwrap()[..8], &MANIFEST_MAGIC);
        assert_eq!(&store.store.info_values().unwrap()[0][..8], &INFO_MAGIC);
        let restored = store.get("session-a").unwrap().unwrap();
        assert_eq!(restored.tokens, vec![1, 2, 3]);
        assert_eq!(restored.pending_tokens, Some(vec![4]));
        assert_eq!(restored.cache_namespace.as_deref(), Some("tenant-a"));
        assert_eq!(restored.last_hidden, vec![10, 20]);
        assert_eq!(restored.stages[0].layer_start, 40);
        assert_eq!(restored.stages[0].kv[0].as_ref().unwrap().latent, vec![1; 12]);
        let mtp = restored.mtp.unwrap();
        assert_eq!(mtp.position, 2);
        assert_eq!(mtp.pending_hidden, vec![30, 40]);
        assert_eq!(mtp.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(mtp.kv[0].as_ref().unwrap().latent, vec![5; 8]);
        assert_eq!(mtp.dsa[0].as_ref().unwrap().keys, vec![7; 4]);
        assert!(mtp.dsa[0].as_ref().unwrap().hadamard);
        let aux = restored.dspark_aux.unwrap();
        assert_eq!(aux.start_position, 1);
        assert_eq!(aux.values, vec![9, 10, 11, 12]);
        let target = restored.dspark_target.unwrap();
        assert_eq!(target.layers[0].start_position, 1);
        assert_eq!(target.layers[0].key.values, vec![1.25, 2.5, 3.75, 4.0]);
        assert_eq!(target.layers[0].value.values, vec![5.25, 6.5]);
        store.delete("session-a").unwrap();
        assert!(!store.contains("session-a"));
        assert!(store.infos().is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disk_prefix按namespace选择最长manifest() {
        let root = std::env::temp_dir().join(format!("zllm-glm52-swap-prefix-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let store = Glm52SwapStore::open(&root, &identity()).unwrap();
        store.put(&snapshot()).unwrap();

        let mut longer = snapshot();
        longer.cache_id = "session-b".to_owned();
        longer.tokens.push(4);
        longer.token_count = 4;
        longer.mtp.as_mut().unwrap().position = 3;
        longer.mtp.as_mut().unwrap().prompt_tokens.push(4);
        longer.dspark_aux.as_mut().unwrap().start_position = 2;
        longer.dspark_target.as_mut().unwrap().layers[0].start_position = 2;
        store.put(&longer).unwrap();

        let mut other = snapshot();
        other.cache_id = "session-c".to_owned();
        other.cache_namespace = Some("tenant-b".to_owned());
        other.tokens.extend([4, 5]);
        other.token_count = 5;
        other.mtp.as_mut().unwrap().position = 4;
        other.mtp.as_mut().unwrap().prompt_tokens.extend([4, 5]);
        other.dspark_aux.as_mut().unwrap().start_position = 3;
        other.dspark_target.as_mut().unwrap().layers[0].start_position = 3;
        store.put(&other).unwrap();

        assert_eq!(store.longest_prefix_cache_id(Some("tenant-a"), &[1, 2, 3, 4, 5, 6]).unwrap().as_deref(), Some("session-b"));
        assert_eq!(store.longest_prefix_cache_id(Some("tenant-b"), &[1, 2, 3, 4, 5, 6]).unwrap().as_deref(), Some("session-c"));
        assert_eq!(store.longest_prefix_cache_id(Some("tenant-c"), &[1, 2, 3, 4, 5, 6]).unwrap(), None);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disk_snapshot支持多会话并行读取() {
        let root = std::env::temp_dir().join(format!("zllm-glm52-swap-parallel-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let store = Arc::new(Glm52SwapStore::open(&root, &identity()).unwrap());
        for index in 0..4 {
            let mut value = snapshot();
            value.cache_id = format!("session-{index}");
            store.put(&value).unwrap();
        }
        let readers = (0..4)
            .map(|index| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || store.get(&format!("session-{index}")).unwrap().unwrap())
            })
            .collect::<Vec<_>>();
        for (index, reader) in readers.into_iter().enumerate() {
            let restored = reader.join().unwrap();
            assert_eq!(restored.cache_id, format!("session-{index}"));
            assert_eq!(restored.tokens, vec![1, 2, 3]);
        }
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mtp_snapshot必须对齐terminal位置() {
        let root = std::env::temp_dir().join(format!("zllm-glm52-swap-invalid-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let store = Glm52SwapStore::open(&root, &identity()).unwrap();
        let mut snapshot = snapshot();
        snapshot.mtp.as_mut().unwrap().position = 1;
        let error = store.put(&snapshot).unwrap_err();
        assert!(error.contains("position=1/2"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn dspark_snapshot拒绝缺失target_cache() {
        let root = std::env::temp_dir().join(format!("zllm-glm52-swap-dspark-invalid-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let store = Glm52SwapStore::open(&root, &identity()).unwrap();
        let mut snapshot = snapshot();
        snapshot.dspark_target = None;
        let error = store.put(&snapshot).unwrap_err();
        assert!(error.contains("必须同时持久化"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }
}
