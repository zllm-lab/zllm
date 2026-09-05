//! DSA union recall 证书探针。
//!
//! 仅在 `ZLLM_DSA_UNION_PROBE=<输出路径>` 设置时启用（实际文件追加 `.dev{id}`），
//! 记录每个 indexer 层 decode 的 Top-K selection；`ZLLM_DSA_UNION_PROBE_SCORES=<N>`
//! 额外为每层前 N 个 decode token 同步记录完整 score（供离线重建 Top-4096/8192
//! 候选集）。默认关闭，关闭时零开销。记录格式：
//!
//! ```text
//! header: b"DSAU0101"
//! record: tag u32 (1=selection, 2=scores)
//!         layer u32, position u64, context_rows u32,
//!         count u32, payload count × u32 (LE)
//! ```

use std::sync::mpsc::{Receiver, SyncSender};

use crate::backend::{BackendError, compute_error};
use crate::kernel::rocm as ops;

const MAGIC: &[u8; 8] = b"DSAU0101";
const TAG_SELECTION: u32 = 1;
const TAG_SCORES: u32 = 2;
const CHANNEL_BOUND: usize = 4096;

struct ProbeConfig {
    path: String,
    score_tokens: usize,
}

fn config() -> Option<&'static ProbeConfig> {
    static CONFIG: std::sync::OnceLock<Option<ProbeConfig>> = std::sync::OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let path = std::env::var("ZLLM_DSA_UNION_PROBE").ok()?;
            if path.is_empty() {
                return None;
            }
            let score_tokens = std::env::var("ZLLM_DSA_UNION_PROBE_SCORES").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
            Some(ProbeConfig { path, score_tokens })
        })
        .as_ref()
}

pub fn enabled() -> bool {
    config().is_some()
}

enum ProbeRecord {
    Selection { layer: u32, position: u64, context_rows: u32, indices: Vec<u32> },
    Scores { layer: u32, position: u64, context_rows: u32, scores: Vec<u32> },
}

struct PendingSelection {
    layer: u32,
    position: u64,
    context_rows: u32,
    top_k: u32,
}

pub struct DsaUnionProbe {
    device_id: i32,
    sender: Option<SyncSender<ProbeRecord>>,
    writer: Option<std::thread::JoinHandle<()>>,
    download: Option<ops::hip::AsyncHostDownload>,
    pending: Option<PendingSelection>,
    score_counts: std::collections::HashMap<usize, usize>,
    score_tokens_per_layer: usize,
}

impl DsaUnionProbe {
    pub fn new(device_id: i32) -> Result<Self, BackendError> {
        let cfg = config().expect("probe 只在 enabled 时创建");
        let path = format!("{}.dev{device_id}", cfg.path);
        let (sender, receiver) = std::sync::mpsc::sync_channel(CHANNEL_BOUND);
        let writer_path = path.clone();
        let writer = std::thread::Builder::new().name(format!("dsa-union-probe-dev{device_id}")).spawn(move || writer_loop(&writer_path, receiver)).map_err(|error| compute_error(format!("DSA union probe writer 启动失败: {error}")))?;
        eprintln!("[dsa-union-probe] device={device_id} output={path} score_tokens={}", cfg.score_tokens);
        Ok(Self { device_id, sender: Some(sender), writer: Some(writer), download: None, pending: None, score_counts: std::collections::HashMap::new(), score_tokens_per_layer: cfg.score_tokens })
    }

    fn harvest(&mut self) -> Result<(), BackendError> {
        let Some(pending) = self.pending.take() else { return Ok(()) };
        let download = self.download.as_mut().expect("pending 必有 download");
        let bytes = download.wait().map_err(compute_error)?;
        let expected = pending.top_k as usize * std::mem::size_of::<u32>();
        if bytes.len() != expected {
            return Err(compute_error(format!("DSA union probe selection bytes={}，期望 {expected}", bytes.len())));
        }
        let indices = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u32>(), pending.top_k as usize) }.to_vec();
        // channel 满说明 writer 已落后于 decode，直接报错比静默丢记录安全。
        self.sender
            .as_ref()
            .expect("harvest 时 sender 存活")
            .try_send(ProbeRecord::Selection { layer: pending.layer, position: pending.position, context_rows: pending.context_rows, indices })
            .map_err(|_| compute_error("DSA union probe writer 积压"))?;
        Ok(())
    }

    /// GPU exact selection 之后调用：异步 D2H indices，等待推迟到下一次 record
    /// （中间隔了至少一层的 GPU 工作，wait 实际是零成本）。
    pub fn record_selection(&mut self, layer: usize, position: usize, context_rows: usize, top_k: usize, selection: &ops::hip::DeviceBuffer) -> Result<(), BackendError> {
        self.harvest()?;
        let bytes = top_k.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| compute_error("DSA union probe selection 大小溢出"))?;
        let mut download = match self.download.take() {
            Some(download) => download,
            None => ops::hip::AsyncHostDownload::new(self.device_id, bytes).map_err(compute_error)?,
        };
        download.enqueue(selection, bytes).map_err(compute_error)?;
        self.download = Some(download);
        self.pending = Some(PendingSelection { layer: layer as u32, position: position as u64, context_rows: context_rows as u32, top_k: top_k as u32 });
        Ok(())
    }

    /// 每层前 N 个 decode token 同步下载完整 score。与既有 shadow 相同的时序约束：
    /// 必须在任何后续 kernel 覆盖 score workspace 之前调用。
    pub fn maybe_record_scores(&mut self, layer: usize, position: usize, context_rows: usize) -> Result<(), BackendError> {
        let count = self.score_counts.entry(layer).or_insert(0);
        if *count >= self.score_tokens_per_layer {
            return Ok(());
        }
        *count += 1;
        let scores = ops::hip::try_download_last_dsa_score_keys(self.device_id, context_rows).map_err(compute_error)?;
        self.sender
            .as_ref()
            .expect("record_scores 时 sender 存活")
            .try_send(ProbeRecord::Scores { layer: layer as u32, position: position as u64, context_rows: context_rows as u32, scores })
            .map_err(|_| compute_error("DSA union probe writer 积压"))?;
        Ok(())
    }
}

impl Drop for DsaUnionProbe {
    fn drop(&mut self) {
        if self.pending.is_some() {
            let _ = self.harvest();
        }
        // 先断开 sender 让 writer 退出 recv 循环，再 join。
        self.sender.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

fn writer_loop(path: &str, receiver: Receiver<ProbeRecord>) {
    let mut records = 0_u64;
    let result = std::fs::File::create(path).and_then(|mut file| {
        use std::io::Write;
        file.write_all(MAGIC)?;
        while let Ok(record) = receiver.recv() {
            match record {
                ProbeRecord::Selection { layer, position, context_rows, indices } => {
                    write_frame(&mut file, TAG_SELECTION, layer, position, context_rows, &indices)?;
                }
                ProbeRecord::Scores { layer, position, context_rows, scores } => {
                    write_frame(&mut file, TAG_SCORES, layer, position, context_rows, &scores)?;
                }
            }
            records += 1;
        }
        file.flush()
    });
    if let Err(error) = result {
        eprintln!("[dsa-union-probe] writer {path} 失败（{records} 条已写）: {error}");
    } else {
        eprintln!("[dsa-union-probe] writer {path} 完成: {records} 条记录");
    }
}

fn write_frame(file: &mut std::fs::File, tag: u32, layer: u32, position: u64, context_rows: u32, payload: &[u32]) -> std::io::Result<()> {
    use std::io::Write;
    file.write_all(&tag.to_le_bytes())?;
    file.write_all(&layer.to_le_bytes())?;
    file.write_all(&position.to_le_bytes())?;
    file.write_all(&context_rows.to_le_bytes())?;
    file.write_all(&(payload.len() as u32).to_le_bytes())?;
    for value in payload {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
