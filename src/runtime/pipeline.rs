//! runtime 节点间分层推理数据面：只传层边界 hidden、控制元数据和最终 token。

use std::io::{Error, ErrorKind};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const PIPELINE_ALPN: &[u8] = b"zllm/pipeline/1";
pub const PIPELINE_PROTOCOL_VERSION: u32 = 1;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_PAYLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PipelineMessageKind {
    Prefill,
    Decode,
    Clear,
    Token,
    Error,
}

#[derive(Clone, Debug)]
pub struct PipelineMessage {
    pub execution_hash: String,
    pub kind: PipelineMessageKind,
    pub layer: usize,
    pub position: usize,
    pub rows: usize,
    pub hidden_size: usize,
    pub token_id: Option<u32>,
    pub error: Option<String>,
    pub payload: Vec<u8>,
}

#[derive(Deserialize, Serialize)]
struct PipelineHeader {
    protocol_version: u32,
    execution_hash: String,
    kind: PipelineMessageKind,
    layer: usize,
    position: usize,
    rows: usize,
    hidden_size: usize,
    token_id: Option<u32>,
    error: Option<String>,
    payload_bytes: usize,
}

pub async fn write_pipeline_message<W>(writer: &mut W, message: &PipelineMessage) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let header = PipelineHeader {
        protocol_version: PIPELINE_PROTOCOL_VERSION,
        execution_hash: message.execution_hash.clone(),
        kind: message.kind,
        layer: message.layer,
        position: message.position,
        rows: message.rows,
        hidden_size: message.hidden_size,
        token_id: message.token_id,
        error: message.error.clone(),
        payload_bytes: message.payload.len(),
    };
    let bytes = serde_json::to_vec(&header).map_err(|error| Error::new(ErrorKind::InvalidData, error))?;
    if bytes.len() > MAX_HEADER_BYTES {
        return Err(Error::new(ErrorKind::InvalidData, "pipeline header 超过 64 KiB"));
    }
    writer.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
    writer.write_all(&bytes).await?;
    writer.write_all(&message.payload).await?;
    writer.flush().await
}

pub async fn read_pipeline_message<R>(reader: &mut R) -> std::io::Result<PipelineMessage>
where
    R: AsyncRead + Unpin,
{
    let mut length = [0; 4];
    reader.read_exact(&mut length).await?;
    let length = u32::from_le_bytes(length) as usize;
    if length == 0 || length > MAX_HEADER_BYTES {
        return Err(Error::new(ErrorKind::InvalidData, "pipeline header 长度无效"));
    }
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes).await?;
    let header: PipelineHeader = serde_json::from_slice(&bytes).map_err(|error| Error::new(ErrorKind::InvalidData, error))?;
    if header.protocol_version != PIPELINE_PROTOCOL_VERSION {
        return Err(Error::new(ErrorKind::InvalidData, format!("pipeline protocol {} 不受支持", header.protocol_version)));
    }
    if header.execution_hash.is_empty() || header.execution_hash.len() > 256 {
        return Err(Error::new(ErrorKind::InvalidData, "execution_hash 长度无效"));
    }
    if header.payload_bytes > MAX_PAYLOAD_BYTES {
        return Err(Error::new(ErrorKind::InvalidData, "pipeline payload 超过 2 GiB"));
    }
    let expected = match header.kind {
        PipelineMessageKind::Prefill | PipelineMessageKind::Decode => header.rows.checked_mul(header.hidden_size).and_then(|elements| elements.checked_mul(2)).ok_or_else(|| Error::new(ErrorKind::InvalidData, "pipeline shape 溢出"))?,
        PipelineMessageKind::Clear | PipelineMessageKind::Token | PipelineMessageKind::Error => 0,
    };
    if expected != header.payload_bytes {
        return Err(Error::new(ErrorKind::InvalidData, format!("pipeline payload={}，期望 {expected}", header.payload_bytes)));
    }
    if header.kind == PipelineMessageKind::Token && header.token_id.is_none() {
        return Err(Error::new(ErrorKind::InvalidData, "pipeline token 缺少 token_id"));
    }
    if header.kind == PipelineMessageKind::Error && header.error.is_none() {
        return Err(Error::new(ErrorKind::InvalidData, "pipeline error 缺少错误信息"));
    }
    let mut payload = vec![0; header.payload_bytes];
    reader.read_exact(&mut payload).await?;
    Ok(PipelineMessage {
        execution_hash: header.execution_hash,
        kind: header.kind,
        layer: header.layer,
        position: header.position,
        rows: header.rows,
        hidden_size: header.hidden_size,
        token_id: header.token_id,
        error: header.error,
        payload,
    })
}
