//! 跨节点模型 stage 的 iroh 双向流：传 BF16 hidden、采样 token 与前机下达的 cache 控制命令。

use std::{
    str::FromStr,
    sync::mpsc,
    time::{Duration, Instant},
};

use iroh::{
    Endpoint, EndpointAddr,
    endpoint::{Connection, QuicTransportConfig, RecvStream, SendStream, presets},
};
use iroh_tickets::endpoint::EndpointTicket;
use tokio::{io::AsyncWriteExt, runtime::Runtime};

use super::iroh::IrohConfig;
use crate::runtime::output::SamplingConfig;

const STAGE_ALPN: &[u8] = b"zllm/stage/1";
const MAGIC: u32 = 0x5a_53_54_47;
const VERSION: u16 = 11;
const STREAM_OPEN: u8 = 0x5a;
const HEADER_BYTES: usize = 48;
const STAGE_KEEP_ALIVE: Duration = Duration::from_secs(5);
const STAGE_MAX_IDLE: Duration = Duration::from_secs(60 * 60);
// 4096-token GLM prefill 会同时携带 96 MiB 主/辅助 hidden 和 32 MiB
// index selection；旧 128 MiB 上限连固定 4 字节 aux metadata 都容不下。
// 这里为更大 chunk 留出余量，但发送端也必须先检查，不能写坏共享 stream。
const MAX_STAGE_PAYLOAD: usize = 512 * 1024 * 1024;

const PREFILL: u16 = 1;
const PREFILL_DONE: u16 = 2;
const DECODE: u16 = 3;
const TOKEN: u16 = 4;
const DELETE: u16 = 5;
const CACHE: u16 = 6;
const OPEN: u16 = 7;
const SWAP_OUT: u16 = 8;
const READY: u16 = 9;
const PERSIST: u16 = 10;
const DEVICE_MEMORY: u16 = 11;
const MTP_CONTEXT: u16 = 12;
const VERIFY: u16 = 13;
const SPECULATIVE: u16 = 14;
const SAMPLE: u16 = 15;
const SAMPLED: u16 = 16;
const SHUTDOWN: u16 = 17;

/// 下游 stage 在常驻权重与输出头完成分配后上报的设备资源快照。
/// `model_units` 由模型 runtime 填入该卡负责的连续计算单元数（Transformer
/// 模型即层数）；传输层和 backend 不解释其模型语义。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageDeviceMemory {
    pub device: i32,
    pub model_units: usize,
    pub available_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId([u8; 16]);

impl RequestId {
    pub fn parse(value: &str) -> Result<Self, String> {
        let value = value.trim().strip_prefix("req_").unwrap_or(value.trim()).replace('-', "");
        if value.len() != 32 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("request_id 必须是 32 位十六进制或 UUID".to_owned());
        }
        let mut bytes = [0_u8; 16];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|error| format!("request_id 解析失败: {error}"))?;
        }
        Ok(Self(bytes))
    }

    pub fn generate_for_test() -> Self {
        let value = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        Self(value.to_le_bytes())
    }

    /// 从 OpenAI `cache_id` 派生 16 字节 `RequestId`，用于 PP 链传播。
    ///
    /// 链头把 `cache_id`(blake3 hex 或客户端自定义串)映射成 `RequestId`，
    /// 随 `StageFrame` 下发给下游；下游 stage 据此 keying 自己的 `TerminalCache`。
    /// 相同 `cache_id` → 相同 `RequestId`(确定性)，让跨请求的 resume 在全链一致命中。
    pub fn from_cache_id(cache_id: &str) -> Self {
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&blake3::hash(cache_id.as_bytes()).as_bytes()[..16]);
        Self(bytes)
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

const STREAM: u16 = 0x100;
const STREAM_ASSIGN: u16 = 0x101;
const CONTINUOUS_STREAM: u16 = 0x102;

#[derive(Debug)]
pub enum StageMessage {
    /// 后续按此顺序流入每个 request 的 hidden；接收端收到首帧即可启动 stage。
    Stream {
        requests: Vec<RequestId>,
    },
    ContinuousStream {
        requests: Vec<RequestId>,
    },
    /// continuous stream 内把新 request 放入已释放的 session slot。
    StreamAssign {
        session: usize,
    },
    Prefill {
        position: usize,
        rows: usize,
        cols: usize,
        values: Vec<u16>,
        selection: Vec<u32>,
        aux_values: Vec<u16>,
        aux_taps: usize,
    },
    PrefillDone {
        tokens: usize,
    },
    Decode {
        cohort: u64,
        cohort_size: usize,
        position: usize,
        cols: usize,
        values: Vec<u16>,
        selection: Vec<u32>,
        aux_values: Vec<u16>,
        aux_taps: usize,
    },
    /// MTP 用完整 prompt token 建立移位输入。
    MtpContext {
        prompt_tokens: Vec<u32>,
        max_decode: usize,
        draft_tokens: usize,
    },
    /// `[当前 token, drafts...]` 一次走完整 target causal append。
    Verify {
        cohort: u64,
        cohort_size: usize,
        position: usize,
        rows: usize,
        cols: usize,
        values: Vec<u16>,
        selection: Vec<u32>,
        aux_values: Vec<u16>,
        aux_taps: usize,
    },
    Token {
        token: u32,
        eos: bool,
    },
    /// 一轮可提交多个输出；`retained_rows` 是上一轮 verify 真正保留的输入行数。
    Speculative {
        tokens: Vec<u32>,
        retained_rows: usize,
        drafts: Vec<u32>,
        eos: bool,
    },
    Delete,
    Cache {
        next_request_id: RequestId,
        tokens: usize,
    },
    /// 前机指定本轮从既有 cache 恢复或从空状态开始。
    Open {
        cache_request_id: Option<RequestId>,
        cached_tokens: usize,
        /// 本轮已经计入 admission 的 cache 行容量；0 兼容旧发送端。
        reserved_rows: usize,
        cache_hit: bool,
        sampling: SamplingConfig,
        tail_sampling: bool,
    },
    /// 链头按当前生成状态下发本步围栏；随机数仍由末段的 SamplingState 产生。
    Sample {
        excluded: Vec<u32>,
    },
    /// 末段完成 final norm、LM head 与采样后只回传 token。
    Sampled {
        cohort: u64,
        cohort_size: usize,
        position: usize,
        token: u32,
        eos: bool,
    },
    /// 前机命令后继把当前 resident session 换出到本机 SSD。
    SwapOut,
    /// 后继完成控制命令后的确认；前机收到后才发送 hidden。
    Ready {
        cached_tokens: usize,
    },
    /// 前机通知后继把单个 resident session 镜像到 SSD。
    Persist,
    /// 链头开始优雅退出；后继按策略持久化全部 resident cache，回复 Ready 后自行退出。
    Shutdown {
        persist: bool,
    },
    DeviceMemory {
        devices: Vec<StageDeviceMemory>,
        /// 下游可同时容纳的 resident pipeline session。旧端不报告时为 None。
        session_capacity: Option<usize>,
    },
}

#[derive(Debug)]
pub struct StageFrame {
    pub request_id: RequestId,
    pub message: StageMessage,
}

pub struct StageTransport {
    runtime: Runtime,
    _endpoint: Endpoint,
    _connection: Connection,
    send: SendStream,
    recv: mpsc::Receiver<Result<(StageFrame, Instant), String>>,
    listener_connections: Option<mpsc::Receiver<(Connection, SendStream, RecvStream)>>,
    pending_listener_connection: Option<(Connection, SendStream, RecvStream)>,
    downstream: Option<(EndpointAddr, Option<String>)>,
}

/// 已发布 ticket、尚未接受上游连接的 stage 监听端。
pub struct StageListener {
    runtime: Runtime,
    endpoint: Endpoint,
    expected: Option<String>,
}

impl Drop for StageTransport {
    fn drop(&mut self) {
        self._connection.close(0u32.into(), b"stage transport shutdown");
    }
}

impl StageListener {
    pub fn accept(self) -> Result<StageTransport, String> {
        let Self { runtime, endpoint, expected } = self;
        let listener_connections = start_listener(&runtime, endpoint.clone(), expected);
        let (connection, send, recv) = listener_connections.recv().map_err(|_| "stage accept 任务已经退出".to_owned())?;
        let recv = start_receiver(&runtime, connection.clone(), recv);
        Ok(StageTransport { runtime, _endpoint: endpoint, _connection: connection, send, recv, listener_connections: Some(listener_connections), pending_listener_connection: None, downstream: None })
    }
}

impl StageTransport {
    pub fn send_stream_assign(&mut self, request_id: RequestId, session: usize) -> Result<(), String> {
        self.send_frame(request_id, STREAM_ASSIGN, session, 0, 0, 0, &[])
    }

    pub fn send_continuous_stream(&mut self, requests: &[RequestId]) -> Result<(), String> {
        let Some(&first) = requests.first() else {
            return Err("continuous stage stream requests 不能为空".to_owned());
        };
        let mut payload = Vec::with_capacity(requests.len() * 16);
        for request in requests {
            payload.extend_from_slice(&request.0);
        }
        self.send_frame(first, CONTINUOUS_STREAM, requests.len(), 0, 0, 0, &payload)
    }

    pub fn send_continuous_stream_end(&mut self, request_id: RequestId) -> Result<(), String> {
        self.send_frame(request_id, CONTINUOUS_STREAM, 0, 0, 0, 0, &[])
    }

    pub fn bind(config: IrohConfig) -> Result<StageListener, String> {
        let secret = config.secret_key.ok_or("stage listener 必须配置固定 secret key")?;
        let node_id = secret.public().to_string();
        let expected = config.expected_peer;
        let bind_addr = config.bind_addr;
        let runtime = runtime()?;
        let endpoint = runtime.block_on(async move {
            let mut builder = Endpoint::builder(presets::N0).secret_key(secret).alpns(vec![STAGE_ALPN.to_vec()]).transport_config(stage_quic_transport_config()?);
            if let Some(bind_addr) = bind_addr {
                // 固定身份还必须固定 UDP 端口，旧 ticket 才能跨进程重启继续连接。
                builder = builder.clear_ip_transports().bind_addr(bind_addr.as_str()).map_err(|error| format!("解析 stage iroh.bind_addr={bind_addr}: {error}"))?;
            }
            let endpoint = builder.bind().await.map_err(|error| format!("绑定 stage endpoint: {error}"))?;
            #[cfg(not(test))]
            tokio::time::timeout(std::time::Duration::from_secs(15), endpoint.online()).await.map_err(|_| "stage listener 连接 iroh 官方 relay 超时".to_owned())?;
            Ok::<_, String>(endpoint)
        })?;
        let ticket = EndpointTicket::new(endpoint.addr()).to_string();
        eprintln!("[stage-listen] node_id={node_id} ticket={ticket}");
        Ok(StageListener { runtime, endpoint, expected })
    }

    pub fn listen(config: IrohConfig) -> Result<Self, String> {
        Self::bind(config)?.accept()
    }

    pub fn connect(ticket: &str, config: IrohConfig) -> Result<Self, String> {
        let ticket = EndpointTicket::from_str(ticket).map_err(|error| format!("解析 stage ticket: {error}"))?;
        let address = ticket.endpoint_addr().clone();
        let expected = config.expected_peer;
        let downstream = (address.clone(), expected.clone());
        let secret = config.secret_key;
        let bind_addr = config.bind_addr;
        let runtime = runtime()?;
        let (endpoint, connection, send, recv) = runtime.block_on(async move {
            let mut builder = Endpoint::builder(presets::N0).transport_config(stage_quic_transport_config()?);
            if let Some(secret) = secret {
                builder = builder.secret_key(secret);
            }
            if let Some(bind_addr) = bind_addr {
                builder = builder.clear_ip_transports().bind_addr(bind_addr.as_str()).map_err(|error| format!("解析 stage client iroh.bind_addr={bind_addr}: {error}"))?;
            }
            let endpoint = builder.bind().await.map_err(|error| format!("绑定 stage client endpoint: {error}"))?;
            let connection = loop {
                match endpoint.connect(address.clone(), STAGE_ALPN).await {
                    Ok(connection) => break connection,
                    Err(error) => {
                        eprintln!("[stage-connect-wait] 下游尚未就绪: {error}");
                        tokio::time::sleep(Duration::from_millis(200)).await;
                    }
                }
            };
            super::iroh::validate_peer(&connection, expected.as_deref(), "downstream")?;
            let (mut send, recv) = connection.open_bi().await.map_err(|error| format!("打开 stage stream: {error}"))?;
            send.write_all(&[STREAM_OPEN]).await.map_err(|error| format!("写入 stage stream opener: {error}"))?;
            send.flush().await.map_err(|error| format!("flush stage stream opener: {error}"))?;
            Ok::<_, String>((endpoint, connection, send, recv))
        })?;
        let recv = start_receiver(&runtime, connection.clone(), recv);
        Ok(Self { runtime, _endpoint: endpoint, _connection: connection, send, recv, listener_connections: None, pending_listener_connection: None, downstream: Some(downstream) })
    }

    /// 连接端复用原固定 UDP endpoint 建立新连接；不会触碰 listener 端的
    /// resident session。模型 runtime 仍需重新接收 DeviceMemory 握手后再发工作。
    pub fn reconnect_downstream(&mut self) -> Result<(), String> {
        let (address, expected) = self.downstream.clone().ok_or("stage listener 端不能主动重连下游")?;
        let (connection, send, recv) = self.runtime.block_on(async {
            let connection = loop {
                match self._endpoint.connect(address.clone(), STAGE_ALPN).await {
                    Ok(connection) => break connection,
                    Err(error) => {
                        eprintln!("[stage-reconnect-wait] 下游尚未就绪: {error}");
                        tokio::time::sleep(Duration::from_millis(200)).await;
                    }
                }
            };
            super::iroh::validate_peer(&connection, expected.as_deref(), "downstream")?;
            let (mut send, recv) = connection.open_bi().await.map_err(|error| format!("重开 stage stream: {error}"))?;
            send.write_all(&[STREAM_OPEN]).await.map_err(|error| format!("写入重连 stage stream opener: {error}"))?;
            send.flush().await.map_err(|error| format!("flush 重连 stage stream opener: {error}"))?;
            Ok::<_, String>((connection, send, recv))
        })?;
        self._connection.close(0u32.into(), b"downstream reconnect");
        self._connection = connection.clone();
        self.send = send;
        self.recv = start_receiver(&self.runtime, connection.clone(), recv);
        eprintln!("[stage-downstream-reconnected] 新连接已建立");
        Ok(())
    }

    /// listener 端保留原 endpoint 与 resident state，始终让最后到达的上游接管。
    pub fn accept_reconnect(&mut self) -> Result<(), String> {
        let connections = self.listener_connections.as_ref().ok_or("stage connect 端不能接受上游重连")?;
        let mut next = match self.pending_listener_connection.take() {
            Some(next) => next,
            None => connections.recv().map_err(|_| "stage accept 任务已经退出".to_owned())?,
        };
        while let Ok(newer) = connections.try_recv() {
            next.0.close(0u32.into(), b"superseded before activation");
            next = newer;
        }
        self._connection.close(0u32.into(), b"superseded by new upstream");
        let (connection, send, recv) = next;
        self._connection = connection.clone();
        self.send = send;
        self.recv = start_receiver(&self.runtime, connection.clone(), recv);
        eprintln!("[stage-upstream-takeover] 新上游已接管");
        Ok(())
    }

    /// 新连接先于旧连接的待收 frame 生效；调用方收到错误后回到统一 reconnect 点。
    fn notice_new_listener_connection(&mut self) -> bool {
        let Some(connections) = self.listener_connections.as_ref() else {
            return false;
        };
        let Ok(mut next) = connections.try_recv() else {
            return false;
        };
        while let Ok(newer) = connections.try_recv() {
            next.0.close(0u32.into(), b"superseded before activation");
            next = newer;
        }
        if let Some(pending) = self.pending_listener_connection.replace(next) {
            pending.0.close(0u32.into(), b"superseded before activation");
        }
        self._connection.close(0u32.into(), b"superseded by new upstream");
        true
    }

    pub fn send_prefill(&mut self, request_id: RequestId, position: usize, rows: usize, cols: usize, values: &[u16], selection: &[u32]) -> Result<(), String> {
        self.send_hidden(request_id, PREFILL, None, position, rows, cols, values, selection, &[], 0)
    }

    pub fn send_prefill_aux(&mut self, request_id: RequestId, position: usize, rows: usize, cols: usize, values: &[u16], selection: &[u32], aux_values: &[u16], aux_taps: usize) -> Result<(), String> {
        self.send_hidden(request_id, PREFILL, None, position, rows, cols, values, selection, aux_values, aux_taps)
    }

    pub fn send_prefill_done(&mut self, request_id: RequestId, tokens: usize) -> Result<(), String> {
        self.send_frame(request_id, PREFILL_DONE, tokens, 0, 0, 0, &[])
    }

    pub fn send_decode(&mut self, request_id: RequestId, position: usize, cols: usize, values: &[u16], selection: &[u32]) -> Result<(), String> {
        self.send_decode_cohort(request_id, 0, 1, position, cols, values, selection)
    }

    pub fn send_decode_cohort(&mut self, request_id: RequestId, cohort: u64, cohort_size: usize, position: usize, cols: usize, values: &[u16], selection: &[u32]) -> Result<(), String> {
        self.send_hidden(request_id, DECODE, Some((cohort, cohort_size)), position, 1, cols, values, selection, &[], 0)
    }

    pub fn send_decode_cohort_aux(&mut self, request_id: RequestId, cohort: u64, cohort_size: usize, position: usize, cols: usize, values: &[u16], selection: &[u32], aux_values: &[u16], aux_taps: usize) -> Result<(), String> {
        self.send_hidden(request_id, DECODE, Some((cohort, cohort_size)), position, 1, cols, values, selection, aux_values, aux_taps)
    }

    pub fn send_verify(&mut self, request_id: RequestId, position: usize, rows: usize, cols: usize, values: &[u16], selection: &[u32]) -> Result<(), String> {
        self.send_verify_cohort(request_id, 0, 1, position, rows, cols, values, selection)
    }

    pub fn send_verify_cohort(&mut self, request_id: RequestId, cohort: u64, cohort_size: usize, position: usize, rows: usize, cols: usize, values: &[u16], selection: &[u32]) -> Result<(), String> {
        if rows == 0 {
            return Err("MTP verify rows 必须大于 0".to_owned());
        }
        self.send_hidden(request_id, VERIFY, Some((cohort, cohort_size)), position, rows, cols, values, selection, &[], 0)
    }

    pub fn send_verify_cohort_aux(&mut self, request_id: RequestId, cohort: u64, cohort_size: usize, position: usize, rows: usize, cols: usize, values: &[u16], selection: &[u32], aux_values: &[u16], aux_taps: usize) -> Result<(), String> {
        if rows == 0 {
            return Err("verify rows 必须大于 0".to_owned());
        }
        self.send_hidden(request_id, VERIFY, Some((cohort, cohort_size)), position, rows, cols, values, selection, aux_values, aux_taps)
    }

    pub fn send_mtp_context(&mut self, request_id: RequestId, prompt_tokens: &[u32], max_decode: usize, draft_tokens: usize) -> Result<(), String> {
        if prompt_tokens.is_empty() || max_decode == 0 || draft_tokens == 0 {
            return Err("MTP context 的 prompt、max_decode 与 draft_tokens 必须非空".to_owned());
        }
        let payload = encode_u32_values(prompt_tokens);
        self.send_frame(request_id, MTP_CONTEXT, max_decode, draft_tokens, 0, 0, &payload)
    }

    pub fn send_token(&mut self, request_id: RequestId, token: u32, eos: bool) -> Result<(), String> {
        self.send_frame(request_id, TOKEN, 0, u32::from(eos) as usize, 0, token, &[])
    }

    pub fn send_speculative(&mut self, request_id: RequestId, tokens: &[u32], retained_rows: usize, drafts: &[u32], eos: bool) -> Result<(), String> {
        if tokens.is_empty() || (retained_rows != 0 && retained_rows != tokens.len()) {
            return Err(format!("MTP speculative 输出非法: tokens={} retained_rows={retained_rows}", tokens.len()));
        }
        let mut payload = encode_u32_values(tokens);
        payload.extend_from_slice(&encode_u32_values(drafts));
        self.send_frame(request_id, SPECULATIVE, retained_rows, tokens.len(), drafts.len(), u32::from(eos), &payload)
    }

    pub fn send_sample(&mut self, request_id: RequestId, excluded: &[u32]) -> Result<(), String> {
        self.send_frame(request_id, SAMPLE, 0, 0, excluded.len(), 0, &encode_u32_values(excluded))
    }

    pub fn send_sampled(&mut self, request_id: RequestId, cohort: u64, cohort_size: usize, position: usize, token: u32, eos: bool) -> Result<(), String> {
        if cohort_size == 0 || (cohort == 0 && cohort_size != 1) {
            return Err(format!("sampled cohort 参数非法: id={cohort} size={cohort_size}"));
        }
        self.send_frame(request_id, SAMPLED, position, cohort_size, usize::from(eos), token, &cohort.to_le_bytes())
    }

    #[track_caller]
    pub fn send_delete(&mut self, request_id: RequestId) -> Result<(), String> {
        let caller = std::panic::Location::caller();
        eprintln!("[stage-delete] request_id={request_id} caller={}:{}", caller.file(), caller.line());
        self.send_frame(request_id, DELETE, 0, 0, 0, 0, &[])
    }

    /// 把当前 PP session 改名为新的 terminal cache id，供下一次追加 prefill 命中。
    pub fn send_cache(&mut self, request_id: RequestId, next_request_id: RequestId, tokens: usize) -> Result<(), String> {
        self.send_frame(request_id, CACHE, tokens, 0, 0, 0, &next_request_id.0)
    }

    pub fn send_open(&mut self, request_id: RequestId, cache_request_id: Option<RequestId>, cached_tokens: usize, cache_hit: bool, sampling: SamplingConfig, tail_sampling: bool) -> Result<(), String> {
        self.send_open_reserved(request_id, cache_request_id, cached_tokens, 0, cache_hit, sampling, tail_sampling)
    }

    pub fn send_open_reserved(&mut self, request_id: RequestId, cache_request_id: Option<RequestId>, cached_tokens: usize, reserved_rows: usize, cache_hit: bool, sampling: SamplingConfig, tail_sampling: bool) -> Result<(), String> {
        let mut payload = encode_sampling(sampling)?.to_vec();
        if let Some(id) = cache_request_id {
            payload.extend_from_slice(&id.0);
        }
        self.send_frame(request_id, OPEN, cached_tokens, usize::from(cache_hit), usize::from(tail_sampling), u32::try_from(reserved_rows).map_err(|_| "stage Open reserved_rows 超过 u32".to_owned())?, &payload)
    }

    pub fn send_swap_out(&mut self, request_id: RequestId) -> Result<(), String> {
        self.send_frame(request_id, SWAP_OUT, 0, 0, 0, 0, &[])
    }

    pub fn send_ready(&mut self, request_id: RequestId, cached_tokens: usize) -> Result<(), String> {
        self.send_frame(request_id, READY, cached_tokens, 0, 0, 0, &[])
    }

    pub fn send_persist(&mut self, request_id: RequestId) -> Result<(), String> {
        self.send_frame(request_id, PERSIST, 0, 0, 0, 0, &[])
    }

    pub fn send_shutdown(&mut self, request_id: RequestId, persist: bool) -> Result<(), String> {
        self.send_frame(request_id, SHUTDOWN, usize::from(persist), 0, 0, 0, &[])
    }

    pub fn send_device_memory(&mut self, devices: &[StageDeviceMemory], session_capacity: Option<usize>) -> Result<(), String> {
        if devices.is_empty() {
            return Err("stage device memory 不能为空".to_owned());
        }
        if session_capacity == Some(0) {
            return Err("stage session capacity 必须大于 0".to_owned());
        }
        let payload = encode_device_memory(devices)?;
        self.send_frame(RequestId([0; 16]), DEVICE_MEMORY, devices.len(), session_capacity.unwrap_or(0), 0, 0, &payload)
    }

    pub fn recv(&mut self) -> Result<StageFrame, String> {
        loop {
            if self.notice_new_listener_connection() {
                return Err("检测到新上游连接，旧连接已关闭".to_owned());
            }
            match self.recv.recv_timeout(Duration::from_millis(20)) {
                Ok(frame) => return frame.map(|(frame, _)| frame),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err("stage 接收任务已经退出".to_owned()),
            }
        }
    }

    pub fn try_recv(&mut self) -> Result<Option<StageFrame>, String> {
        if self.notice_new_listener_connection() {
            return Err("检测到新上游连接，旧连接已关闭".to_owned());
        }
        match self.recv.try_recv() {
            Ok(frame) => frame.map(|(frame, _)| Some(frame)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("stage 接收任务已经退出".to_owned()),
        }
    }

    /// 返回 frame 以及异步收包任务完成解析后在本地队列中的驻留时间。
    /// 仅供调度诊断使用，避免用跨机器 wall clock 混淆网络与协调线程阻塞。
    pub fn try_recv_timed(&mut self) -> Result<Option<(StageFrame, Duration)>, String> {
        if self.notice_new_listener_connection() {
            return Err("检测到新上游连接，旧连接已关闭".to_owned());
        }
        match self.recv.try_recv() {
            Ok(frame) => frame.map(|(frame, received)| Some((frame, received.elapsed()))),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("stage 接收任务已经退出".to_owned()),
        }
    }

    pub fn connection_diagnostics(&self) -> String {
        connection_diagnostics(&self._connection)
    }

    #[allow(clippy::too_many_arguments)]
    fn send_hidden(&mut self, request_id: RequestId, kind: u16, cohort: Option<(u64, usize)>, position: usize, rows: usize, cols: usize, values: &[u16], selection: &[u32], aux_values: &[u16], aux_taps: usize) -> Result<(), String> {
        if let Some((cohort, cohort_size)) = cohort
            && (cohort_size == 0 || (cohort == 0 && cohort_size != 1))
        {
            return Err(format!("stage cohort 参数非法: id={cohort} size={cohort_size}"));
        }
        let expected = rows.checked_mul(cols).ok_or("stage hidden 元素数溢出")?;
        if values.len() != expected {
            return Err(format!("stage hidden shape=[{rows},{cols}]，实际元素={}", values.len()));
        }
        if (!aux_values.is_empty() && aux_values.len() != expected) || (aux_values.is_empty() != (aux_taps == 0)) {
            return Err(format!("stage aux hidden shape/taps 非法: values={} expected={expected} taps={aux_taps}", aux_values.len()));
        }
        let payload = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
        let selection_payload = unsafe { std::slice::from_raw_parts(selection.as_ptr().cast::<u8>(), std::mem::size_of_val(selection)) };
        let aux_payload = unsafe { std::slice::from_raw_parts(aux_values.as_ptr().cast::<u8>(), std::mem::size_of_val(aux_values)) };
        let mut trailing = Vec::with_capacity(selection_payload.len() + 4 + aux_payload.len() + usize::from(cohort.is_some()) * 12);
        trailing.extend_from_slice(selection_payload);
        trailing.extend_from_slice(&u32::try_from(aux_taps).map_err(|_| "stage aux taps 超过 u32")?.to_le_bytes());
        trailing.extend_from_slice(aux_payload);
        if let Some((cohort, cohort_size)) = cohort {
            trailing.extend_from_slice(&cohort.to_le_bytes());
            trailing.extend_from_slice(&u32::try_from(cohort_size).map_err(|_| "stage cohort size 超过 u32")?.to_le_bytes());
        }
        let selection_count = u32::try_from(selection.len()).map_err(|_| "stage selection 元素数超过 u32")?;
        self.send_frame_parts(request_id, kind, position, rows, cols, selection_count, payload, &trailing)
    }

    #[allow(clippy::too_many_arguments)]
    fn send_frame(&mut self, request_id: RequestId, kind: u16, position: usize, rows: usize, cols: usize, value: u32, payload: &[u8]) -> Result<(), String> {
        self.send_frame_parts(request_id, kind, position, rows, cols, value, payload, &[])
    }

    #[allow(clippy::too_many_arguments)]
    fn send_frame_parts(&mut self, request_id: RequestId, kind: u16, position: usize, rows: usize, cols: usize, value: u32, payload: &[u8], trailing: &[u8]) -> Result<(), String> {
        let position = u64::try_from(position).map_err(|_| "stage position 超过 u64")?;
        let rows = u32::try_from(rows).map_err(|_| "stage rows 超过 u32")?;
        let cols = u32::try_from(cols).map_err(|_| "stage cols 超过 u32")?;
        let payload_len = payload.len().checked_add(trailing.len()).ok_or("stage payload 大小溢出")?;
        validate_stage_payload_len(payload_len)?;
        let payload_len = u32::try_from(payload_len).map_err(|_| "stage payload 超过 u32")?;
        let mut header = [0_u8; HEADER_BYTES];
        header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        header[4..6].copy_from_slice(&VERSION.to_le_bytes());
        header[6..8].copy_from_slice(&kind.to_le_bytes());
        header[8..24].copy_from_slice(&request_id.0);
        header[24..32].copy_from_slice(&position.to_le_bytes());
        header[32..36].copy_from_slice(&rows.to_le_bytes());
        header[36..40].copy_from_slice(&cols.to_le_bytes());
        header[40..44].copy_from_slice(&value.to_le_bytes());
        header[44..48].copy_from_slice(&payload_len.to_le_bytes());
        let send = &mut self.send;
        let started = Instant::now();
        let result = self.runtime.block_on(async move {
            send.write_all(&header).await.map_err(|error| format!("写入 stage header: {error}"))?;
            send.write_all(payload).await.map_err(|error| format!("写入 stage payload: {error}"))?;
            send.write_all(trailing).await.map_err(|error| format!("写入 stage trailing payload: {error}"))?;
            send.flush().await.map_err(|error| format!("flush stage stream: {error}"))
        });
        let elapsed = started.elapsed();
        if elapsed >= Duration::from_millis(500) {
            eprintln!("[stage-send-slow] request_id={request_id} kind={kind} position={position} rows={rows} payload_bytes={payload_len} wall_ms={:.3}", elapsed.as_secs_f64() * 1000.0,);
        }
        result
    }
}

fn connection_diagnostics(connection: &Connection) -> String {
    let stats = connection.stats();
    let paths = connection
        .paths()
        .iter()
        .map(|path| {
            let stats = path.stats();
            format!(
                "selected={} direct={} remote={:?} rtt_ms={:.3} cwnd={} congestion={} spurious={} lost_packets={} lost_bytes={} mtu={}",
                path.is_selected(),
                path.is_ip(),
                path.remote_addr(),
                path.rtt().as_secs_f64() * 1000.0,
                stats.cwnd,
                stats.congestion_events,
                stats.spurious_congestion_events,
                stats.lost_packets,
                stats.lost_bytes,
                stats.current_mtu,
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    format!(
        "lost_packets={} lost_bytes={} stream_blocked_tx={} stream_blocked_rx={} max_stream_data_tx={} max_stream_data_rx={} paths=[{paths}]",
        stats.lost_packets, stats.lost_bytes, stats.frame_tx.stream_data_blocked, stats.frame_rx.stream_data_blocked, stats.frame_tx.max_stream_data, stats.frame_rx.max_stream_data,
    )
}

fn start_receiver(runtime: &Runtime, connection: Connection, recv: RecvStream) -> mpsc::Receiver<Result<(StageFrame, Instant), String>> {
    let (frames, receiver) = mpsc::channel();
    runtime.spawn(async move {
        let mut recv = recv;
        loop {
            let frame = receive_frame(&mut recv).await.map(|frame| (frame, Instant::now())).map_err(|error| format!("{error}; close_reason={:?}; {}", connection.close_reason(), connection_diagnostics(&connection)));
            let failed = frame.is_err();
            if frames.send(frame).is_err() || failed {
                break;
            }
        }
    });
    receiver
}

fn start_listener(runtime: &Runtime, endpoint: Endpoint, expected: Option<String>) -> mpsc::Receiver<(Connection, SendStream, RecvStream)> {
    let (connections, receiver) = mpsc::channel();
    runtime.spawn(async move {
        loop {
            let Some(accepting) = endpoint.accept().await else {
                break;
            };
            let connection = match accepting.await {
                Ok(connection) => connection,
                Err(error) => {
                    eprintln!("[stage-accept-wait] 上游尚未就绪: {error}");
                    continue;
                }
            };
            if let Err(error) = super::iroh::validate_peer(&connection, expected.as_deref(), "upstream") {
                eprintln!("[stage-accept-wait] {error}");
                connection.close(0u32.into(), b"unexpected upstream");
                continue;
            }
            match connection.accept_bi().await {
                Ok((send, mut recv)) => {
                    let mut opener = [0_u8; 1];
                    match recv.read_exact(&mut opener).await {
                        Ok(_) if opener[0] == STREAM_OPEN => {
                            if connections.send((connection, send, recv)).is_err() {
                                break;
                            }
                        }
                        Ok(_) => eprintln!("[stage-accept-wait] 上游 stream opener 不匹配"),
                        Err(error) => eprintln!("[stage-accept-wait] 读取上游 stream opener: {error}"),
                    }
                }
                Err(error) => eprintln!("[stage-accept-wait] 上游 stream 尚未就绪: {error}"),
            }
        }
    });
    receiver
}

/// payload 只封顶防 48 字节 header 声称超大长度;Token/Delete/Ready 等
/// 控制帧是合法的零载荷帧,不能在长度上拒绝。
fn validate_stage_payload_len(payload_len: usize) -> Result<(), String> {
    if payload_len > MAX_STAGE_PAYLOAD {
        return Err(format!("stage payload 长度 {payload_len} 非法(上限 {MAX_STAGE_PAYLOAD})"));
    }
    Ok(())
}

async fn receive_frame(recv: &mut RecvStream) -> Result<StageFrame, String> {
    let mut header = [0_u8; HEADER_BYTES];
    recv.read_exact(&mut header).await.map_err(|error| format!("读取 stage header: {error}"))?;
    if u32::from_le_bytes(header[0..4].try_into().unwrap()) != MAGIC {
        return Err("stage frame magic 不匹配".to_owned());
    }
    if u16::from_le_bytes(header[4..6].try_into().unwrap()) != VERSION {
        return Err("stage frame version 不匹配".to_owned());
    }
    let kind = u16::from_le_bytes(header[6..8].try_into().unwrap());
    let request_id = RequestId(header[8..24].try_into().unwrap());
    let position = usize::try_from(u64::from_le_bytes(header[24..32].try_into().unwrap())).map_err(|_| "stage position 超过 usize")?;
    let rows = u32::from_le_bytes(header[32..36].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(header[36..40].try_into().unwrap()) as usize;
    let value = u32::from_le_bytes(header[40..44].try_into().unwrap());
    let payload_len = u32::from_le_bytes(header[44..48].try_into().unwrap()) as usize;
    validate_stage_payload_len(payload_len).map_err(|message| format!("{message}; kind={kind} request_id={request_id} position={position} rows={rows} cols={cols} value={value}"))?;
    let mut payload = vec![0_u8; payload_len];
    recv.read_exact(&mut payload).await.map_err(|error| format!("读取 stage payload: {error}"))?;
    match kind {
        PREFILL | DECODE | VERIFY => {
            let elements = rows.checked_mul(cols).ok_or("stage hidden 元素数溢出")?;
            let hidden_bytes = elements.checked_mul(2).ok_or("stage hidden 字节数溢出")?;
            let selection_bytes = (value as usize).checked_mul(4).ok_or("stage selection 字节数溢出")?;
            let cohort_bytes = usize::from(kind != PREFILL) * 12;
            let aux_taps_offset = hidden_bytes.checked_add(selection_bytes).ok_or("stage aux metadata 偏移溢出")?;
            if payload_len < aux_taps_offset + 4 {
                return Err("stage aux metadata 缺失".to_owned());
            }
            let aux_taps = u32::from_le_bytes(payload[aux_taps_offset..aux_taps_offset + 4].try_into().unwrap()) as usize;
            let aux_bytes = usize::from(aux_taps != 0).checked_mul(hidden_bytes).ok_or("stage aux hidden 字节数溢出")?;
            if payload_len
                != hidden_bytes.checked_add(selection_bytes).and_then(|bytes| bytes.checked_add(4)).and_then(|bytes| bytes.checked_add(aux_bytes)).and_then(|bytes| bytes.checked_add(cohort_bytes)).ok_or("stage payload 字节数溢出")?
            {
                return Err(format!("stage hidden 字节数错误: payload={payload_len} shape=[{rows},{cols}]"));
            }
            let values = payload[..hidden_bytes].chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect();
            let selection_end = hidden_bytes + selection_bytes;
            let selection = payload[hidden_bytes..selection_end].chunks_exact(4).map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())).collect();
            let aux_start = selection_end + 4;
            let aux_end = aux_start + aux_bytes;
            let aux_values = payload[aux_start..aux_end].chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect();
            let cohort = (kind != PREFILL).then(|| u64::from_le_bytes(payload[aux_end..aux_end + 8].try_into().unwrap())).unwrap_or(0);
            let cohort_size = (kind != PREFILL).then(|| u32::from_le_bytes(payload[aux_end + 8..aux_end + 12].try_into().unwrap()) as usize).unwrap_or(1);
            if kind != PREFILL && (cohort_size == 0 || (cohort == 0 && cohort_size != 1)) {
                return Err(format!("stage cohort 参数非法: id={cohort} size={cohort_size}"));
            }
            let message = match kind {
                PREFILL => StageMessage::Prefill { position, rows, cols, values, selection, aux_values, aux_taps },
                DECODE if rows == 1 => StageMessage::Decode { cohort, cohort_size, position, cols, values, selection, aux_values, aux_taps },
                DECODE => return Err(format!("stage decode rows={rows}，期望 1")),
                VERIFY if rows > 0 => StageMessage::Verify { cohort, cohort_size, position, rows, cols, values, selection, aux_values, aux_taps },
                VERIFY => return Err("MTP verify rows 必须大于 0".to_owned()),
                _ => unreachable!(),
            };
            Ok(StageFrame { request_id, message })
        }
        STREAM if payload_len == position.checked_mul(16).ok_or("stage stream request 数溢出")? => {
            let requests = payload.chunks_exact(16).map(|bytes| RequestId(bytes.try_into().unwrap())).collect();
            Ok(StageFrame { request_id, message: StageMessage::Stream { requests } })
        }
        STREAM_ASSIGN if payload_len == 0 => Ok(StageFrame { request_id, message: StageMessage::StreamAssign { session: position } }),
        CONTINUOUS_STREAM if payload_len == position.checked_mul(16).ok_or("continuous stage stream request 数溢出")? => {
            let requests = payload.chunks_exact(16).map(|bytes| RequestId(bytes.try_into().unwrap())).collect();
            Ok(StageFrame { request_id, message: StageMessage::ContinuousStream { requests } })
        }
        PREFILL_DONE if payload_len == 0 => Ok(StageFrame { request_id, message: StageMessage::PrefillDone { tokens: position } }),
        TOKEN if payload_len == 0 => Ok(StageFrame { request_id, message: StageMessage::Token { token: value, eos: rows != 0 } }),
        SAMPLE if rows == 0 && value == 0 && payload_len == cols.checked_mul(4).ok_or("sample payload 长度溢出")? => Ok(StageFrame { request_id, message: StageMessage::Sample { excluded: decode_u32_values(&payload) } }),
        SAMPLED if payload_len == 8 && rows > 0 && cols <= 1 => {
            let cohort = u64::from_le_bytes(payload.as_slice().try_into().unwrap());
            if cohort == 0 && rows != 1 {
                return Err(format!("sampled cohort 参数非法: id={cohort} size={rows}"));
            }
            Ok(StageFrame { request_id, message: StageMessage::Sampled { cohort, cohort_size: rows, position, token: value, eos: cols != 0 } })
        }
        MTP_CONTEXT => decode_mtp_context(position, rows, cols, value, &payload).map(|message| StageFrame { request_id, message }),
        SPECULATIVE => decode_speculative(position, rows, cols, value, &payload).map(|message| StageFrame { request_id, message }),
        DELETE if payload_len == 0 => Ok(StageFrame { request_id, message: StageMessage::Delete }),
        CACHE if payload_len == 16 => Ok(StageFrame { request_id, message: StageMessage::Cache { next_request_id: RequestId(payload.as_slice().try_into().unwrap()), tokens: position } }),
        OPEN if (payload_len == 16 || payload_len == 32) && cols <= 1 => Ok(StageFrame {
            request_id,
            message: StageMessage::Open {
                sampling: decode_sampling(&payload[..16])?,
                cache_request_id: (payload_len == 32).then(|| RequestId(payload[16..32].try_into().unwrap())),
                cached_tokens: position,
                reserved_rows: value as usize,
                cache_hit: rows != 0,
                tail_sampling: cols != 0,
            },
        }),
        SWAP_OUT if payload_len == 0 => Ok(StageFrame { request_id, message: StageMessage::SwapOut }),
        READY if payload_len == 0 => Ok(StageFrame { request_id, message: StageMessage::Ready { cached_tokens: position } }),
        PERSIST if payload_len == 0 => Ok(StageFrame { request_id, message: StageMessage::Persist }),
        SHUTDOWN if payload_len == 0 && position <= 1 => Ok(StageFrame { request_id, message: StageMessage::Shutdown { persist: position != 0 } }),
        DEVICE_MEMORY if payload_len == position.checked_mul(24).ok_or("stage device memory 数量溢出")? => {
            Ok(StageFrame { request_id, message: StageMessage::DeviceMemory { devices: decode_device_memory(&payload)?, session_capacity: (rows != 0).then_some(rows) } })
        }
        _ => Err(format!("未知 stage frame kind={kind} payload={payload_len}")),
    }
}

fn encode_device_memory(devices: &[StageDeviceMemory]) -> Result<Vec<u8>, String> {
    let mut payload = Vec::with_capacity(devices.len().checked_mul(24).ok_or("stage device memory 大小溢出")?);
    for device in devices {
        payload.extend_from_slice(&device.device.to_le_bytes());
        payload.extend_from_slice(&u32::try_from(device.model_units).map_err(|_| "stage model units 超过 u32")?.to_le_bytes());
        payload.extend_from_slice(&device.available_bytes.to_le_bytes());
        payload.extend_from_slice(&device.total_bytes.to_le_bytes());
    }
    Ok(payload)
}

fn encode_u32_values(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn decode_u32_values(payload: &[u8]) -> Vec<u32> {
    payload.chunks_exact(4).map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())).collect()
}

fn decode_mtp_context(max_decode: usize, draft_tokens: usize, cols: usize, value: u32, payload: &[u8]) -> Result<StageMessage, String> {
    if max_decode == 0 || draft_tokens == 0 || cols != 0 || value != 0 || !payload.len().is_multiple_of(4) {
        return Err("MTP context frame 非法".to_owned());
    }
    let prompt_tokens = decode_u32_values(payload);
    if prompt_tokens.is_empty() {
        return Err("MTP context prompt 不能为空".to_owned());
    }
    Ok(StageMessage::MtpContext { prompt_tokens, max_decode, draft_tokens })
}

fn decode_speculative(retained_rows: usize, token_count: usize, draft_count: usize, value: u32, payload: &[u8]) -> Result<StageMessage, String> {
    let expected = token_count.checked_add(draft_count).and_then(|count| count.checked_mul(4)).ok_or("MTP speculative payload 溢出")?;
    if token_count == 0 || payload.len() != expected || (retained_rows != 0 && retained_rows != token_count) {
        return Err(format!("MTP speculative frame 非法: payload={} tokens={token_count} drafts={draft_count} retained_rows={retained_rows}", payload.len()));
    }
    let values = decode_u32_values(payload);
    Ok(StageMessage::Speculative { tokens: values[..token_count].to_vec(), retained_rows, drafts: values[token_count..].to_vec(), eos: value != 0 })
}

fn decode_device_memory(payload: &[u8]) -> Result<Vec<StageDeviceMemory>, String> {
    if !payload.len().is_multiple_of(24) {
        return Err(format!("stage device memory payload={} 不是 24 的倍数", payload.len()));
    }
    payload
        .chunks_exact(24)
        .map(|bytes| {
            Ok(StageDeviceMemory {
                device: i32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                model_units: u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize,
                available_bytes: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                total_bytes: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            })
        })
        .collect()
}

fn encode_sampling(sampling: SamplingConfig) -> Result<[u8; 16], String> {
    let sampling = sampling.validate()?;
    let mut payload = [0_u8; 16];
    payload[..4].copy_from_slice(&sampling.temperature.to_le_bytes());
    payload[4..8].copy_from_slice(&sampling.top_p.to_le_bytes());
    payload[8..16].copy_from_slice(&sampling.seed.to_le_bytes());
    Ok(payload)
}

fn decode_sampling(payload: &[u8]) -> Result<SamplingConfig, String> {
    if payload.len() != 16 {
        return Err(format!("stage sampling payload={}，期望 16", payload.len()));
    }
    SamplingConfig { temperature: f32::from_le_bytes(payload[..4].try_into().unwrap()), top_p: f32::from_le_bytes(payload[4..8].try_into().unwrap()), seed: u64::from_le_bytes(payload[8..16].try_into().unwrap()) }.validate()
}

fn runtime() -> Result<Runtime, String> {
    tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(|error| format!("创建 stage tokio runtime: {error}"))
}

fn stage_quic_transport_config() -> Result<QuicTransportConfig, String> {
    // stage 是长期复用的内网双向流。Iroh 默认 30 秒 connection idle 在模型加载或
    // 低流量服务期过短；显式 keepalive 维持路径，较长上限仍允许故障最终收口。
    let idle = STAGE_MAX_IDLE.try_into().map_err(|error| format!("stage QUIC idle timeout 无效: {error}"))?;
    Ok(QuicTransportConfig::builder().max_idle_timeout(Some(idle)).keep_alive_interval(STAGE_KEEP_ALIVE).build())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_cache_id_is_deterministic() {
        assert_eq!(RequestId::from_cache_id("abc123"), RequestId::from_cache_id("abc123"));
    }

    #[test]
    fn from_cache_id_distinct_inputs() {
        assert_ne!(RequestId::from_cache_id("session-A"), RequestId::from_cache_id("session-B"));
    }

    #[test]
    fn from_cache_id_nonzero() {
        let id = RequestId::from_cache_id("deterministic");
        assert_ne!(format!("{id}"), "00000000000000000000000000000000");
    }

    #[test]
    fn device_memory_payload_roundtrips() {
        let devices = vec![StageDeviceMemory { device: 0, model_units: 6, available_bytes: 17 << 30, total_bytes: 48 << 30 }, StageDeviceMemory { device: 5, model_units: 5, available_bytes: 21 << 30, total_bytes: 48 << 30 }];
        assert_eq!(decode_device_memory(&encode_device_memory(&devices).unwrap()).unwrap(), devices);
    }

    #[test]
    fn sampling_payload_roundtrips() {
        let sampling = SamplingConfig { temperature: 0.6, top_p: 0.95, seed: 42 };
        assert_eq!(decode_sampling(&encode_sampling(sampling).unwrap()).unwrap(), sampling);
    }

    #[test]
    fn mtp_context_payload_roundtrips() {
        let payload = encode_u32_values(&[11, 22, 33]);
        assert!(matches!(decode_mtp_context(64, 3, 0, 0, &payload).unwrap(), StageMessage::MtpContext { prompt_tokens, max_decode: 64, draft_tokens: 3 } if prompt_tokens == [11, 22, 33]));
        assert!(decode_mtp_context(0, 3, 0, 0, &payload).is_err());
    }

    #[test]
    fn speculative_payload_roundtrips_and_checks_retained_rows() {
        let payload = encode_u32_values(&[101, 102, 201, 202]);
        assert!(matches!(decode_speculative(2, 2, 2, 0, &payload).unwrap(), StageMessage::Speculative { tokens, retained_rows: 2, drafts, eos: false } if tokens == [101, 102] && drafts == [201, 202]));
        assert!(decode_speculative(1, 2, 2, 0, &payload).is_err());
    }

    #[test]
    fn zero_payload_control_frames_are_legal() {
        assert!(validate_stage_payload_len(0).is_ok());
    }

    #[test]
    fn oversized_payload_is_rejected() {
        assert!(validate_stage_payload_len(MAX_STAGE_PAYLOAD).is_ok());
        assert!(validate_stage_payload_len(MAX_STAGE_PAYLOAD + 1).is_err());
    }

    #[test]
    fn glm52_prefill_aux_selection_payload_is_legal() {
        let payload = 4096 * 6144 * 2 * 2 + 4096 * 2048 * 4 + 4;
        assert_eq!(payload, 128 * 1024 * 1024 + 4);
        assert!(validate_stage_payload_len(payload).is_ok());
    }

    #[test]
    fn decode_verify_cohort_metadata_roundtrips() {
        let listener = StageTransport::bind(IrohConfig { secret_key: Some(iroh::SecretKey::from_bytes(&[9; 32])), bind_addr: None, expected_peer: None }).unwrap();
        let ticket = EndpointTicket::new(listener.endpoint.addr()).to_string();
        let accepted = std::thread::spawn(move || listener.accept().unwrap());
        let mut sender = StageTransport::connect(&ticket, IrohConfig::default()).unwrap();
        let mut receiver = accepted.join().unwrap();
        let request = RequestId::generate_for_test();
        sender.send_decode_cohort(request, 17, 4, 12, 2, &[1, 2], &[]).unwrap();
        assert!(matches!(receiver.recv().unwrap().message, StageMessage::Decode { cohort: 17, cohort_size: 4, position: 12, cols: 2, values, .. } if values == [1, 2]));
        sender.send_verify_cohort(request, 18, 3, 13, 4, 2, &[1, 2, 3, 4, 5, 6, 7, 8], &[]).unwrap();
        assert!(matches!(receiver.recv().unwrap().message, StageMessage::Verify { cohort: 18, cohort_size: 3, position: 13, rows: 4, cols: 2, values, .. } if values.len() == 8));
    }

    #[test]
    fn dspark_aux_hidden_roundtrips() {
        let listener = StageTransport::bind(IrohConfig { secret_key: Some(iroh::SecretKey::from_bytes(&[7; 32])), bind_addr: None, expected_peer: None }).unwrap();
        let ticket = EndpointTicket::new(listener.endpoint.addr()).to_string();
        let accepted = std::thread::spawn(move || listener.accept().unwrap());
        let mut sender = StageTransport::connect(&ticket, IrohConfig::default()).unwrap();
        let mut receiver = accepted.join().unwrap();
        let request = RequestId::generate_for_test();
        sender.send_prefill_aux(request, 3, 2, 2, &[1, 2, 3, 4], &[], &[5, 6, 7, 8], 5).unwrap();
        assert!(matches!(receiver.recv().unwrap().message, StageMessage::Prefill { position: 3, rows: 2, cols: 2, aux_values, aux_taps: 5, .. } if aux_values == [5, 6, 7, 8]));
        sender.send_decode_cohort_aux(request, 9, 2, 5, 2, &[1, 2], &[], &[3, 4], 5).unwrap();
        assert!(matches!(receiver.recv().unwrap().message, StageMessage::Decode { cohort: 9, aux_values, aux_taps: 5, .. } if aux_values == [3, 4]));
        sender.send_verify_cohort_aux(request, 10, 2, 6, 2, 2, &[1, 2, 3, 4], &[], &[5, 6, 7, 8], 5).unwrap();
        assert!(matches!(receiver.recv().unwrap().message, StageMessage::Verify { cohort: 10, aux_values, aux_taps: 5, .. } if aux_values == [5, 6, 7, 8]));
    }

    #[test]
    fn tail_sampling_messages_roundtrip() {
        let listener = StageTransport::bind(IrohConfig { secret_key: Some(iroh::SecretKey::from_bytes(&[6; 32])), bind_addr: None, expected_peer: None }).unwrap();
        let ticket = EndpointTicket::new(listener.endpoint.addr()).to_string();
        let accepted = std::thread::spawn(move || listener.accept().unwrap());
        let mut head = StageTransport::connect(&ticket, IrohConfig::default()).unwrap();
        let mut tail = accepted.join().unwrap();
        let request = RequestId::generate_for_test();
        head.send_open_reserved(request, None, 7, 1024, true, SamplingConfig::greedy(9), true).unwrap();
        assert!(matches!(tail.recv().unwrap().message, StageMessage::Open { cached_tokens: 7, reserved_rows: 1024, cache_hit: true, tail_sampling: true, .. }));
        head.send_open(request, None, 7, true, SamplingConfig::greedy(9), true).unwrap();
        assert!(matches!(tail.recv().unwrap().message, StageMessage::Open { cached_tokens: 7, reserved_rows: 0, cache_hit: true, tail_sampling: true, .. }));
        head.send_sample(request, &[3, 5, 8]).unwrap();
        assert!(matches!(tail.recv().unwrap().message, StageMessage::Sample { excluded } if excluded == [3, 5, 8]));
        tail.send_sampled(request, 17, 4, 12, 42, false).unwrap();
        assert!(matches!(head.recv().unwrap().message, StageMessage::Sampled { cohort: 17, cohort_size: 4, position: 12, token: 42, eos: false }));
    }

    #[test]
    fn continuous_stream_end_ack_roundtrips_with_independent_control_id() {
        let listener = StageTransport::bind(IrohConfig { secret_key: Some(iroh::SecretKey::from_bytes(&[8; 32])), bind_addr: None, expected_peer: None }).unwrap();
        let ticket = EndpointTicket::new(listener.endpoint.addr()).to_string();
        let accepted = std::thread::spawn(move || listener.accept().unwrap());
        let mut head = StageTransport::connect(&ticket, IrohConfig::default()).unwrap();
        let mut tail = accepted.join().unwrap();
        let request = RequestId::from_cache_id("request");
        let control = RequestId::from_cache_id("continuous-control");

        head.send_continuous_stream(&[request]).unwrap();
        assert!(matches!(tail.recv().unwrap().message, StageMessage::ContinuousStream { requests } if requests == [request]));
        head.send_continuous_stream_end(control).unwrap();
        let end = tail.recv().unwrap();
        assert_eq!(end.request_id, control);
        assert!(matches!(end.message, StageMessage::ContinuousStream { requests } if requests.is_empty()));
        tail.send_ready(control, 0).unwrap();
        assert!(matches!(head.recv().unwrap(), StageFrame { request_id, message: StageMessage::Ready { cached_tokens: 0 } } if request_id == control));
    }

    #[test]
    fn shutdown_roundtrips_and_waits_for_ready() {
        let listener = StageTransport::bind(IrohConfig { secret_key: Some(iroh::SecretKey::from_bytes(&[5; 32])), bind_addr: None, expected_peer: None }).unwrap();
        let ticket = EndpointTicket::new(listener.endpoint.addr()).to_string();
        let accepted = std::thread::spawn(move || listener.accept().unwrap());
        let mut head = StageTransport::connect(&ticket, IrohConfig::default()).unwrap();
        let mut tail = accepted.join().unwrap();
        let control = RequestId::from_cache_id("stage-shutdown");

        head.send_shutdown(control, true).unwrap();
        assert!(matches!(tail.recv().unwrap(), StageFrame { request_id, message: StageMessage::Shutdown { persist: true } } if request_id == control));
        tail.send_ready(control, 0).unwrap();
        assert!(matches!(head.recv().unwrap(), StageFrame { request_id, message: StageMessage::Ready { cached_tokens: 0 } } if request_id == control));
    }

    #[test]
    fn downstream_reconnect_reuses_endpoint_and_preempts_old_stream() {
        let listener = StageTransport::bind(IrohConfig { secret_key: Some(iroh::SecretKey::from_bytes(&[7; 32])), bind_addr: None, expected_peer: None }).unwrap();
        let ticket = EndpointTicket::new(listener.endpoint.addr()).to_string();
        let accepted = std::thread::spawn(move || listener.accept().unwrap());
        let mut client = StageTransport::connect(&ticket, IrohConfig::default()).unwrap();
        let mut server = accepted.join().unwrap();
        client.reconnect_downstream().unwrap();

        let error = server.recv().unwrap_err();
        assert!(error.contains("新上游连接") || error.contains("connection lost"), "{error}");
        server.accept_reconnect().unwrap();
        server.send_device_memory(&[StageDeviceMemory { device: 0, model_units: 40, available_bytes: 17, total_bytes: 48 }], Some(16)).unwrap();
        assert!(matches!(client.recv().unwrap().message, StageMessage::DeviceMemory { devices, session_capacity: Some(16) } if devices.len() == 1 && devices[0].model_units == 40));
    }
}
