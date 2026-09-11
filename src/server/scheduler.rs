//! iroh 内网调度平面。只传递节点、请求和推理事件，不传递模型内部状态。

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    error::Error,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use iroh::{Endpoint, endpoint::presets};
use iroh_tickets::endpoint::EndpointTicket;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    sync::{Mutex, mpsc},
    task::JoinHandle,
    time::timeout,
};

pub const SCHEDULER_ALPN: &[u8] = b"zllm/scheduler/1";
pub const SCHEDULER_PROTOCOL_VERSION: u32 = 9;
const MAX_MESSAGE_BYTES: usize = 80 * 1024 * 1024;
pub(super) const MAX_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
static ARTIFACT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

type DynError = Box<dyn Error + Send + Sync>;

pub use crate::kv_cache::terminal_cache::TerminalInfo as CacheInfo;
pub use crate::runtime::session::{AtomicCounterU64, KvCacheDeviceCapacity, NodeCapabilities, RuntimeStatus as NodeRuntime, ToolCall, ToolCallDelta, ToolFunction};
pub use crate::runtime::session::{
    TerminalResume, activate_terminal_append, client_replayable_response, conversation_hash, request_resume_boundary, request_terminal_resume, resume_terminal_append, resume_terminal_session, retain_terminal_session, terminal_cache_id,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NodeMessage {
    Register { protocol_version: u32, api_key: Option<String>, model: String, max_concurrency: usize, caches: Vec<CacheInfo>, capabilities: NodeCapabilities, runtime: NodeRuntime },
    Heartbeat { caches: Vec<CacheInfo>, runtime: NodeRuntime },
    RuntimeChanged { runtime: NodeRuntime },
    Event { request_id: String, event: InferenceEvent },
    TaskStatus { task_id: String, status: String, error: Option<String>, outputs: Vec<ArtifactDescriptor> },
    TaskProgress { task_id: String, progress: TaskProgress },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SchedulerMessage {
    Registered {
        node_id: String,
        heartbeat_seconds: u64,
        task_ids: Vec<String>,
    },
    NewPrefill {
        request_id: String,
        model: String,
        request: Value,
    },
    Cancel {
        request_id: String,
    },
    NewTask {
        task_id: String,
        model: String,
        task_kind: String,
        request: Value,
    },
    ArtifactCommitted {
        task_id: String,
        artifact_id: String,
    },
    /// append 请求在 owner 满载排队期间 pin 终点 cache,节点换出循环跳过被 pin
    /// 条目,排到队时命中内存而不是 swap 慢路径;`UnpinCache` 解除。
    PinCache {
        cache_id: String,
    },
    UnpinCache {
        cache_id: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskProgress {
    pub phase: String,
    pub completed: usize,
    pub total: usize,
    pub elapsed_seconds: f64,
    pub phase_eta_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactDescriptor {
    pub id: String,
    pub file_name: String,
    pub content_type: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactHeader {
    pub task_id: String,
    pub artifact: ArtifactDescriptor,
    pub output_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct StoredArtifact {
    pub id: String,
    pub file_name: String,
    pub content_type: String,
    pub bytes: u64,
    pub url: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TaskView {
    pub id: String,
    pub model: String,
    pub task_kind: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<TaskProgress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub metadata: Value,
    pub outputs: Vec<StoredArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    Started,
    Token { token_id: u32, text: String },
    ToolCallDelta { delta: ToolCallDelta },
    ToolCall { index: usize, tool_call: ToolCall },
    Completed { finish_reason: String, prompt_tokens: usize, completion_tokens: usize },
    Error { message: String },
}

impl InferenceEvent {
    pub(crate) fn terminal(&self) -> bool {
        matches!(self, Self::Completed { .. } | Self::Error { .. })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AvailableNode {
    pub node_id: String,
    pub model: String,
    pub caches: Vec<CacheInfo>,
    pub active_requests: usize,
    pub max_concurrency: usize,
    /// 负载均衡视图:当前在跑的请求数(等于 `active_requests`,管理接口字段别名,让
    /// 仪表板一眼看清"当前/最大负载")。`#[serde(default)]` 让旧版 wire 节点心跳不被新字段拒绝。
    #[serde(default)]
    pub current_load: usize,
    /// 负载均衡视图:最大并发槽位(等于 `max_concurrency`,管理接口字段别名)。
    #[serde(default)]
    pub max_load: usize,
    pub capabilities: NodeCapabilities,
    pub runtime: NodeRuntime,
    pub registered_at: u64,
    /// 注册顺序（与 `registered_at` 同秒内也能区分），`cache_owner` 用它做 owner 排序 tiebreak。
    /// 不进 wire（`#[serde(default)]`），仅本地排序使用。
    #[serde(default)]
    pub registration_seq: u64,
    pub last_seen: u64,
}

struct RegisteredNode {
    view: AvailableNode,
    connection: u64,
    last_dispatched: u64,
    /// 已下发、尚未被 node 最新负载快照覆盖的准入压力。
    pending_pressure: usize,
    commands: NodeCommands,
    /// 持有 iroh 连接防 drop 关闭；standalone 同进程节点没有 transport。
    _transport: Option<iroh::endpoint::Connection>,
}

/// 节点命令通道。iroh 节点只收 wire 消息；standalone 本地节点的 NewPrefill
/// 额外携带请求事件通道，token 数据面直达 HTTP 处理侧，绕过 scheduler 逐跳路由。
#[derive(Clone)]
enum NodeCommands {
    Wire(mpsc::Sender<SchedulerMessage>),
    Local(mpsc::Sender<LocalNodeCommand>),
}

/// standalone 本地节点的命令：wire 消息原样转发；NewPrefill 附带事件直发通道。
#[derive(Debug)]
pub enum LocalNodeCommand {
    Wire(SchedulerMessage),
    NewPrefill { request_id: String, model: String, request: Value, events: mpsc::Sender<InferenceEvent> },
}

impl NodeCommands {
    /// 发送非 NewPrefill 命令（Cancel/NewTask/ArtifactCommitted）。false = 节点通道已关闭。
    async fn send(&self, message: SchedulerMessage) -> bool {
        match self {
            Self::Wire(tx) => tx.send(message).await.is_ok(),
            Self::Local(tx) => tx.send(LocalNodeCommand::Wire(message)).await.is_ok(),
        }
    }
}

/// 一次推理执行的事件分发中枢。节点数据面(wire 的 publish 或本地节点的直连
/// 通道)把事件送入 entry,分发 task 追加历史并转发给全部订阅者;同一请求的
/// 重复提交(重试)在这里合并——新订阅者在锁内重放完整历史后实时跟进,
/// 第一个 token 延迟只剩已生成部分的重放耗时。
struct EventHub {
    entry: mpsc::Sender<InferenceEvent>,
    state: std::sync::Mutex<HubState>,
}

struct HubState {
    history: Vec<InferenceEvent>,
    subscribers: Vec<(String, mpsc::Sender<InferenceEvent>)>,
    /// 节点认识的 request_id;向节点发 Cancel 必须用它。
    primary_request_id: String,
    /// 终态已进入历史(与 push 在同一临界区内置位),此后不再接受合并订阅。
    finished: bool,
}

impl EventHub {
    fn new(entry: mpsc::Sender<InferenceEvent>, primary_request_id: String) -> Arc<Self> {
        Arc::new(Self { entry, state: std::sync::Mutex::new(HubState { history: Vec::new(), subscribers: Vec::new(), primary_request_id, finished: false }) })
    }

    /// 重试合并:锁内重放历史并加入订阅。容量 ≥ 历史长度保证 try_send 必成;
    /// 终态已到(finished)则只重放不订阅——receiver 在 tx drop 后自然关闭。
    /// 与 distribute 的 push/clone 临界区互斥,事件不重不漏:订阅先于 push 则
    /// 实时收到,晚于 push 则历史里已有。
    fn subscribe(&self, request_id: String) -> mpsc::Receiver<InferenceEvent> {
        let mut state = self.state.lock().expect("事件枢纽锁中毒");
        let (sender, receiver) = mpsc::channel(64 + state.history.len().max(1));
        for event in &state.history {
            let _ = sender.try_send(event.clone());
        }
        if !state.finished {
            state.subscribers.push((request_id, sender));
        }
        receiver
    }

    /// 分发一个事件(可能终态);订阅者已关闭时移除。
    async fn distribute(&self, event: InferenceEvent) {
        let terminal = event.terminal();
        let subscribers = {
            let mut state = self.state.lock().expect("事件枢纽锁中毒");
            state.history.push(event.clone());
            if terminal {
                state.finished = true;
            }
            state.subscribers.clone()
        };
        for (request_id, sender) in subscribers {
            if sender.send(event.clone()).await.is_err() {
                if let Ok(mut state) = self.state.lock() {
                    state.subscribers.retain(|(existing, _)| existing != &request_id);
                }
            }
        }
    }

    /// 节点异常断开时对所有订阅者补一个终态错误;已 finished 则无事可做。
    async fn abort(&self, message: &str) {
        let subscribers = {
            let mut state = self.state.lock().expect("事件枢纽锁中毒");
            if state.finished {
                return;
            }
            state.finished = true;
            let event = InferenceEvent::Error { message: message.to_owned() };
            state.history.push(event.clone());
            state.subscribers.clone()
        };
        for (_, sender) in subscribers {
            let _ = sender.send(InferenceEvent::Error { message: message.to_owned() }).await;
        }
    }

    /// 摘除一个订阅者;返回 (primary_request_id, 是否已无订阅者)。
    /// writer 的生死由订阅者计数决定:全部订阅者断开才应该取消节点执行。
    fn unsubscribe(&self, request_id: &str) -> (String, bool) {
        let mut state = self.state.lock().expect("事件枢纽锁中毒");
        state.subscribers.retain(|(existing, _)| existing != request_id);
        (state.primary_request_id.clone(), state.subscribers.is_empty() && !state.finished)
    }
}

struct InflightRequest {
    node_id: String,
    hub: Arc<EventHub>,
    cache_id: Option<String>,
    cancelled: bool,
    /// 是否占用节点并发槽位(primary=true;重试合并条目没有真实 NewPrefill,不占)。
    counted: bool,
}

struct InflightTask {
    node_id: String,
    view: TaskView,
    /// 原始任务请求。节点断连后孤儿任务改派到新节点时需要重发 NewTask。
    request: Value,
    expected_outputs: BTreeMap<String, ArtifactDescriptor>,
    expected_output_count: usize,
}

/// standalone 同进程节点与 scheduler 之间的内存通道端点（节点侧持有）。
/// 消息类型与 wire 协议一致；iroh 路径的 JSON/QUIC 在这里被省略。
pub struct LocalNodeChannels {
    pub commands: mpsc::Receiver<LocalNodeCommand>,
    pub messages: mpsc::Sender<NodeMessage>,
}

/// 被新注册节点接管的孤儿任务;调用方在锁外重发 NewTask。
struct AdoptedTask {
    task_id: String,
    task_kind: String,
    request: Value,
}

#[derive(Default)]
struct RegistryState {
    nodes: BTreeMap<String, RegisteredNode>,
    requests: HashMap<String, InflightRequest>,
    tasks: BTreeMap<String, InflightTask>,
    next_connection: u64,
    next_dispatch: u64,
    /// 单调递增的注册序号，让 `cache_owner` 在 `registered_at` 同秒内也能稳定排序。
    /// 进程内本地状态，不跨 wire 同步。
    next_registration_seq: u64,
}

#[derive(Clone)]
pub struct Scheduler {
    state: Arc<Mutex<RegistryState>>,
    artifact_dir: Arc<PathBuf>,
    public_base_url: Arc<str>,
    dispatch_wait: std::time::Duration,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(&SchedulerConfig::default())
    }
}

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub artifact_dir: PathBuf,
    pub public_base_url: String,
    pub dispatch_wait: std::time::Duration,
    pub iroh: super::iroh::IrohConfig,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { artifact_dir: std::env::temp_dir().join("zllm-artifacts"), public_base_url: "http://127.0.0.1:8000".to_owned(), dispatch_wait: std::time::Duration::from_secs(30), iroh: super::iroh::IrohConfig::default() }
    }
}

#[derive(Debug)]
pub enum DispatchError {
    NoAvailableNode(String),
    NodeDisconnected,
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAvailableNode(model) => write!(formatter, "模型 {model} 没有可用节点"),
            Self::NodeDisconnected => formatter.write_str("选中的节点已经断开"),
        }
    }
}

impl Error for DispatchError {}

impl Scheduler {
    pub fn new(config: &SchedulerConfig) -> Self {
        Self { state: Arc::new(Mutex::new(RegistryState::default())), artifact_dir: Arc::new(config.artifact_dir.clone()), public_base_url: Arc::from(config.public_base_url.trim_end_matches('/')), dispatch_wait: config.dispatch_wait }
    }

    #[allow(clippy::too_many_arguments)]
    async fn register(
        &self,
        node_id: String,
        model: String,
        max_concurrency: usize,
        caches: Vec<CacheInfo>,
        capabilities: NodeCapabilities,
        runtime: NodeRuntime,
        commands: NodeCommands,
        transport: Option<iroh::endpoint::Connection>,
    ) -> (u64, Vec<String>, Vec<AdoptedTask>) {
        let now = unix_seconds();
        let registered_id = node_id.clone();
        let (connection, task_ids, adopted) = {
            let mut state = self.state.lock().await;
            state.next_connection = state.next_connection.wrapping_add(1).max(1);
            let connection = state.next_connection;
            state.next_registration_seq = state.next_registration_seq.wrapping_add(1).max(1);
            let registration_seq = state.next_registration_seq;
            let task_ids = state.tasks.iter().filter(|(_, task)| task.node_id == node_id && !matches!(task.view.status.as_str(), "succeeded" | "failed" | "cancelled")).map(|(task_id, _)| task_id.clone()).collect::<Vec<_>>();
            let active_requests = task_ids.len();
            let max_concurrency = registered_max_concurrency(max_concurrency, &capabilities);
            // KV 容量不足一个 reservation page 时 max_concurrency 会算成 0,调度过滤(active < max)将永远跳过该节点;
            // 0 本身就是"不可调度"标记,这里把原因打成日志,避免节点"注册成功却从不接单"无法排查。
            if max_concurrency == 0 {
                eprintln!(
                    "[scheduler] 节点 {node_id} KV 容量不足: min_token_capacity={} page_tokens={}，max_concurrency=0，节点不会被调度",
                    capabilities.kv_cache_devices.iter().map(|device| device.token_capacity).min().unwrap_or(0),
                    capabilities.kv_reservation_page_tokens,
                );
            }
            state.nodes.insert(
                node_id.clone(),
                RegisteredNode {
                    view: AvailableNode {
                        node_id,
                        model,
                        caches,
                        active_requests,
                        max_concurrency,
                        // 负载均衡视图与 active_requests/max_concurrency 同步初始化,后续增量更新
                        // 留给 `nodes()` 出口处统一派生,避免在每个 += 1 / -= 1 路径上重复维护。
                        current_load: active_requests,
                        max_load: max_concurrency,
                        capabilities,
                        runtime,
                        registered_at: now,
                        registration_seq,
                        last_seen: now,
                    },
                    connection,
                    last_dispatched: 0,
                    pending_pressure: 0,
                    commands,
                    _transport: transport,
                },
            );
            (connection, task_ids, adopt_orphaned_tasks(&mut state, &registered_id))
        };
        (connection, task_ids, adopted)
    }

    async fn update_heartbeat(&self, node_id: &str, connection: u64, caches: Vec<CacheInfo>, runtime: NodeRuntime) {
        let mut state = self.state.lock().await;
        if let Some(node) = state.nodes.get_mut(node_id).filter(|node| node.connection == connection) {
            node.view.caches = caches;
            node.view.runtime = runtime;
            node.pending_pressure = 0;
            node.view.last_seen = unix_seconds();
        }
    }

    async fn update_runtime(&self, node_id: &str, connection: u64, runtime: NodeRuntime) {
        let mut state = self.state.lock().await;
        if let Some(node) = state.nodes.get_mut(node_id).filter(|node| node.connection == connection) {
            node.view.runtime = runtime;
            node.pending_pressure = 0;
            node.view.last_seen = unix_seconds();
        }
    }

    async fn unregister(&self, node_id: &str, connection: u64) {
        let (pending, tasks) = {
            let mut state = self.state.lock().await;
            if state.nodes.get(node_id).is_none_or(|node| node.connection != connection) {
                return;
            }
            state.nodes.remove(node_id);
            let request_ids = state.requests.iter().filter_map(|(request_id, request)| (request.node_id == node_id).then_some(request_id.clone())).collect::<Vec<_>>();
            // 同一 hub 可能挂多个条目(primary + 重试合并),去重后统一补终态错误。
            let mut pending = Vec::new();
            for request_id in request_ids {
                if let Some(request) = state.requests.remove(&request_id)
                    && !pending.iter().any(|hub: &Arc<EventHub>| Arc::ptr_eq(hub, &request.hub))
                {
                    pending.push(request.hub);
                }
            }
            let tasks = state
                .tasks
                .iter()
                .filter(|(_, task)| task.node_id == node_id && !matches!(task.view.status.as_str(), "succeeded" | "failed" | "cancelled"))
                .map(|(task_id, task)| (task_id.clone(), task.view.task_kind.clone()))
                .collect::<Vec<_>>();
            (pending, tasks)
        };
        for hub in pending {
            hub.abort("推理节点连接已断开").await;
        }
        for (task_id, task_kind) in tasks {
            let (status, error) = if task_kind == "video_generation" { ("queued", "执行节点连接已断开，等待持久化任务恢复") } else { ("failed", "执行节点连接已断开") };
            self.update_task_status("", &task_id, status, Some(error.to_owned()), Vec::new()).await;
        }
    }

    pub async fn nodes(&self) -> Vec<AvailableNode> {
        // 出口处把 `current_load` / `max_load` 与 `active_requests` / `max_concurrency` 同步,
        // 内部并发增量只动 `active_requests`,不重复维护两个字段。
        self.state
            .lock()
            .await
            .nodes
            .values()
            .map(|node| {
                let mut view = node.view.clone();
                view.current_load = view.active_requests;
                view.max_load = view.max_concurrency;
                view
            })
            .collect()
    }

    pub async fn models(&self) -> Vec<String> {
        self.state.lock().await.nodes.values().map(|node| node.view.model.clone()).collect::<BTreeSet<_>>().into_iter().collect()
    }

    pub async fn dispatch(&self, request_id: String, model: String, cache_id: Option<&str>, request: Value) -> Result<mpsc::Receiver<InferenceEvent>, DispatchError> {
        let events = {
            let mut state = self.state.lock().await;
            // 同 cache_id 已有 writer = 同一请求的重复提交:正常对话的 cache_id
            // 随对话历史单调前进,只有重试会重放旧 id。合并到现有事件流——历史
            // 重放 + 实时跟进,不发 NewPrefill、不占节点槽位、无需排队等待。
            if let Some(cache_id) = cache_id
                && let Some(existing) = state.requests.values().find(|request| request.cache_id.as_deref() == Some(cache_id))
            {
                let hub = existing.hub.clone();
                let node_id = existing.node_id.clone();
                let receiver = hub.subscribe(request_id.clone());
                state.requests.insert(request_id.clone(), InflightRequest { node_id, hub, cache_id: Some(cache_id.to_owned()), cancelled: false, counted: false });
                return Ok(receiver);
            }
            let cached_nodes = cache_id
                .map_or_else(Vec::new, |cache_id| state.nodes.iter().filter(|(_, node)| node.view.model == model && node.view.caches.iter().any(|cache| cache.cache_id == cache_id)).map(|(node_id, _)| node_id.clone()).collect::<Vec<_>>());
            // 已有 cache 的 append 使用独立准入，节点再次检查真实命中；
            // miss 重建仍走新 prompt 的四份预算，不能只凭客户端 cache_id 放行。
            let admission_pressure = if cached_nodes.is_empty() { 4 } else { 1 };
            // cache 已存在时只在持有者中选择：内存优先于本机 SSD；全部满载则排队，
            // 不 fallback 到无 cache 节点重算并制造第二个 writer。
            let selected = if let Some(cache_id) = cache_id.filter(|_| !cached_nodes.is_empty()) {
                cached_nodes.into_iter().filter(|node_id| state.nodes.get(node_id).is_some_and(|node| node_has_capacity(node, admission_pressure))).min_by_key(|node_id| {
                    let node = &state.nodes[node_id];
                    (cache_location_rank(&node.view.runtime, cache_id), node.view.runtime.scheduling_pressure(), node.last_dispatched, node.view.registration_seq)
                })
            } else {
                state
                    .nodes
                    .iter()
                    .filter(|(_, node)| node.view.model == model && node_has_capacity(node, admission_pressure))
                    .min_by_key(|(_, node)| dispatch_order(&node.view, node.last_dispatched, None))
                    .map(|(node_id, _)| node_id.clone())
            };
            let Some(node_id) = selected else {
                return Err(DispatchError::NoAvailableNode(model));
            };
            state.next_dispatch = state.next_dispatch.wrapping_add(1).max(1);
            let dispatch = state.next_dispatch;
            let node = state.nodes.get_mut(&node_id).expect("刚选择的节点必须存在");
            node.last_dispatched = dispatch;
            node.pending_pressure = node.pending_pressure.saturating_add(admission_pressure);
            let commands = node.commands.clone();
            let (entry_tx, entry_rx) = mpsc::channel(64);
            let hub = EventHub::new(entry_tx.clone(), request_id.clone());
            let receiver = hub.subscribe(request_id.clone());
            state.requests.insert(request_id.clone(), InflightRequest { node_id: node_id.clone(), hub: hub.clone(), cache_id: cache_id.map(str::to_owned), cancelled: false, counted: true });
            if let Some(node) = state.nodes.get_mut(&node_id) {
                node.view.active_requests += 1;
            }
            // try_send 让"登记 inflight + 入 node 命令队列"在同一无 await 临界区
            // 完成；handler future 不可能停在已登记但 NewPrefill 尚未入队的状态。
            // 本地节点的 NewPrefill 直接携带事件入口,token 数据面不再经过 scheduler。
            let retry_model = model.clone();
            let sent = match &commands {
                NodeCommands::Wire(tx) => tx.try_send(SchedulerMessage::NewPrefill { request_id: request_id.clone(), model, request }).map_err(|error| match error {
                    mpsc::error::TrySendError::Full(_) => DispatchError::NoAvailableNode(retry_model.clone()),
                    mpsc::error::TrySendError::Closed(_) => DispatchError::NodeDisconnected,
                }),
                NodeCommands::Local(tx) => tx.try_send(LocalNodeCommand::NewPrefill { request_id: request_id.clone(), model, request, events: entry_tx }).map_err(|error| match error {
                    mpsc::error::TrySendError::Full(_) => DispatchError::NoAvailableNode(retry_model.clone()),
                    mpsc::error::TrySendError::Closed(_) => DispatchError::NodeDisconnected,
                }),
            };
            if let Err(error) = sent {
                state.requests.remove(&request_id);
                if let Some(node) = state.nodes.get_mut(&node_id) {
                    node.view.active_requests = node.view.active_requests.saturating_sub(1);
                    node.pending_pressure = node.pending_pressure.saturating_sub(admission_pressure);
                }
                return Err(error);
            }
            self.spawn_hub_dispatcher(entry_rx, hub);
            receiver
        };
        Ok(events)
    }

    /// hub 分发 task:消费节点事件入口,追加历史并转发给全部订阅者;终态后
    /// 摘除该 hub 的合并条目(primary 条目由 publish/note_terminal 摘除)。
    /// 入口关闭而未终态(节点进程消失)时补发终态错误,订阅者不会挂死。
    fn spawn_hub_dispatcher(&self, mut entry: mpsc::Receiver<InferenceEvent>, hub: Arc<EventHub>) {
        let scheduler = self.clone();
        tokio::spawn(async move {
            while let Some(event) = entry.recv().await {
                let terminal = event.terminal();
                hub.distribute(event).await;
                if terminal {
                    scheduler.reap_hub_entries(&hub).await;
                    return;
                }
            }
            hub.abort("推理节点事件流已关闭").await;
            scheduler.reap_hub_entries(&hub).await;
        });
    }

    /// 摘除 hub 上未被终态簿记清理的重试合并条目(不影响节点槽位计数)。
    async fn reap_hub_entries(&self, hub: &Arc<EventHub>) {
        self.state.lock().await.requests.retain(|_, request| !Arc::ptr_eq(&request.hub, hub) || request.counted);
    }

    /// 查 `cache_id` 的全部持有者节点及命令通道(`cache_owner` 只返回最早一个)。
    async fn cache_holders(&self, cache_id: &str) -> Vec<(String, NodeCommands)> {
        self.state.lock().await.nodes.iter().filter(|(_, node)| node.view.caches.iter().any(|cache| cache.cache_id == cache_id)).map(|(node_id, node)| (node_id.clone(), node.commands.clone())).collect()
    }

    /// 节点真实并发槽位暂满时等待流式 runtime 释放容量；模型未注册则立即返回。
    /// 同 cache_id 的重试请求在 dispatch 内即时合并,不进入这条等待路径。
    /// append 排队期间对 cache 持有者下发 pin:排队等的是"槽位",不能再让
    /// LRU 换出把等待目标挪到 SSD,排到队时退化为 swap-in 慢路径。
    pub async fn dispatch_wait(&self, request_id: String, model: String, cache_id: Option<&str>, request: Value) -> Result<mpsc::Receiver<InferenceEvent>, DispatchError> {
        let deadline = tokio::time::Instant::now() + self.dispatch_wait;
        let mut pinned: Vec<(String, NodeCommands)> = Vec::new();
        let result = loop {
            match self.dispatch(request_id.clone(), model.clone(), cache_id, request.clone()).await {
                Err(DispatchError::NoAvailableNode(_)) => {
                    let (model_registered, cache_hit) = {
                        let state = self.state.lock().await;
                        let mut nodes = state.nodes.values().filter(|node| node.view.model == model).peekable();
                        let registered = nodes.peek().is_some();
                        (registered, nodes.any(|node| cache_id.is_some_and(|id| node.view.caches.iter().any(|cache| cache.cache_id == id))))
                    };
                    // 命中请求等待持有节点释放槽位；未命中仍按配置超时返回 503。
                    if !model_registered || !cache_hit && tokio::time::Instant::now() >= deadline {
                        break Err(DispatchError::NoAvailableNode(model));
                    }
                    if pinned.is_empty()
                        && let Some(cache_id) = cache_id
                        && let Some(holder_commands) = self.try_pin_holders(cache_id).await
                    {
                        pinned = holder_commands;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                result => break result,
            }
        };
        if !pinned.is_empty()
            && let Some(cache_id) = cache_id
        {
            // 解 pin 对象覆盖 pin 时刻与退出时刻持有者的并集:窗口内持有者变化时
            // 多发的 unpin 在节点侧是无害 no-op。
            let mut targets = self.cache_holders(cache_id).await;
            for (node_id, commands) in pinned {
                if !targets.iter().any(|(existing, _)| existing == &node_id) {
                    targets.push((node_id, commands));
                }
            }
            for (_, commands) in targets {
                let _ = commands.send(SchedulerMessage::UnpinCache { cache_id: cache_id.to_owned() }).await;
            }
        }
        result
    }

    /// 首次排队时 pin 该 cache 的全部持有者(内存与 SSD 持有者都算,最终选中谁
    /// 由容量与 location rank 决定);无持有者返回 None,保持未 pin 状态。
    async fn try_pin_holders(&self, cache_id: &str) -> Option<Vec<(String, NodeCommands)>> {
        let holders = self.cache_holders(cache_id).await;
        if holders.is_empty() {
            return None;
        }
        for (_, commands) in &holders {
            let _ = commands.send(SchedulerMessage::PinCache { cache_id: cache_id.to_owned() }).await;
        }
        Some(holders)
    }

    /// 查 `cache_id` 的 owner node + CacheInfo（供 `GET /v1/caches/{cache_id}`）。
    /// 线性会话下每个 cache_id 仅一个 owner；多 node 误报时返回最先注册的：
    /// 用 `(registered_at, registration_seq)` 字典序比较，前者秒级粗排，后者处理同秒内的注册突发。
    pub async fn cache_owner(&self, cache_id: &str) -> Option<(String, CacheInfo)> {
        let state = self.state.lock().await;
        state
            .nodes
            .iter()
            .filter_map(|(node_id, node)| node.view.caches.iter().find(|cache| cache.cache_id == cache_id).map(|cache| (node_id.clone(), cache.clone(), node.view.registered_at, node.view.registration_seq)))
            .min_by_key(|(_, _, registered_at, registration_seq)| (*registered_at, *registration_seq))
            .map(|(node_id, cache, _, _)| (node_id, cache))
    }

    async fn publish(&self, node_id: &str, request_id: &str, event: InferenceEvent) {
        let terminal = event.terminal();
        let entry = {
            let mut state = self.state.lock().await;
            let Some(request) = state.requests.get(request_id).filter(|request| request.node_id == node_id) else {
                return;
            };
            let entry = request.hub.entry.clone();
            if terminal {
                let request = state.requests.remove(request_id).expect("刚确认请求存在");
                if let Some(node) = state.nodes.get_mut(&request.node_id) {
                    node.view.active_requests = node.view.active_requests.saturating_sub(1);
                }
            }
            entry
        };
        let _ = entry.send(event).await;
    }

    pub async fn cancel(&self, request_id: &str) {
        let command = {
            let mut state = self.state.lock().await;
            let Some(request) = state.requests.get_mut(request_id) else {
                return;
            };
            if request.cancelled {
                return;
            }
            request.cancelled = true;
            let hub = request.hub.clone();
            let node_id = request.node_id.clone();
            // 只摘除该请求的事件订阅;writer 的生死由订阅者计数决定——重试
            // 订阅者还在时不向节点发 Cancel,执行继续为存活的连接产出。
            let (primary_request_id, last_subscriber) = hub.unsubscribe(request_id);
            last_subscriber.then(|| state.nodes.get(&node_id).map(|node| node.commands.clone())).flatten().map(|commands| (commands, primary_request_id))
        };
        if let Some((commands, primary_request_id)) = command {
            let _ = commands.send(SchedulerMessage::Cancel { request_id: primary_request_id }).await;
        }
    }

    /// standalone 在同进程组合唯一 node：注册、事件流与命令下发走内存通道，
    /// 消息类型与 wire 协议一致，调度、cache、取消逻辑全部复用，只是不走
    /// QUIC 环回与 JSON 序列化。节点侧通道关闭（节点退出）时自动 unregister。
    ///
    /// 本地节点的 token/Started/ToolCall 经 NewPrefill 携带的通道直达 HTTP 处理侧；
    /// 控制通道只剩 terminal 簿记（释放调度槽位），不再做事件转发。
    pub fn attach_local_node(&self, node_id: String) -> LocalNodeChannels {
        let (command_tx, command_rx) = mpsc::channel::<LocalNodeCommand>(64);
        let (message_tx, mut message_rx) = mpsc::channel::<NodeMessage>(128);
        let scheduler = self.clone();
        tokio::spawn(async move {
            let mut connection = None;
            while let Some(message) = message_rx.recv().await {
                match message {
                    NodeMessage::Register { model, max_concurrency, caches, capabilities, runtime, .. } => {
                        if connection.is_some() {
                            eprintln!("[scheduler] 本地节点 {node_id} 重复注册，忽略");
                            continue;
                        }
                        let (connection_id, task_ids, adopted) = scheduler.register(node_id.clone(), model.clone(), max_concurrency, caches, capabilities, runtime, NodeCommands::Local(command_tx.clone()), None).await;
                        connection = Some(connection_id);
                        if command_tx.send(LocalNodeCommand::Wire(SchedulerMessage::Registered { node_id: node_id.clone(), heartbeat_seconds: 5, task_ids })).await.is_err() {
                            break;
                        }
                        for task in adopted {
                            if command_tx.send(LocalNodeCommand::Wire(SchedulerMessage::NewTask { task_id: task.task_id, model: model.clone(), task_kind: task.task_kind, request: task.request })).await.is_err() {
                                break;
                            }
                        }
                        eprintln!("[scheduler] 节点上线 id={node_id} model={model} (local)");
                    }
                    NodeMessage::Heartbeat { caches, runtime } => {
                        if let Some(connection_id) = connection {
                            scheduler.update_heartbeat(&node_id, connection_id, caches, runtime).await;
                        }
                    }
                    NodeMessage::RuntimeChanged { runtime } => {
                        if let Some(connection_id) = connection {
                            scheduler.update_runtime(&node_id, connection_id, runtime).await;
                        }
                    }
                    NodeMessage::Event { request_id, event } => {
                        if event.terminal() {
                            scheduler.note_terminal(&node_id, &request_id).await;
                        } else {
                            // 数据面应已直达请求通道；非终态事件出现在控制面说明节点实现漂移。
                            eprintln!("[scheduler] 本地节点经控制面发送非终态事件 request={request_id}，已丢弃");
                        }
                    }
                    NodeMessage::TaskStatus { task_id, status, error, outputs } => {
                        scheduler.update_task_status(&node_id, &task_id, &status, error, outputs).await;
                    }
                    NodeMessage::TaskProgress { task_id, progress } => {
                        scheduler.update_task_progress(&node_id, &task_id, progress).await;
                    }
                }
            }
            if let Some(connection_id) = connection {
                scheduler.unregister(&node_id, connection_id).await;
            }
            eprintln!("[scheduler] 节点下线 id={node_id} (local)");
        });
        LocalNodeChannels { commands: command_rx, messages: message_tx }
    }

    /// 本地节点终态簿记：摘除 inflight、释放调度槽位。数据面不经 scheduler，
    /// 与 publish 的差别是不再向请求通道转发事件。
    async fn note_terminal(&self, node_id: &str, request_id: &str) {
        let mut state = self.state.lock().await;
        if let Some(request) = state.requests.remove(request_id).filter(|request| request.node_id == node_id)
            && let Some(node) = state.nodes.get_mut(&request.node_id)
        {
            node.view.active_requests = node.view.active_requests.saturating_sub(1);
        }
    }

    #[cfg(test)]
    pub(super) async fn track_test_request(&self, request_id: &str) {
        let (entry, _entry_rx) = mpsc::channel(1);
        let hub = EventHub::new(entry, request_id.to_owned());
        self.state.lock().await.requests.insert(request_id.to_owned(), InflightRequest { node_id: String::new(), hub, cache_id: None, cancelled: false, counted: false });
    }

    #[cfg(test)]
    pub(super) async fn test_request_active(&self, request_id: &str) -> bool {
        self.state.lock().await.requests.contains_key(request_id)
    }

    #[cfg(test)]
    pub(super) async fn test_request_cancelled(&self, request_id: &str) -> bool {
        self.state.lock().await.requests.get(request_id).is_some_and(|request| request.cancelled)
    }

    pub async fn dispatch_task(&self, task_id: String, model: String, task_kind: String, request: Value, metadata: Value) -> Result<TaskView, DispatchError> {
        let now = unix_seconds();
        let view = TaskView { id: task_id.clone(), model: model.clone(), task_kind: task_kind.clone(), status: "queued".to_owned(), progress: None, error: None, created_at: now, updated_at: now, metadata, outputs: Vec::new() };
        let commands = {
            let mut state = self.state.lock().await;
            // 跟文本路径同元组（无 cache_id → cache_miss=false）：先按 current_batch_tokens，再 active_requests，
            // 最后 last_dispatched 单调计数器做 tiebreak，取代 last_seen（与 active_requests 共变动会反复重排）。
            let selected = state
                .nodes
                .iter()
                .filter(|(_, node)| node.view.model == model && node.view.active_requests < node.view.max_concurrency && node.view.capabilities.task_kinds.iter().any(|kind| kind == &task_kind))
                .min_by_key(|(_, node)| dispatch_order(&node.view, node.last_dispatched, None))
                .map(|(node_id, _)| node_id.clone());
            let Some(node_id) = selected else {
                return Err(DispatchError::NoAvailableNode(model));
            };
            state.next_dispatch = state.next_dispatch.wrapping_add(1).max(1);
            let dispatch = state.next_dispatch;
            if let Some(node) = state.nodes.get_mut(&node_id) {
                node.last_dispatched = dispatch;
            }
            let commands = state.nodes.get(&node_id).expect("刚选择的节点必须存在").commands.clone();
            state.tasks.insert(task_id.clone(), InflightTask { node_id: node_id.clone(), view: view.clone(), request: request.clone(), expected_outputs: BTreeMap::new(), expected_output_count: 0 });
            if let Some(node) = state.nodes.get_mut(&node_id) {
                node.view.active_requests += 1;
            }
            commands
        };
        if !commands.send(SchedulerMessage::NewTask { task_id: task_id.clone(), model, task_kind, request }).await {
            self.update_task_status("", &task_id, "failed", Some("选中的节点已经断开".to_owned()), Vec::new()).await;
            return Err(DispatchError::NodeDisconnected);
        }
        Ok(view)
    }

    async fn update_task_status(&self, node_id: &str, task_id: &str, status: &str, error: Option<String>, outputs: Vec<ArtifactDescriptor>) {
        if !matches!(status, "queued" | "running" | "uploading" | "succeeded" | "failed" | "cancelled") {
            return;
        }
        let mut state = self.state.lock().await;
        let Some(task) = state.tasks.get_mut(task_id) else {
            return;
        };
        if !node_id.is_empty() && task.node_id != node_id {
            return;
        }
        let was_terminal = matches!(task.view.status.as_str(), "succeeded" | "failed" | "cancelled");
        if was_terminal {
            return;
        }
        let (status, error, expected_outputs) = if status == "succeeded" {
            ("failed", Some("节点不能直接声明任务成功，必须等待全部 artifact commit".to_owned()), None)
        } else if status == "uploading" {
            match validate_artifact_descriptors(outputs) {
                Ok(expected) if task.expected_outputs.is_empty() || task.expected_outputs == expected => (status, error, Some(expected)),
                Ok(_) => ("failed", Some("节点重报的 artifact 清单与原清单不一致".to_owned()), None),
                Err(message) => ("failed", Some(message), None),
            }
        } else if outputs.is_empty() {
            (status, error, None)
        } else {
            ("failed", Some(format!("task status={status} 不能携带 artifact 清单")), None)
        };
        task.view.status = status.to_owned();
        task.view.updated_at = unix_seconds();
        task.view.error = error;
        if let Some(expected) = expected_outputs {
            task.expected_output_count = expected.len();
            task.expected_outputs = expected;
        }
        let owner = task.node_id.clone();
        let terminal = matches!(status, "succeeded" | "failed" | "cancelled");
        if terminal && let Some(node) = state.nodes.get_mut(&owner) {
            node.view.active_requests = node.view.active_requests.saturating_sub(1);
        }
    }

    async fn update_task_progress(&self, node_id: &str, task_id: &str, progress: TaskProgress) {
        let mut state = self.state.lock().await;
        let Some(task) = state.tasks.get_mut(task_id) else {
            return;
        };
        if task.node_id != node_id || matches!(task.view.status.as_str(), "succeeded" | "failed" | "cancelled") {
            return;
        }
        task.view.progress = Some(progress);
        task.view.updated_at = unix_seconds();
    }

    pub async fn task(&self, task_id: &str) -> Option<TaskView> {
        self.state.lock().await.tasks.get(task_id).map(|task| task.view.clone())
    }

    pub async fn tasks(&self, task_kind: Option<&str>) -> Vec<TaskView> {
        self.state.lock().await.tasks.values().filter(|task| task_kind.is_none_or(|kind| task.view.task_kind == kind)).map(|task| task.view.clone()).collect()
    }

    pub async fn cancel_task(&self, task_id: &str) -> Option<TaskView> {
        let (commands, view) = {
            let mut state = self.state.lock().await;
            let task = state.tasks.get_mut(task_id)?;
            if matches!(task.view.status.as_str(), "succeeded" | "failed" | "cancelled") {
                return Some(task.view.clone());
            }
            task.view.status = "cancelled".to_owned();
            task.view.updated_at = unix_seconds();
            let node_id = task.node_id.clone();
            let view = task.view.clone();
            let commands = state.nodes.get_mut(&node_id).map(|node| {
                node.view.active_requests = node.view.active_requests.saturating_sub(1);
                node.commands.clone()
            });
            (commands, view)
        };
        if let Some(commands) = commands {
            let _ = commands.send(SchedulerMessage::Cancel { request_id: task_id.to_owned() }).await;
        }
        Some(view)
    }

    pub async fn remove_task(&self, task_id: &str) -> Option<TaskView> {
        let task = {
            let mut state = self.state.lock().await;
            let task = state.tasks.get(task_id)?;
            if !matches!(task.view.status.as_str(), "succeeded" | "failed" | "cancelled") {
                return None;
            }
            state.tasks.remove(task_id).map(|task| task.view)
        }?;
        if let Some(directory) = self.artifact_path(task_id, "output").and_then(|path| path.parent().map(PathBuf::from)) {
            let _ = tokio::fs::remove_dir_all(directory).await;
        }
        Some(task)
    }

    pub fn artifact_path(&self, task_id: &str, artifact_id: &str) -> Option<PathBuf> {
        if !safe_id(task_id) || !safe_id(artifact_id) {
            return None;
        }
        Some(self.artifact_dir.join(task_id).join(artifact_id))
    }

    async fn receive_artifact(&self, node_id: &str, connection: u64, mut recv: iroh::endpoint::RecvStream) -> Result<(), DynError> {
        let header_bytes = recv.read_u32().await? as usize;
        if header_bytes == 0 || header_bytes > 64 * 1024 {
            return Err(format!("artifact header {header_bytes} bytes 非法").into());
        }
        let mut encoded = vec![0u8; header_bytes];
        recv.read_exact(&mut encoded).await?;
        let header: ArtifactHeader = serde_json::from_slice(&encoded)?;
        if header.artifact.bytes == 0 || header.artifact.bytes > MAX_ARTIFACT_BYTES || header.output_count == 0 {
            return Err(format!("artifact {} bytes={} output_count={} 非法", header.artifact.id, header.artifact.bytes, header.output_count).into());
        }
        let commands = {
            let state = self.state.lock().await;
            let task = state.tasks.get(&header.task_id).filter(|task| task.node_id == node_id).ok_or_else(|| format!("artifact task {} 不属于节点 {node_id}", header.task_id))?;
            let node = state.nodes.get(node_id).filter(|node| node.connection == connection).ok_or_else(|| format!("artifact 节点 {node_id} 连接已失效"))?;
            let expected = task.expected_outputs.get(&header.artifact.id).ok_or_else(|| format!("artifact {} 不在任务 {} 的预期输出中", header.artifact.id, header.task_id))?;
            if expected != &header.artifact {
                return Err(format!("artifact {} 描述与任务声明不一致: actual={:?} expected={expected:?}", header.artifact.id, header.artifact).into());
            }
            if header.output_count != task.expected_output_count || task.expected_output_count != task.expected_outputs.len() {
                return Err(format!("artifact {} output_count={}，任务预期 {}/{}", header.artifact.id, header.output_count, task.expected_output_count, task.expected_outputs.len()).into());
            }
            if matches!(task.view.status.as_str(), "failed" | "cancelled") {
                return Err(format!("artifact task {} 不属于节点 {node_id}", header.task_id).into());
            }
            node.commands.clone()
        };
        let final_path = self.artifact_path(&header.task_id, &header.artifact.id).ok_or("artifact id 非法")?;
        let directory = final_path.parent().ok_or("artifact 路径没有父目录")?;
        tokio::fs::create_dir_all(directory).await?;
        let sequence = ARTIFACT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = directory.join(format!(".{}.{}-{sequence}.part", header.artifact.id, connection));
        let mut file = tokio::fs::File::create(&temporary).await?;
        let copied = tokio::io::copy(&mut recv.take(header.artifact.bytes), &mut file).await?;
        if copied != header.artifact.bytes {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(format!("artifact {} 只收到 {copied}/{} bytes", header.artifact.id, header.artifact.bytes).into());
        }
        file.flush().await?;
        file.sync_all().await?;
        drop(file);
        tokio::fs::rename(&temporary, &final_path).await?;
        tokio::fs::File::open(directory).await?.sync_all().await?;

        {
            let mut state = self.state.lock().await;
            let task = state.tasks.get_mut(&header.task_id).ok_or_else(|| format!("artifact task {} 已不存在", header.task_id))?;
            if matches!(task.view.status.as_str(), "failed" | "cancelled") {
                return Err(format!("artifact task {} 已终止", header.task_id).into());
            }
            task.view.outputs.retain(|output| output.id != header.artifact.id);
            task.view.outputs.push(StoredArtifact {
                id: header.artifact.id.clone(),
                file_name: header.artifact.file_name,
                content_type: header.artifact.content_type,
                bytes: header.artifact.bytes,
                url: format!("{}/v1/artifacts/{}/{}", self.public_base_url, header.task_id, header.artifact.id),
            });
            task.view.updated_at = unix_seconds();
            if task.view.outputs.len() == task.expected_output_count {
                let was_terminal = matches!(task.view.status.as_str(), "succeeded" | "failed" | "cancelled");
                task.view.status = "succeeded".to_owned();
                let owner = task.node_id.clone();
                if !was_terminal && let Some(node) = state.nodes.get_mut(&owner) {
                    node.view.active_requests = node.view.active_requests.saturating_sub(1);
                }
            }
        }
        if !commands.send(SchedulerMessage::ArtifactCommitted { task_id: header.task_id, artifact_id: header.artifact.id }).await {
            return Err("artifact committed ACK 发送失败".into());
        }
        Ok(())
    }
}

fn registered_max_concurrency(requested: usize, capabilities: &NodeCapabilities) -> usize {
    let requested = requested.clamp(1, NodeRuntime::MAX_CONCURRENT_SESSIONS);
    if capabilities.kv_cache_devices.is_empty() {
        return requested;
    }
    let page_tokens = capabilities.kv_reservation_page_tokens.max(1);
    let min_tokens = capabilities.kv_cache_devices.iter().map(|device| device.token_capacity).min().unwrap_or(0);
    requested.min(min_tokens / page_tokens)
}

fn node_has_capacity(node: &RegisteredNode, additional: usize) -> bool {
    node.view.active_requests < node.view.max_concurrency
        && if additional == 1 {
            node.view.runtime.decode.saturating_add(node.pending_pressure) <= NodeRuntime::APPEND_DECODE_LIMIT
        } else {
            node.view.runtime.scheduling_pressure().saturating_add(node.pending_pressure).saturating_add(additional) <= NodeRuntime::MAX_SCHEDULING_PRESSURE
        }
}

/// 节点断连时 video_generation 等持久化任务会置回 queued 等待原节点恢复；若原
/// 节点再未上线且没有别的接管者，任务会永久滞留。新节点注册时把这类孤儿任务
/// (node 已消失、同模型、新节点声明了对应 task_kind 且并发未满)立即改派。
/// NewTask 由调用方在锁外发送；发送失败意味着新节点也断开，这些任务会在它的
/// unregister 中重新进入 queued，等待下一个接管者。
fn adopt_orphaned_tasks(state: &mut RegistryState, node_id: &str) -> Vec<AdoptedTask> {
    let Some(node) = state.nodes.get(node_id) else {
        return Vec::new();
    };
    let (model, task_kinds, mut slots) = (node.view.model.clone(), node.view.capabilities.task_kinds.clone(), node.view.max_concurrency.saturating_sub(node.view.active_requests));
    let mut adopted = Vec::new();
    if slots == 0 || task_kinds.is_empty() {
        return adopted;
    }
    for (task_id, task) in state.tasks.iter_mut() {
        if slots == 0 {
            break;
        }
        if task.view.status != "queued" || state.nodes.contains_key(&task.node_id) || task.view.model != model || !task_kinds.contains(&task.view.task_kind) {
            continue;
        }
        task.node_id = node_id.to_owned();
        task.view.error = None;
        task.view.updated_at = unix_seconds();
        slots -= 1;
        if let Some(node) = state.nodes.get_mut(node_id) {
            node.view.active_requests += 1;
        }
        adopted.push(AdoptedTask { task_id: task_id.clone(), task_kind: task.view.task_kind.clone(), request: task.request.clone() });
    }
    adopted
}

fn dispatch_order(node: &AvailableNode, last_dispatched: u64, cache_id: Option<&str>) -> (bool, usize, usize, u64) {
    let cache_miss = cache_id.is_some_and(|cache_id| !node.caches.iter().any(|cache| cache.cache_id == cache_id));
    (cache_miss, node.runtime.current_batch_tokens, node.active_requests, last_dispatched)
}

fn cache_location_rank(runtime: &NodeRuntime, cache_id: &str) -> u8 {
    if runtime.memory_cache_ids.iter().any(|id| id == cache_id) {
        0
    } else if runtime.ssd_cache_ids.iter().any(|id| id == cache_id) {
        1
    } else {
        2
    }
}

pub struct SchedulerService {
    scheduler: Scheduler,
    endpoint: Endpoint,
    accept_task: JoinHandle<()>,
    ticket: String,
}

impl SchedulerService {
    pub async fn bind(api_key: Option<String>, config: SchedulerConfig) -> Result<Self, DynError> {
        let mut builder = Endpoint::builder(presets::N0).alpns(vec![SCHEDULER_ALPN.to_vec()]);
        if let Some(secret_key) = config.iroh.secret_key.clone() {
            builder = builder.secret_key(secret_key);
        }
        if let Some(bind_addr) = config.iroh.bind_addr.as_deref() {
            // 固定身份还必须固定 UDP 端口，节点保存的 scheduler ticket 才能跨服务重启复用。
            builder = builder.clear_ip_transports().bind_addr(bind_addr).map_err(|error| std::io::Error::other(format!("解析 scheduler iroh.bind_addr={bind_addr}: {error}")))?;
        }
        let endpoint = builder.bind().await?;
        #[cfg(not(test))]
        tokio::time::timeout(Duration::from_secs(15), endpoint.online()).await.map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "scheduler 连接 iroh 官方 relay 超时"))?;
        let ticket = EndpointTicket::new(endpoint.addr()).to_string();
        let scheduler = Scheduler::new(&config);
        let accept_task = tokio::spawn(accept_loop(endpoint.clone(), scheduler.clone(), api_key));
        Ok(Self { scheduler, endpoint, accept_task, ticket })
    }

    pub fn scheduler(&self) -> Scheduler {
        self.scheduler.clone()
    }

    pub fn ticket(&self) -> &str {
        &self.ticket
    }

    pub async fn shutdown(self) {
        self.endpoint.close().await;
        let _ = self.accept_task.await;
    }
}

async fn accept_loop(endpoint: Endpoint, scheduler: Scheduler, api_key: Option<String>) {
    while let Some(accepting) = endpoint.accept().await {
        let scheduler = scheduler.clone();
        let api_key = api_key.clone();
        tokio::spawn(async move {
            let result = async {
                let connection = accepting.await?;
                handle_connection(connection, scheduler, api_key).await
            }
            .await;
            if let Err(error) = result {
                eprintln!("[scheduler] iroh 节点连接失败: {error}");
            }
            Ok::<(), DynError>(())
        });
    }
}

async fn handle_connection(connection: iroh::endpoint::Connection, scheduler: Scheduler, api_key: Option<String>) -> Result<(), DynError> {
    let peer_id = connection.remote_id().to_string();
    let (send, recv) = timeout(Duration::from_secs(10), connection.accept_bi()).await??;
    let mut reader = BufReader::new(recv);
    let first: NodeMessage = timeout(Duration::from_secs(10), read_json_line(&mut reader)).await??;
    let NodeMessage::Register { protocol_version, api_key: supplied_key, model, max_concurrency, caches, capabilities, runtime } = first else {
        return Err("iroh 首条消息必须是 register".into());
    };
    if protocol_version != SCHEDULER_PROTOCOL_VERSION {
        return Err(format!("scheduler protocol {protocol_version} 不受支持").into());
    }
    if !authorized(api_key.as_deref(), supplied_key.as_deref()) {
        return Err("scheduler API key 无效".into());
    }
    if model.trim().is_empty() {
        return Err("节点 model 不能为空".into());
    }

    let (command_tx, mut command_rx) = mpsc::channel::<SchedulerMessage>(64);
    let writer = tokio::spawn(async move {
        let mut send = send;
        while let Some(message) = command_rx.recv().await {
            write_json_line(&mut send, &message).await?;
        }
        send.finish()?;
        Ok::<(), DynError>(())
    });
    let (connection_id, task_ids, adopted) = scheduler.register(peer_id.clone(), model.clone(), max_concurrency, caches, capabilities, runtime, NodeCommands::Wire(command_tx.clone()), Some(connection.clone())).await;
    command_tx.send(SchedulerMessage::Registered { node_id: peer_id.clone(), heartbeat_seconds: 5, task_ids }).await.map_err(|_| "节点注册确认发送失败")?;
    for task in adopted {
        // 改派失败仅意味着新节点已断开;任务的 node_id 已指向它,随其 unregister 重新排队。
        if command_tx.send(SchedulerMessage::NewTask { task_id: task.task_id, model: model.clone(), task_kind: task.task_kind, request: task.request }).await.is_err() {
            break;
        }
    }
    eprintln!("[scheduler] 节点上线 id={peer_id} model={model}");

    let artifact_scheduler = scheduler.clone();
    let artifact_peer = peer_id.clone();
    let artifact_connection = connection.clone();
    let artifacts = tokio::spawn(async move {
        while let Ok(recv) = artifact_connection.accept_uni().await {
            if let Err(error) = artifact_scheduler.receive_artifact(&artifact_peer, connection_id, recv).await {
                eprintln!("[scheduler] 节点 {artifact_peer} artifact 接收失败: {error}");
                // 没有失败 ACK；继续保留连接只会让节点永久等待 committed。
                artifact_connection.close(2u32.into(), b"artifact protocol error");
                break;
            }
        }
    });

    let result = async {
        loop {
            let message: NodeMessage = read_json_line(&mut reader).await?;
            match message {
                NodeMessage::Heartbeat { caches, runtime } => {
                    scheduler.update_heartbeat(&peer_id, connection_id, caches, runtime).await;
                }
                NodeMessage::RuntimeChanged { runtime } => scheduler.update_runtime(&peer_id, connection_id, runtime).await,
                NodeMessage::Event { request_id, event } => {
                    scheduler.publish(&peer_id, &request_id, event).await;
                }
                NodeMessage::TaskStatus { task_id, status, error, outputs } => {
                    scheduler.update_task_status(&peer_id, &task_id, &status, error, outputs).await;
                }
                NodeMessage::TaskProgress { task_id, progress } => {
                    scheduler.update_task_progress(&peer_id, &task_id, progress).await;
                }
                NodeMessage::Register { .. } => return Err::<(), DynError>("同一连接不能重复注册".into()),
            }
        }
    }
    .await;
    scheduler.unregister(&peer_id, connection_id).await;
    drop(command_tx);
    writer.abort();
    artifacts.abort();
    eprintln!("[scheduler] 节点下线 id={peer_id} model={model}");
    result
}

pub async fn write_json_line<W, T>(writer: &mut W, value: &T) -> Result<(), DynError>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let mut bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(format!("scheduler 消息 {} bytes 超过上限", bytes.len()).into());
    }
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn read_json_line<R, T>(mut reader: &mut BufReader<R>) -> Result<T, DynError>
where
    R: tokio::io::AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut line = Vec::new();
    // take 封顶:对端发送永不带换行的字节流时,read_until 不再无界累积内存。
    use tokio::io::AsyncBufReadExt;
    let mut capped = (&mut reader).take((MAX_MESSAGE_BYTES + 1) as u64);
    let bytes = capped.read_until(b'\n', &mut line).await?;
    if bytes == 0 {
        return Err("scheduler 连接已经关闭".into());
    }
    if line.len() > MAX_MESSAGE_BYTES {
        return Err(format!("scheduler 消息 {} bytes 超过上限", line.len()).into());
    }
    Ok(serde_json::from_slice(&line)?)
}

fn authorized(expected: Option<&str>, supplied: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    let Some(supplied) = supplied else {
        return false;
    };
    if expected.len() != supplied.len() {
        return false;
    }
    expected.as_bytes().iter().zip(supplied.as_bytes()).fold(0u8, |difference, (left, right)| difference | (left ^ right)) == 0
}

fn unix_seconds() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub(super) fn safe_id(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn validate_artifact_descriptors(outputs: Vec<ArtifactDescriptor>) -> Result<BTreeMap<String, ArtifactDescriptor>, String> {
    if outputs.is_empty() || outputs.len() > 256 {
        return Err(format!("artifact 清单数量 {} 非法", outputs.len()));
    }
    let mut expected = BTreeMap::new();
    for output in outputs {
        validate_artifact_descriptor(&output)?;
        if expected.contains_key(&output.id) {
            return Err(format!("artifact id 非法或重复: {:?}", output.id));
        }
        expected.insert(output.id.clone(), output);
    }
    Ok(expected)
}

pub(super) fn validate_artifact_descriptor(output: &ArtifactDescriptor) -> Result<(), String> {
    let file_name = Path::new(&output.file_name);
    if !safe_id(&output.id) {
        return Err(format!("artifact id 非法: {:?}", output.id));
    }
    if file_name.file_name().is_none_or(|name| name != file_name.as_os_str()) || matches!(output.file_name.as_str(), "." | "..") {
        return Err(format!("artifact {} file_name 不是单一文件名", output.id));
    }
    if output.content_type.trim().is_empty() || output.content_type.chars().any(char::is_control) {
        return Err(format!("artifact {} content_type 非法", output.id));
    }
    if output.bytes == 0 || output.bytes > MAX_ARTIFACT_BYTES {
        return Err(format!("artifact {} bytes={} 非法", output.id, output.bytes));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use serde_json::json;

    use super::*;

    /// 最小终态:只关心 terminal_tokens,encode/decode 不参与(无 swap)。
    struct AlignedState(Vec<u32>);
    impl crate::kv_cache::terminal_cache::TerminalSnapshot for AlignedState {
        type Resources = ();
        fn encode(&self) -> Result<Vec<u8>, String> {
            Ok(Vec::new())
        }
        fn decode(_bytes: &[u8], _resources: &()) -> Result<Self, String> {
            unreachable!()
        }
        fn terminal_tokens(&self) -> &[u32] {
            &self.0
        }
        fn info(&self) -> &CacheInfo {
            static EMPTY: std::sync::OnceLock<CacheInfo> = std::sync::OnceLock::new();
            EMPTY.get_or_init(CacheInfo::default)
        }
    }

    fn info_with_id(cache_id: &str) -> CacheInfo {
        CacheInfo { cache_id: cache_id.to_owned(), ..CacheInfo::default() }
    }

    #[test]
    fn resume_terminal_append命中与未命中() {
        let first = json!({ "model": "m", "messages": [{"role": "user", "content": "问"}] });
        let cache_id = terminal_cache_id(&first, "答", &[]).unwrap();
        let mut request = first.clone();
        request["messages"].as_array_mut().unwrap().push(json!({"role": "assistant", "content": "答"}));
        request["messages"].as_array_mut().unwrap().push(json!({"role": "user", "content": "继续"}));
        request["cache_id"] = json!(cache_id);

        // 命中:返回终态与渲染闭包产出的增量段(拼接式,不做前缀比对)。
        let mut sessions = crate::kv_cache::terminal_cache::TerminalSessions::<AlignedState>::new(2, None);
        assert!(retain_terminal_session(&mut sessions, AlignedState(vec![1, 2, 3]), info_with_id(&cache_id)).is_some());
        let (state, suffix) = resume_terminal_append(
            &mut sessions,
            &request,
            |assistant| {
                assert_eq!(assistant, 1, "assistant 边界下标(messages 里第 1 位)");
                Ok(vec![4, 5])
            },
            &(),
        )
        .unwrap()
        .expect("append 必须命中");
        assert_eq!(state.0, vec![1, 2, 3]);
        assert_eq!(suffix, vec![4, 5]);

        // cache_id 未携带 → None。
        let mut no_id = request.clone();
        no_id["cache_id"].take();
        let mut sessions = crate::kv_cache::terminal_cache::TerminalSessions::<AlignedState>::new(2, None);
        assert!(retain_terminal_session(&mut sessions, AlignedState(vec![1, 2, 3]), info_with_id(&cache_id)).is_some());
        assert!(resume_terminal_append(&mut sessions, &no_id, |_assistant| Ok(Vec::new()), &()).unwrap().is_none());
    }

    #[test]
    fn terminal_resume统一验证assistant边界() {
        let mut request = json!({
            "model": "glm-5.2",
            "messages": [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "答案"},
                {"role": "user", "content": "继续"}
            ]
        });
        let (cache_id, assistant) = request_resume_boundary(&request).unwrap().unwrap();
        request["cache_id"] = Value::String(cache_id.clone());
        assert_eq!(request_terminal_resume(&request).unwrap(), TerminalResume::Match { cache_id, assistant });
        request["cache_id"] = Value::String("wrong".to_owned());
        assert!(matches!(request_terminal_resume(&request).unwrap(), TerminalResume::Mismatch { requested, .. } if requested == "wrong"));
    }

    #[test]
    fn terminal_hash跨模型排除私有reasoning() {
        let previous = json!({
            "model": "reasoning-model",
            "reasoning_effort": "high",
            "_zllm_cache_namespace": "tenant-a",
            "messages": [{
                "role": "user", "content": "问题", "reasoning_content": null, "name": null,
                "tool_call_id": null, "tool_calls": null
            }]
        });
        let cache_id = terminal_cache_id(&previous, "私有推理</think>最终答案", &[]).unwrap();
        let next = json!({
            "model": "reasoning-model",
            "reasoning_effort": "high",
            "cache_id": cache_id,
            "_zllm_cache_namespace": "tenant-a",
            "messages": [
                {
                    "role": "user", "content": "问题", "reasoning_content": null, "name": null,
                    "tool_call_id": null, "tool_calls": null
                },
                {
                    "role": "assistant", "content": "最终答案", "reasoning_content": null, "name": null,
                    "tool_call_id": null, "tool_calls": null
                },
                {
                    "role": "user", "content": "继续", "reasoning_content": null, "name": null,
                    "tool_call_id": null, "tool_calls": null
                }
            ]
        });
        assert!(matches!(request_terminal_resume(&next).unwrap(), TerminalResume::Match { .. }));
        assert_eq!(client_replayable_response(&previous, "未结束的私有推理"), "");
        assert_eq!(client_replayable_response(&json!({}), "原始输出</think>正文"), "原始输出正文");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn iroh_register_dispatch_and_stream_events() {
        let service = SchedulerService::bind(None, SchedulerConfig::default()).await.unwrap();
        let scheduler = service.scheduler();
        let ticket = EndpointTicket::from_str(service.ticket()).unwrap();
        let client = Endpoint::builder(presets::N0).clear_relay_transports().bind().await.unwrap();
        let connection = client.connect(ticket.endpoint_addr().clone(), SCHEDULER_ALPN).await.unwrap();
        let (mut send, recv) = connection.open_bi().await.unwrap();
        write_json_line(
            &mut send,
            &NodeMessage::Register {
                protocol_version: SCHEDULER_PROTOCOL_VERSION,
                api_key: None,
                model: "ornith".to_owned(),
                max_concurrency: 2,
                capabilities: NodeCapabilities::default(),
                runtime: NodeRuntime::default(),
                caches: vec![CacheInfo { cache_id: "session-a".to_owned(), model_key: "ornith-test".to_owned(), cache_format: "gqa-int8".to_owned(), last_layer: 7, prompt_tokens: 128, bytes: 4096, modified_unix: 1 }],
            },
        )
        .await
        .unwrap();
        let mut reader = BufReader::new(recv);
        let registered: SchedulerMessage = read_json_line(&mut reader).await.unwrap();
        assert!(matches!(registered, SchedulerMessage::Registered { .. }));
        assert_eq!(scheduler.nodes().await.len(), 1);

        let request_id = "req_test".to_owned();
        let mut events = scheduler.dispatch(request_id.clone(), "ornith".to_owned(), Some("session-a"), json!({"model":"ornith","messages":[{"role":"user","content":"hi"}]})).await.unwrap();
        let command: SchedulerMessage = read_json_line(&mut reader).await.unwrap();
        assert!(matches!(command, SchedulerMessage::NewPrefill { ref request_id, .. } if request_id == "req_test"));

        for event in [InferenceEvent::Started, InferenceEvent::Token { token_id: 42, text: "ok".to_owned() }, InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 1 }] {
            write_json_line(&mut send, &NodeMessage::Event { request_id: request_id.clone(), event }).await.unwrap();
        }
        assert!(matches!(events.recv().await, Some(InferenceEvent::Started)));
        assert!(matches!(events.recv().await, Some(InferenceEvent::Token { token_id: 42, ref text }) if text == "ok"));
        assert!(matches!(events.recv().await, Some(InferenceEvent::Completed { completion_tokens: 1, .. })));
        assert_eq!(scheduler.nodes().await[0].active_requests, 0);

        // cache_owner 能查到注册的 cache（供 GET /v1/caches/{cache_id}）。
        assert!(scheduler.cache_owner("session-a").await.is_some(), "cache_owner 应找到已注册的 session-a");
        write_json_line(&mut send, &NodeMessage::Heartbeat { caches: Vec::new(), runtime: NodeRuntime::default() }).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while scheduler.cache_owner("session-a").await.is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        // cancel 只给旧 writer 发撤销信号(订阅者清零才发);同 cache_id 的重试
        // 请求即时合并到现有事件流(tee),不排队、不占节点执行槽。
        let cancelled_events = scheduler.dispatch("req_cancelled_writer".to_owned(), "ornith".to_owned(), Some("retry-cache"), json!({"model":"ornith"})).await.unwrap();
        assert!(matches!(read_json_line::<_, SchedulerMessage>(&mut reader).await.unwrap(), SchedulerMessage::NewPrefill { ref request_id, .. } if request_id == "req_cancelled_writer"));
        scheduler.cancel("req_cancelled_writer").await;
        assert!(matches!(read_json_line::<_, SchedulerMessage>(&mut reader).await.unwrap(), SchedulerMessage::Cancel { ref request_id } if request_id == "req_cancelled_writer"));
        let mut retry_tee = scheduler.dispatch("req_early_retry".to_owned(), "ornith".to_owned(), Some("retry-cache"), json!({"model":"ornith"})).await.unwrap();
        assert_eq!(scheduler.nodes().await[0].active_requests, 1, "合并请求不占节点执行槽");
        drop(cancelled_events);

        let retry_cache = CacheInfo { cache_id: "retry-cache".to_owned(), model_key: "ornith-test".to_owned(), cache_format: "gqa-int8".to_owned(), last_layer: 7, prompt_tokens: 96, bytes: 3072, modified_unix: 2 };
        write_json_line(&mut send, &NodeMessage::Heartbeat { caches: vec![retry_cache], runtime: NodeRuntime::default() }).await.unwrap();
        write_json_line(&mut send, &NodeMessage::Event { request_id: "req_cancelled_writer".to_owned(), event: InferenceEvent::Completed { finish_reason: "cancelled".to_owned(), prompt_tokens: 128, completion_tokens: 0 } }).await.unwrap();
        assert!(matches!(retry_tee.recv().await, Some(InferenceEvent::Completed { ref finish_reason, .. }) if finish_reason == "cancelled"), "合并订阅者应收到重放的终态");
        tokio::time::timeout(Duration::from_secs(1), async {
            while scheduler.nodes().await[0].active_requests != 0 || scheduler.cache_owner("retry-cache").await.is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let mut retry_events = scheduler.dispatch("req_retry_after_checkpoint".to_owned(), "ornith".to_owned(), Some("retry-cache"), json!({"model":"ornith"})).await.unwrap();
        assert!(matches!(read_json_line::<_, SchedulerMessage>(&mut reader).await.unwrap(), SchedulerMessage::NewPrefill { ref request_id, .. } if request_id == "req_retry_after_checkpoint"));
        write_json_line(&mut send, &NodeMessage::Event { request_id: "req_retry_after_checkpoint".to_owned(), event: InferenceEvent::Error { message: "test complete".to_owned() } }).await.unwrap();
        assert!(matches!(retry_events.recv().await, Some(InferenceEvent::Error { .. })));

        connection.close(0u32.into(), b"test complete");
        client.close().await;
        service.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn 断连后的video孤儿任务由新节点接管() {
        let service = SchedulerService::bind(None, SchedulerConfig::default()).await.unwrap();
        let scheduler = service.scheduler();
        let ticket = EndpointTicket::from_str(service.ticket()).unwrap();
        let capabilities = |kinds: &[&str]| NodeCapabilities { task_kinds: kinds.iter().map(|kind| (*kind).to_owned()).collect(), ..NodeCapabilities::default() };

        // 节点 A 接单后断连:video_generation 任务回 queued 等待恢复。
        let client_a = Endpoint::builder(presets::N0).clear_relay_transports().bind().await.unwrap();
        let connection_a = client_a.connect(ticket.endpoint_addr().clone(), SCHEDULER_ALPN).await.unwrap();
        let (mut send_a, recv_a) = connection_a.open_bi().await.unwrap();
        write_json_line(
            &mut send_a,
            &NodeMessage::Register {
                protocol_version: SCHEDULER_PROTOCOL_VERSION,
                api_key: None,
                model: "h3".to_owned(),
                max_concurrency: 2,
                capabilities: capabilities(&["video_generation"]),
                runtime: NodeRuntime::default(),
                caches: Vec::new(),
            },
        )
        .await
        .unwrap();
        let mut reader_a = BufReader::new(recv_a);
        let _: SchedulerMessage = read_json_line(&mut reader_a).await.unwrap();
        scheduler.dispatch_task("task_orphan".to_owned(), "h3".to_owned(), "video_generation".to_owned(), json!({"prompt": "x"}), json!({})).await.unwrap();
        assert!(matches!(read_json_line::<_, SchedulerMessage>(&mut reader_a).await.unwrap(), SchedulerMessage::NewTask { ref task_id, .. } if task_id == "task_orphan"));

        connection_a.close(0u32.into(), b"gone");
        client_a.close().await;
        tokio::time::timeout(Duration::from_secs(2), async {
            while !scheduler.nodes().await.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let task = scheduler.task("task_orphan").await.unwrap();
        assert_eq!(task.status, "queued", "video 任务断连后应等待恢复而非失败: {task:?}");

        // 新节点上线即接管孤儿任务,重发 NewTask;否则任务会永久滞留。
        let client_b = Endpoint::builder(presets::N0).clear_relay_transports().bind().await.unwrap();
        let connection_b = client_b.connect(ticket.endpoint_addr().clone(), SCHEDULER_ALPN).await.unwrap();
        let (mut send_b, recv_b) = connection_b.open_bi().await.unwrap();
        write_json_line(
            &mut send_b,
            &NodeMessage::Register {
                protocol_version: SCHEDULER_PROTOCOL_VERSION,
                api_key: None,
                model: "h3".to_owned(),
                max_concurrency: 2,
                capabilities: capabilities(&["video_generation"]),
                runtime: NodeRuntime::default(),
                caches: Vec::new(),
            },
        )
        .await
        .unwrap();
        let mut reader_b = BufReader::new(recv_b);
        let _: SchedulerMessage = read_json_line(&mut reader_b).await.unwrap();
        let adopted = tokio::time::timeout(Duration::from_secs(2), read_json_line::<_, SchedulerMessage>(&mut reader_b)).await.unwrap().unwrap();
        assert!(
            matches!(adopted, SchedulerMessage::NewTask { ref task_id, ref task_kind, ref request, .. } if task_id == "task_orphan" && task_kind == "video_generation" && request["prompt"] == "x"),
            "新节点应收到孤儿任务改派: {adopted:?}"
        );
        let task = scheduler.task("task_orphan").await.unwrap();
        assert_eq!(task.status, "queued");
        assert!(task.error.is_none(), "改派后应清除等待恢复的说明: {task:?}");
        assert_eq!(scheduler.nodes().await[0].active_requests, 1);

        connection_b.close(0u32.into(), b"done");
        client_b.close().await;
        service.shutdown().await;
    }

    #[test]
    fn dispatch_prefers_cache_then_load_then_active_requests() {
        let cache = CacheInfo { cache_id: "session-a".to_owned(), model_key: "glm-5.2".to_owned(), cache_format: "mla".to_owned(), last_layer: 77, prompt_tokens: 128, bytes: 4096, modified_unix: 1 };
        let node = |load, active, caches: Vec<CacheInfo>| AvailableNode {
            node_id: String::new(),
            model: "glm-5.2".to_owned(),
            caches,
            active_requests: active,
            max_concurrency: 4,
            current_load: active,
            max_load: 4,
            capabilities: NodeCapabilities::default(),
            runtime: NodeRuntime { current_batch_tokens: load, ..NodeRuntime::default() },
            registered_at: 0,
            registration_seq: 0,
            last_seen: 0,
        };
        let cached = node(10_000, 3, vec![cache]);
        let idle = node(0, 0, Vec::new());
        assert!(dispatch_order(&cached, 9, Some("session-a")) < dispatch_order(&idle, 0, Some("session-a")));

        let lighter = node(64, 2, Vec::new());
        let heavier = node(128, 0, Vec::new());
        assert!(dispatch_order(&lighter, 9, None) < dispatch_order(&heavier, 0, None));

        let fewer = node(64, 1, Vec::new());
        let more = node(64, 2, Vec::new());
        assert!(dispatch_order(&fewer, 9, None) < dispatch_order(&more, 0, None));
    }

    #[test]
    fn registration_uses_kv_capacity_but_caps_node_at_fourteen_sessions() {
        let capabilities = NodeCapabilities {
            kv_cache_devices: vec![
                KvCacheDeviceCapacity { device: "gpu0".to_owned(), available_bytes: 8 << 30, bytes_per_token: 4096, token_capacity: 2_097_152 },
                KvCacheDeviceCapacity { device: "gpu1".to_owned(), available_bytes: 2 << 30, bytes_per_token: 4096, token_capacity: 524_288 },
            ],
            kv_reservation_page_tokens: 1024,
            ..NodeCapabilities::default()
        };
        assert_eq!(registered_max_concurrency(1024, &capabilities), 14);
        assert_eq!(registered_max_concurrency(128, &capabilities), 14);
        assert_eq!(registered_max_concurrency(128, &NodeCapabilities::default()), 14);
    }

    #[test]
    fn runtime_pressure_uses_prefill_four_to_one_ratio() {
        let runtime = NodeRuntime { new_prefill: 2, append_prefill: 3, decode: 5, ..NodeRuntime::default() };
        assert_eq!(runtime.scheduling_pressure(), 25);
    }

    #[test]
    fn cache_location_prefers_memory_then_ssd() {
        let runtime = NodeRuntime { memory_cache_ids: vec!["memory".to_owned()], ssd_cache_ids: vec!["ssd".to_owned()], ..NodeRuntime::default() };
        assert_eq!(cache_location_rank(&runtime, "memory"), 0);
        assert_eq!(cache_location_rank(&runtime, "ssd"), 1);
        assert_eq!(cache_location_rank(&runtime, "missing"), 2);
    }

    #[test]
    fn artifact_descriptor_rejects_duplicates_and_paths() {
        let descriptor = ArtifactDescriptor { id: "video".to_owned(), file_name: "video.mp4".to_owned(), content_type: "video/mp4".to_owned(), bytes: 1024 };
        assert_eq!(validate_artifact_descriptors(vec![descriptor.clone()]).unwrap().len(), 1);
        assert!(validate_artifact_descriptors(vec![descriptor.clone(), descriptor]).unwrap_err().contains("重复"));
        let outside = ArtifactDescriptor { id: "audio".to_owned(), file_name: "../audio.wav".to_owned(), content_type: "audio/wav".to_owned(), bytes: 1024 };
        assert!(validate_artifact_descriptors(vec![outside]).unwrap_err().contains("单一文件名"));
    }

    /// 本地节点注册辅助:用 `attach_local_node` 的内存通道避开 iroh。
    /// 等到 `Registered` 命令出现在 commands 流上,说明 register 已生效,可以开始 dispatch。
    /// 返回整个 `LocalNodeChannels`,让调用方持有 `messages` Sender 避免 spawned task 立即退出。
    async fn register_local_node(scheduler: &Scheduler, node_id: &str, model: &str, max_concurrency: usize, caches: Vec<CacheInfo>) -> LocalNodeChannels {
        let mut channels = scheduler.attach_local_node(node_id.to_owned());
        channels
            .messages
            .send(NodeMessage::Register { protocol_version: SCHEDULER_PROTOCOL_VERSION, api_key: None, model: model.to_owned(), max_concurrency, caches, capabilities: NodeCapabilities::default(), runtime: NodeRuntime::default() })
            .await
            .unwrap();
        // 收到 Registered 才认为 register 路径在 scheduler 侧跑完,后续 dispatch 不会落到不存在的节点。
        match tokio::time::timeout(Duration::from_secs(1), channels.commands.recv()).await {
            Ok(Some(LocalNodeCommand::Wire(SchedulerMessage::Registered { node_id: ref registered, .. }))) if registered == node_id => {}
            other => panic!("本地节点 {node_id} 注册未确认: {other:?}"),
        }
        channels
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn available_nodes_exposes_current_load_and_max_load() {
        // /v1/nodes 必须能从内部状态直接算出 available_nodes / total_max_load / total_current_load,
        // 不用 server 解析 AvailableNode 序列;这里只校验 Scheduler::nodes() 出口处派生正确。
        let scheduler = Scheduler::default();
        let _a = register_local_node(&scheduler, "node-a", "glm-5.2", 4, Vec::new()).await;
        let _b = register_local_node(&scheduler, "node-b", "glm-5.2", 8, Vec::new()).await;
        let nodes = scheduler.nodes().await;
        assert_eq!(nodes.len(), 2);
        for node in &nodes {
            assert_eq!(node.current_load, node.active_requests, "{} current_load 应等于 active_requests", node.node_id);
            assert_eq!(node.max_load, node.max_concurrency, "{} max_load 应等于 max_concurrency", node.node_id);
        }
        let total_max_load: usize = nodes.iter().map(|n| n.max_load).sum();
        let total_current_load: usize = nodes.iter().map(|n| n.current_load).sum();
        assert_eq!(total_max_load, 12);
        assert_eq!(total_current_load, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn new_prefill按四点压力在十上限排队() {
        let scheduler = Scheduler::default();
        let _node = register_local_node(&scheduler, "node", "glm-5.2", 22, Vec::new()).await;
        let mut receivers = Vec::new();
        for index in 0..2 {
            receivers.push(scheduler.dispatch(format!("req-{index}"), "glm-5.2".to_owned(), None, json!({"model":"glm-5.2"})).await.unwrap());
        }
        assert!(matches!(scheduler.dispatch("req-2".to_owned(), "glm-5.2".to_owned(), None, json!({"model":"glm-5.2"})).await, Err(DispatchError::NoAvailableNode(_))));
        assert_eq!(scheduler.nodes().await[0].active_requests, 2);
        drop(receivers);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn append_admission_at_thirteen_does_not_bypass_new_prompt_limit() {
        let scheduler = Scheduler::default();
        let cache = CacheInfo { cache_id: "session-a".to_owned(), model_key: "glm-5.2".to_owned(), cache_format: "mla".to_owned(), last_layer: 77, prompt_tokens: 4096, bytes: 4096, modified_unix: 1 };
        let _node = register_local_node(&scheduler, "node", "glm-5.2", 14, vec![cache]).await;
        {
            let mut state = scheduler.state.lock().await;
            let node = state.nodes.get_mut("node").unwrap();
            node.view.runtime.decode = 13;
            node.view.active_requests = 13;
        }
        assert!(matches!(scheduler.dispatch("new".to_owned(), "glm-5.2".to_owned(), None, json!({})).await, Err(DispatchError::NoAvailableNode(_))));
        assert!(matches!(scheduler.dispatch("missing".to_owned(), "glm-5.2".to_owned(), Some("missing"), json!({})).await, Err(DispatchError::NoAvailableNode(_))));
        let _events = scheduler.dispatch("append".to_owned(), "glm-5.2".to_owned(), Some("session-a"), json!({})).await.unwrap();
        assert_eq!(scheduler.nodes().await[0].active_requests, 14);
        // 新的 cache_id 不会被同 writer 合流掩盖容量边界。
        let mut state = scheduler.state.lock().await;
        let node = state.nodes.get_mut("node").unwrap();
        node.view.runtime.decode = 14;
        assert!(!node_has_capacity(node, 1));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dispatch_routes_to_cache_owner_when_under_capacity() {
        // cache_id 命中 + owner 未满载 → 必须落到 owner 节点(无视另一节点的负载更小)。
        let scheduler = Scheduler::default();
        let cache = CacheInfo { cache_id: "session-a".to_owned(), model_key: "glm-5.2".to_owned(), cache_format: "mla".to_owned(), last_layer: 77, prompt_tokens: 128, bytes: 4096, modified_unix: 1 };
        let mut owner = register_local_node(&scheduler, "owner", "glm-5.2", 4, vec![cache]).await;
        let _idle = register_local_node(&scheduler, "idle", "glm-5.2", 4, Vec::new()).await;
        let events = scheduler.dispatch("req_1".to_owned(), "glm-5.2".to_owned(), Some("session-a"), json!({"model": "glm-5.2"})).await.unwrap();
        // owner 节点命令通道应先收到 NewPrefill;idle 节点不应收到任何东西。
        // 本地节点路径下 NewPrefill 走独立变体,不包 Wire(参 scheduler.rs NodeCommands::Local 分支)。
        match tokio::time::timeout(Duration::from_millis(200), owner.commands.recv()).await {
            Ok(Some(LocalNodeCommand::NewPrefill { ref request_id, .. })) if request_id == "req_1" => {}
            other => panic!("owner 节点应先收到 NewPrefill: {other:?}"),
        }
        // 清理:推 terminal 让 active_requests 归零,避免 drop 节点时仍在 inflight。
        drop(events);
        scheduler.publish("owner", "req_1", InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 1, completion_tokens: 0 }).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn 同cache_id重试即时合并tee且订阅计数决定cancel() {
        let scheduler = Scheduler::default();
        let mut node = register_local_node(&scheduler, "node", "glm-5.2", 4, Vec::new()).await;
        let _first = scheduler.dispatch("req-1".to_owned(), "glm-5.2".to_owned(), Some("session-a"), json!({"model":"glm-5.2"})).await.unwrap();
        match tokio::time::timeout(Duration::from_millis(200), node.commands.recv()).await {
            Ok(Some(LocalNodeCommand::NewPrefill { ref request_id, .. })) if request_id == "req-1" => {}
            other => panic!("节点应收到一次 NewPrefill: {other:?}"),
        }
        // writer 进行中,事件已流入历史
        scheduler.publish("node", "req-1", InferenceEvent::Started).await;
        scheduler.publish("node", "req-1", InferenceEvent::Token { token_id: 1, text: "你".to_owned() }).await;
        // 同 cache_id 重试到达:立即合并,历史重放,不再排队等待
        let mut retry = scheduler.dispatch("req-1-retry".to_owned(), "glm-5.2".to_owned(), Some("session-a"), json!({"model":"glm-5.2"})).await.unwrap();
        assert!(matches!(retry.recv().await, Some(InferenceEvent::Started)), "重试订阅者应立即重放历史");
        assert!(matches!(retry.recv().await, Some(InferenceEvent::Token { ref text, .. }) if text == "你"));
        assert_eq!(scheduler.nodes().await[0].active_requests, 1, "合并请求不占节点执行槽");
        assert!(tokio::time::timeout(Duration::from_millis(100), node.commands.recv()).await.is_err(), "合并请求不应触发第二次 NewPrefill");
        // 第一个连接断开:重试订阅者仍存活,不得向节点发 Cancel
        scheduler.cancel("req-1").await;
        assert!(tokio::time::timeout(Duration::from_millis(50), node.commands.recv()).await.is_err(), "仍有订阅者时不应发 Cancel");
        // 终态:存活订阅者收到完整终态,合并条目与槽位全部清理
        scheduler.publish("node", "req-1", InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 1 }).await;
        assert!(matches!(retry.recv().await, Some(InferenceEvent::Completed { .. })));
        tokio::time::timeout(Duration::from_secs(1), async {
            while scheduler.nodes().await[0].active_requests != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn append排队超过未命中超时仍等待持有者() {
        let scheduler = Scheduler::new(&SchedulerConfig { dispatch_wait: Duration::from_millis(30), ..SchedulerConfig::default() });
        let cache = CacheInfo { cache_id: "session-a".to_owned(), model_key: "glm-5.2".to_owned(), cache_format: "mla".to_owned(), last_layer: 77, prompt_tokens: 128, bytes: 4096, modified_unix: 1 };
        let mut owner = register_local_node(&scheduler, "owner", "glm-5.2", 1, vec![cache]).await;
        let mut idle = register_local_node(&scheduler, "idle", "glm-5.2", 1, Vec::new()).await;
        scheduler.state.lock().await.nodes.get_mut("owner").unwrap().view.active_requests = 1;
        let waiting = scheduler.dispatch_wait("req-queued".to_owned(), "glm-5.2".to_owned(), Some("session-a"), json!({}));
        tokio::pin!(waiting);
        assert!(tokio::time::timeout(Duration::from_millis(100), &mut waiting).await.is_err());
        assert!(matches!(owner.commands.try_recv(), Ok(LocalNodeCommand::Wire(SchedulerMessage::PinCache { cache_id })) if cache_id == "session-a"));
        assert!(idle.commands.try_recv().is_err(), "命中请求不能分派到无缓存的空闲节点");
        scheduler.state.lock().await.nodes.get_mut("owner").unwrap().view.active_requests = 0;
        let _events = tokio::time::timeout(Duration::from_secs(1), waiting).await.unwrap().unwrap();
        assert!(matches!(owner.commands.try_recv(), Ok(LocalNodeCommand::NewPrefill { .. })));
        assert!(matches!(owner.commands.try_recv(), Ok(LocalNodeCommand::Wire(SchedulerMessage::UnpinCache { cache_id })) if cache_id == "session-a"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn 未命中且容量满仍按配置超时() {
        let scheduler = Scheduler::new(&SchedulerConfig { dispatch_wait: Duration::from_millis(30), ..SchedulerConfig::default() });
        let _owner = register_local_node(&scheduler, "owner", "glm-5.2", 1, Vec::new()).await;
        scheduler.state.lock().await.nodes.get_mut("owner").unwrap().view.active_requests = 1;
        let result = tokio::time::timeout(Duration::from_secs(1), scheduler.dispatch_wait("miss".to_owned(), "glm-5.2".to_owned(), Some("absent"), json!({}))).await.unwrap();
        assert!(matches!(result, Err(DispatchError::NoAvailableNode(_))));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cache_owner满载时不跨节点分叉() {
        let scheduler = Scheduler::default();
        let cache = CacheInfo { cache_id: "session-a".to_owned(), model_key: "glm-5.2".to_owned(), cache_format: "mla".to_owned(), last_layer: 77, prompt_tokens: 128, bytes: 4096, modified_unix: 1 };
        let _owner = register_local_node(&scheduler, "owner", "glm-5.2", 1, vec![cache]).await;
        let mut idle = register_local_node(&scheduler, "idle", "glm-5.2", 1, Vec::new()).await;
        scheduler.state.lock().await.nodes.get_mut("owner").unwrap().view.active_requests = 1;

        let result = scheduler.dispatch("resume".to_owned(), "glm-5.2".to_owned(), Some("session-a"), json!({"model":"glm-5.2"})).await;
        assert!(matches!(result, Err(DispatchError::NoAvailableNode(_))));
        assert!(tokio::time::timeout(Duration::from_millis(50), idle.commands.recv()).await.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dispatch_falls_back_to_idle_when_owner_full() {
        // owner max_concurrency=1,idle max=4。idle 故意报一个更大的 current_batch_tokens,
        // 让首次 dispatch 的 dispatch_order (cache_miss=false, current_batch_tokens, active, last_dispatched)
        // 必然选 owner;owner 满载后再 dispatch → filter 排除 owner,落到 idle。
        let scheduler = Scheduler::default();
        let mut owner = register_local_node(&scheduler, "owner", "glm-5.2", 1, Vec::new()).await;
        let mut idle_channels = scheduler.attach_local_node("idle".to_owned());
        idle_channels
            .messages
            .send(NodeMessage::Register {
                protocol_version: SCHEDULER_PROTOCOL_VERSION,
                api_key: None,
                model: "glm-5.2".to_owned(),
                max_concurrency: 4,
                caches: Vec::new(),
                capabilities: NodeCapabilities::default(),
                runtime: NodeRuntime { current_batch_tokens: 1024, ..NodeRuntime::default() },
            })
            .await
            .unwrap();
        // 等 idle 注册完成(Registered 命令),避免 race。
        match tokio::time::timeout(Duration::from_secs(1), idle_channels.commands.recv()).await {
            Ok(Some(LocalNodeCommand::Wire(SchedulerMessage::Registered { node_id, .. }))) if node_id == "idle" => {}
            other => panic!("idle 注册未确认: {other:?}"),
        }

        // 首次 dispatch:两个都未满,idle current_batch_tokens=1024 拉低其 rank,owner 必胜。
        let _pending_fill = scheduler.dispatch("req_fill".to_owned(), "glm-5.2".to_owned(), None, json!({"model": "glm-5.2"})).await.unwrap();
        match tokio::time::timeout(Duration::from_millis(200), owner.commands.recv()).await {
            Ok(Some(LocalNodeCommand::NewPrefill { ref request_id, .. })) if request_id == "req_fill" => {}
            other => panic!("首次 dispatch 应落到 owner: {other:?}"),
        }
        assert_eq!(scheduler.nodes().await.iter().find(|n| n.node_id == "owner").unwrap().active_requests, 1);

        // 第二次 dispatch:owner 1/1 filter 排除,落到 idle。
        let pending_after = scheduler.dispatch("req_after".to_owned(), "glm-5.2".to_owned(), None, json!({"model": "glm-5.2"})).await.unwrap();
        match tokio::time::timeout(Duration::from_millis(200), idle_channels.commands.recv()).await {
            Ok(Some(LocalNodeCommand::NewPrefill { ref request_id, .. })) if request_id == "req_after" => {}
            other => panic!("owner 满载时应落到 idle: {other:?}"),
        }
        // 清理两个 inflight,避免 drop scheduler 时序竞争。
        drop(pending_after);
        scheduler.publish("idle", "req_after", InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 1, completion_tokens: 0 }).await;
        scheduler.publish("owner", "req_fill", InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 1, completion_tokens: 0 }).await;
    }
}
