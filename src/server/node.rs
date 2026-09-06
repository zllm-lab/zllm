//! zLLM 工作节点：组合 iroh 调度连接与本地模型执行器。

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    error::Error,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, UNIX_EPOCH},
};

use iroh::{Endpoint, endpoint::presets};
use iroh_tickets::endpoint::EndpointTicket;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    io::BufReader,
    sync::mpsc,
    task::JoinHandle,
    time::{interval, sleep},
};

use crate::{
    kv_cache::{FjallValueStore, terminal_cache::TerminalInfo as CacheInfo},
    runtime::session::{AtomicCounterU64, NodeCapabilities, RuntimeStatus as NodeRuntime, ToolCallDelta},
};

use super::{
    iroh::IrohConfig,
    scheduler::{
        ArtifactDescriptor, ArtifactHeader, InferenceEvent, LocalNodeCommand, NodeMessage, SCHEDULER_ALPN, SCHEDULER_PROTOCOL_VERSION, Scheduler, SchedulerMessage, TaskProgress, read_json_line, safe_id, validate_artifact_descriptor,
        write_json_line,
    },
};

pub type DynError = Box<dyn Error + Send + Sync>;
pub use crate::runtime::session::{BatchTokenGuard, GenerationSummary, TextStream, TextStreamUpdate, parse_stops};

/// 节点与调度器之间的上游通道：跨进程走 iroh ticket；standalone 同进程
/// 直连 Scheduler,省掉 QUIC 环回与 JSON 序列化。
pub enum NodeUpstream {
    Ticket(String),
    Local(Scheduler),
}

pub struct NodeConfig {
    pub upstream: NodeUpstream,
    pub api_key: Option<String>,
    pub cache_dir: PathBuf,
    pub persist_kv_cache: bool,
    pub iroh: IrohConfig,
    pub max_concurrency: Option<usize>,
    pub model_alias: Option<String>,
    pub terminal_cache_global_entries: usize,
    pub terminal_cache_prefix_rounds: usize,
}

pub trait NodeEngine {
    fn model_key(&self) -> &'static str;
    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>);
    fn refresh_runtime(&self) {}
    fn terminal_cache_infos(&self) -> Vec<CacheInfo>;
    fn shutdown(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// 返回该实现能够同时持有并协作推进的独立推理 session 数，不是串行队列长度。
    fn max_concurrency(&self) -> usize;
    /// 单请求生成钩子：串行实现只需覆盖它，`generate_batch` 的默认实现会逐请求
    /// 调用并组装结果。`request_id` 供工具调用流等按请求分流的能力使用。
    fn generate_one(&mut self, request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        let _ = (request_id, request, cancellation, on_token);
        Err(format!("模型 {} 不支持单请求生成", self.model_key()))
    }
    /// 一批请求内部必须由模型实现按 prefill/decode phase 协作推进；NodeExecutor 不提供串行 fallback。
    /// `intake` 不等待新请求，只返回调用时已经收到的请求；未收到输入的工作不得进入设备调度。
    /// 默认实现为串行逐请求 map(max_concurrency=1 的模型全部适用)。
    fn generate_batch(
        &mut self,
        requests: Vec<NodeBatchRequest>,
        _intake: &mut dyn FnMut(usize) -> Vec<NodeBatchRequest>,
        on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        _on_tool_call_delta: &mut dyn FnMut(&str, ToolCallDelta) -> bool,
        _on_runtime_changed: &mut dyn FnMut(),
        _on_result: &mut dyn FnMut(NodeBatchResult),
    ) -> Vec<NodeBatchResult> {
        requests
            .into_iter()
            .map(|request| {
                let request_id = request.request_id;
                let result = self.generate_one(&request_id, &request.request, &request.cancellation, &mut |token, text| on_token(&request_id, token, text));
                NodeBatchResult { request_id, result }
            })
            .collect()
    }
    fn execute_task(&mut self, task_kind: &str, request: &Value, output_dir: &Path, cancellation: &AtomicBool, on_progress: &mut dyn FnMut(TaskProgress)) -> Result<Vec<GeneratedArtifact>, String> {
        let _ = (request, output_dir, cancellation, on_progress);
        Err(format!("模型 {} 不支持任务 {task_kind}", self.model_key()))
    }
    /// 终点 cache 的 pin 共享句柄;scheduler 在 append 排队期间 pin 换出保护。
    /// 返回 None 的实现没有 resident 换出路径,pin 是 no-op。
    fn terminal_cache_pins(&self) -> Option<Arc<Mutex<HashSet<String>>>> {
        None
    }
}

pub struct NodeBatchRequest {
    pub request_id: String,
    pub request: Value,
    pub cancellation: Arc<AtomicBool>,
}

pub struct NodeBatchResult {
    pub request_id: String,
    pub result: Result<GenerationSummary, String>,
}

pub struct GeneratedArtifact {
    pub id: String,
    pub file_name: String,
    pub content_type: String,
    pub path: PathBuf,
}

pub type NodeEngineFactory = Box<dyn FnOnce(Arc<Mutex<NodeRuntime>>, Arc<AtomicCounterU64>) -> Result<Box<dyn NodeEngine>, DynError> + Send>;

#[derive(Clone)]
struct NodeExecutor {
    commands: std::sync::mpsc::Sender<EngineCommand>,
    runtime_caches: Arc<Mutex<Vec<CacheInfo>>>,
    model_key: String,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
    accelerator_allocated: Arc<dyn Fn() -> u64 + Send + Sync>,
    max_concurrency: usize,
    /// engine 终点 cache 的 pin 共享句柄;None = 模型没有换出路径(如无 swap)。
    pins: Option<Arc<Mutex<HashSet<String>>>>,
}

struct ExecutionCommand {
    request_id: String,
    request: Value,
    cancellation: Arc<AtomicBool>,
    events: EventSink,
    active_requests: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    runtime_caches: Arc<Mutex<Vec<CacheInfo>>>,
    cache_dir: PathBuf,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
    accelerator_allocated: Arc<dyn Fn() -> u64 + Send + Sync>,
}

/// 推理事件出口。iroh 节点一切经 scheduler 路由（Wire）；standalone 本地节点
/// 数据面（Started/Token/ToolCall/Completed/Error）直达请求事件通道，控制面
/// （heartbeat、terminal 簿记）仍走 scheduler 内存通道，省掉逐 token 路由跳。
#[derive(Clone)]
enum EventSink {
    Wire(mpsc::Sender<NodeMessage>),
    Direct { events: mpsc::Sender<InferenceEvent>, control: mpsc::Sender<NodeMessage> },
}

impl EventSink {
    /// 数据面事件。try 语义：满队列失败由调用方按事件重要性兜底。
    fn try_event(&self, request_id: &str, event: InferenceEvent) -> bool {
        match self {
            Self::Wire(tx) => tx.try_send(NodeMessage::Event { request_id: request_id.to_owned(), event }).is_ok(),
            Self::Direct { events, .. } => events.try_send(event).is_ok(),
        }
    }

    /// 不能丢的数据面事件（ToolCall/Completed/Error）：瞬时满队列时阻塞兜底。
    /// Direct 模式的终态事件同时给 scheduler 发簿记副本（释放调度槽位）。
    fn send_event_reliable(&self, request_id: &str, event: InferenceEvent) -> bool {
        match self {
            Self::Wire(tx) => send_node_message_reliable(tx, NodeMessage::Event { request_id: request_id.to_owned(), event }),
            Self::Direct { events, control } => {
                let terminal = event.terminal();
                let delivered = match events.try_send(event.clone()) {
                    Ok(()) => true,
                    Err(mpsc::error::TrySendError::Full(event)) => events.blocking_send(event).is_ok(),
                    Err(mpsc::error::TrySendError::Closed(_)) => false,
                };
                if terminal {
                    let _ = control.try_send(NodeMessage::Event { request_id: request_id.to_owned(), event });
                }
                delivered
            }
        }
    }

    /// 控制面消息（heartbeat）。Wire 与事件同通道；Direct 走 scheduler 内存通道。
    fn send_control_reliable(&self, message: NodeMessage) -> bool {
        match self {
            Self::Wire(tx) => send_node_message_reliable(tx, message),
            Self::Direct { control, .. } => send_node_message_reliable(control, message),
        }
    }
}

/// engine 在独立 OS 线程运行；不能丢的控制/终态消息在瞬时满队列时阻塞兜底。
fn send_node_message_reliable(events: &mpsc::Sender<NodeMessage>, message: NodeMessage) -> bool {
    match events.try_send(message) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(message)) => events.blocking_send(message).is_ok(),
        Err(mpsc::error::TrySendError::Closed(_)) => false,
    }
}

enum EngineCommand {
    Inference(ExecutionCommand),
    Task(TaskExecutionCommand),
    Shutdown(std::sync::mpsc::SyncSender<Result<(), String>>),
}

struct TaskExecutionCommand {
    task_id: String,
    task_kind: String,
    request: Value,
    output_dir: PathBuf,
    cancellation: Arc<AtomicBool>,
    events: mpsc::Sender<NodeMessage>,
    artifacts: mpsc::Sender<ArtifactUpload>,
    active_requests: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    task_store: Arc<FjallValueStore>,
    interruption: Arc<AtomicBool>,
}

struct ArtifactUpload {
    task_id: String,
    output_count: usize,
    artifact: GeneratedArtifact,
}

struct ArtifactCommitState {
    output_count: usize,
    uploaded: HashSet<String>,
    committed: HashSet<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct PersistedArtifact {
    id: String,
    file_name: String,
    content_type: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct PersistedTask {
    task_id: String,
    task_kind: String,
    request: Value,
    state: String,
    artifacts: Vec<PersistedArtifact>,
}

impl NodeExecutor {
    fn start(factory: NodeEngineFactory, configured_max_concurrency: Option<usize>) -> Result<Self, DynError> {
        let (command_tx, command_rx) = std::sync::mpsc::channel::<EngineCommand>();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(String, NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>, usize, Option<Arc<Mutex<HashSet<String>>>>), String>>(1);
        let runtime_caches = Arc::new(Mutex::new(Vec::new()));
        let runtime = Arc::new(Mutex::new(NodeRuntime::default()));
        let compute_steps = Arc::new(AtomicCounterU64::new(0));
        let engine_runtime = runtime.clone();
        let engine_compute_steps = compute_steps.clone();
        let engine_caches = runtime_caches.clone();
        std::thread::Builder::new().name("zllm-model-engine".to_owned()).spawn(move || {
            let max_concurrency;
            let mut engine = match factory(engine_runtime, engine_compute_steps) {
                Ok(engine) => {
                    let model_key = engine.model_key().to_owned();
                    let (capabilities, accelerator_allocated) = engine.startup_info();
                    if let Ok(mut caches) = engine_caches.lock() {
                        *caches = engine.terminal_cache_infos();
                    }
                    max_concurrency = node_max_concurrency(engine.max_concurrency(), configured_max_concurrency);
                    let _ = ready_tx.send(Ok((model_key, capabilities, accelerator_allocated, max_concurrency, engine.terminal_cache_pins())));
                    engine
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error.to_string()));
                    return;
                }
            };
            let mut deferred = VecDeque::new();
            loop {
                let command = match deferred.pop_front() {
                    Some(command) => command,
                    None => match command_rx.recv() {
                        Ok(command) => command,
                        Err(_) => break,
                    },
                };
                match command {
                    EngineCommand::Inference(command) => {
                        ExecutionCommand::execute_batch(vec![command], engine.as_mut(), &command_rx, &mut deferred, max_concurrency);
                    }
                    EngineCommand::Task(command) => command.execute(engine.as_mut()),
                    EngineCommand::Shutdown(reply) => {
                        let result = engine.shutdown();
                        let _ = reply.send(result);
                        // 进程即将退出，GPU 驱动资源由操作系统回收；此处析构 ROCm 对象会触发错误的 C++ 虚函数顺序。
                        std::mem::forget(engine);
                        return;
                    }
                }
            }
        })?;
        match ready_rx.recv()? {
            Ok((model_key, capabilities, accelerator_allocated, max_concurrency, pins)) => Ok(Self { commands: command_tx, runtime_caches, model_key, capabilities, runtime, compute_steps, accelerator_allocated, max_concurrency, pins }),
            Err(error) => Err(error.into()),
        }
    }

    /// 响应 scheduler 的 PinCache/UnpinCache:直接写共享句柄,不经 engine 命令
    /// 队列(满载 batch 期间队列会推迟命令,而 pin 恰恰在这个窗口必须生效)。
    /// 上限限流防止排队风暴锁死 resident;无换出路径的模型忽略并打日志。
    fn set_pin(&self, cache_id: &str, pinned: bool) {
        let Some(pins) = &self.pins else {
            eprintln!("[zllm-node] 模型 {} 无终点 cache 换出路径,忽略 pin cache_id={cache_id}", self.model_key);
            return;
        };
        let Ok(mut guard) = pins.lock() else { return };
        if pinned {
            if guard.len() >= crate::kv_cache::terminal_cache::TERMINAL_PIN_LIMIT && !guard.contains(cache_id) {
                eprintln!("[zllm-node] pin 超过上限 {} 忽略 cache_id={cache_id}", crate::kv_cache::terminal_cache::TERMINAL_PIN_LIMIT);
                return;
            }
            guard.insert(cache_id.to_owned());
        } else {
            guard.remove(cache_id);
        }
    }

    fn submit(&self, command: ExecutionCommand) -> Result<(), String> {
        if !command.events.try_event(&command.request_id, InferenceEvent::Started) {
            return Err("scheduler event 队列无法接收推理 Started".to_owned());
        }
        self.commands.send(EngineCommand::Inference(command)).map_err(|_| "模型执行线程已经退出".to_owned())
    }

    fn submit_task(&self, command: TaskExecutionCommand) -> Result<(), String> {
        self.commands.send(EngineCommand::Task(command)).map_err(|_| "模型执行线程已经退出".to_owned())
    }

    fn shutdown(&self) -> Result<(), String> {
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        self.commands.send(EngineCommand::Shutdown(reply_tx)).map_err(|_| "模型执行线程已经退出，无法持久化 cache".to_owned())?;
        reply_rx.recv().map_err(|_| "模型执行线程未返回 cache 持久化结果".to_owned())?
    }

    fn runtime_status(&self) -> NodeRuntime {
        runtime_status(&self.runtime, &self.compute_steps, &self.accelerator_allocated)
    }
}

fn runtime_status(runtime: &Arc<Mutex<NodeRuntime>>, compute_steps: &Arc<AtomicCounterU64>, accelerator_allocated: &Arc<dyn Fn() -> u64 + Send + Sync>) -> NodeRuntime {
    let mut runtime = runtime.lock().map(|runtime| runtime.clone()).unwrap_or_default();
    runtime.accelerator_allocated_bytes = accelerator_allocated();
    runtime.compute_steps_total = compute_steps.load();
    runtime
}

impl ExecutionCommand {
    fn finish_inference(&self, result: Result<GenerationSummary, String>) {
        // engine 在独立 OS 线程运行。token 可以用 try_send 施加取消背压，但完成阶段
        // 的 cache heartbeat、tool call 和 terminal event 不能因瞬时满队列而静默丢失。
        match result {
            Ok(summary) => {
                // 先把新 checkpoint 发布进 heartbeat，再发 terminal event。Scheduler
                // 收到 terminal 才释放同 cache_id writer，immediate retry 因而必定能看见
                // 已提交 cache，不会在另一节点从零分叉。
                if let Some(cache) = summary.cache
                    && let Ok(mut caches) = self.runtime_caches.lock()
                {
                    caches.retain(|existing| existing.cache_id != cache.cache_id);
                    caches.push(cache);
                    drop(caches);
                    let runtime = runtime_status(&self.runtime, &self.compute_steps, &self.accelerator_allocated);
                    let _ = self.events.send_control_reliable(NodeMessage::Heartbeat { caches: available_caches(&self.cache_dir, &self.runtime_caches), runtime });
                }
                for (index, tool_call) in summary.tool_calls.into_iter().enumerate() {
                    if !self.events.send_event_reliable(&self.request_id, InferenceEvent::ToolCall { index, tool_call }) {
                        break;
                    }
                }
                self.events.send_event_reliable(&self.request_id, InferenceEvent::Completed { finish_reason: summary.finish_reason, prompt_tokens: summary.prompt_tokens, completion_tokens: summary.completion_tokens });
            }
            Err(message) => {
                eprintln!("[inference-error] request_id={} message={message}", self.request_id);
                self.events.send_event_reliable(&self.request_id, InferenceEvent::Error { message });
            }
        }
        if let Ok(mut active) = self.active_requests.lock() {
            active.remove(&self.request_id);
        }
    }

    fn execute_batch(commands: Vec<Self>, engine: &mut dyn NodeEngine, receiver: &std::sync::mpsc::Receiver<EngineCommand>, deferred: &mut VecDeque<EngineCommand>, max_concurrency: usize) {
        let requests = commands.iter().map(|command| NodeBatchRequest { request_id: command.request_id.clone(), request: command.request.clone(), cancellation: command.cancellation.clone() }).collect();
        let commands = RefCell::new(commands);
        let completed = RefCell::new(HashSet::new());
        let results = engine.generate_batch(
            requests,
            &mut |available| {
                let mut requests = Vec::new();
                for _ in 0..available.min(max_concurrency) {
                    match receiver.try_recv() {
                        Ok(EngineCommand::Inference(command)) => {
                            requests.push(NodeBatchRequest { request_id: command.request_id.clone(), request: command.request.clone(), cancellation: command.cancellation.clone() });
                            commands.borrow_mut().push(command);
                        }
                        Ok(command) => deferred.push_back(command),
                        Err(std::sync::mpsc::TryRecvError::Empty | std::sync::mpsc::TryRecvError::Disconnected) => break,
                    }
                }
                requests
            },
            &mut |request_id, token_id, text| commands.borrow().iter().find(|command| command.request_id == request_id).is_some_and(|command| command.events.try_event(request_id, InferenceEvent::Token { token_id, text })),
            &mut |request_id, delta| commands.borrow().iter().find(|command| command.request_id == request_id).is_some_and(|command| command.events.try_event(request_id, InferenceEvent::ToolCallDelta { delta })),
            &mut || {
                if let Some(command) = commands.borrow().first() {
                    let runtime = runtime_status(&command.runtime, &command.compute_steps, &command.accelerator_allocated);
                    let _ = command.events.send_control_reliable(NodeMessage::RuntimeChanged { runtime });
                }
            },
            &mut |result| {
                if completed.borrow().contains(&result.request_id) {
                    return;
                }
                if let Some(command) = commands.borrow().iter().find(|command| command.request_id == result.request_id) {
                    command.finish_inference(result.result);
                    completed.borrow_mut().insert(result.request_id);
                }
            },
        );
        let commands = commands.into_inner();
        let completed = completed.into_inner();
        engine.refresh_runtime();
        if let Some(command) = commands.first()
            && let Ok(mut caches) = command.runtime_caches.lock()
        {
            *caches = engine.terminal_cache_infos();
        }
        if let Some(command) = commands.first() {
            let new_runtime = runtime_status(&command.runtime, &command.compute_steps, &command.accelerator_allocated);
            let _ = command.events.send_control_reliable(NodeMessage::Heartbeat { caches: available_caches(&command.cache_dir, &command.runtime_caches), runtime: new_runtime });
        }
        let mut results = results.into_iter().map(|result| (result.request_id, result.result)).collect::<HashMap<_, _>>();
        for command in commands {
            if completed.contains(&command.request_id) {
                continue;
            }
            let result = results.remove(&command.request_id).unwrap_or_else(|| Err("模型 batch 未返回该请求结果".to_owned()));
            command.finish_inference(result);
        }
    }
}

impl TaskExecutionCommand {
    fn execute(self, engine: &mut dyn NodeEngine) {
        let Self { task_id, task_kind, request, output_dir, cancellation, events, artifacts, active_requests, task_store, interruption } = self;
        let send_status = |status: &str, error: Option<String>, outputs: Vec<ArtifactDescriptor>| events.try_send(NodeMessage::TaskStatus { task_id: task_id.clone(), status: status.to_owned(), error, outputs }).is_ok();
        let mut send_progress = |progress| {
            if events.try_send(NodeMessage::TaskProgress { task_id: task_id.clone(), progress }).is_err() {
                cancellation.store(true, Ordering::Release);
            }
        };
        let mut persisted = PersistedTask { task_id: task_id.clone(), task_kind: task_kind.clone(), request: request.clone(), state: "running".to_owned(), artifacts: Vec::new() };
        if cancellation.load(Ordering::Acquire) {
            discard_persisted_task(&task_store, &output_dir, &task_id);
            send_status("cancelled", None, Vec::new());
            if let Ok(mut active) = active_requests.lock() {
                active.remove(&task_id);
            }
            return;
        }
        if let Err(message) = persist_task(&task_store, &persisted) {
            send_status("failed", Some(message), Vec::new());
            if let Ok(mut active) = active_requests.lock() {
                active.remove(&task_id);
            }
            return;
        }
        if cancellation.load(Ordering::Acquire) {
            discard_persisted_task(&task_store, &output_dir, &task_id);
            send_status("cancelled", None, Vec::new());
            if let Ok(mut active) = active_requests.lock() {
                active.remove(&task_id);
            }
            return;
        }
        eprintln!("[zllm-node] task={task_id} kind={task_kind} 开始执行 output={}", output_dir.display());
        if !send_status("running", None, Vec::new()) {
            cancellation.store(true, Ordering::Release);
        }
        match engine.execute_task(&task_kind, &request, &output_dir, &cancellation, &mut send_progress) {
            Ok(_) if cancellation.load(Ordering::Acquire) => {
                discard_persisted_task(&task_store, &output_dir, &task_id);
                send_status("cancelled", None, Vec::new());
            }
            Ok(generated) if generated.is_empty() => {
                eprintln!("[zllm-node] task={task_id} 失败: 任务没有生成任何产物");
                discard_persisted_task(&task_store, &output_dir, &task_id);
                send_status("failed", Some("任务没有生成任何产物".to_owned()), Vec::new());
            }
            Ok(mut generated) => {
                eprintln!("[zllm-node] task={task_id} 执行完成 artifacts={}", generated.len());
                let output_count = generated.len();
                let descriptors = match describe_generated_artifacts(&output_dir, &mut generated) {
                    Ok(descriptors) => descriptors,
                    Err(message) => {
                        eprintln!("[zllm-node] task={task_id} 产物无效: {message}");
                        discard_persisted_task(&task_store, &output_dir, &task_id);
                        send_status("failed", Some(message), Vec::new());
                        if let Ok(mut active) = active_requests.lock() {
                            active.remove(&task_id);
                        }
                        return;
                    }
                };
                persisted.state = "uploading".to_owned();
                persisted.artifacts = generated.iter().map(|artifact| PersistedArtifact { id: artifact.id.clone(), file_name: artifact.file_name.clone(), content_type: artifact.content_type.clone() }).collect();
                if let Err(message) = persist_task(&task_store, &persisted) {
                    send_status("failed", Some(message), Vec::new());
                } else if cancellation.load(Ordering::Acquire) {
                    discard_persisted_task(&task_store, &output_dir, &task_id);
                    send_status("cancelled", None, Vec::new());
                } else if send_status("uploading", None, descriptors) {
                    for artifact in generated {
                        if artifacts.try_send(ArtifactUpload { task_id: task_id.clone(), output_count, artifact }).is_err() {
                            send_status("failed", Some("artifact 上传队列已经关闭".to_owned()), Vec::new());
                            break;
                        }
                    }
                }
            }
            Err(message) => {
                eprintln!("[zllm-node] task={task_id} 执行失败: {message}");
                let recoverable = interruption.load(Ordering::Acquire) || events.is_closed();
                let explicitly_deleted = cancellation.load(Ordering::Acquire) && matches!(task_store.get(&task_id), Ok(None));
                if recoverable && !explicitly_deleted {
                    persisted.state = "queued".to_owned();
                    persisted.artifacts.clear();
                    let error = persist_task(&task_store, &persisted).err().unwrap_or(message);
                    send_status("queued", Some(error), Vec::new());
                } else if cancellation.load(Ordering::Acquire) {
                    let _ = task_store.remove(&task_id);
                    send_status("cancelled", None, Vec::new());
                } else {
                    let _ = task_store.remove(&task_id);
                    send_status("failed", Some(message), Vec::new());
                }
            }
        }
        if let Ok(mut active) = active_requests.lock() {
            active.remove(&task_id);
        }
    }
}

pub async fn run_node(config: NodeConfig, factory: NodeEngineFactory) -> Result<(), DynError> {
    std::fs::create_dir_all(&config.cache_dir)?;
    let task_store = Arc::new(FjallValueStore::open(config.cache_dir.join("task-state"), "video_tasks").map_err(|error| -> DynError { error.into() })?);
    let cancellations = Arc::new(Mutex::new(HashMap::<String, Arc<AtomicBool>>::new()));
    let mut executor = NodeExecutor::start(factory, config.max_concurrency)?;
    if let Some(model_alias) = config.model_alias {
        executor.model_key = model_alias;
    }

    let endpoint = match config.upstream {
        NodeUpstream::Local(scheduler) => {
            eprintln!("[zllm-node] {} 已就绪，本地直连调度器", executor.model_key);
            let interruption = Arc::new(AtomicBool::new(false));
            let serve = serve_local(scheduler, executor.clone(), config.cache_dir.clone(), cancellations.clone(), task_store.clone(), interruption.clone());
            tokio::select! {
                result = serve => {
                    if let Err(error) = result {
                        eprintln!("[zllm-node] 本地调度会话结束: {error}");
                    }
                }
                _ = super::termination_signal() => {
                    interruption.store(true, Ordering::Release);
                    cancel_active(&cancellations);
                }
            }
            None
        }
        NodeUpstream::Ticket(ticket) => {
            let ticket = EndpointTicket::from_str(&ticket)?;
            let scheduler_addr = ticket.endpoint_addr().clone();
            let expected_scheduler = config.iroh.expected_peer.clone();
            eprintln!("[zllm-node] {} 已就绪，开始注册调度器", executor.model_key);
            let mut endpoint_builder = Endpoint::builder(presets::N0);
            if let Some(secret_key) = config.iroh.secret_key {
                endpoint_builder = endpoint_builder.secret_key(secret_key);
            }
            if let Some(bind_addr) = config.iroh.bind_addr.as_deref() {
                endpoint_builder = endpoint_builder.clear_ip_transports().bind_addr(bind_addr).map_err(|error| format!("解析 node iroh.bind_addr={bind_addr}: {error}"))?;
            }
            let endpoint = endpoint_builder.bind().await?;
            loop {
                let connection = tokio::select! {
                    connection = endpoint.connect(scheduler_addr.clone(), SCHEDULER_ALPN) => connection,
                    _ = super::termination_signal() => {
                        cancel_active(&cancellations);
                        break;
                    }
                };
                let connection = match connection {
                    Ok(connection) => {
                        super::iroh::validate_peer(&connection, expected_scheduler.as_deref(), "scheduler")?;
                        connection
                    }
                    Err(error) => {
                        eprintln!("[zllm-node] 连接 scheduler 失败: {error}，2 秒后重试");
                        tokio::select! {
                            _ = sleep(Duration::from_secs(2)) => continue,
                            _ = super::termination_signal() => {
                                cancel_active(&cancellations);
                                break;
                            },
                        }
                    }
                };
                let interruption = Arc::new(AtomicBool::new(false));
                let serve = serve_connection(connection, executor.clone(), config.cache_dir.clone(), config.api_key.clone(), cancellations.clone(), task_store.clone(), interruption.clone());
                tokio::select! {
                    result = serve => {
                        if let Err(error) = result {
                            eprintln!("[zllm-node] scheduler 连接结束: {error}");
                        }
                        sleep(Duration::from_secs(1)).await;
                    }
                    _ = super::termination_signal() => {
                        interruption.store(true, Ordering::Release);
                        cancel_active(&cancellations);
                        break;
                    },
                }
            }
            Some(endpoint)
        }
    };
    cancel_active(&cancellations);
    let _ = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            if cancellations.lock().map(|active| active.is_empty()).unwrap_or(true) {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    })
    .await;
    executor.shutdown().map_err(|error| -> DynError { error.into() })?;
    if let Some(endpoint) = endpoint {
        // 失效的直连地址可能让 iroh close 长时间等待；任务 journal 已落盘后不应阻塞进程退出。
        let _ = tokio::time::timeout(Duration::from_secs(5), endpoint.close()).await;
    }
    Ok(())
}

/// 一条节点会话（iroh 或 standalone 本地）共享的命令分发状态。
/// transport 外壳只负责消息收发；业务分发两种通道完全一致。
struct NodeSession {
    executor: NodeExecutor,
    cache_dir: PathBuf,
    cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    task_store: Arc<FjallValueStore>,
    interruption: Arc<AtomicBool>,
    outgoing: mpsc::Sender<NodeMessage>,
    artifacts: mpsc::Sender<ArtifactUpload>,
    artifact_acks: mpsc::Sender<(String, String)>,
}

impl NodeSession {
    async fn handle(&self, message: SchedulerMessage) -> Result<(), DynError> {
        let Self { executor, cache_dir, cancellations, task_store, interruption, outgoing, artifacts, artifact_acks } = self;
        match message {
            SchedulerMessage::Registered { node_id, heartbeat_seconds, task_ids } => {
                eprintln!("[zllm-node] 注册完成 id={node_id} heartbeat={heartbeat_seconds}s");
                if let Err(error) = recover_tasks(executor, cache_dir, outgoing, artifacts, cancellations, task_store, interruption, &task_ids).await {
                    eprintln!("[zllm-node] 恢复视频任务失败: {error}");
                }
            }
            SchedulerMessage::NewPrefill { request_id, model, request } => {
                self.handle_new_prefill(request_id, model, request, EventSink::Wire(outgoing.clone())).await?;
            }
            SchedulerMessage::Cancel { request_id } => {
                signal_cancellation(cancellations, &request_id)?;
                match task_store.get(&request_id) {
                    Ok(Some(_)) => discard_persisted_task(task_store, &cache_dir.join("tasks").join(&request_id), &request_id),
                    Ok(None) => {}
                    Err(error) => eprintln!("[zllm-node] task={request_id} 查询取消任务状态失败: {error}"),
                }
            }
            SchedulerMessage::NewTask { task_id, model, task_kind, request } => {
                if model != executor.model_key || !executor.capabilities.task_kinds.iter().any(|kind| kind == &task_kind) {
                    outgoing
                        .send(NodeMessage::TaskStatus { task_id, status: "failed".to_owned(), error: Some(format!("节点模型 {} 不支持 {model}/{task_kind}", executor.model_key)), outputs: Vec::new() })
                        .await
                        .map_err(|_| "scheduler writer 已关闭")?;
                    return Ok(());
                }
                let cancellation = Arc::new(AtomicBool::new(false));
                cancellations.lock().map_err(|_| "cancel map 锁中毒")?.insert(task_id.clone(), cancellation.clone());
                let output_dir = cache_dir.join("tasks").join(&task_id);
                let persisted = PersistedTask { task_id: task_id.clone(), task_kind: task_kind.clone(), request: request.clone(), state: "queued".to_owned(), artifacts: Vec::new() };
                if let Err(message) = persist_task(task_store, &persisted).and_then(|_| prepare_output_dir(&output_dir)).and_then(|_| {
                    executor.submit_task(TaskExecutionCommand {
                        task_id: task_id.clone(),
                        task_kind,
                        request,
                        output_dir,
                        cancellation,
                        events: outgoing.clone(),
                        artifacts: artifacts.clone(),
                        active_requests: cancellations.clone(),
                        task_store: task_store.clone(),
                        interruption: interruption.clone(),
                    })
                }) {
                    cancellations.lock().map_err(|_| "cancel map 锁中毒")?.remove(&task_id);
                    outgoing.send(NodeMessage::TaskStatus { task_id, status: "queued".to_owned(), error: Some(message), outputs: Vec::new() }).await.map_err(|_| "scheduler writer 已关闭")?;
                }
            }
            SchedulerMessage::PinCache { cache_id } => executor.set_pin(&cache_id, true),
            SchedulerMessage::UnpinCache { cache_id } => executor.set_pin(&cache_id, false),
            SchedulerMessage::ArtifactCommitted { task_id, artifact_id } => {
                artifact_acks.send((task_id, artifact_id)).await.map_err(|_| "artifact ACK 队列已经关闭")?;
            }
        }
        Ok(())
    }

    /// standalone 本地命令:NewPrefill 携带请求事件通道,数据面直达 HTTP 处理侧。
    async fn handle_local(&self, message: LocalNodeCommand) -> Result<(), DynError> {
        match message {
            LocalNodeCommand::Wire(message) => self.handle(message).await,
            LocalNodeCommand::NewPrefill { request_id, model, request, events } => self.handle_new_prefill(request_id, model, request, EventSink::Direct { events, control: self.outgoing.clone() }).await,
        }
    }

    async fn handle_new_prefill(&self, request_id: String, model: String, request: Value, events: EventSink) -> Result<(), DynError> {
        let executor = &self.executor;
        if model != executor.model_key {
            events.send_event_reliable(&request_id, InferenceEvent::Error { message: format!("节点模型 {} 不能执行 {model}", executor.model_key) });
            return Ok(());
        }
        let cancellation = Arc::new(AtomicBool::new(false));
        self.cancellations.lock().map_err(|_| "cancel map 锁中毒")?.insert(request_id.clone(), cancellation.clone());
        if let Err(message) = executor.submit(ExecutionCommand {
            request_id: request_id.clone(),
            request,
            cancellation,
            events,
            active_requests: self.cancellations.clone(),
            runtime_caches: executor.runtime_caches.clone(),
            cache_dir: self.cache_dir.clone(),
            runtime: executor.runtime.clone(),
            compute_steps: executor.compute_steps.clone(),
            accelerator_allocated: executor.accelerator_allocated.clone(),
        }) {
            self.cancellations.lock().map_err(|_| "cancel map 锁中毒")?.remove(&request_id);
            self.outgoing.send(NodeMessage::Event { request_id, event: InferenceEvent::Error { message } }).await.map_err(|_| "scheduler writer 已关闭")?;
        }
        Ok(())
    }
}

fn register_message(executor: &NodeExecutor, cache_dir: &Path, api_key: Option<String>) -> NodeMessage {
    NodeMessage::Register {
        protocol_version: SCHEDULER_PROTOCOL_VERSION,
        api_key,
        model: executor.model_key.clone(),
        max_concurrency: executor.max_concurrency,
        caches: available_caches(cache_dir, &executor.runtime_caches),
        capabilities: executor.capabilities.clone(),
        runtime: executor.runtime_status(),
    }
}

fn spawn_heartbeat(outgoing: mpsc::Sender<NodeMessage>, cache_dir: PathBuf, executor: NodeExecutor) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(5));
        ticker.tick().await;
        loop {
            ticker.tick().await;
            if outgoing.send(NodeMessage::Heartbeat { caches: available_caches(&cache_dir, &executor.runtime_caches), runtime: executor.runtime_status() }).await.is_err() {
                break;
            }
        }
    })
}

/// standalone 同进程会话：与 scheduler 之间走内存通道，无 iroh/JSON。
/// 本地节点只声明 text_generation，dispatch_task 不会向它派发带产物的任务；
/// artifact 通道保留只为让共享的 task 路径可编译，收到上传即报错（不可达）。
async fn serve_local(scheduler: Scheduler, executor: NodeExecutor, cache_dir: PathBuf, cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>, task_store: Arc<FjallValueStore>, interruption: Arc<AtomicBool>) -> Result<(), DynError> {
    let channels = scheduler.attach_local_node("local".to_owned());
    let outgoing_tx = channels.messages;
    let mut incoming = channels.commands;
    outgoing_tx.send(register_message(&executor, &cache_dir, None)).await.map_err(|_| "scheduler 本地通道已关闭")?;

    let (artifact_tx, mut artifact_rx) = mpsc::channel::<ArtifactUpload>(8);
    let (artifact_ack_tx, mut artifact_ack_rx) = mpsc::channel::<(String, String)>(32);
    let artifact_writer = tokio::spawn(async move {
        while let Some(upload) = artifact_rx.recv().await {
            eprintln!("[zllm-node] task={} standalone 本地节点不支持 artifact 上传", upload.task_id);
        }
        // 保持 ACK 接收端存活，避免共享路径的 send 报错。
        let _ = artifact_ack_rx.recv().await;
    });
    let heartbeat = spawn_heartbeat(outgoing_tx.clone(), cache_dir.clone(), executor.clone());

    let session = NodeSession { executor, cache_dir, cancellations: cancellations.clone(), task_store, interruption: interruption.clone(), outgoing: outgoing_tx, artifacts: artifact_tx, artifact_acks: artifact_ack_tx };
    let result: Result<(), DynError> = async {
        while let Some(message) = incoming.recv().await {
            session.handle_local(message).await?;
        }
        Ok(())
    }
    .await;
    interruption.store(true, Ordering::Release);
    if let Ok(active) = cancellations.lock() {
        for cancellation in active.values() {
            cancellation.store(true, Ordering::Release);
        }
    }
    heartbeat.abort();
    artifact_writer.abort();
    result
}

async fn serve_connection(
    connection: iroh::endpoint::Connection,
    executor: NodeExecutor,
    cache_dir: PathBuf,
    api_key: Option<String>,
    cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    task_store: Arc<FjallValueStore>,
    interruption: Arc<AtomicBool>,
) -> Result<(), DynError> {
    let (send, recv) = connection.open_bi().await?;
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<NodeMessage>(128);
    let mut writer = tokio::spawn(async move {
        let mut send = send;
        while let Some(message) = outgoing_rx.recv().await {
            write_json_line(&mut send, &message).await?;
        }
        send.finish()?;
        Ok::<(), DynError>(())
    });
    outgoing_tx.send(register_message(&executor, &cache_dir, api_key)).await.map_err(|_| "scheduler writer 已关闭")?;

    let (artifact_tx, mut artifact_rx) = mpsc::channel::<ArtifactUpload>(8);
    let (artifact_ack_tx, mut artifact_ack_rx) = mpsc::channel::<(String, String)>(32);
    let artifact_connection = connection.clone();
    let artifact_events = outgoing_tx.clone();
    let artifact_store = task_store.clone();
    let artifact_writer = tokio::spawn(async move {
        let mut tasks = HashMap::<String, ArtifactCommitState>::new();
        loop {
            tokio::select! {
                upload = artifact_rx.recv() => {
                    let Some(upload) = upload else { break };
                    let task_id = upload.task_id.clone();
                    let artifact_id = upload.artifact.id.clone();
                    let output_count = upload.output_count;
                    let task = tasks.entry(task_id.clone()).or_insert_with(|| ArtifactCommitState { output_count, uploaded: HashSet::new(), committed: HashSet::new() });
                    if task.output_count != output_count {
                        let _ = artifact_events.send(NodeMessage::TaskStatus { task_id, status: "queued".to_owned(), error: Some(format!("artifact output_count 从 {} 变为 {output_count}，等待恢复", task.output_count)), outputs: Vec::new() }).await;
                        continue;
                    }
                    match upload_artifact(&artifact_connection, upload).await {
                        Ok(()) => {
                            task.uploaded.insert(artifact_id);
                        }
                        Err(error) => {
                            let _ = artifact_events.send(NodeMessage::TaskStatus { task_id, status: "queued".to_owned(), error: Some(format!("artifact 上传中断，等待恢复: {error}")), outputs: Vec::new() }).await;
                            // 强制重连会触发持久化任务恢复；保持当前连接只会让 queued 任务永久等待。
                            artifact_connection.close(1u32.into(), b"artifact upload failed");
                        }
                    }
                }
                committed = artifact_ack_rx.recv() => {
                    let Some((task_id, artifact_id)) = committed else { break };
                    let Some(task) = tasks.get_mut(&task_id) else {
                        eprintln!("[zllm-node] task={task_id} 收到未知 artifact={artifact_id} committed ACK");
                        continue;
                    };
                    if !task.uploaded.contains(&artifact_id) {
                        eprintln!("[zllm-node] task={task_id} artifact={artifact_id} 在本地上传完成前收到 committed ACK");
                        continue;
                    }
                    task.committed.insert(artifact_id);
                    if task.committed.len() == task.output_count {
                        if let Err(error) = artifact_store.remove(&task_id) {
                            eprintln!("[zllm-node] task={task_id} 清理持久化状态失败: {error}");
                        }
                        tasks.remove(&task_id);
                    }
                }
            }
        }
    });

    let heartbeat = spawn_heartbeat(outgoing_tx.clone(), cache_dir.clone(), executor.clone());

    let session = NodeSession { executor, cache_dir, cancellations: cancellations.clone(), task_store, interruption: interruption.clone(), outgoing: outgoing_tx.clone(), artifacts: artifact_tx, artifact_acks: artifact_ack_tx };
    let mut reader = BufReader::new(recv);
    let result: Result<(), DynError> = {
        let reader = async {
            loop {
                let message = read_json_line::<_, SchedulerMessage>(&mut reader).await?;
                session.handle(message).await?;
            }
        };
        tokio::pin!(reader);
        // 写半边失败时调度器会立即摘除节点；必须结束读循环，才能回到外层重连。
        // 只等待 scheduler 输入会把进程永久留在失联连接上，形成仍占 GPU 的僵尸节点。
        tokio::select! {
            result = &mut reader => result,
            result = &mut writer => match result {
                Ok(Ok(())) => Err("scheduler writer 意外结束".into()),
                Ok(Err(error)) => Err(format!("scheduler writer 失败: {error}").into()),
                Err(error) => Err(format!("scheduler writer task 失败: {error}").into()),
            },
        }
    };
    interruption.store(true, Ordering::Release);
    if let Ok(active) = cancellations.lock() {
        for cancellation in active.values() {
            cancellation.store(true, Ordering::Release);
        }
    }
    heartbeat.abort();
    artifact_writer.abort();
    drop(outgoing_tx);
    writer.abort();
    result
}

fn node_max_concurrency(supported: usize, configured: Option<usize>) -> usize {
    configured.filter(|value| *value > 0).map_or(supported.max(1), |configured| configured.min(supported.max(1)))
}

fn persist_task(store: &FjallValueStore, task: &PersistedTask) -> Result<(), String> {
    let bytes = serde_json::to_vec(task).map_err(|error| format!("序列化视频任务 {}: {error}", task.task_id))?;
    store.put(&task.task_id, &bytes)
}

fn discard_persisted_task(task_store: &FjallValueStore, output_dir: &Path, task_id: &str) {
    if let Err(error) = task_store.remove(task_id) {
        eprintln!("[zllm-node] task={task_id} 删除持久化状态失败: {error}");
    }
    if output_dir.exists()
        && let Err(error) = std::fs::remove_dir_all(output_dir)
    {
        eprintln!("[zllm-node] task={task_id} 删除任务目录 {} 失败: {error}", output_dir.display());
    }
}

fn prepare_output_dir(output_dir: &Path) -> Result<(), String> {
    match std::fs::remove_dir_all(output_dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("清理旧任务目录 {}: {error}", output_dir.display())),
    }
    std::fs::create_dir_all(output_dir).map_err(|error| format!("创建任务目录 {}: {error}", output_dir.display()))
}

fn describe_generated_artifacts(output_dir: &Path, artifacts: &mut [GeneratedArtifact]) -> Result<Vec<ArtifactDescriptor>, String> {
    let output_root = output_dir.canonicalize().map_err(|error| format!("解析任务目录 {}: {error}", output_dir.display()))?;
    let mut ids = HashSet::with_capacity(artifacts.len());
    artifacts
        .iter_mut()
        .map(|artifact| {
            if !safe_id(&artifact.id) || !ids.insert(artifact.id.clone()) {
                return Err(format!("artifact id 非法或重复: {:?}", artifact.id));
            }
            let expected = output_dir.join(&artifact.file_name).canonicalize().map_err(|error| format!("解析 artifact {}: {error}", artifact.path.display()))?;
            let actual = artifact.path.canonicalize().map_err(|error| format!("解析 artifact {}: {error}", artifact.path.display()))?;
            if actual != expected || !actual.starts_with(&output_root) {
                return Err(format!("artifact {} 不在任务目录内: {}", artifact.id, artifact.path.display()));
            }
            let metadata = actual.metadata().map_err(|error| format!("读取 artifact {} metadata: {error}", artifact.id))?;
            if !metadata.is_file() {
                return Err(format!("artifact {} bytes={} 非法", artifact.id, metadata.len()));
            }
            let descriptor = ArtifactDescriptor { id: artifact.id.clone(), file_name: artifact.file_name.clone(), content_type: artifact.content_type.clone(), bytes: metadata.len() };
            validate_artifact_descriptor(&descriptor)?;
            // 后续上传直接打开已验证的真实路径，避免再次跟随原始符号链接。
            artifact.path = actual;
            Ok(descriptor)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
async fn recover_tasks(
    executor: &NodeExecutor,
    cache_dir: &Path,
    events: &mpsc::Sender<NodeMessage>,
    artifacts: &mpsc::Sender<ArtifactUpload>,
    active_requests: &Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    task_store: &Arc<FjallValueStore>,
    interruption: &Arc<AtomicBool>,
    task_ids: &[String],
) -> Result<(), String> {
    for bytes in task_store.values()? {
        let mut task = match serde_json::from_slice::<PersistedTask>(&bytes) {
            Ok(task) => task,
            Err(error) => {
                eprintln!("[zllm-node] 跳过损坏的视频任务记录: {error}");
                continue;
            }
        };
        let output_dir = cache_dir.join("tasks").join(&task.task_id);
        if !task_ids.iter().any(|task_id| task_id == &task.task_id) {
            eprintln!("[zllm-node] task={} 不在调度器恢复列表，清理本地持久化状态", task.task_id);
            discard_persisted_task(task_store, &output_dir, &task.task_id);
            continue;
        }
        if task.task_kind != "video_generation" || !executor.capabilities.task_kinds.iter().any(|kind| kind == &task.task_kind) {
            continue;
        }
        loop {
            let active = active_requests.lock().map_err(|_| "cancel map 锁中毒")?.contains_key(&task.task_id);
            if !active {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
        if task.state == "uploading" && !task.artifacts.is_empty() && task.artifacts.iter().all(|artifact| output_dir.join(&artifact.file_name).is_file()) {
            let mut generated = task
                .artifacts
                .iter()
                .map(|artifact| GeneratedArtifact { id: artifact.id.clone(), file_name: artifact.file_name.clone(), content_type: artifact.content_type.clone(), path: output_dir.join(&artifact.file_name) })
                .collect::<Vec<_>>();
            if let Ok(descriptors) = describe_generated_artifacts(&output_dir, &mut generated) {
                events.send(NodeMessage::TaskStatus { task_id: task.task_id.clone(), status: "uploading".to_owned(), error: None, outputs: descriptors }).await.map_err(|_| "scheduler writer 已关闭")?;
                let output_count = generated.len();
                for artifact in generated {
                    artifacts.send(ArtifactUpload { task_id: task.task_id.clone(), output_count, artifact }).await.map_err(|_| "artifact 上传队列已经关闭")?;
                }
                eprintln!("[zllm-node] task={} 恢复 artifact 上传", task.task_id);
                continue;
            }
            eprintln!("[zllm-node] task={} 持久化产物无效，重新执行", task.task_id);
        }
        prepare_output_dir(&output_dir)?;
        task.state = "queued".to_owned();
        task.artifacts.clear();
        persist_task(task_store, &task)?;
        let cancellation = Arc::new(AtomicBool::new(false));
        active_requests.lock().map_err(|_| "cancel map 锁中毒")?.insert(task.task_id.clone(), cancellation.clone());
        if let Err(message) = executor.submit_task(TaskExecutionCommand {
            task_id: task.task_id.clone(),
            task_kind: task.task_kind,
            request: task.request,
            output_dir,
            cancellation,
            events: events.clone(),
            artifacts: artifacts.clone(),
            active_requests: active_requests.clone(),
            task_store: task_store.clone(),
            interruption: interruption.clone(),
        }) {
            active_requests.lock().map_err(|_| "cancel map 锁中毒")?.remove(&task.task_id);
            return Err(message);
        }
        eprintln!("[zllm-node] task={} 从 fjall 恢复执行", task.task_id);
    }
    Ok(())
}

fn cancel_active(active_requests: &Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>) {
    if let Ok(active) = active_requests.lock() {
        for cancellation in active.values() {
            cancellation.store(true, Ordering::Release);
        }
    }
}

fn signal_cancellation(active_requests: &Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>, request_id: &str) -> Result<(), &'static str> {
    let cancellation = active_requests.lock().map_err(|_| "cancel map 锁中毒")?.get(request_id).cloned();
    if let Some(cancellation) = cancellation {
        cancellation.store(true, Ordering::Release);
    }
    Ok(())
}

async fn upload_artifact(connection: &iroh::endpoint::Connection, upload: ArtifactUpload) -> Result<(), DynError> {
    let mut file = tokio::fs::File::open(&upload.artifact.path).await?;
    let metadata = file.metadata().await?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!("产物 {} 不是非空文件", upload.artifact.path.display()).into());
    }
    let descriptor = ArtifactDescriptor { id: upload.artifact.id, file_name: upload.artifact.file_name, content_type: upload.artifact.content_type, bytes: metadata.len() };
    validate_artifact_descriptor(&descriptor)?;
    let header = serde_json::to_vec(&ArtifactHeader { task_id: upload.task_id, artifact: descriptor, output_count: upload.output_count })?;
    let header_len = u32::try_from(header.len()).map_err(|_| "artifact header 超过 u32")?;
    let mut send = connection.open_uni().await?;
    tokio::io::AsyncWriteExt::write_u32(&mut send, header_len).await?;
    tokio::io::AsyncWriteExt::write_all(&mut send, &header).await?;
    let copied = tokio::io::copy(&mut file, &mut send).await?;
    if copied != metadata.len() {
        return Err(format!("产物 {} 上传期间长度从 {} 变为 {copied}", upload.artifact.path.display(), metadata.len()).into());
    }
    send.finish()?;
    Ok(())
}

#[derive(Deserialize)]
struct CacheManifest {
    model_key: String,
    cache_format: String,
    last_layer: usize,
    prompt_tokens: usize,
}

fn scan_caches(root: &Path) -> Vec<CacheInfo> {
    let mut directories = vec![(".".to_owned(), root.to_owned())];
    if let Ok(entries) = std::fs::read_dir(root) {
        directories.extend(entries.filter_map(Result::ok).filter_map(|entry| entry.file_type().ok().filter(|kind| kind.is_dir()).map(|_| (entry.file_name().to_string_lossy().into_owned(), entry.path()))));
    }
    let mut caches = directories
        .into_iter()
        .filter_map(|(cache_id, directory)| {
            let manifest_path = directory.join("manifest.json");
            let manifest: CacheManifest = serde_json::from_slice(&std::fs::read(manifest_path).ok()?).ok()?;
            let (bytes, modified_unix) = directory_stats(&directory);
            Some(CacheInfo { cache_id, model_key: manifest.model_key, cache_format: manifest.cache_format, last_layer: manifest.last_layer, prompt_tokens: manifest.prompt_tokens, bytes, modified_unix })
        })
        .collect::<Vec<_>>();
    caches.sort_by(|left, right| left.cache_id.cmp(&right.cache_id));
    caches
}

fn available_caches(root: &Path, runtime_caches: &Arc<Mutex<Vec<CacheInfo>>>) -> Vec<CacheInfo> {
    // 用 root 一级子项的 max mtime 做缓存键：一级文件/目录变更会推 mtime；
    // scan_caches 的目录递归 stats 跳过开销大，缓存命中时只 walk root 一层。
    let cached = scan_cache().lock().ok().and_then(|cache| {
        let mtime = root_max_mtime(root);
        cache.as_ref().filter(|(stored_root, stored_mtime, _)| stored_root == root && *stored_mtime == mtime).map(|(_, _, infos)| (mtime, infos.clone()))
    });
    let mut caches = match cached {
        Some((_mtime, infos)) => infos,
        None => {
            let infos = scan_caches(root);
            let mtime = root_max_mtime(root);
            if let Ok(mut cache) = scan_cache().lock() {
                *cache = Some((root.to_path_buf(), mtime, infos.clone()));
            }
            infos
        }
    };
    if let Ok(runtime) = runtime_caches.lock() {
        caches.extend(runtime.iter().cloned());
    }
    caches.sort_by(|left, right| left.cache_id.cmp(&right.cache_id));
    caches.dedup_by(|left, right| left.cache_id == right.cache_id);
    truncate_reported_caches(caches)
}

/// 节点向 scheduler 报告的 cache 列表硬上限,远高于单节点常用量(默认
/// `terminal_cache_global_entries=64`)。截断是负载均衡视图的边界保护,不是
/// 节点内存 LRU 上限;后者由 `terminal_cache_global_entries` 决定。
pub const MAX_REPORTED_CACHES: usize = 10240;

/// 按 cache_id 字典序保留前 `MAX_REPORTED_CACHES` 条;超限部分不参与跨节点命中分配,
/// 让 `cache_owner` / dispatch 命中查找的视图稳定。`available_caches` 的最后一步。
pub(super) fn truncate_reported_caches(mut caches: Vec<CacheInfo>) -> Vec<CacheInfo> {
    if caches.len() > MAX_REPORTED_CACHES {
        caches.truncate(MAX_REPORTED_CACHES);
    }
    caches
}

fn root_max_mtime(root: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(root) else { return 0 };
    entries.filter_map(Result::ok).filter_map(|entry| entry.metadata().ok().and_then(|m| m.modified().ok()).map(|t| t.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()))).max().unwrap_or(0)
}

#[allow(clippy::type_complexity)]
static SCAN_CACHE: std::sync::OnceLock<std::sync::Mutex<Option<(PathBuf, u64, Vec<CacheInfo>)>>> = std::sync::OnceLock::new();

// 第一次 available_caches 调用时初始化全局缓存；之后所有节点复用同一把锁。
fn scan_cache() -> &'static std::sync::Mutex<Option<(PathBuf, u64, Vec<CacheInfo>)>> {
    SCAN_CACHE.get_or_init(|| std::sync::Mutex::new(None))
}

fn directory_stats(root: &Path) -> (u64, u64) {
    let mut bytes = 0u64;
    let mut modified = 0u64;
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                bytes = bytes.saturating_add(metadata.len());
                modified = modified.max(metadata.modified().ok().and_then(|time| time.duration_since(UNIX_EPOCH).ok()).map_or(0, |duration| duration.as_secs()));
            }
        }
    }
    (bytes, modified)
}

pub use crate::runtime::session::{ContentPiece, content_pieces, local_image_paths, split_local_images, text_content, with_content_parts};
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_stream_holds_cross_token_stop() {
        let mut stream = TextStream::new(&["</stop>".to_owned()]);
        let first = stream.push(b"answer</st");
        assert!(!first.stopped);
        let second = stream.push(b"op>hidden");
        assert!(second.stopped);
        assert_eq!(format!("{}{}", first.chunk.unwrap_or_default(), second.chunk.unwrap_or_default()), "answer");
        assert_eq!(stream.text(), "answer</stop>hidden");
    }

    #[test]
    fn text_stream_keeps_utf8_character_whole() {
        let mut stream = TextStream::new(&[]);
        let bytes = "你".as_bytes();
        assert!(stream.push(&bytes[..2]).chunk.is_none());
        assert_eq!(stream.push(&bytes[2..]).chunk.as_deref(), Some("你"));
        assert!(stream.finish().is_none());
    }

    #[test]
    fn checkpoint_heartbeat_precedes_terminal_event() {
        let (events, mut receiver) = mpsc::channel(4);
        let request_id = "req_checkpoint".to_owned();
        let active_requests = Arc::new(Mutex::new(HashMap::from([(request_id.clone(), Arc::new(AtomicBool::new(false)))])));
        let command = ExecutionCommand {
            request_id: request_id.clone(),
            request: Value::Null,
            cancellation: Arc::new(AtomicBool::new(false)),
            events: EventSink::Wire(events),
            active_requests: active_requests.clone(),
            runtime_caches: Arc::new(Mutex::new(Vec::new())),
            cache_dir: PathBuf::from("target/nonexistent-node-cache-test"),
            runtime: Arc::new(Mutex::new(NodeRuntime::default())),
            compute_steps: Arc::new(AtomicCounterU64::new(0)),
            accelerator_allocated: Arc::new(|| 0),
        };
        let cache = CacheInfo { cache_id: "session-checkpoint".to_owned(), model_key: "glm-5.2".to_owned(), cache_format: "test".to_owned(), last_layer: 77, prompt_tokens: 128, bytes: 4096, modified_unix: 1 };
        command.finish_inference(Ok(GenerationSummary { finish_reason: "cancelled".to_owned(), prompt_tokens: 256, completion_tokens: 0, cache: Some(cache), tool_calls: Vec::new() }));

        match receiver.try_recv().unwrap() {
            NodeMessage::Heartbeat { caches, .. } => assert!(caches.iter().any(|cache| cache.cache_id == "session-checkpoint" && cache.prompt_tokens == 128)),
            message => panic!("checkpoint 后第一条消息应是 heartbeat，实际 {message:?}"),
        }
        assert!(matches!(receiver.try_recv().unwrap(), NodeMessage::Event { event: InferenceEvent::Completed { ref finish_reason, .. }, .. } if finish_reason == "cancelled"));
        assert!(!active_requests.lock().unwrap().contains_key(&request_id));
    }

    #[test]
    fn terminal_event_waits_for_a_full_node_queue() {
        let (events, mut receiver) = mpsc::channel(1);
        events.try_send(NodeMessage::Event { request_id: "queued".to_owned(), event: InferenceEvent::Started }).unwrap();
        let terminal = std::thread::spawn(move || {
            let message = NodeMessage::Event { request_id: "terminal".to_owned(), event: InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 1 } };
            match events.try_send(message) {
                Ok(()) => true,
                Err(mpsc::error::TrySendError::Full(message)) => events.blocking_send(message).is_ok(),
                Err(mpsc::error::TrySendError::Closed(_)) => false,
            }
        });

        assert!(matches!(receiver.blocking_recv(), Some(NodeMessage::Event { request_id, event: InferenceEvent::Started }) if request_id == "queued"));
        assert!(matches!(receiver.blocking_recv(), Some(NodeMessage::Event { request_id, event: InferenceEvent::Completed { .. } }) if request_id == "terminal"));
        assert!(terminal.join().unwrap());
    }

    #[test]
    fn scheduler_cancel_sets_node_atomic_flag() {
        let cancellation = Arc::new(AtomicBool::new(false));
        let active = Arc::new(Mutex::new(HashMap::from([("req_cancel".to_owned(), cancellation.clone())])));
        signal_cancellation(&active, "req_cancel").unwrap();
        assert!(cancellation.load(Ordering::Acquire));
    }

    #[test]
    fn content_pieces解析图文混排并保留顺序() {
        let content = serde_json::json!([
            {"type": "text", "text": "看这两张图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "http://example.com/2.png"}},
            {"type": "text", "text": "说说区别"}
        ]);
        let pieces = content_pieces(Some(&content)).unwrap();
        assert!(matches!(&pieces[0], ContentPiece::Text(text) if text == "看这两张图"));
        assert!(matches!(&pieces[1], ContentPiece::Image { url } if url == "data:image/png;base64,AAA"));
        assert!(matches!(&pieces[2], ContentPiece::Image { url } if url == "http://example.com/2.png"));
        assert!(matches!(&pieces[3], ContentPiece::Text(text) if text == "说说区别"));
        // 纯字符串与空 content 与 text_content 行为一致。
        assert!(matches!(content_pieces(Some(&serde_json::json!("hi"))).unwrap()[..1], [ContentPiece::Text(ref text)] if text == "hi"));
        assert!(content_pieces(None).unwrap().is_empty());
        // 缺字段与非图文 type 拒绝。
        assert!(content_pieces(Some(&serde_json::json!([{"type": "image_url"}]))).is_err());
        assert!(content_pieces(Some(&serde_json::json!([{"type": "audio"}]))).is_err());
        assert!(content_pieces(Some(&serde_json::json!([{"text": "x"}]))).is_err());
    }

    #[test]
    fn 本地图片路径不会隐式改变结构化请求语义() {
        let directory = std::env::temp_dir().join(format!("zllm-content-path-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let first = directory.join("first.png");
        let second = directory.join("second image.webp");
        std::fs::write(&first, b"image").unwrap();
        std::fs::write(&second, b"image").unwrap();
        let content = format!("先看{}，再看 `{}`，不要重复 {}", first.display(), second.display(), first.display());
        let paths = local_image_paths(&content);
        assert_eq!(paths, [first.to_string_lossy().as_ref(), second.to_string_lossy().as_ref()]);
        let pieces = content_pieces(Some(&serde_json::json!(content))).unwrap();
        assert!(matches!(&pieces[..], [ContentPiece::Text(text)] if text.contains("first.png") && text.contains("second image.webp")));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn with_content_parts物化图像并保序透传文本() {
        // 1×1 PNG(魔数可识别)的 base64;解码后交由回调按序消费。
        let png = {
            let mut bytes = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(2, 2, image::Rgb([1, 2, 3]))).write_to(&mut bytes, image::ImageFormat::Png).unwrap();
            bytes.into_inner()
        };
        let encoded = {
            const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut output = String::new();
            for chunk in png.chunks(3) {
                let group = (chunk[0] as u32) << 16 | (chunk.get(1).copied().unwrap_or(0) as u32) << 8 | (chunk.get(2).copied().unwrap_or(0) as u32);
                output.push(ALPHABET[(group >> 18) as usize & 63] as char);
                output.push(ALPHABET[(group >> 12) as usize & 63] as char);
                output.push(if chunk.len() > 1 { ALPHABET[(group >> 6) as usize & 63] as char } else { '=' });
                output.push(if chunk.len() > 2 { ALPHABET[group as usize & 63] as char } else { '=' });
            }
            output
        };
        let pieces = vec![ContentPiece::Text("前".to_owned()), ContentPiece::Image { url: format!("data:image/png;base64,{encoded}") }, ContentPiece::Text("后".to_owned())];
        let summary = with_content_parts(&pieces, |parts| {
            let mut texts = Vec::new();
            let mut images = Vec::new();
            for part in parts {
                match part {
                    crate::vision::ContentPart::Text(text) => texts.push(text.to_owned()),
                    crate::vision::ContentPart::Image(image) => images.push((image.width, image.height)),
                }
            }
            Ok(format!("texts={texts:?} images={images:?}"))
        })
        .unwrap();
        assert_eq!(summary, "texts=[\"前\", \"后\"] images=[(2, 2)]");
        // 图像 URL 无法物化时报错并带序号。
        let bad = vec![ContentPiece::Text("x".to_owned()), ContentPiece::Image { url: "not a path or base64 !!!".to_owned() }];
        assert!(with_content_parts(&bad, |_| Ok(())).is_err());
    }

    #[test]
    fn available_caches_truncate_to_10240() {
        // 节点心跳里报告的 cache 列表硬上限 10240,超过按 cache_id 字典序保留前 10240 条。
        // `truncate_reported_caches` 只截断不重排;`available_caches` 在调用前已 sort_by cache_id,
        // 所以截断后第一条必须是字典序最小。测试以这个事实构造输入。
        let mut caches: Vec<CacheInfo> =
            (0..10_300).map(|index| CacheInfo { cache_id: format!("c{index:06}"), model_key: "model".to_owned(), cache_format: "mla".to_owned(), last_layer: 1, prompt_tokens: 1, bytes: 1, modified_unix: 0 }).collect();
        // 模拟 `available_caches` 的 sort + dedup + truncate 顺序:先 sort_by cache_id,再 truncate。
        caches.sort_by(|left, right| left.cache_id.cmp(&right.cache_id));
        let truncated = truncate_reported_caches(caches);
        assert_eq!(truncated.len(), 10240);
        assert_eq!(truncated[0].cache_id, "c000000");
        assert_eq!(truncated[10_239].cache_id, "c010239");
    }

    #[test]
    fn available_caches_below_cap_passes_through() {
        // 不超过上限时原样返回(顺序与输入无关,truncate_reported_caches 不重排)。
        let caches: Vec<CacheInfo> = (0..10).map(|index| CacheInfo { cache_id: format!("c{index:04}"), model_key: "model".to_owned(), cache_format: "mla".to_owned(), last_layer: 1, prompt_tokens: 1, bytes: 1, modified_unix: 0 }).collect();
        let truncated = truncate_reported_caches(caches.clone());
        assert_eq!(truncated.len(), caches.len());
        assert_eq!(truncated[0].cache_id, caches[0].cache_id);
    }
}
