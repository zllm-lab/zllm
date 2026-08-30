//! HTTP 服务基础与路由：请求/响应公共类型、鉴权、校验、取消与生命周期；
//! Chat Completions 与 Responses 协议分别位于 `chat.rs` 与 `responses.rs`。

use std::{
    convert::Infallible,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};

use scheduler::{DispatchError, InferenceEvent, SCHEDULER_ALPN, Scheduler, SchedulerService};

mod anthropic;
mod chat;
use chat::chat_completions;
mod h3;
pub mod iroh;
pub mod node;
mod responses;
pub mod scheduler;
pub mod stage_transport;
use h3::{create_video_generation, delete_video_generation, download_artifact, list_video_generation, query_video_generation};

const MAX_REQUEST_BYTES: usize = 80 * 1024 * 1024;

#[derive(Clone, Debug, Default)]
pub struct ServerConfig {
    pub api_keys: Vec<String>,
    /// 不校验 API key 时仍使用稳定 namespace，保证本地服务重启后 cache id 不漂移。
    pub anonymous_cache_namespace: Option<String>,
    pub node_api_key: Option<String>,
    pub models: Vec<String>,
    /// 对外别名 -> 后台真实模型；请求模型命中别名时重写后再调度（见 `resolve_model_alias`）。
    pub model_aliases: std::collections::HashMap<String, String>,
    pub anthropic_family_tiers: std::collections::HashMap<String, String>,
    pub scheduler: scheduler::SchedulerConfig,
}

#[derive(Clone)]
struct ServerState {
    config: Arc<ServerConfig>,
    scheduler: Scheduler,
    scheduler_ticket: Option<Arc<str>>,
    created: u64,
    request_sequence: Arc<AtomicU64>,
    response_continuations: Arc<Mutex<responses::ResponseContinuations>>,
}

impl ServerState {
    fn request_id(&self) -> String {
        format!("req_{:016x}{:016x}", self.created, self.request_sequence.fetch_add(1, Ordering::Relaxed),)
    }
}

/// 一次前台推理的共享取消状态。stream producer 负责在模型终态时 disarm，
/// HTTP body 负责在客户端断开、未消费或中途 drop 时触发 Scheduler cancel。
#[derive(Clone)]
struct InferenceCancellation {
    inner: Arc<InferenceCancellationInner>,
}

struct InferenceCancellationInner {
    scheduler: Scheduler,
    request_id: String,
    armed: AtomicBool,
}

impl InferenceCancellation {
    fn new(scheduler: Scheduler, request_id: String) -> Self {
        Self { inner: Arc::new(InferenceCancellationInner { scheduler, request_id, armed: AtomicBool::new(true) }) }
    }

    fn disarm(&self) {
        self.inner.armed.store(false, Ordering::Release);
    }

    async fn cancel(&self) {
        if self.inner.armed.swap(false, Ordering::AcqRel) {
            self.inner.scheduler.cancel(&self.inner.request_id).await;
        }
    }

    fn cancel_detached(&self) {
        if !self.inner.armed.swap(false, Ordering::AcqRel) {
            return;
        }
        let scheduler = self.inner.scheduler.clone();
        let request_id = self.inner.request_id.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move { scheduler.cancel(&request_id).await });
        }
    }
}

/// handler future 被 axum 丢弃时同步触发取消；正常终态必须显式 disarm。
struct InferenceLease {
    cancellation: Option<InferenceCancellation>,
}

impl InferenceLease {
    fn new(scheduler: Scheduler, request_id: String) -> Self {
        Self { cancellation: Some(InferenceCancellation::new(scheduler, request_id)) }
    }

    fn into_cancellation(mut self) -> InferenceCancellation {
        self.cancellation.take().expect("推理 lease 只能转交一次")
    }

    fn disarm(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            cancellation.disarm();
        }
    }

    async fn cancel(mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            cancellation.cancel().await;
        }
    }
}

impl Drop for InferenceLease {
    fn drop(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            cancellation.cancel_detached();
        }
    }
}

/// 取消绑定在真实 HTTP body 上，而不是依赖 producer 下一次 send 才间接发现断连。
struct CancelOnDropStream<S> {
    inner: S,
    cancellation: InferenceCancellation,
}

impl<S> CancelOnDropStream<S> {
    fn new(inner: S, cancellation: InferenceCancellation) -> Self {
        Self { inner, cancellation }
    }
}

impl<S: Stream + Unpin> Stream for CancelOnDropStream<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl<S> Drop for CancelOnDropStream<S> {
    fn drop(&mut self) {
        self.cancellation.cancel_detached();
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    thinking: Option<Value>,
    reasoning_effort: Option<String>,
    thinking_token_budget: Option<i64>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    max_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
    n: Option<u32>,
    stop: Option<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    response_format: Option<Value>,
    stream_options: Option<Value>,
    cache_id: Option<String>,
    repeat_loop_breaker: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: Option<Value>,
    reasoning_content: Option<String>,
    name: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Option<Value>,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    anthropic_family_tier: Option<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorObject,
}

#[derive(Serialize)]
struct ErrorObject {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
    param: Option<String>,
    code: &'static str,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    kind: &'static str,
    param: Option<String>,
    code: &'static str,
}

impl ApiError {
    fn invalid(message: impl Into<String>, param: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: message.into(), kind: "invalid_request_error", param: Some(param.into()), code: "invalid_request" }
    }

    fn into_response(self, request_id: &str) -> Response {
        with_request_id((self.status, Json(ErrorResponse { error: ErrorObject { message: self.message, kind: self.kind, param: self.param, code: self.code } })).into_response(), request_id)
    }
}

pub fn router(config: ServerConfig) -> Router {
    router_with_scheduler(config, Scheduler::default(), None, None)
}

/// 未匹配路由也先读干请求体再应答:上游反代(caddy)还在向本服务转发大 body 时,
/// 直接关闭连接会撞 broken pipe,把本应返回的 404 变成客户端看到的 502 并触发重试。
async fn drain_not_found(request: axum::extract::Request) -> Response {
    let _ = axum::body::to_bytes(request.into_body(), MAX_REQUEST_BYTES).await;
    ApiError { status: StatusCode::NOT_FOUND, message: "未知路径".to_owned(), kind: "invalid_request_error", param: None, code: "not_found" }.into_response("req_unknown")
}

fn router_with_scheduler(config: ServerConfig, scheduler: Scheduler, scheduler_ticket: Option<String>, continuations_dir: Option<std::path::PathBuf>) -> Router {
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let response_continuations = continuations_dir.map(|dir| responses::ResponseContinuations::open(&dir)).unwrap_or_default();
    let state =
        ServerState { config: Arc::new(config), scheduler, scheduler_ticket: scheduler_ticket.map(Arc::from), created, request_sequence: Arc::new(AtomicU64::new(1)), response_continuations: Arc::new(Mutex::new(response_continuations)) };
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/nodes", get(nodes))
        .route("/v1/caches/{cache_id}", get(cache_lookup))
        .route("/v1/scheduler", get(scheduler_info))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/responses", post(responses::responses).get(responses::responses_ws))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v2/video_generation", post(create_video_generation))
        .route("/v2/query/video_generation", get(list_video_generation))
        .route("/v2/query/video_generation/{task_id}", get(query_video_generation))
        .route("/v2/video_generation/{task_id}", delete(delete_video_generation))
        .route("/v1/artifacts/{task_id}/{artifact_id}", get(download_artifact))
        .fallback(drain_not_found)
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(state)
}

pub async fn serve(listener: tokio::net::TcpListener, config: ServerConfig) -> std::io::Result<()> {
    let scheduler_service = SchedulerService::bind(config.node_api_key.clone(), config.scheduler.clone()).await.map_err(std::io::Error::other)?;
    let ticket = scheduler_service.ticket().to_owned();
    println!("zLLM iroh scheduler ticket: {ticket}");
    let continuations_dir = config.scheduler.artifact_dir.join("response-continuations");
    let app = router_with_scheduler(config, scheduler_service.scheduler(), Some(ticket), Some(continuations_dir));
    let result = axum::serve(listener, app).with_graceful_shutdown(termination_signal()).await;
    scheduler_service.shutdown().await;
    result
}

/// standalone 在同一进程内组合 HTTP scheduler 与唯一 node；二者经内存通道直连
/// （消息类型与 wire 协议一致），不走 iroh/QUIC/JSON 环回，decode 每 token 省一跳序列化。
pub async fn serve_standalone<F, Fut>(listener: tokio::net::TcpListener, config: ServerConfig, start_node: F) -> std::io::Result<()>
where
    F: FnOnce(scheduler::Scheduler) -> Fut,
    Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'static,
{
    let scheduler = scheduler::Scheduler::new(&config.scheduler);
    let mut node = tokio::spawn(start_node(scheduler.clone()));
    let continuations_dir = config.scheduler.artifact_dir.join("response-continuations");
    let app = router_with_scheduler(config, scheduler, None, Some(continuations_dir));
    let http = async move { axum::serve(listener, app).with_graceful_shutdown(termination_signal()).await };
    tokio::pin!(http);
    let shutdown = termination_signal();
    tokio::pin!(shutdown);
    let result = tokio::select! {
        biased;
        _ = &mut shutdown => {
            // standalone 的 HTTP 和 node 都收到同一个广播信号。中央分支
            // 先被选中，等 node 完成分布式 cache 握手，不把正常退出误报为崩溃。
            let http_result = (&mut http).await;
            match (&mut node).await {
                Ok(Ok(())) => http_result,
                Ok(Err(error)) => Err(std::io::Error::other(format!("standalone node 优雅退出失败: {error}"))),
                Err(error) => Err(std::io::Error::other(format!("standalone node task 退出失败: {error}"))),
            }
        }
        result = &mut http => result,
        result = &mut node => Err(std::io::Error::other(match result {
            Ok(Ok(())) => "standalone node 意外退出".to_owned(),
            Ok(Err(error)) => format!("standalone node 失败: {error}"),
            Err(error) => format!("standalone node task 失败: {error}"),
        })),
    };
    node.abort();
    result
}

#[cfg(unix)]
pub(super) async fn termination_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("注册 SIGTERM 失败");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
}

#[cfg(not(unix))]
pub(super) async fn termination_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn health(State(state): State<ServerState>) -> Response {
    let request_id = state.request_id();
    let nodes = state.scheduler.nodes().await.len();
    with_request_id(
        Json(json!({
            "status": "ok",
            "model_backend": if nodes == 0 { "disconnected" } else { "connected" },
            "available_nodes": nodes
        }))
        .into_response(),
        &request_id,
    )
}

async fn models(State(state): State<ServerState>, mut headers: HeaderMap) -> Response {
    let request_id = state.request_id();
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let mut model_ids = state.config.models.clone();
    model_ids.extend(state.config.model_aliases.keys().cloned());
    model_ids.extend(state.scheduler.models().await);
    model_ids.sort();
    model_ids.dedup();
    let data = model_ids.iter().map(|id| ModelObject { id: id.clone(), object: "model", created: state.created, owned_by: "zllm", anthropic_family_tier: state.config.anthropic_family_tiers.get(id).cloned() }).collect();
    with_request_id(Json(ModelsResponse { object: "list", data }).into_response(), &request_id)
}

/// `GET /v1/nodes` — 负载均衡管理视图:实时可用节点、最大负载、当前负载、cache 数量。
/// 免鉴权(明确的产品取舍,见 README "管理接口" 一行);若需要受保护,
/// 在反代 / ingress 层加 ACL,不要在 handler 内塞回 auth,避免与 `api_keys` 的语义耦合。
async fn nodes(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    let request_id = state.request_id();
    let _ = headers;
    let nodes = state.scheduler.nodes().await;
    // 顶层摘要 + 顶层 data 列表(保持向后兼容);每条 node 自带 current_load / max_load 字段。
    let summary = json!({
        "object": "list",
        "available_nodes": nodes.len(),
        "total_max_load": nodes.iter().map(|node| node.max_load).sum::<usize>(),
        "total_current_load": nodes.iter().map(|node| node.current_load).sum::<usize>(),
        "data": &nodes,
    });
    with_request_id(Json(summary).into_response(), &request_id)
}

/// `GET /v1/caches/{cache_id}` — 查 cache 当前在哪个 node（含已换出到 swap 的仍算该 owner 持有）。
async fn cache_lookup(State(state): State<ServerState>, Path(cache_id): Path<String>, mut headers: HeaderMap) -> Response {
    let request_id = state.request_id();
    let namespace = match authorize(&state.config, &mut headers) {
        Ok(namespace) => namespace,
        Err(error) => return error.into_response(&request_id),
    };
    let internal_cache_id = scoped_cache_id(namespace.as_deref(), &cache_id);
    match state.scheduler.cache_owner(&internal_cache_id).await {
        Some((node_id, cache)) => with_request_id(
            Json(json!({
                "cache_id": cache_id,
                "model_key": cache.model_key,
                "node_id": node_id,
                "cache_format": cache.cache_format,
                "last_layer": cache.last_layer,
                "prompt_tokens": cache.prompt_tokens,
                "bytes": cache.bytes,
                "modified_unix": cache.modified_unix,
            }))
            .into_response(),
            &request_id,
        ),
        None => ApiError { status: StatusCode::NOT_FOUND, message: format!("cache_id {cache_id} 未找到"), kind: "not_found", param: None, code: "cache_not_found" }.into_response(&request_id),
    }
}

async fn scheduler_info(State(state): State<ServerState>, mut headers: HeaderMap) -> Response {
    let request_id = state.request_id();
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let Some(ticket) = state.scheduler_ticket.as_deref() else {
        return ApiError { status: StatusCode::SERVICE_UNAVAILABLE, message: "iroh scheduler 未启动".to_owned(), kind: "server_error", param: None, code: "scheduler_not_ready" }.into_response(&request_id);
    };
    with_request_id(
        Json(json!({
            "object": "zllm.scheduler",
            "transport": "iroh",
            "alpn": String::from_utf8_lossy(SCHEDULER_ALPN),
            "ticket": ticket,
        }))
        .into_response(),
        &request_id,
    )
}

/// 请求模型命中配置别名时返回后台真实模型；cache hash 与节点调度统一按真实模型，
/// 同一会话经别名或真实模型访问共享同一份 KV cache。
pub(super) fn resolve_model_alias<'a>(config: &'a ServerConfig, model: &'a str) -> &'a str {
    config.model_aliases.get(model).map(String::as_str).unwrap_or(model)
}

fn resume_cache_id(request: &ChatCompletionRequest) -> Option<String> {
    let value = serde_json::to_value(request).ok()?;
    crate::runtime::session::request_resume_boundary(&value).ok().flatten().map(|(cache_id, _)| cache_id)
}

fn request_cache_id(request: &ChatCompletionRequest) -> Option<String> {
    resume_cache_id(request)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_error(error: DispatchError) -> ApiError {
    ApiError { status: StatusCode::SERVICE_UNAVAILABLE, message: error.to_string(), kind: "server_error", param: None, code: "model_not_ready" }
}

async fn await_started(events: &mut mpsc::Receiver<InferenceEvent>) -> Result<(), ApiError> {
    match tokio::time::timeout(Duration::from_secs(30), events.recv()).await {
        Ok(Some(InferenceEvent::Started)) => Ok(()),
        Ok(Some(InferenceEvent::Error { message })) => Err(ApiError { status: StatusCode::BAD_GATEWAY, message, kind: "server_error", param: None, code: "inference_error" }),
        Ok(Some(_)) => Err(ApiError { status: StatusCode::BAD_GATEWAY, message: "节点未发送 Started 就返回了推理数据".to_owned(), kind: "server_error", param: None, code: "protocol_error" }),
        Ok(None) => Err(ApiError { status: StatusCode::BAD_GATEWAY, message: "节点在推理开始前关闭了事件流".to_owned(), kind: "server_error", param: None, code: "node_disconnected" }),
        Err(_) => Err(ApiError { status: StatusCode::GATEWAY_TIMEOUT, message: "等待节点开始推理超时".to_owned(), kind: "server_error", param: None, code: "inference_start_timeout" }),
    }
}

fn authorize(config: &ServerConfig, headers: &mut HeaderMap) -> Result<Option<String>, ApiError> {
    if config.api_keys.is_empty() {
        return Ok(config.anonymous_cache_namespace.clone());
    }
    if !headers.contains_key(header::AUTHORIZATION)
        && let Some(api_key) = headers.get("x-api-key").cloned()
    {
        let mut value = b"Bearer ".to_vec();
        value.extend_from_slice(api_key.as_bytes());
        if let Ok(value) = HeaderValue::from_bytes(&value) {
            headers.insert(header::AUTHORIZATION, value);
        }
    }
    let supplied = headers.get(header::AUTHORIZATION).and_then(|value| value.to_str().ok()).and_then(|value| value.strip_prefix("Bearer "));
    let accepted = supplied.is_some_and(|value| config.api_keys.iter().fold(false, |accepted, expected| accepted | constant_time_eq(value.as_bytes(), expected.as_bytes())));
    if let Some(value) = supplied.filter(|_| accepted) {
        return Ok(Some(blake3::hash(value.as_bytes()).to_hex().to_string()));
    }
    Err(ApiError { status: StatusCode::UNAUTHORIZED, message: "Authorization Bearer token 无效".to_owned(), kind: "invalid_request_error", param: None, code: "invalid_api_key" })
}

pub fn scoped_cache_id(namespace: Option<&str>, cache_id: &str) -> String {
    crate::runtime::session::scoped_cache_id(namespace, cache_id)
}

fn attach_cache_namespace(request: &mut Value, namespace: Option<&str>) {
    if let (Some(namespace), Some(request)) = (namespace, request.as_object_mut()) {
        request.insert("_zllm_cache_namespace".to_owned(), Value::String(namespace.to_owned()));
    }
}

fn validate_chat_request(request: &ChatCompletionRequest, _models: &[String]) -> Result<(), ApiError> {
    if request.model.trim().is_empty() {
        return Err(ApiError::invalid("model 不能为空", "model"));
    }
    if request.messages.is_empty() {
        return Err(ApiError::invalid("messages 至少需要一条消息", "messages"));
    }
    for (index, message) in request.messages.iter().enumerate() {
        validate_message(message, index)?;
    }
    if request.temperature.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value)) {
        return Err(ApiError::invalid("temperature 必须位于 [0,2]", "temperature"));
    }
    if request.top_p.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(ApiError::invalid("top_p 必须位于 [0,1]", "top_p"));
    }
    if request.thinking.as_ref().is_some_and(|value| !value.as_object().and_then(|value| value.get("type")).and_then(Value::as_str).is_some_and(|value| matches!(value, "enabled" | "disabled"))) {
        return Err(ApiError::invalid("thinking.type 必须是 enabled / disabled", "thinking"));
    }
    if request.reasoning_effort.as_deref().is_some_and(|value| !matches!(value, "low" | "medium" | "high" | "max" | "xhigh")) {
        return Err(ApiError::invalid("reasoning_effort 必须是 low / medium / high / max / xhigh", "reasoning_effort"));
    }
    if request.thinking_token_budget.is_some_and(|value| value < -1) {
        return Err(ApiError::invalid("thinking_token_budget 必须是 -1 或非负整数", "thinking_token_budget"));
    }
    if request.max_tokens == Some(0) {
        return Err(ApiError::invalid("max_tokens 必须大于 0", "max_tokens"));
    }
    if request.max_completion_tokens == Some(0) {
        return Err(ApiError::invalid("max_completion_tokens 必须大于 0", "max_completion_tokens"));
    }
    if request.n.is_some_and(|value| value == 0) {
        return Err(ApiError::invalid("n 必须大于 0", "n"));
    }
    validate_stop(request.stop.as_ref())?;
    validate_tools(request.tools.as_deref())?;
    validate_tool_choice(request.tool_choice.as_ref(), request.tools.as_deref())?;
    validate_optional_object(request.response_format.as_ref(), "response_format")?;
    validate_optional_object(request.stream_options.as_ref(), "stream_options")?;
    Ok(())
}

fn validate_message(message: &ChatMessage, index: usize) -> Result<(), ApiError> {
    let param = format!("messages.{index}");
    if !matches!(message.role.as_str(), "developer" | "system" | "user" | "assistant" | "tool") {
        return Err(ApiError::invalid(format!("{} 的 role '{}' 无效", param, message.role), format!("{param}.role")));
    }
    if message.name.as_ref().is_some_and(|name| name.trim().is_empty()) {
        return Err(ApiError::invalid("message name 不能为空", format!("{param}.name")));
    }
    if message.role == "tool" && message.tool_call_id.as_ref().is_none_or(|id| id.trim().is_empty()) {
        return Err(ApiError::invalid("tool 消息必须提供 tool_call_id", format!("{param}.tool_call_id")));
    }
    if message.tool_calls.is_some() && message.role != "assistant" {
        return Err(ApiError::invalid("只有 assistant 消息可以包含 tool_calls", format!("{param}.tool_calls")));
    }
    if let Some(tool_calls) = message.tool_calls.as_ref() {
        validate_message_tool_calls(tool_calls, &param)?;
    }
    let content_optional = message.role == "assistant" && message.tool_calls.is_some();
    match message.content.as_ref() {
        Some(Value::String(_)) => {}
        Some(Value::Array(parts)) if !parts.is_empty() => {
            for (part_index, part) in parts.iter().enumerate() {
                let Some(kind) = part.as_object().and_then(|part| part.get("type")).and_then(Value::as_str) else {
                    return Err(ApiError::invalid("多模态 content part 必须是带 type 的对象", format!("{param}.content.{part_index}")));
                };
                if kind.trim().is_empty() {
                    return Err(ApiError::invalid("content part type 不能为空", format!("{param}.content.{part_index}.type")));
                }
            }
        }
        None | Some(Value::Null) if content_optional => {}
        _ => return Err(ApiError::invalid("content 必须是字符串或非空多模态数组", format!("{param}.content"))),
    }
    Ok(())
}

fn validate_message_tool_calls(value: &Value, param: &str) -> Result<(), ApiError> {
    let calls = value.as_array().filter(|calls| !calls.is_empty()).ok_or_else(|| ApiError::invalid("assistant.tool_calls 必须是非空数组", format!("{param}.tool_calls")))?;
    for (index, call) in calls.iter().enumerate() {
        let call_param = format!("{param}.tool_calls.{index}");
        let object = call.as_object().ok_or_else(|| ApiError::invalid("tool_call 必须是对象", &call_param))?;
        if object.get("id").and_then(Value::as_str).is_none_or(|id| id.trim().is_empty()) {
            return Err(ApiError::invalid("tool_call.id 不能为空", format!("{call_param}.id")));
        }
        if object.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ApiError::invalid("tool_call.type 必须为 function", format!("{call_param}.type")));
        }
        let function = object.get("function").and_then(Value::as_object).ok_or_else(|| ApiError::invalid("tool_call.function 必须是对象", format!("{call_param}.function")))?;
        if function.get("name").and_then(Value::as_str).is_none_or(|name| name.trim().is_empty()) {
            return Err(ApiError::invalid("tool_call.function.name 不能为空", format!("{call_param}.function.name")));
        }
        if !function.get("arguments").is_some_and(Value::is_string) {
            return Err(ApiError::invalid("tool_call.function.arguments 必须是 JSON 字符串", format!("{call_param}.function.arguments")));
        }
    }
    Ok(())
}

fn validate_tools(tools: Option<&[Value]>) -> Result<(), ApiError> {
    let Some(tools) = tools else {
        return Ok(());
    };
    for (index, tool) in tools.iter().enumerate() {
        let object = tool.as_object().ok_or_else(|| ApiError::invalid("tool 必须是对象", format!("tools.{index}")))?;
        if object.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ApiError::invalid("当前 tool type 必须为 function", format!("tools.{index}.type")));
        }
        let function = object.get("function").and_then(Value::as_object).ok_or_else(|| ApiError::invalid("tool.function 必须是对象", format!("tools.{index}.function")))?;
        if function.get("name").and_then(Value::as_str).is_none_or(|name| name.trim().is_empty()) {
            return Err(ApiError::invalid("tool function name 不能为空", format!("tools.{index}.function.name")));
        }
    }
    Ok(())
}

fn validate_tool_choice(choice: Option<&Value>, tools: Option<&[Value]>) -> Result<(), ApiError> {
    let available = tools.unwrap_or_default();
    match choice {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(value)) if matches!(value.as_str(), "none" | "auto") => Ok(()),
        Some(Value::String(value)) if value == "required" && !available.is_empty() => Ok(()),
        Some(Value::String(value)) if value == "required" => Err(ApiError::invalid("tool_choice=required 时 tools 不能为空", "tool_choice")),
        Some(Value::Object(object))
            if object.get("type").and_then(Value::as_str) == Some("function")
                && object
                    .get("function")
                    .and_then(Value::as_object)
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .is_some_and(|name| available.iter().any(|tool| tool.get("function").and_then(|function| function.get("name")).and_then(Value::as_str) == Some(name))) =>
        {
            Ok(())
        }
        _ => Err(ApiError::invalid("tool_choice 必须是 none/auto/required 或指定 function 的对象", "tool_choice")),
    }
}

fn validate_stop(stop: Option<&Value>) -> Result<(), ApiError> {
    match stop {
        None | Some(Value::Null) | Some(Value::String(_)) => Ok(()),
        Some(Value::Array(values)) if values.len() <= 4 && values.iter().all(Value::is_string) => Ok(()),
        _ => Err(ApiError::invalid("stop 必须是字符串或最多 4 个字符串的数组", "stop")),
    }
}

fn validate_optional_object(value: Option<&Value>, param: &'static str) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_object()) {
        return Err(ApiError::invalid(format!("{param} 必须是对象"), param));
    }
    Ok(())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right).fold(0u8, |difference, (left, right)| difference | (left ^ right)) == 0
}

/// JSON 允许原样携带 NEL/Line Separator/Paragraph Separator，但 Python
/// Requests 等常见 SSE 客户端会用 Unicode `splitlines()` 拆流，把它们误判为
/// event 边界并产生截断 JSON。在线流协议统一转义这三个码点；解析后的文本不变。
pub(super) fn sse_json(value: &Value) -> String {
    let encoded = value.to_string();
    if !encoded.contains(['\u{0085}', '\u{2028}', '\u{2029}']) {
        return encoded;
    }
    encoded.replace('\u{0085}', "\\u0085").replace('\u{2028}', "\\u2028").replace('\u{2029}', "\\u2029")
}

/// SSE 是 UTF-8 文本；显式声明 charset，避免客户端按 Latin-1 解码后把
/// 多字节字符中的 `0x85` 等字节误当成 Unicode 换行并截断 event JSON。
pub(super) fn sse_utf8(mut response: Response) -> Response {
    response.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    response
}

fn with_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::chat::{collect_completion, stream_completion, tool_argument_schemas};
    use super::responses::{ResponseContinuation, collect_response_api, response_chat_request, responses_request, stream_response_api};
    use super::scheduler::ToolCall;
    use super::*;
    use std::collections::HashMap;

    fn test_state() -> ServerState {
        ServerState {
            config: Arc::new(ServerConfig::default()),
            scheduler: Scheduler::default(),
            scheduler_ticket: None,
            created: 1,
            request_sequence: Arc::new(AtomicU64::new(1)),
            response_continuations: Arc::new(Mutex::new(responses::ResponseContinuations::default())),
        }
    }

    fn test_lease(state: &ServerState, request_id: &str) -> InferenceLease {
        InferenceLease::new(state.scheduler.clone(), request_id.to_owned())
    }

    #[test]
    fn sse_json_escapes_unicode_line_boundaries_without_changing_value() {
        let value = json!({"text": "a\u{0085}b\u{2028}c\u{2029}d"});
        let encoded = sse_json(&value);
        assert!(!encoded.contains(['\u{0085}', '\u{2028}', '\u{2029}']));
        assert!(encoded.contains("\\u0085"));
        assert!(encoded.contains("\\u2028"));
        assert!(encoded.contains("\\u2029"));
        assert_eq!(serde_json::from_str::<Value>(&encoded).unwrap(), value);
    }

    async fn wait_cancelled(scheduler: &Scheduler, request_id: &str) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !scheduler.test_request_cancelled(request_id).await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("body/future drop 应在一秒内传播 Scheduler cancel");
        assert!(scheduler.test_request_active(request_id).await, "cancel 只能标记 inflight，必须等 node terminal 才释放");
    }

    #[test]
    fn authorize_accepts_any_key_and_scopes_cache() {
        let config = ServerConfig { api_keys: vec!["sk-zllm-first".to_owned(), "sk-zllm-second".to_owned()], ..ServerConfig::default() };
        let mut valid = HeaderMap::new();
        valid.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer sk-zllm-second"));
        let namespace = authorize(&config, &mut valid).unwrap().unwrap();
        let first = scoped_cache_id(Some(&namespace), "shared-cache");
        let mut request = json!({"model": "glm-5.2"});
        attach_cache_namespace(&mut request, Some(&namespace));
        assert_eq!(request.get("_zllm_cache_namespace").and_then(Value::as_str), Some(namespace.as_str()));

        let mut other = HeaderMap::new();
        other.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer sk-zllm-first"));
        let other_namespace = authorize(&config, &mut other).unwrap().unwrap();
        assert_ne!(first, scoped_cache_id(Some(&other_namespace), "shared-cache"));

        let mut invalid = HeaderMap::new();
        invalid.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer sk-zllm-invalid"));
        assert!(authorize(&config, &mut invalid).is_err());
    }

    #[test]
    fn anonymous_service_uses_fixed_cache_namespace() {
        let config = ServerConfig { anonymous_cache_namespace: Some("standalone".to_owned()), ..ServerConfig::default() };
        let mut first = HeaderMap::new();
        let mut second = HeaderMap::new();
        second.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer ignored"));
        assert_eq!(authorize(&config, &mut first).unwrap().as_deref(), Some("standalone"));
        assert_eq!(authorize(&config, &mut second).unwrap().as_deref(), Some("standalone"));
    }

    #[test]
    fn models按配置输出anthropic家族tier() {
        let configured = serde_json::to_value(ModelObject { id: "deepseek-v4-flash".to_owned(), object: "model", created: 1, owned_by: "zllm", anthropic_family_tier: Some("sonnet".to_owned()) }).unwrap();
        assert_eq!(configured["anthropic_family_tier"], "sonnet");

        let plain = serde_json::to_value(ModelObject { id: "glm-5.2".to_owned(), object: "model", created: 1, owned_by: "zllm", anthropic_family_tier: None }).unwrap();
        assert!(plain.get("anthropic_family_tier").is_none());
    }

    #[tokio::test]
    async fn 模型别名下发并解析() {
        let mut config = ServerConfig::default();
        config.models = vec!["glm-5.2".to_owned()];
        config.model_aliases = HashMap::from([("claude-opus-5-2".to_owned(), "glm-5.2".to_owned())]);
        config.anthropic_family_tiers = HashMap::from([("claude-opus-5-2".to_owned(), "opus".to_owned())]);
        assert_eq!(resolve_model_alias(&config, "claude-opus-5-2"), "glm-5.2");
        assert_eq!(resolve_model_alias(&config, "glm-5.2"), "glm-5.2");

        let state = ServerState { config: Arc::new(config), ..test_state() };
        let response = models(State(state), HeaderMap::new()).await;
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let entries = value["data"].as_array().unwrap();
        let ids: Vec<&str> = entries.iter().map(|entry| entry["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"glm-5.2") && ids.contains(&"claude-opus-5-2"));
        let alias = entries.iter().find(|entry| entry["id"] == "claude-opus-5-2").unwrap();
        assert_eq!(alias["anthropic_family_tier"], "opus");
    }

    #[test]
    fn responses别名重写为后台模型() {
        let mut config = ServerConfig::default();
        config.model_aliases = HashMap::from([("claude-opus-5-2".to_owned(), "glm-5.2".to_owned())]);
        let state = ServerState { config: Arc::new(config), ..test_state() };
        let mut request = responses_request(true);
        request.model = "claude-opus-5-2".to_owned();
        let dispatch = responses::prepare_response_dispatch(&state, None, request).unwrap();
        assert_eq!(dispatch.model, "glm-5.2");
        assert_eq!(dispatch.request_value["model"], "glm-5.2");
        assert_eq!(dispatch.request.model, "claude-opus-5-2");
    }

    #[tokio::test]
    async fn anthropic_count_tokens按内容估算() {
        let state = test_state();
        let request = json!({
            "model": "claude-opus-5-2",
            "system": "你是助手",
            "messages": [
                {"role": "user", "content": "你好世界"},
                {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "完成"}]}
            ],
            "tools": [{"name": "Read", "description": "read file", "input_schema": {"type": "object"}}]
        });
        let payload = serde_json::from_value::<anthropic::AnthropicCountTokensRequest>(request).unwrap();
        let response = anthropic::count_tokens(State(state), HeaderMap::new(), Ok(Json(payload))).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let tokens = value["input_tokens"].as_u64().unwrap();
        assert!(tokens > 20, "估算应包含 system/messages/tools 的内容,实际 {tokens}");
    }

    #[test]
    fn anthropic估算cjk重于ascii() {
        assert_eq!(anthropic::anthropic_estimate_tokens("你好世界"), 5);
        assert_eq!(anthropic::anthropic_estimate_tokens("hello!"), 2); // 6 字符 / 4 + 1
        assert!(anthropic::anthropic_estimate_tokens("你好世界") > anthropic::anthropic_estimate_tokens("abcd"));
    }

    #[tokio::test]
    async fn 未知路径先读干body再返回404() {
        let request = axum::extract::Request::builder().uri("/v1/nonexistent").body(axum::body::Body::from("x".repeat(4096))).expect("测试请求构建失败");
        let response = drain_not_found(request).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn terminal_cache_id_matches_chat_message_serialization() {
        let previous: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Reply with exactly CACHE-ONE."}]
        }))
        .unwrap();
        let terminal = scheduler::terminal_cache_id(&serde_json::to_value(&previous).unwrap(), "CACHE-ONE", &[]).unwrap();
        let next: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": "Reply with exactly CACHE-ONE."},
                {"role": "assistant", "content": "CACHE-ONE"},
                {"role": "user", "content": "Reply with exactly CACHE-TWO."}
            ]
        }))
        .unwrap();
        assert_eq!(request_cache_id(&next).as_deref(), Some(terminal.as_str()));
    }

    #[tokio::test]
    async fn chat_completion_stream_is_sse_and_ends_with_done() {
        let state = test_state();
        let (events, receiver) = mpsc::channel(8);
        events.send(InferenceEvent::Started).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 42, text: "你好".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 1 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_test");
        let response = stream_completion(state, "req_test".to_owned(), "glm-5.2".to_owned(), HashMap::new(), false, receiver, lease).await;
        assert_eq!(response.headers().get(header::CONTENT_TYPE).and_then(|value| value.to_str().ok()), Some("text/event-stream; charset=utf-8"));
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("你好"));
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn chat_completion_stream_separates_reasoning_before_think_end() {
        let state = test_state();
        let (events, receiver) = mpsc::channel(8);
        events.send(InferenceEvent::Started).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 1, text: "先分析 ```python\n错误草稿\n```".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 2, text: "</think>".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 3, text: "```python\nprint(1)\n```".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 3 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_reasoning_stream");
        let response = stream_completion(state, "req_reasoning_stream".to_owned(), "glm-5.2".to_owned(), HashMap::new(), true, receiver, lease).await;
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains(r#""reasoning_content":"先分析 ```python\n错误草稿\n```""#));
        assert!(body.contains(r#""content":"```python\nprint(1)\n```""#));
        assert!(!body.contains("</think>"));
    }

    #[tokio::test]
    async fn chat_completion_json_separates_reasoning_before_think_end() {
        let state = test_state();
        let (events, receiver) = mpsc::channel(8);
        events.send(InferenceEvent::Started).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 1, text: "analysis".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 2, text: "</think>answer".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 2 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_reasoning_json");
        let response = collect_completion(state, "req_reasoning_json".to_owned(), "glm-5.2".to_owned(), HashMap::new(), true, receiver, lease).await;
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["reasoning_content"], "analysis");
        assert_eq!(value["choices"][0]["message"]["content"], "answer");
    }

    #[tokio::test]
    async fn chat_completion_stream_emits_incremental_tool_arguments_once() {
        let state = test_state();
        let tool_call = ToolCall { id: "call_bash".to_owned(), kind: "function".to_owned(), function: scheduler::ToolFunction { name: "Bash".to_owned(), arguments: r#"{"command":"python3"}"#.to_owned() } };
        let (events, receiver) = mpsc::channel(8);
        events.send(InferenceEvent::Started).await.unwrap();
        events.send(InferenceEvent::ToolCallDelta { delta: scheduler::ToolCallDelta { index: 0, id: Some("call_bash".to_owned()), name: Some("Bash".to_owned()), arguments: r#"{"command":"python"#.to_owned() } }).await.unwrap();
        events.send(InferenceEvent::ToolCallDelta { delta: scheduler::ToolCallDelta { index: 0, id: None, name: None, arguments: r#"3"}"#.to_owned() } }).await.unwrap();
        events.send(InferenceEvent::ToolCall { index: 0, tool_call }).await.unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "tool_calls".to_owned(), prompt_tokens: 3, completion_tokens: 8 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_tool_delta");
        let response = stream_completion(
            state,
            "req_tool_delta".to_owned(),
            "glm-5.2".to_owned(),
            tool_argument_schemas(&[json!({"type":"function","function":{"name":"zcode","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}})]),
            false,
            receiver,
            lease,
        )
        .await;
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert_eq!(body.matches("call_bash").count(), 1, "最终完整 ToolCall 不能重复发送");
        assert!(body.contains("call_bash"));
        assert!(body.contains("Bash"));
        assert!(body.contains("python"));
        assert!(body.contains("3\\\"}"));
        assert!(body.contains("\"finish_reason\":\"tool_calls\""));
    }

    #[tokio::test]
    async fn anthropic_stream_converts_raw_xml_tool_call_for_zcode() {
        let state = test_state();
        let (events, receiver) = mpsc::channel(8);
        events.send(InferenceEvent::Started).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 1, text: "<tool_ca".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 2, text: "ll>WebSearch<arg_key>query</arg_key><arg_value>Radeon P2P DMA</arg_value></tool_call>".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 2 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_zcode_xml");
        let chat = stream_completion(
            state,
            "req_zcode_xml".to_owned(),
            "glm-5.2".to_owned(),
            tool_argument_schemas(&[json!({"type":"function","function":{"name":"WebSearch","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}})]),
            false,
            receiver,
            lease,
        )
        .await;
        let response = anthropic::anthropic_stream_response(chat, "glm-5.2".to_owned());
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains(r#""type":"tool_use""#));
        assert!(body.contains(r#""name":"WebSearch""#));
        assert!(body.contains(r#""partial_json":"{\"query\":\"Radeon P2P DMA\"}""#));
        assert!(body.contains(r#""stop_reason":"tool_use""#));
        assert!(!body.contains("<tool_call>"));
    }

    #[tokio::test]
    async fn anthropic_stream_does_not_repair_malformed_dsml_for_zcode() {
        let state = test_state();
        let (events, receiver) = mpsc::channel(8);
        events.send(InferenceEvent::Started).await.unwrap();
        events
            .send(InferenceEvent::Token {
                token_id: 1,
                text: r#"Let me先看一下。<｜DSML｜tool\_c优化化 <｜DSML｜tool\_calls>
<｜DSML｜invrule name="Read">
<｜DSML｜parameter name="file\_path" string="true">docs/qwen36-dspark-metal.md\</｜DSML｜parameter>
\</｜DSML｜inv>
\</｜DSML｜tool\_calls>"#
                    .to_owned(),
            })
            .await
            .unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 2 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_zcode_dsml_variant");
        let chat = stream_completion(
            state,
            "req_zcode_dsml_variant".to_owned(),
            "deepseek-v4-flash".to_owned(),
            tool_argument_schemas(&[json!({"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}})]),
            false,
            receiver,
            lease,
        )
        .await;
        let response = anthropic::anthropic_stream_response(chat, "deepseek-v4-flash".to_owned());
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(!body.contains(r#""type":"tool_use""#));
        assert!(!body.contains(r#""stop_reason":"tool_use""#));
        assert!(!body.contains("invrule"));
        assert!(!body.contains("DSML"));
    }

    #[tokio::test]
    async fn chat_completion_converts_raw_xml_tool_call() {
        let state = test_state();
        let (events, receiver) = mpsc::channel(4);
        events.send(InferenceEvent::Started).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 1, text: "<tool_call>WebSearch<arg_key>query</arg_key><arg_value>Radeon P2P DMA</arg_value></tool_call>".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 1 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_xml_json");
        let response = collect_completion(
            state,
            "req_xml_json".to_owned(),
            "glm-5.2".to_owned(),
            tool_argument_schemas(&[json!({"type":"function","function":{"name":"WebSearch","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}})]),
            false,
            receiver,
            lease,
        )
        .await;
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert!(value["choices"][0]["message"]["content"].is_null());
        assert_eq!(value["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "WebSearch");
        assert_eq!(value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], r#"{"query":"Radeon P2P DMA"}"#);
        assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn chat_completion_converts_raw_deepseek_dsml_tool_call() {
        let state = test_state();
        let (events, receiver) = mpsc::channel(4);
        events.send(InferenceEvent::Started).await.unwrap();
        events
            .send(InferenceEvent::Token {
                token_id: 1,
                text: "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Read\">\n<｜DSML｜parameter name=\"path\" string=\"true\">docs/qwen36-dspark-metal.md</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>".to_owned(),
            })
            .await
            .unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 1 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_dsml_json");
        let response = collect_completion(
            state,
            "req_dsml_json".to_owned(),
            "deepseek-v4-flash".to_owned(),
            tool_argument_schemas(&[json!({"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}})]),
            false,
            receiver,
            lease,
        )
        .await;
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert!(value["choices"][0]["message"]["content"].is_null());
        assert_eq!(value["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "Read");
        assert_eq!(value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], r#"{"path":"docs/qwen36-dspark-metal.md"}"#);
        assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn responses_stream_is_sse_and_previous_response_reuses_cache() {
        let state = test_state();
        let mut request = responses_request(true);
        request.instructions = Some("始终简短回答".to_owned());
        request.cache_id = Some("旧输入-cache".to_owned());
        let chat_request = response_chat_request(&request, None).unwrap();
        let messages = chat_request.messages.clone();
        let cache_request = serde_json::to_value(&chat_request).unwrap();
        let expected_cache_request = cache_request.clone();
        let (events, receiver) = mpsc::channel(4);
        events.send(InferenceEvent::Started).await.unwrap();
        events.send(InferenceEvent::Token { token_id: 42, text: "世界".to_owned() }).await.unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 1 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_test");
        let namespace = "test-namespace";
        let response = stream_response_api(state.clone(), "req_test".to_owned(), "resp_test".to_owned(), "resp_test".to_owned(), Some(namespace.to_owned()), request, messages, cache_request, receiver, lease).await;
        assert_eq!(response.headers().get(header::CONTENT_TYPE).and_then(|value| value.to_str().ok()), Some("text/event-stream; charset=utf-8"));
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.completed"));

        let previous = state.response_continuations.lock().unwrap().get("resp_test").unwrap();
        let terminal_hash = crate::runtime::session::terminal_cache_id(&expected_cache_request, "世界", &[]).unwrap();
        assert_eq!(previous.cache_id, scoped_cache_id(Some(namespace), &terminal_hash));
        assert_ne!(previous.cache_id, "旧输入-cache");
        let mut next = responses_request(false);
        next.input = Value::String("继续".to_owned());
        next.instructions = Some("始终简短回答".to_owned());
        next.previous_response_id = Some("resp_test".to_owned());
        let resumed = response_chat_request(&next, Some(&previous)).unwrap();
        assert_eq!(resumed.cache_id.as_deref(), Some(previous.cache_id.as_str()));
        let mut replay_request = resumed.clone();
        replay_request.cache_id = None;
        assert_eq!(request_cache_id(&replay_request).as_deref(), Some(terminal_hash.as_str()), "重放完整消息时也必须定位到上一轮 terminal cache");
        assert_eq!(resumed.messages.last().and_then(|message| message.content.as_ref()).and_then(Value::as_str), Some("继续"));
        assert!(resumed.messages.iter().any(|message| message.role == "assistant" && message.content.as_ref().and_then(Value::as_str) == Some("世界")));
        assert_eq!(resumed.messages.iter().filter(|message| message.role == "developer").count(), 1);

        next.instructions = Some("改用详细回答".to_owned());
        let replaced = response_chat_request(&next, Some(&previous)).unwrap();
        assert_eq!(replaced.messages.first().and_then(|message| message.content.as_ref()).and_then(Value::as_str), Some("改用详细回答"));
        assert!(!replaced.messages.iter().any(|message| message.content.as_ref().and_then(Value::as_str) == Some("始终简短回答")));
    }

    #[tokio::test]
    async fn responses_stream_emits_function_call_and_keeps_tool_continuation() {
        let state = test_state();
        let mut request = responses_request(true);
        request.tools = Some(vec![json!({
            "type": "function",
            "name": "read_file",
            "description": "读取文件",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        })]);
        let chat_request = response_chat_request(&request, None).unwrap();
        let messages = chat_request.messages.clone();
        let cache_request = serde_json::to_value(chat_request).unwrap();
        let tool_call = ToolCall { id: "call_test".to_owned(), kind: "function".to_owned(), function: scheduler::ToolFunction { name: "read_file".to_owned(), arguments: r#"{"path":"/tmp/a"}"#.to_owned() } };
        let (events, receiver) = mpsc::channel(8);
        events.send(InferenceEvent::Started).await.unwrap();
        events.send(InferenceEvent::ToolCallDelta { delta: scheduler::ToolCallDelta { index: 0, id: Some("call_test".to_owned()), name: Some("read_file".to_owned()), arguments: r#"{"path":"/tmp"#.to_owned() } }).await.unwrap();
        events.send(InferenceEvent::ToolCallDelta { delta: scheduler::ToolCallDelta { index: 0, id: None, name: None, arguments: r#"/a"}"#.to_owned() } }).await.unwrap();
        events.send(InferenceEvent::ToolCall { index: 0, tool_call }).await.unwrap();
        events.send(InferenceEvent::Completed { finish_reason: "tool_calls".to_owned(), prompt_tokens: 3, completion_tokens: 8 }).await.unwrap();
        drop(events);

        let lease = test_lease(&state, "req_tool");
        let response = stream_response_api(state.clone(), "req_tool".to_owned(), "resp_tool".to_owned(), "resp_tool".to_owned(), None, request, messages, cache_request, receiver, lease).await;
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let stream_events = body.lines().filter_map(|line| line.strip_prefix("data: ")).map(|data| serde_json::from_str::<Value>(data).unwrap()).collect::<Vec<_>>();
        assert_eq!(stream_events.iter().filter_map(|event| event.get("sequence_number").and_then(Value::as_u64)).collect::<Vec<_>>(), (0..stream_events.len() as u64).collect::<Vec<_>>());
        let function_item_ids = stream_events
            .iter()
            .filter(|event| {
                matches!(
                    event.get("type").and_then(Value::as_str),
                    Some("response.output_item.added" | "response.function_call_arguments.delta" | "response.function_call_arguments.done" | "response.output_item.done" | "response.completed")
                )
            })
            .filter_map(|event| event.pointer("/item/id").or_else(|| event.get("item_id")).or_else(|| event.pointer("/response/output/0/id")).and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert_eq!(function_item_ids, vec!["fc_test"; function_item_ids.len()]);
        assert!(body.contains("event: response.function_call_arguments.delta"));
        assert_eq!(body.matches("event: response.function_call_arguments.delta").count(), 2);
        assert!(body.contains("event: response.function_call_arguments.done"));
        assert!(body.contains(r#""call_id":"call_test""#));
        assert!(body.contains(r#""name":"read_file""#));

        let previous = state.response_continuations.lock().unwrap().get("resp_tool").unwrap();
        let assistant = previous.messages.last().unwrap();
        assert!(assistant.content.is_none());
        assert_eq!(assistant.tool_calls.as_ref().and_then(Value::as_array).map(Vec::len), Some(1));
    }

    #[tokio::test]
    async fn anthropic_system_stream_and_cache_id_are_forwarded() {
        let request = anthropic::AnthropicMessagesRequest {
            model: "glm-5.2".to_owned(),
            max_tokens: 8,
            messages: vec![
                json!({"role": "system", "content": [{"type": "text", "text": "消息内系统指令"}]}),
                json!({"role": "user", "content": [{"type": "text", "text": "你"}, {"type": "text", "text": "好"}]}),
                json!({"role": "assistant", "content": [{"type": "text", "text": "世"}, {"type": "text", "text": "界"}]}),
                json!({"role": "user", "content": "继续"}),
            ],
            stream: true,
            system: Some(json!([{"type": "text", "text": "顶层系统指令"}])),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            cache_id: Some("cache-test".to_owned()),
        };
        let chat = anthropic::anthropic_to_chat_request(request).unwrap();
        assert_eq!(chat.cache_id.as_deref(), Some("cache-test"));
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[0].content.as_ref().and_then(Value::as_str), Some("顶层系统指令"));
        assert_eq!(chat.messages[1].role, "system");
        assert_eq!(chat.messages[1].content.as_ref().and_then(Value::as_str), Some("消息内系统指令"));
        assert_eq!(chat.messages[2].content.as_ref().and_then(Value::as_str), Some("你好"));
        assert_eq!(chat.messages[3].content.as_ref().and_then(Value::as_str), Some("世界"));
        let mut automatic = chat.clone();
        automatic.cache_id = None;
        assert!(resume_cache_id(&automatic).is_some(), "普通 Anthropic 全历史请求应自动定位上一轮 cache");

        let source = concat!(
            "data: {\"id\":\"req_test\",\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"req_test\",\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"世界\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"req_test\",\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n"
        );
        let response = Response::builder().header(header::CONTENT_TYPE, "text/event-stream").body(axum::body::Body::from(source)).unwrap();
        let response = anthropic::anthropic_stream_response(response, "glm-5.2".to_owned());
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("event: message_stop"));
        assert!(body.contains("世界"));
    }

    #[tokio::test]
    async fn anthropic长prefill直接向外层发送心跳() {
        let pending = futures_util::stream::pending::<Result<axum::body::Bytes, std::io::Error>>();
        let response = Response::builder().header(header::CONTENT_TYPE, "text/event-stream").body(axum::body::Body::from_stream(pending)).unwrap();
        let response = anthropic::anthropic_stream_response(response, "glm-5.2".to_owned());
        let mut body = response.into_body().into_data_stream();
        let heartbeat = tokio::time::timeout(Duration::from_secs(1), body.next()).await.expect("Anthropic 外层应立即发送首个心跳").unwrap().unwrap();
        assert_eq!(heartbeat.as_ref(), b": keep-alive\n\n");
    }

    #[tokio::test]
    async fn anthropic_stream_translates_function_call_to_tool_use() {
        let source = concat!(
            "data: {\"id\":\"req_tool\",\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"req_tool\",\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_test\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/a\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"req_tool\",\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":8}}\n\n",
            "data: [DONE]\n\n"
        );
        let response = Response::builder().header(header::CONTENT_TYPE, "text/event-stream").body(axum::body::Body::from(source)).unwrap();
        let response = anthropic::anthropic_stream_response(response, "glm-5.2".to_owned());
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains(r#""type":"tool_use""#));
        assert!(body.contains(r#""id":"call_test""#));
        assert!(body.contains(r#""name":"read_file""#));
        assert!(body.contains(r#""type":"input_json_delta""#));
        assert!(body.contains(r#""stop_reason":"tool_use""#));
    }

    #[tokio::test]
    async fn chat_stream_body_drop_cancels_scheduler_request() {
        let state = test_state();
        state.scheduler.track_test_request("req_chat_drop").await;
        let (events, receiver) = mpsc::channel(2);
        events.send(InferenceEvent::Started).await.unwrap();
        let lease = test_lease(&state, "req_chat_drop");
        let response = stream_completion(state.clone(), "req_chat_drop".to_owned(), "glm-5.2".to_owned(), HashMap::new(), false, receiver, lease).await;
        drop(response);
        wait_cancelled(&state.scheduler, "req_chat_drop").await;
        drop(events);
    }

    #[tokio::test]
    async fn responses_stream_body_drop_cancels_scheduler_request() {
        let state = test_state();
        state.scheduler.track_test_request("req_responses_drop").await;
        let request = responses_request(true);
        let chat_request = response_chat_request(&request, None).unwrap();
        let messages = chat_request.messages.clone();
        let cache_request = serde_json::to_value(chat_request).unwrap();
        let (events, receiver) = mpsc::channel(2);
        events.send(InferenceEvent::Started).await.unwrap();
        let lease = test_lease(&state, "req_responses_drop");
        let response = stream_response_api(state.clone(), "req_responses_drop".to_owned(), "resp_responses_drop".to_owned(), "resp_responses_drop".to_owned(), None, request, messages, cache_request, receiver, lease).await;
        drop(response);
        wait_cancelled(&state.scheduler, "req_responses_drop").await;
        drop(events);
    }

    #[tokio::test]
    async fn anthropic_stream_body_drop_cancels_scheduler_request() {
        let state = test_state();
        state.scheduler.track_test_request("req_anthropic_drop").await;
        let (events, receiver) = mpsc::channel(2);
        events.send(InferenceEvent::Started).await.unwrap();
        let lease = test_lease(&state, "req_anthropic_drop");
        let chat = stream_completion(state.clone(), "req_anthropic_drop".to_owned(), "glm-5.2".to_owned(), HashMap::new(), false, receiver, lease).await;
        let response = anthropic::anthropic_stream_response(chat, "glm-5.2".to_owned());
        drop(response);
        wait_cancelled(&state.scheduler, "req_anthropic_drop").await;
        drop(events);
    }

    #[tokio::test]
    async fn non_stream_handler_future_drop_cancels_scheduler_request() {
        let state = test_state();
        state.scheduler.track_test_request("req_json_drop").await;
        let (_events, receiver) = mpsc::channel(1);
        let lease = test_lease(&state, "req_json_drop");
        let task = tokio::spawn(collect_completion(state.clone(), "req_json_drop".to_owned(), "glm-5.2".to_owned(), HashMap::new(), false, receiver, lease));
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;
        wait_cancelled(&state.scheduler, "req_json_drop").await;

        state.scheduler.track_test_request("req_response_json_drop").await;
        let request = responses_request(false);
        let chat_request = response_chat_request(&request, None).unwrap();
        let messages = chat_request.messages.clone();
        let cache_request = serde_json::to_value(chat_request).unwrap();
        let (_events, receiver) = mpsc::channel(1);
        let lease = test_lease(&state, "req_response_json_drop");
        let task = tokio::spawn(collect_response_api(state.clone(), "req_response_json_drop".to_owned(), "resp_response_json_drop".to_owned(), "resp_response_json_drop".to_owned(), None, request, messages, cache_request, receiver, lease));
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;
        wait_cancelled(&state.scheduler, "req_response_json_drop").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_stream_tcp_disconnect_emits_scheduler_cancel() {
        use std::str::FromStr;

        use ::iroh::{Endpoint, endpoint::presets};
        use iroh_tickets::endpoint::EndpointTicket;
        use tokio::io::AsyncWriteExt;

        let service = SchedulerService::bind(None, scheduler::SchedulerConfig { dispatch_wait: Duration::from_secs(1), ..scheduler::SchedulerConfig::default() }).await.unwrap();
        let scheduler = service.scheduler();
        let ticket = EndpointTicket::from_str(service.ticket()).unwrap();
        let node = Endpoint::builder(presets::N0).clear_relay_transports().bind().await.unwrap();
        let connection = node.connect(ticket.endpoint_addr().clone(), SCHEDULER_ALPN).await.unwrap();
        let (mut node_send, node_recv) = connection.open_bi().await.unwrap();
        scheduler::write_json_line(
            &mut node_send,
            &scheduler::NodeMessage::Register {
                protocol_version: scheduler::SCHEDULER_PROTOCOL_VERSION,
                api_key: None,
                model: "glm-5.2".to_owned(),
                max_concurrency: 1,
                caches: Vec::new(),
                capabilities: scheduler::NodeCapabilities::default(),
                runtime: scheduler::NodeRuntime::default(),
            },
        )
        .await
        .unwrap();
        let mut node_recv = tokio::io::BufReader::new(node_recv);
        assert!(matches!(scheduler::read_json_line::<_, scheduler::SchedulerMessage>(&mut node_recv).await.unwrap(), scheduler::SchedulerMessage::Registered { .. }));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_scheduler(ServerConfig { models: vec!["glm-5.2".to_owned()], ..ServerConfig::default() }, scheduler.clone(), None, None);
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        let body = r#"{"model":"glm-5.2","messages":[{"role":"user","content":"long prefill"}],"stream":false}"#;
        let request = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
        socket.write_all(request.as_bytes()).await.unwrap();

        let request_id = match tokio::time::timeout(Duration::from_secs(2), scheduler::read_json_line::<_, scheduler::SchedulerMessage>(&mut node_recv)).await.unwrap().unwrap() {
            scheduler::SchedulerMessage::NewPrefill { request_id, .. } => request_id,
            message => panic!("非流 HTTP 请求应下发 NewPrefill，实际 {message:?}"),
        };
        scheduler::write_json_line(&mut node_send, &scheduler::NodeMessage::Event { request_id: request_id.clone(), event: InferenceEvent::Started }).await.unwrap();
        socket.shutdown().await.unwrap();
        drop(socket);

        let cancel = tokio::time::timeout(Duration::from_secs(2), scheduler::read_json_line::<_, scheduler::SchedulerMessage>(&mut node_recv)).await.expect("真实 TCP 断连应 drop 非流 handler lease").unwrap();
        assert!(matches!(cancel, scheduler::SchedulerMessage::Cancel { request_id: ref cancelled } if cancelled == &request_id));
        assert_eq!(scheduler.nodes().await[0].active_requests, 1, "Cancel 后仍等待 node terminal");
        scheduler::write_json_line(&mut node_send, &scheduler::NodeMessage::Event { request_id, event: InferenceEvent::Completed { finish_reason: "cancelled".to_owned(), prompt_tokens: 1, completion_tokens: 0 } }).await.unwrap();

        connection.close(0u32.into(), b"test complete");
        node.close().await;
        server.abort();
        service.shutdown().await;
    }

    #[test]
    fn responses_background_is_rejected_before_dispatch() {
        let mut request = responses_request(false);
        request.background = Some(true);
        let error = response_chat_request(&request, None).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("background"));
    }

    /// GLM-5.2 节点把"边界后没有新 user/tool 消息"的续接按"继续上一轮"从终点状态
    /// 续写,不再抛 "resume 边界之后没有新消息"。server 侧必须放行 dispatch,
    /// 否则 Codex 的 compaction / function_call-only 轮次会被 400 杀掉。
    #[test]
    fn responses_continuation_without_new_message_is_dispatched() {
        // 上一轮的 continuation,messages 末尾是 assistant
        let previous = ResponseContinuation {
            model: "glm-5.2".to_owned(),
            cache_id: "cache-x".to_owned(),
            instructions: None,
            messages: vec![
                ChatMessage { role: "user".to_owned(), content: Some(Value::String("ping".to_owned())), name: None, reasoning_content: None, tool_call_id: None, tool_calls: None },
                ChatMessage { role: "assistant".to_owned(), content: Some(Value::String("Pong!".to_owned())), name: None, reasoning_content: None, tool_call_id: None, tool_calls: None },
            ],
        };
        // Codex 多轮 context compaction:input 只有 compaction_trigger,新 user/tool 消息没注入
        let mut request = responses_request(false);
        request.input = json!([{"type": "compaction_trigger"}]);
        let chat = response_chat_request(&request, Some(&previous)).unwrap();
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages.last().unwrap().role, "assistant");

        // 同上,只有 function_call item 时注入成 assistant,同样放行
        request.input = json!([{"type": "function_call", "call_id": "call_x", "name": "noop", "arguments": "{}"}]);
        let chat = response_chat_request(&request, Some(&previous)).unwrap();
        assert_eq!(chat.messages.last().unwrap().role, "assistant");

        // 正常续接:新 user 消息注入,上一轮末尾 assistant 之后有新消息
        request.input = json!("continue");
        let chat = response_chat_request(&request, Some(&previous)).unwrap();
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(chat.messages.last().unwrap().role, "user");
    }

    #[test]
    fn reasoning_controls_validate_and_responses_forward_them() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "solve"}],
            "reasoning_effort": "high",
            "thinking_token_budget": 128,
            "repeat_loop_breaker": false
        }))
        .unwrap();
        validate_chat_request(&request, &[]).unwrap();
        assert_eq!(serde_json::to_value(&request).unwrap().get("repeat_loop_breaker"), Some(&Value::Bool(false)), "显式关闭重复 guard 必须透传到节点");

        let mut responses = responses_request(false);
        responses.reasoning = Some(json!({"effort": "max"}));
        responses.thinking_token_budget = Some(-1);
        let forwarded = response_chat_request(&responses, None).unwrap();
        assert_eq!(forwarded.reasoning_effort.as_deref(), Some("max"));
        assert_eq!(forwarded.thinking_token_budget, Some(-1));
        assert_eq!(forwarded.thinking, Some(json!({"type": "enabled"})));

        let plain = response_chat_request(&responses_request(false), None).unwrap();
        assert_eq!(plain.thinking, Some(json!({"type": "disabled"})), "Responses 未显式请求 reasoning 时必须关闭模型默认 thinking");

        let invalid: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "solve"}],
            "reasoning_effort": "ultra"
        }))
        .unwrap();
        assert_eq!(validate_chat_request(&invalid, &[]).unwrap_err().param.as_deref(), Some("reasoning_effort"));

        // 校验放宽:OpenAI 标准四档 low / medium / high / max 全部合法
        for effort in ["low", "medium", "high", "max"] {
            let valid: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "glm-5.2",
                "messages": [{"role": "user", "content": "solve"}],
                "reasoning_effort": effort
            }))
            .unwrap();
            validate_chat_request(&valid, &[]).unwrap();
        }
    }

    #[test]
    fn responses_lite_additional_tools_become_chat_tools() {
        // Codex Responses Lite: additional_tools input 前缀 + developer 指令消息,顶层 tools 为空。
        let mut request = responses_request(true);
        request.input = json!([
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "namespace", "name": "functions", "description": "", "tools": [
                    {"type": "function", "name": "get_weather", "description": "查天气", "parameters": {"type": "object"}}
                ]},
                {"type": "web_search"}
            ]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "始终简短回答"}]},
            {"type": "message", "role": "user", "content": "你好"}
        ]);
        let chat = response_chat_request(&request, None).unwrap();
        let tools = chat.tools.expect("additional_tools 应合并进 tools");
        assert_eq!(tools.len(), 1, "namespace 展开 function,web_search 跳过");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
        assert_eq!(chat.messages.iter().map(|message| message.role.as_str()).collect::<Vec<_>>(), ["developer", "user"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn responses_websocket_streams_events_and_reuses_connection() {
        use ::iroh::{Endpoint, endpoint::presets};
        use futures_util::SinkExt;
        use iroh_tickets::endpoint::EndpointTicket;
        use std::str::FromStr;
        use tokio_tungstenite::tungstenite::Message as ClientMessage;

        let service = SchedulerService::bind(None, scheduler::SchedulerConfig { dispatch_wait: Duration::from_secs(1), ..scheduler::SchedulerConfig::default() }).await.unwrap();
        let scheduler = service.scheduler();
        let ticket = EndpointTicket::from_str(service.ticket()).unwrap();
        let node = Endpoint::builder(presets::N0).clear_relay_transports().bind().await.unwrap();
        let connection = node.connect(ticket.endpoint_addr().clone(), SCHEDULER_ALPN).await.unwrap();
        let (mut node_send, node_recv) = connection.open_bi().await.unwrap();
        scheduler::write_json_line(
            &mut node_send,
            &scheduler::NodeMessage::Register {
                protocol_version: scheduler::SCHEDULER_PROTOCOL_VERSION,
                api_key: None,
                model: "glm-5.2".to_owned(),
                max_concurrency: 1,
                caches: Vec::new(),
                capabilities: scheduler::NodeCapabilities::default(),
                runtime: scheduler::NodeRuntime::default(),
            },
        )
        .await
        .unwrap();
        let mut node_recv = tokio::io::BufReader::new(node_recv);
        assert!(matches!(scheduler::read_json_line::<_, scheduler::SchedulerMessage>(&mut node_recv).await.unwrap(), scheduler::SchedulerMessage::Registered { .. }));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_scheduler(ServerConfig { models: vec!["glm-5.2".to_owned()], ..ServerConfig::default() }, scheduler.clone(), None, None);
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let (mut client, _upgrade) = tokio_tungstenite::connect_async(format!("ws://{address}/v1/responses")).await.unwrap();

        // 第一轮:未知 type 只回 error 帧,会话保持。
        client.send(ClientMessage::Text(r#"{"type":"response.update"}"#.into())).await.unwrap();
        let error = read_ws_json(&mut client).await;
        assert_eq!(error["type"], "error");
        assert_eq!(error["status"], 400);

        // 第二轮:Codex Responses Lite 形态的 response.create。
        let create = json!({
            "type": "response.create",
            "model": "glm-5.2",
            "instructions": "",
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": [
                    {"type": "function", "name": "get_weather", "description": "查天气", "parameters": {"type": "object"}}
                ]},
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "始终简短回答"}]},
                {"type": "message", "role": "user", "content": "你好"}
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "store": false,
            "stream": true,
            "client_metadata": {"session_id": "ws-test"}
        });
        client.send(ClientMessage::Text(create.to_string().into())).await.unwrap();
        let (request_id, request) = match tokio::time::timeout(Duration::from_secs(2), scheduler::read_json_line::<_, scheduler::SchedulerMessage>(&mut node_recv)).await.unwrap().unwrap() {
            scheduler::SchedulerMessage::NewPrefill { request_id, request, .. } => (request_id, request),
            message => panic!("ws response.create 应下发 NewPrefill，实际 {message:?}"),
        };
        assert_eq!(request["tools"][0]["function"]["name"], "get_weather", "additional_tools 应转换进调度请求");
        scheduler::write_json_line(&mut node_send, &scheduler::NodeMessage::Event { request_id: request_id.clone(), event: InferenceEvent::Started }).await.unwrap();
        scheduler::write_json_line(&mut node_send, &scheduler::NodeMessage::Event { request_id: request_id.clone(), event: InferenceEvent::Token { token_id: 0, text: "你好".to_owned() } }).await.unwrap();
        scheduler::write_json_line(&mut node_send, &scheduler::NodeMessage::Event { request_id: request_id.clone(), event: InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 2 } })
            .await
            .unwrap();

        let mut saw_delta = false;
        let mut completed = None;
        for _ in 0..16 {
            let event = read_ws_json(&mut client).await;
            match event["type"].as_str() {
                Some("response.output_text.delta") if event["delta"] == "你好" => saw_delta = true,
                Some("response.completed") => {
                    completed = Some(event);
                    break;
                }
                _ => {}
            }
        }
        assert!(saw_delta, "ws 流应包含 output_text.delta");
        let completed = completed.expect("ws 流应以 response.completed 结束");
        assert_eq!(completed["response"]["usage"]["output_tokens"], 2);

        // 第三轮:同一连接继续下一个 response.create,证明终端事件后未断开。
        client
            .send(ClientMessage::Text(json!({"type": "response.create", "model": "glm-5.2", "instructions": "", "input": [{"type": "message", "role": "user", "content": "继续"}], "store": false, "stream": true}).to_string().into()))
            .await
            .unwrap();
        let request_id = match tokio::time::timeout(Duration::from_secs(2), scheduler::read_json_line::<_, scheduler::SchedulerMessage>(&mut node_recv)).await.unwrap().unwrap() {
            scheduler::SchedulerMessage::NewPrefill { request_id, .. } => request_id,
            message => panic!("连接复用应再次下发 NewPrefill，实际 {message:?}"),
        };
        scheduler::write_json_line(&mut node_send, &scheduler::NodeMessage::Event { request_id: request_id.clone(), event: InferenceEvent::Started }).await.unwrap();
        scheduler::write_json_line(&mut node_send, &scheduler::NodeMessage::Event { request_id, event: InferenceEvent::Completed { finish_reason: "stop".to_owned(), prompt_tokens: 1, completion_tokens: 0 } }).await.unwrap();
        for _ in 0..16 {
            if read_ws_json(&mut client).await["type"] == "response.completed" {
                break;
            }
        }

        client.close(None).await.unwrap();
        connection.close(0u32.into(), b"test complete");
        node.close().await;
        server.abort();
        service.shutdown().await;
    }

    async fn read_ws_json(client: &mut tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>) -> Value {
        tokio::time::timeout(Duration::from_secs(5), client.next())
            .await
            .expect("ws 帧应在 5 秒内到达")
            .expect("ws 流不应提前关闭")
            .expect("ws 读不应出错")
            .into_text()
            .expect("Responses 事件应为文本帧")
            .parse()
            .expect("Responses 事件应为 JSON")
    }

    /// `/v1/nodes` 免鉴权(管理接口,负载均衡视图);即使 api_keys 配了,
    /// 不带 Authorization 也应返回 200 和完整摘要。无节点时返回空数据 + 零聚合值。
    #[tokio::test(flavor = "multi_thread")]
    async fn nodes_endpoint_is_unauthenticated_and_returns_load_summary() {
        // 与生产一致:即便 ServerConfig 配了 api_keys,/v1/nodes 也要 200。
        let mut config = ServerConfig::default();
        config.api_keys = vec!["sk-zllm-test".to_owned()];
        let state = ServerState { config: Arc::new(config), ..test_state() };

        // 0 节点:顶层摘要应全 0,data 数组空。
        let response = nodes(State(state.clone()), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["available_nodes"], 0);
        assert_eq!(value["total_max_load"], 0);
        assert_eq!(value["total_current_load"], 0);
        assert!(value["data"].as_array().unwrap().is_empty());

        // 1 节点:current_load / max_load 字段穿透到 per-node 视图。
        let mut channels = state.scheduler.attach_local_node("n1".to_owned());
        channels
            .messages
            .send(scheduler::NodeMessage::Register {
                protocol_version: scheduler::SCHEDULER_PROTOCOL_VERSION,
                api_key: None,
                model: "glm-5.2".to_owned(),
                max_concurrency: 8,
                caches: Vec::new(),
                capabilities: scheduler::NodeCapabilities::default(),
                runtime: scheduler::NodeRuntime::default(),
            })
            .await
            .unwrap();
        // 等 register 在 scheduler 侧完成,避免 race;同时必须持有 channels 让 spawned task 不退出。
        match tokio::time::timeout(Duration::from_secs(1), channels.commands.recv()).await {
            Ok(Some(scheduler::LocalNodeCommand::Wire(scheduler::SchedulerMessage::Registered { .. }))) => {}
            other => panic!("本地节点 n1 注册未确认: {other:?}"),
        }

        let response = nodes(State(state.clone()), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::OK, "管理接口免 auth,即使 api_keys 已配");
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["available_nodes"], 1);
        assert_eq!(value["total_max_load"], 8);
        assert_eq!(value["total_current_load"], 0);
        let nodes = value["data"].as_array().unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0]["node_id"], "n1");
        assert_eq!(nodes[0]["max_load"], 8);
        assert_eq!(nodes[0]["current_load"], 0);
    }
}
