//! Responses 协议：请求与 Chat 协议的相互转换、SSE/ws 事件推送与会话续接存储。

use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    extract::{
        State,
        rejection::JsonRejection,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use super::chat::{parsed_tool_call, tool_argument_schemas};
use super::scheduler::{self, InferenceEvent, ToolCall};
use super::{
    ApiError, CancelOnDropStream, ChatCompletionRequest, ChatMessage, InferenceCancellation, InferenceLease, MAX_REQUEST_BYTES, ServerState, attach_cache_namespace, authorize, await_started, dispatch_error, request_cache_id,
    resolve_model_alias, scoped_cache_id, sse_utf8, validate_chat_request, with_request_id,
};
use crate::runtime::tool::ToolDialect;
use fjall::{Database, Keyspace, KeyspaceCreateOptions};
use serde::Serialize;

/// 把推理流里的 `<think>...</think>` 拆成 (reasoning, answer) 段;和 chat 路径同款,
/// 这里要求 caller 自行用 `request.reasoning` / `request.thinking_token_budget` 判定
/// `in_reasoning` 初值。
fn split_reasoning_delta<'a>(text: &'a str, in_reasoning: &mut bool) -> (&'a str, &'a str) {
    if !*in_reasoning {
        return ("", text);
    }
    let Some(end) = text.find("</think>") else { return (text, "") };
    *in_reasoning = false;
    (&text[..end], &text[end + "</think>".len()..])
}

fn reasoning_enabled(request: &ResponsesRequest) -> bool {
    request.reasoning.as_ref().and_then(|reasoning| reasoning.get("effort")).is_some_and(|effort| !effort.is_null()) || request.thinking_token_budget.is_some()
}

fn zcode_client(headers: &HeaderMap) -> bool {
    headers.contains_key("x-zcode-app-version") || headers.get("user-agent").and_then(|value| value.to_str().ok()).and_then(|value| value.split('/').next()).is_some_and(|name| name.eq_ignore_ascii_case("zcode"))
}

fn apply_zcode_reasoning_compat(zcode: bool, request: &mut ResponsesRequest) {
    if request.reasoning.is_none() && request.thinking_token_budget.is_none() && request.model.to_ascii_lowercase().starts_with("deepseek-v4") && zcode {
        request.reasoning = Some(json!({"effort": "max"}));
    }
}

/// 单进程内存中保留的续接上限；超过后按插入顺序淘汰最旧会话。
const MAX_RESPONSE_CONTINUATIONS: usize = 1024;

/// Responses 未指定输出上限时允许使用完整 1M 上下文；各 runtime 会减去已有输入 token。
const DEFAULT_MAX_TOTAL_TOKENS: u64 = 1_048_576;

/// Responses `previous_response_id` 的协议元数据；只保存消息与 cache key，不持有模型 KV。
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct ResponseContinuation {
    pub(super) model: String,
    pub(super) messages: Vec<ChatMessage>,
    pub(super) cache_id: String,
    pub(super) instructions: Option<String>,
}

/// 续接落盘载体;Database 必须与 Keyspace 同生命周期。
struct PersistedContinuations {
    _database: Database,
    entries: Keyspace,
}

#[derive(Default)]
pub(super) struct ResponseContinuations {
    pub(super) entries: HashMap<String, ResponseContinuation>,
    pub(super) order: VecDeque<String>,
    /// 调度器重启后按 key 懒恢复续接:旧线程静默续跑,节点 KV 仍在则从 SSD
    /// 恢复,否则按恢复出的消息重新 prefill。None 时退回纯内存。
    persisted: Option<PersistedContinuations>,
}

impl ResponseContinuations {
    /// 打开持久化目录;失败只告警并退回纯内存,不影响服务。
    pub(super) fn open(directory: &std::path::Path) -> Self {
        let open = || -> Result<Option<PersistedContinuations>, String> {
            let database = Database::builder(directory).open().map_err(|error| format!("打开续接持久化目录 {}: {error}", directory.display()))?;
            let entries = database.keyspace("continuations", KeyspaceCreateOptions::default).map_err(|error| format!("创建续接 keyspace 失败: {error}"))?;
            Ok(Some(PersistedContinuations { _database: database, entries }))
        };
        match open() {
            Ok(persisted) => Self { persisted, ..Self::default() },
            Err(error) => {
                eprintln!("[responses] {error},续接退回纯内存(重启丢历史)");
                Self::default()
            }
        }
    }

    pub(super) fn get(&mut self, response_id: &str) -> Option<ResponseContinuation> {
        if let Some(continuation) = self.entries.get(response_id) {
            return Some(continuation.clone());
        }
        let persisted = self.persisted.as_ref()?;
        let bytes = persisted.entries.get(response_id.as_bytes()).ok()??;
        let continuation = serde_json::from_slice::<ResponseContinuation>(&bytes).ok()?;
        // 懒恢复只写内存;不重写磁盘,淘汰与更新仍走常规路径。
        self.entries.insert(response_id.to_owned(), continuation.clone());
        self.order.push_back(response_id.to_owned());
        Some(continuation)
    }

    fn insert(&mut self, response_id: String, continuation: ResponseContinuation) {
        self.entries.remove(&response_id);
        self.order.retain(|existing| existing != &response_id);
        if let Some(persisted) = self.persisted.as_ref()
            && let Ok(value) = serde_json::to_vec(&continuation)
        {
            let _ = persisted.entries.insert(response_id.as_bytes(), &value);
        }
        while self.entries.len() >= MAX_RESPONSE_CONTINUATIONS {
            let Some(oldest) = self.order.pop_front() else { break };
            if let Some(persisted) = self.persisted.as_ref() {
                let _ = persisted.entries.remove(oldest.as_bytes());
            }
            self.entries.remove(&oldest);
        }
        self.order.push_back(response_id.clone());
        self.entries.insert(response_id, continuation);
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ResponsesRequest {
    pub(super) model: String,
    pub(super) input: Value,
    pub(super) instructions: Option<String>,
    #[serde(default)]
    pub(super) stream: bool,
    pub(super) max_output_tokens: Option<u64>,
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) seed: Option<u64>,
    pub(super) reasoning: Option<Value>,
    pub(super) thinking_token_budget: Option<i64>,
    pub(super) tools: Option<Vec<Value>>,
    pub(super) tool_choice: Option<Value>,
    pub(super) parallel_tool_calls: Option<bool>,
    pub(super) previous_response_id: Option<String>,
    pub(super) background: Option<bool>,
    pub(super) store: Option<bool>,
    pub(super) metadata: Option<Value>,
    pub(super) text: Option<Value>,
    pub(super) truncation: Option<String>,
    pub(super) cache_id: Option<String>,
}

pub(super) struct ResponseDispatch {
    pub(super) request: ResponsesRequest,
    pub(super) cache_id: Option<String>,
    pub(super) continuation_messages: Vec<ChatMessage>,
    /// 未注入租户 namespace 的完整 Chat 请求，用于生成终态 cache identity。
    pub(super) cache_request: Value,
    pub(super) request_value: Value,
    pub(super) model: String,
}

pub(super) async fn responses(State(state): State<ServerState>, mut headers: HeaderMap, payload: Result<Json<ResponsesRequest>, JsonRejection>) -> Response {
    let request_id = state.request_id();
    let zcode = zcode_client(&headers);
    let namespace = match authorize(&state.config, &mut headers) {
        Ok(namespace) => namespace,
        Err(error) => return error.into_response(&request_id),
    };
    let mut request = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return ApiError::invalid(format!("请求 JSON 无效: {}", error.body_text()), "body").into_response(&request_id);
        }
    };
    apply_zcode_reasoning_compat(zcode, &mut request);
    let ResponseDispatch { request, cache_id, continuation_messages, cache_request, request_value, model } = match prepare_response_dispatch(&state, namespace.as_deref(), request) {
        Ok(dispatch) => dispatch,
        Err(error) => return error.into_response(&request_id),
    };
    let mut lease = InferenceLease::new(state.scheduler.clone(), request_id.clone());
    let events = match state.scheduler.dispatch_wait(request_id.clone(), model, cache_id.as_deref(), request_value).await {
        Ok(events) => events,
        Err(error) => {
            lease.disarm();
            return dispatch_error(error).into_response(&request_id);
        }
    };
    let response_id = request_id.replacen("req_", "resp_", 1);
    let continuation_id = scoped_cache_id(namespace.as_deref(), &response_id);
    if request.stream {
        stream_response_api(state, request_id, response_id, continuation_id, namespace, request, continuation_messages, cache_request, events, lease).await
    } else {
        collect_response_api(state, request_id, response_id, continuation_id, namespace, request, continuation_messages, cache_request, events, lease).await
    }
}

/// Codex Responses-over-WebSocket 传输（`ws://…/v1/responses`）：单连接串行复用，
/// 每轮一条 `response.create` 文本帧，服务端每个事件一帧回传，终端事件
/// （completed/incomplete/failed）后连接保持打开等待下一轮。
pub(super) async fn responses_ws(State(state): State<ServerState>, mut headers: HeaderMap, upgrade: WebSocketUpgrade) -> Response {
    let request_id = state.request_id();
    let zcode = zcode_client(&headers);
    let namespace = match authorize(&state.config, &mut headers) {
        Ok(namespace) => namespace,
        Err(error) => return error.into_response(&request_id),
    };
    // 长 context 的单帧请求可达数十 MB，帧上限与 HTTP body 上限对齐。
    upgrade.max_message_size(MAX_REQUEST_BYTES).on_upgrade(move |socket| async move { responses_ws_session(state, namespace, zcode, socket).await })
}

async fn responses_ws_session(state: ServerState, namespace: Option<String>, zcode: bool, mut socket: WebSocket) {
    while let Some(Ok(message)) = socket.recv().await {
        match message {
            Message::Text(text) => {
                if !run_ws_response(&state, namespace.as_deref(), zcode, &text, &mut socket).await {
                    break;
                }
            }
            Message::Ping(payload) => {
                if socket.send(Message::Pong(payload)).await.is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

/// 处理一轮 `response.create`。返回 false 表示连接已不可用，会话结束；
/// 请求级错误按 Codex ws 协议发 `{type:"error", status, error}` 帧后继续等下一轮。
async fn run_ws_response(state: &ServerState, namespace: Option<&str>, zcode: bool, text: &str, socket: &mut WebSocket) -> bool {
    let request_id = state.request_id();
    let value = match serde_json::from_str::<Value>(text) {
        Ok(value) => value,
        Err(error) => return send_ws_error(socket, &ApiError::invalid(format!("请求 JSON 无效: {error}"), "body")).await,
    };
    if value.get("type").and_then(Value::as_str) != Some("response.create") {
        return send_ws_error(socket, &ApiError::invalid("ws 请求只支持 response.create", "type")).await;
    }
    // ws 请求与 HTTP POST 同构，多出的 client_metadata/include 等字段由反序列化忽略。
    let mut request = match serde_json::from_value::<ResponsesRequest>(value) {
        Ok(request) => request,
        Err(error) => return send_ws_error(socket, &ApiError::invalid(format!("请求 JSON 无效: {error}"), "body")).await,
    };
    apply_zcode_reasoning_compat(zcode, &mut request);
    // 纯准备阶段返回全量拥有值，continuation 锁不会跨越后面的 await。
    let ResponseDispatch { request, cache_id, continuation_messages, cache_request, request_value, model } = match prepare_response_dispatch(state, namespace, request) {
        Ok(dispatch) => dispatch,
        Err(error) => return send_ws_error(socket, &error).await,
    };
    let mut lease = InferenceLease::new(state.scheduler.clone(), request_id.clone());
    let mut events = match state.scheduler.dispatch_wait(request_id.clone(), model, cache_id.as_deref(), request_value).await {
        Ok(events) => events,
        Err(error) => {
            lease.disarm();
            return send_ws_error(socket, &dispatch_error(error)).await;
        }
    };
    if let Err(error) = await_started(&mut events).await {
        lease.cancel().await;
        return send_ws_error(socket, &error).await;
    }
    let response_id = request_id.replacen("req_", "resp_", 1);
    let continuation_id = scoped_cache_id(namespace, &response_id);
    let cancellation = lease.into_cancellation();
    let mut events_rx = spawn_response_producer(state.response_continuations.clone(), request_id, response_id, continuation_id, namespace.map(str::to_owned), request, continuation_messages, cache_request, events, cancellation.clone());
    // 心跳兼作断连探测：长 prefill 期间没有事件，必须周期性写 socket，
    // 否则客户端 idle 超时（Codex 默认 300s）会先于 400K 档 TTFT 触发。
    let mut keep_alive = tokio::time::interval(Duration::from_secs(10));
    keep_alive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            event = events_rx.recv() => {
                let Some(event) = event else { return true };
                if socket.send(Message::Text(event.data.into())).await.is_err() {
                    cancellation.cancel_detached();
                    return false;
                }
            }
            _ = keep_alive.tick() => {
                if socket.send(Message::Text(json!({"type": "keep-alive"}).to_string().into())).await.is_err() {
                    cancellation.cancel_detached();
                    return false;
                }
            }
        }
    }
}

/// Codex ws 客户端把该帧映射回等价 HTTP 状态码的 transport 错误。
async fn send_ws_error(socket: &mut WebSocket, error: &ApiError) -> bool {
    let frame = json!({"type": "error", "status": error.status.as_u16(), "error": {"code": error.code, "message": error.message, "param": error.param}});
    socket.send(Message::Text(frame.to_string().into())).await.is_ok()
}

pub(super) fn prepare_response_dispatch(state: &ServerState, namespace: Option<&str>, mut request: ResponsesRequest) -> Result<ResponseDispatch, ApiError> {
    if let Some(cache_id) = request.cache_id.take() {
        request.cache_id = Some(scoped_cache_id(namespace, &cache_id));
    }
    // previous_response_id miss 降级: 找不到 continuation 不返 400,把请求当新请求处理,
    // 让 client (Codex / ZCode) 不被一次 cache evict / 重启 卡住。
    // `previous` 是 `Option<ResponseContinuation>`:None=没传 previous_response_id 或 cache miss。
    let previous = if let Some(previous_response_id) = request.previous_response_id.as_deref() {
        let continuation_id = scoped_cache_id(namespace, previous_response_id);
        let mut continuations =
            state.response_continuations.lock().map_err(|_| ApiError { status: StatusCode::INTERNAL_SERVER_ERROR, message: "response continuation 锁中毒".to_owned(), kind: "server_error", param: None, code: "continuation_error" })?;
        continuations.get(&continuation_id)
    } else {
        None
    };
    let mut chat_request = response_chat_request(&request, previous.as_ref())?;
    // 别名重写为后台真实模型:调度、cache hash 与节点侧一致;回显与续接记录仍用请求别名,
    // 续接一致性检查(request.model)不受影响。
    chat_request.model = resolve_model_alias(&state.config, &request.model).to_owned();
    validate_chat_request(&chat_request, &state.config.models)?;
    if chat_request.cache_id.is_none() {
        chat_request.cache_id = request_cache_id(&chat_request).map(|cache_id| scoped_cache_id(namespace, &cache_id));
    }
    request.cache_id = chat_request.cache_id.clone();
    let cache_id = chat_request.cache_id.clone();
    let continuation_messages = chat_request.messages.clone();
    let cache_request =
        serde_json::to_value(&chat_request).map_err(|error| ApiError { status: StatusCode::INTERNAL_SERVER_ERROR, message: format!("序列化调度请求失败: {error}"), kind: "server_error", param: None, code: "scheduler_error" })?;
    let mut request_value = cache_request.clone();
    attach_cache_namespace(&mut request_value, namespace);
    let model = chat_request.model.clone();
    Ok(ResponseDispatch { request, cache_id, continuation_messages, cache_request, request_value, model })
}

pub(super) fn response_chat_request(request: &ResponsesRequest, previous: Option<&ResponseContinuation>) -> Result<ChatCompletionRequest, ApiError> {
    if request.background == Some(true) {
        return Err(ApiError::invalid("当前不支持 background responses", "background"));
    }
    if let Some(previous) = previous {
        if previous.model != request.model {
            return Err(ApiError::invalid(format!("previous_response_id 属于模型 {}，不能用于 {}", previous.model, request.model), "previous_response_id"));
        }
        if request.cache_id.as_deref().is_some_and(|cache_id| cache_id != previous.cache_id) {
            return Err(ApiError::invalid("cache_id 与 previous_response_id 指向的 cache 不一致", "cache_id"));
        }
    }
    let instructions = request.instructions.as_ref().filter(|instructions| !instructions.is_empty());
    let mut messages = previous.map(|previous| previous.messages.clone()).unwrap_or_default();
    if previous.is_none() {
        if let Some(instructions) = instructions {
            messages.push(ChatMessage { role: "developer".to_owned(), content: Some(Value::String(instructions.clone())), reasoning_content: None, name: None, tool_call_id: None, tool_calls: None });
        }
    } else if previous.and_then(|previous| previous.instructions.as_ref()) != instructions {
        let old_instructions = previous.and_then(|previous| previous.instructions.as_deref());
        if messages.first().is_some_and(|message| message.role == "developer" && message.content.as_ref().and_then(Value::as_str) == old_instructions) {
            messages.remove(0);
        }
        if let Some(instructions) = instructions {
            messages.insert(0, ChatMessage { role: "developer".to_owned(), content: Some(Value::String(instructions.clone())), reasoning_content: None, name: None, tool_call_id: None, tool_calls: None });
        }
    }
    let mut input_tools = append_response_input(&mut messages, &request.input)?;
    let mut response_tools = request.tools.clone().unwrap_or_default();
    response_tools.append(&mut input_tools);
    let tools = if response_tools.is_empty() { None } else { Some(response_tools_to_chat(&response_tools)) };
    let tool_choice = response_tool_choice_to_chat(request.tool_choice.as_ref())?;
    let reasoning_effort = match request.reasoning.as_ref() {
        None | Some(Value::Null) => None,
        Some(Value::Object(reasoning)) => match reasoning.get("effort") {
            None | Some(Value::Null) => None,
            // 接受 OpenAI 标准四档:low / medium / high / max,后端把 low/medium 映射成 High
            // (Glm52ReasoningEffort 只有 High/Max 两 tier,GLM-5.2 在 prompt template 里
            // 也只认这俩,加 enum variant 改动大、收益小)。
            Some(Value::String(effort)) if matches!(effort.as_str(), "low" | "medium" | "high" | "max") => Some(effort.clone()),
            Some(_) => return Err(ApiError::invalid("reasoning.effort 必须是 low / medium / high / max", "reasoning.effort")),
        },
        Some(_) => return Err(ApiError::invalid("reasoning 必须是对象", "reasoning")),
    };
    // 边界后没有新 user/tool 消息的续接(compaction_trigger / function_call 等元数据
    // item)不再 400:glm52 节点会按"继续上一轮"从终点状态续写,session 不再被杀。
    // Responses 的 reasoning 是显式 opt-in。不能依赖模型自己的默认值，否则
    // DeepSeek 会把私有思考生成到普通 output_text，客户端只能当正文显示。
    let thinking_enabled = reasoning_effort.is_some() || request.thinking_token_budget.is_some();
    let thinking = Some(json!({"type": if thinking_enabled { "enabled" } else { "disabled" }}));
    Ok(ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        stream: request.stream,
        temperature: request.temperature,
        top_p: request.top_p,
        seed: request.seed,
        thinking,
        reasoning_effort,
        thinking_token_budget: request.thinking_token_budget,
        enable_thinking: if thinking_enabled { None } else { Some(false) },
        max_tokens: None,
        max_completion_tokens: Some(request.max_output_tokens.unwrap_or(DEFAULT_MAX_TOTAL_TOKENS)),
        n: Some(1),
        stop: None,
        tools,
        tool_choice,
        response_format: request.text.clone(),
        stream_options: None,
        cache_id: previous.map(|previous| previous.cache_id.clone()).or_else(|| request.cache_id.clone()),
        repeat_loop_breaker: None,
        prefill_chunk_size: None,
    })
}

/// 把 Responses `input` item 追加为 chat 消息，并返回 `additional_tools` item 声明的
/// 工具列表（Codex Responses Lite 把工具放 input 前缀而不是顶层 `tools` 字段）。
fn append_response_input(messages: &mut Vec<ChatMessage>, input: &Value) -> Result<Vec<Value>, ApiError> {
    let mut additional_tools = Vec::new();
    match input {
        Value::String(text) if !text.is_empty() => {
            messages.push(ChatMessage { role: "user".to_owned(), content: Some(Value::String(text.clone())), reasoning_content: None, name: None, tool_call_id: None, tool_calls: None });
            Ok(additional_tools)
        }
        Value::Array(items) if !items.is_empty() => {
            for (index, item) in items.iter().enumerate() {
                let object = item.as_object().ok_or_else(|| ApiError::invalid("input item 必须是对象", format!("input.{index}")))?;
                match object.get("type").and_then(Value::as_str).unwrap_or("message") {
                    "message" => {
                        let role = object.get("role").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("message item 缺少 role", format!("input.{index}.role")))?;
                        let content = response_content_to_chat(object.get("content").ok_or_else(|| ApiError::invalid("message item 缺少 content", format!("input.{index}.content")))?, index)?;
                        messages.push(ChatMessage { role: role.to_owned(), content: Some(content), reasoning_content: None, name: None, tool_call_id: None, tool_calls: None });
                    }
                    "additional_tools" => {
                        if let Some(tools) = object.get("tools").and_then(Value::as_array) {
                            additional_tools.extend(tools.iter().cloned());
                        }
                    }
                    "function_call" => {
                        let call_id = object.get("call_id").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("function_call 缺少 call_id", format!("input.{index}.call_id")))?;
                        let name = object.get("name").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("function_call 缺少 name", format!("input.{index}.name")))?;
                        let arguments = object.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        messages.push(ChatMessage {
                            role: "assistant".to_owned(),
                            content: None,
                            reasoning_content: None,
                            name: None,
                            tool_call_id: None,
                            tool_calls: Some(json!([{
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments}
                            }])),
                        });
                    }
                    "function_call_output" => {
                        let call_id = object.get("call_id").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("function_call_output 缺少 call_id", format!("input.{index}.call_id")))?;
                        let output = match object.get("output") {
                            Some(Value::String(output)) => output.clone(),
                            Some(output) => output.to_string(),
                            None => return Err(ApiError::invalid("function_call_output 缺少 output", format!("input.{index}.output"))),
                        };
                        messages.push(ChatMessage { role: "tool".to_owned(), content: Some(Value::String(output)), reasoning_content: None, name: None, tool_call_id: Some(call_id.to_owned()), tool_calls: None });
                    }
                    // Codex 上下文压缩触发信号，不影响对话内容，直接跳过。
                    "compaction_trigger" => {}
                    kind => {
                        return Err(ApiError::invalid(format!("当前不支持 input item type '{kind}'"), format!("input.{index}.type")));
                    }
                }
            }
            Ok(additional_tools)
        }
        _ => Err(ApiError::invalid("input 必须是非空字符串或非空 item 数组", "input")),
    }
}

fn response_content_to_chat(content: &Value, item_index: usize) -> Result<Value, ApiError> {
    match content {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(parts) if !parts.is_empty() => {
            let mut text = String::new();
            let mut images = Vec::<Value>::new();
            for (part_index, part) in parts.iter().enumerate() {
                let kind = part.get("type").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("content part 缺少 type", format!("input.{item_index}.content.{part_index}.type")))?;
                match kind {
                    "input_text" | "output_text" | "text" => {
                        let part_text = part.get("text").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("文本 content part 缺少 text", format!("input.{item_index}.content.{part_index}.text")))?;
                        text.push_str(part_text);
                    }
                    // 多模态图片输入：image_url 兼容字符串与 {"url": ...} 两种写法，
                    // 转成 chat 协议的 image_url parts（data URI），供视觉塔消费。
                    "input_image" => {
                        let url = match part.get("image_url") {
                            Some(Value::String(url)) => url.clone(),
                            Some(url) => url.get("url").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("input_image 缺少 image_url.url", format!("input.{item_index}.content.{part_index}.image_url")))?.to_owned(),
                            None => return Err(ApiError::invalid("input_image 缺少 image_url", format!("input.{item_index}.content.{part_index}.image_url"))),
                        };
                        images.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    }
                    other => {
                        return Err(ApiError::invalid(format!("当前 Responses 接口不支持 content part '{other}'"), format!("input.{item_index}.content.{part_index}.type")));
                    }
                }
            }
            if images.is_empty() {
                return Ok(Value::String(text));
            }
            let mut chat_parts = Vec::<Value>::new();
            if !text.is_empty() {
                chat_parts.push(json!({"type": "text", "text": text}));
            }
            chat_parts.extend(images);
            Ok(Value::Array(chat_parts))
        }
        _ => Err(ApiError::invalid("message content 必须是字符串或非空文本数组", format!("input.{item_index}.content"))),
    }
}

fn response_tools_to_chat(tools: &[Value]) -> Vec<Value> {
    let mut converted = Vec::new();
    for tool in tools {
        let Some(object) = tool.as_object() else { continue };
        match object.get("type").and_then(Value::as_str) {
            Some("function") => {
                let mut function = object.clone();
                function.remove("type");
                converted.push(json!({"type": "function", "function": function}));
            }
            // namespace 是 Responses Lite 的工具分组,内层才是可执行工具,递归展开一层。
            Some("namespace") => {
                if let Some(inner) = object.get("tools").and_then(Value::as_array) {
                    converted.extend(response_tools_to_chat(inner));
                }
            }
            // 跳过 web_search / code_interpreter / custom 等后端无法执行的内置 tool。
            _ => {}
        }
    }
    converted
}

fn response_tool_choice_to_chat(choice: Option<&Value>) -> Result<Option<Value>, ApiError> {
    match choice {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(choice)) => Ok(Some(Value::String(choice.clone()))),
        Some(Value::Object(object)) if object.get("type").and_then(Value::as_str) == Some("function") => {
            let name = object.get("name").and_then(Value::as_str).ok_or_else(|| ApiError::invalid("function tool_choice 缺少 name", "tool_choice.name"))?;
            Ok(Some(json!({"type": "function", "function": {"name": name}})))
        }
        _ => Err(ApiError::invalid("tool_choice 格式无效", "tool_choice")),
    }
}

fn save_response_continuation(
    store: &Arc<Mutex<ResponseContinuations>>,
    response_id: &str,
    fallback_cache_id: Option<&str>,
    cache_namespace: Option<&str>,
    model: &str,
    cache_request: &Value,
    instructions: Option<&str>,
    mut messages: Vec<ChatMessage>,
    text: &str,
    tool_calls: &[ToolCall],
) {
    let content = if text.is_empty() && !tool_calls.is_empty() { None } else { Some(Value::String(text.to_owned())) };
    messages.push(ChatMessage { role: "assistant".to_owned(), content, reasoning_content: None, name: None, tool_call_id: None, tool_calls: (!tool_calls.is_empty()).then(|| json!(tool_calls)) });
    let cache_id = if let Some(cache_id) = fallback_cache_id {
        cache_id.to_owned()
    } else {
        let Ok(cache_id) = crate::runtime::session::terminal_cache_id(cache_request, text, tool_calls) else {
            return;
        };
        scoped_cache_id(cache_namespace, &cache_id)
    };
    if let Ok(mut store) = store.lock() {
        store.insert(response_id.to_owned(), ResponseContinuation { model: model.to_owned(), messages, cache_id, instructions: instructions.filter(|instructions| !instructions.is_empty()).map(str::to_owned) });
    }
}

pub(super) async fn stream_response_api(
    state: ServerState,
    request_id: String,
    response_id: String,
    continuation_id: String,
    cache_namespace: Option<String>,
    request: ResponsesRequest,
    continuation_messages: Vec<ChatMessage>,
    cache_request: Value,
    mut events: mpsc::Receiver<InferenceEvent>,
    lease: InferenceLease,
) -> Response {
    if let Err(error) = await_started(&mut events).await {
        lease.cancel().await;
        return error.into_response(&request_id);
    }
    let cancellation = lease.into_cancellation();
    let rx = spawn_response_producer(state.response_continuations.clone(), request_id.clone(), response_id, continuation_id, cache_namespace, request, continuation_messages, cache_request, events, cancellation.clone());
    let stream = CancelOnDropStream::new(ReceiverStream::new(rx).map(|event| Ok::<_, Infallible>(Event::default().event(event.kind).data(event.data))), cancellation);
    sse_utf8(with_request_id(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)).text("keep-alive")).into_response(), &request_id))
}

/// Responses 协议事件：kind 供 SSE `event:` 行使用，data 是完整事件 JSON，ws 传输直接作为文本帧。
struct ResponseApiEvent {
    kind: String,
    data: String,
}

struct ResponseEventSender {
    sender: mpsc::Sender<ResponseApiEvent>,
    sequence_number: u64,
}

impl ResponseEventSender {
    fn new(sender: mpsc::Sender<ResponseApiEvent>) -> Self {
        Self { sender, sequence_number: 0 }
    }

    async fn send(&mut self, kind: &str, mut payload: Value) -> bool {
        let Some(object) = payload.as_object_mut() else {
            return false;
        };
        object.insert("sequence_number".to_owned(), self.sequence_number.into());
        self.sequence_number += 1;
        self.sender.send(ResponseApiEvent { kind: kind.to_owned(), data: super::sse_json(&payload) }).await.is_ok()
    }
}

/// 把推理事件解码为 Responses 协议事件，SSE 与 ws 两种传输共用。终端事件
/// （completed/incomplete/failed）后通道关闭，消费者收到 None 表示本轮结束。
fn spawn_response_producer(
    continuations: Arc<Mutex<ResponseContinuations>>,
    request_id: String,
    response_id: String,
    continuation_id: String,
    cache_namespace: Option<String>,
    request: ResponsesRequest,
    continuation_messages: Vec<ChatMessage>,
    cache_request: Value,
    mut events: mpsc::Receiver<InferenceEvent>,
    cancellation: InferenceCancellation,
) -> mpsc::Receiver<ResponseApiEvent> {
    let (tx, rx) = mpsc::channel(32);
    let failed_id = request_id;
    let producer_cancellation = cancellation;
    tokio::spawn(async move {
        let mut tx = ResponseEventSender::new(tx);
        let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let message_id = format!("msg_{}", response_id.trim_start_matches("resp_"));
        let reasoning_id = format!("rs_{}", response_id.trim_start_matches("resp_"));
        let tool_schemas = tool_argument_schemas(&response_tools_to_chat(request.tools.as_deref().unwrap_or_default()));
        let mut completed = false;
        let mut output = Vec::<Value>::new();
        let mut text = String::new();
        let mut reasoning_text = String::new();
        // Responses 协议:thinking 必须以独立 `output[type=reasoning]` item 暴露,
        // Codex / ZCode 才渲染 <think> 标签。GLM-5.2 在 reasoning 开启时一定从 <think>
        // 开始,所以和 chat 路径一样,request 显式 opt-in 时进入拆分支。
        let mut in_reasoning = reasoning_enabled(&request);
        let mut tool_calls = Vec::<ToolCall>::new();
        let mut streamed_tools = HashMap::<usize, (usize, ToolCall)>::new();
        let mut message_started = false;
        let mut message_done = false;
        let created_response = response_envelope(&request, &response_id, created, "in_progress", Vec::new(), None, None, None);
        if !send_response_event(
            &mut tx,
            "response.created",
            json!({
                "type": "response.created",
                "response": created_response
            }),
        )
        .await
            || !send_response_event(
                &mut tx,
                "response.in_progress",
                json!({
                    "type": "response.in_progress",
                    "response": response_envelope(
                        &request,
                        &response_id,
                        created,
                        "in_progress",
                        Vec::new(),
                        None,
                        None,
                        None,
                    )
                }),
            )
            .await
        {
            producer_cancellation.cancel().await;
            return;
        }

        'inference: loop {
            let event = match events.recv().await {
                Some(event) => event,
                None => break 'inference,
            };
            match event {
                InferenceEvent::Started => {}
                InferenceEvent::Token { text: delta, .. } => {
                    // 先把 thinking 拆出去: reasoning 部分累积到 reasoning_text,
                    // 留给完成时插入 output[type=reasoning] item; text 部分继续走 message。
                    let (reasoning_delta, text_delta) = split_reasoning_delta(&delta, &mut in_reasoning);
                    reasoning_text.push_str(reasoning_delta);
                    if text_delta.is_empty() {
                        continue;
                    }
                    if !message_started {
                        if !start_response_message(&mut tx, &message_id, output.len()).await {
                            break 'inference;
                        }
                        message_started = true;
                    }
                    text.push_str(text_delta);
                    if !send_response_event(
                        &mut tx,
                        "response.output_text.delta",
                        json!({
                            "type": "response.output_text.delta",
                            "item_id": message_id,
                            "output_index": output.len(),
                            "content_index": 0,
                            "delta": text_delta
                        }),
                    )
                    .await
                    {
                        break 'inference;
                    }
                }
                InferenceEvent::ToolCallDelta { delta } => {
                    if message_started && !message_done {
                        if !finish_response_message(&mut tx, &message_id, output.len(), &text).await {
                            break 'inference;
                        }
                        output.push(response_message_item(&message_id, &text));
                        message_done = true;
                    }
                    if !streamed_tools.contains_key(&delta.index) {
                        let (Some(id), Some(name)) = (delta.id.clone(), delta.name.clone()) else {
                            break 'inference;
                        };
                        let output_index = output.len();
                        let tool_call = ToolCall { id, kind: "function".to_owned(), function: scheduler::ToolFunction { name, arguments: String::new() } };
                        if !start_response_function_call(&mut tx, output_index, &tool_call).await {
                            break 'inference;
                        }
                        let mut pending = response_function_call_item(&tool_call);
                        pending["status"] = Value::String("in_progress".to_owned());
                        output.push(pending);
                        streamed_tools.insert(delta.index, (output_index, tool_call));
                    }
                    let Some((output_index, tool_call)) = streamed_tools.get_mut(&delta.index) else {
                        break 'inference;
                    };
                    if !delta.arguments.is_empty() {
                        tool_call.function.arguments.push_str(&delta.arguments);
                        if !send_response_function_call_delta(&mut tx, *output_index, tool_call, &delta.arguments).await {
                            break 'inference;
                        }
                    }
                }
                InferenceEvent::ToolCall { index, tool_call } => {
                    if let Some((output_index, _)) = streamed_tools.remove(&index) {
                        if !finish_response_function_call(&mut tx, output_index, &tool_call).await {
                            break 'inference;
                        }
                        output[output_index] = response_function_call_item(&tool_call);
                        tool_calls.push(tool_call);
                    } else {
                        if message_started && !message_done {
                            if !finish_response_message(&mut tx, &message_id, output.len(), &text).await {
                                break 'inference;
                            }
                            output.push(response_message_item(&message_id, &text));
                            message_done = true;
                        }
                        let output_index = output.len();
                        if !send_response_function_call(&mut tx, output_index, &tool_call).await {
                            break 'inference;
                        }
                        output.push(response_function_call_item(&tool_call));
                        tool_calls.push(tool_call);
                    }
                }
                InferenceEvent::Completed { finish_reason, prompt_tokens, completion_tokens } => {
                    producer_cancellation.disarm();
                    let (text, parsed_tool_calls) = ToolDialect::Auto.split_output(&text, &tool_schemas);
                    let native_tool_count = tool_calls.len();
                    tool_calls.extend(parsed_tool_calls.into_iter().enumerate().map(|(index, call)| parsed_tool_call(&response_id, index, call)));
                    if !message_started && output.is_empty() {
                        if !start_response_message(&mut tx, &message_id, 0).await || !finish_response_message(&mut tx, &message_id, 0, "").await {
                            break 'inference;
                        }
                        output.push(response_message_item(&message_id, ""));
                    } else if message_started && !message_done {
                        if !finish_response_message(&mut tx, &message_id, output.len(), &text).await {
                            break 'inference;
                        }
                        output.push(response_message_item(&message_id, &text));
                    }
                    for tool_call in &tool_calls[native_tool_count..] {
                        let output_index = output.len();
                        if !send_response_function_call(&mut tx, output_index, tool_call).await {
                            break 'inference;
                        }
                        output.push(response_function_call_item(tool_call));
                    }
                    // 把 thinking 作为独立 `output[type=reasoning]` item 前插:
                    // Codex / ZCode 靠这一项渲染 <think> 标签;若模型没产生 thinking
                    // (或请求没开 reasoning),这里就什么都不做。
                    if !reasoning_text.is_empty() {
                        output.insert(0, response_reasoning_item(&reasoning_id, &reasoning_text));
                    }
                    let (status, incomplete) = response_status(&finish_reason);
                    let usage = response_usage(prompt_tokens, completion_tokens);
                    let kind = if status == "incomplete" { "response.incomplete" } else { "response.completed" };
                    // A 方案: 永远存 server-side continuation,与 OpenAI `store` 字段解耦
                    // (zLLM 没有 OpenAI dashboard 概念,`store` 不再控制 continuation 存储)
                    let fallback_cache_id = if finish_reason == "cancelled" { request.cache_id.as_deref() } else { None };
                    save_response_continuation(
                        &continuations,
                        &continuation_id,
                        fallback_cache_id,
                        cache_namespace.as_deref(),
                        &request.model,
                        &cache_request,
                        request.instructions.as_deref(),
                        continuation_messages.clone(),
                        &text,
                        &tool_calls,
                    );
                    let response = response_envelope(&request, &response_id, created, status, output, Some(usage), None, incomplete);
                    let _ = send_response_event(&mut tx, kind, json!({"type": kind, "response": response})).await;
                    completed = true;
                    break;
                }
                InferenceEvent::Error { message } => {
                    producer_cancellation.disarm();
                    // Responses 客户端对 response.failed.error 的透传并不一致；
                    // 先发标准 error 事件，避免调用方把真实推理错误误报为空响应。
                    let stream_error = json!({"type": "error", "sequence_number": 0, "error": {"type": "server_error", "code": "inference_error", "message": message, "param": null}});
                    let _ = send_response_event(&mut tx, "error", stream_error).await;
                    let error = json!({"code": "inference_error", "message": message});
                    let response = response_envelope(&request, &response_id, created, "failed", output, None, Some(error), None);
                    let _ = send_response_event(&mut tx, "response.failed", json!({"type": "response.failed", "response": response})).await;
                    completed = true;
                    break;
                }
            }
        }
        if !completed {
            // 同 chat 路径:异常结束必须带 terminal 事件,Responses 客户端靠
            // response.failed/response.completed 判定完成。
            let message = "inference stream closed without completion";
            let stream_error = json!({"type": "error", "sequence_number": 0, "error": {"type": "server_error", "code": "server_error", "message": message, "param": null}});
            let _ = send_response_event(&mut tx, "error", stream_error).await;
            let _ = send_response_event(&mut tx, "response.failed", json!({"type": "response.failed", "response": {"id": failed_id, "error": {"code": "server_error", "message": message}}})).await;
            producer_cancellation.cancel().await;
        }
    });
    rx
}

pub(super) async fn collect_response_api(
    state: ServerState,
    request_id: String,
    response_id: String,
    continuation_id: String,
    cache_namespace: Option<String>,
    request: ResponsesRequest,
    continuation_messages: Vec<ChatMessage>,
    cache_request: Value,
    mut events: mpsc::Receiver<InferenceEvent>,
    mut lease: InferenceLease,
) -> Response {
    if let Err(error) = await_started(&mut events).await {
        lease.cancel().await;
        return error.into_response(&request_id);
    }
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let message_id = format!("msg_{}", response_id.trim_start_matches("resp_"));
    let reasoning_id = format!("rs_{}", response_id.trim_start_matches("resp_"));
    let mut text = String::new();
    let mut reasoning_text = String::new();
    // 非流式: 与流式保持一致,request 显式 opt-in reasoning 时才拆 <think>。
    let mut in_reasoning = reasoning_enabled(&request);
    let mut tool_calls = Vec::<ToolCall>::new();
    while let Some(event) = events.recv().await {
        match event {
            InferenceEvent::Token { text: delta, .. } => {
                let (reasoning_delta, text_delta) = split_reasoning_delta(&delta, &mut in_reasoning);
                reasoning_text.push_str(reasoning_delta);
                text.push_str(text_delta);
            }
            InferenceEvent::ToolCallDelta { .. } => {}
            InferenceEvent::ToolCall { tool_call, .. } => tool_calls.push(tool_call),
            InferenceEvent::Completed { finish_reason, prompt_tokens, completion_tokens } => {
                lease.disarm();
                let tool_schemas = tool_argument_schemas(&response_tools_to_chat(request.tools.as_deref().unwrap_or_default()));
                let (text, parsed_tool_calls) = ToolDialect::Auto.split_output(&text, &tool_schemas);
                tool_calls.extend(parsed_tool_calls.into_iter().enumerate().map(|(index, call)| parsed_tool_call(&request_id, index, call)));
                let mut output = Vec::new();
                // thinking 独立成 `output[type=reasoning]` item,Codex / ZCode 渲染 <think> 标签靠它
                if !reasoning_text.is_empty() {
                    output.push(response_reasoning_item(&reasoning_id, &reasoning_text));
                }
                if !text.is_empty() || tool_calls.is_empty() {
                    output.push(response_message_item(&message_id, &text));
                }
                output.extend(tool_calls.iter().map(response_function_call_item));
                let (status, incomplete) = response_status(&finish_reason);
                // A 方案: 永远存 server-side continuation(同 ws 路径)
                let fallback_cache_id = if finish_reason == "cancelled" { request.cache_id.as_deref() } else { None };
                save_response_continuation(
                    &state.response_continuations,
                    &continuation_id,
                    fallback_cache_id,
                    cache_namespace.as_deref(),
                    &request.model,
                    &cache_request,
                    request.instructions.as_deref(),
                    continuation_messages,
                    &text,
                    &tool_calls,
                );
                let response = response_envelope(&request, &response_id, created, status, output, Some(response_usage(prompt_tokens, completion_tokens)), None, incomplete);
                return with_request_id(Json(response).into_response(), &request_id);
            }
            InferenceEvent::Error { message } => {
                lease.disarm();
                return ApiError { status: StatusCode::BAD_GATEWAY, message, kind: "server_error", param: None, code: "inference_error" }.into_response(&request_id);
            }
            InferenceEvent::Started => {}
        }
    }
    lease.cancel().await;
    ApiError { status: StatusCode::BAD_GATEWAY, message: "节点在完成 response 前关闭了事件流".to_owned(), kind: "server_error", param: None, code: "node_disconnected" }.into_response(&request_id)
}

async fn send_response_event(sender: &mut ResponseEventSender, kind: &str, payload: Value) -> bool {
    sender.send(kind, payload).await
}

async fn start_response_message(sender: &mut ResponseEventSender, message_id: &str, output_index: usize) -> bool {
    send_response_event(
        sender,
        "response.output_item.added",
        json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {"id": message_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
        }),
    )
    .await
        && send_response_event(
            sender,
            "response.content_part.added",
            json!({
                "type": "response.content_part.added",
                "item_id": message_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}
            }),
        )
        .await
}

async fn finish_response_message(sender: &mut ResponseEventSender, message_id: &str, output_index: usize, text: &str) -> bool {
    let item = response_message_item(message_id, text);
    send_response_event(
        sender,
        "response.output_text.done",
        json!({
            "type": "response.output_text.done",
            "item_id": message_id,
            "output_index": output_index,
            "content_index": 0,
            "text": text
        }),
    )
    .await
        && send_response_event(
            sender,
            "response.content_part.done",
            json!({
                "type": "response.content_part.done",
                "item_id": message_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []}
            }),
        )
        .await
        && send_response_event(
            sender,
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item
            }),
        )
        .await
}

async fn send_response_function_call(sender: &mut ResponseEventSender, output_index: usize, tool_call: &ToolCall) -> bool {
    start_response_function_call(sender, output_index, tool_call).await
        && send_response_function_call_delta(sender, output_index, tool_call, &tool_call.function.arguments).await
        && finish_response_function_call(sender, output_index, tool_call).await
}

async fn start_response_function_call(sender: &mut ResponseEventSender, output_index: usize, tool_call: &ToolCall) -> bool {
    let mut pending = response_function_call_item(tool_call);
    pending["status"] = Value::String("in_progress".to_owned());
    pending["arguments"] = Value::String(String::new());
    send_response_event(
        sender,
        "response.output_item.added",
        json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": pending
        }),
    )
    .await
}

async fn send_response_function_call_delta(sender: &mut ResponseEventSender, output_index: usize, tool_call: &ToolCall, delta: &str) -> bool {
    send_response_event(
        sender,
        "response.function_call_arguments.delta",
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": response_function_item_id(tool_call),
            "output_index": output_index,
            "delta": delta
        }),
    )
    .await
}

async fn finish_response_function_call(sender: &mut ResponseEventSender, output_index: usize, tool_call: &ToolCall) -> bool {
    send_response_event(
        sender,
        "response.function_call_arguments.done",
        json!({
            "type": "response.function_call_arguments.done",
            "item_id": response_function_item_id(tool_call),
            "output_index": output_index,
            "arguments": tool_call.function.arguments
        }),
    )
    .await
        && send_response_event(
            sender,
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": response_function_call_item(tool_call)
            }),
        )
        .await
}

fn response_message_item(message_id: &str, text: &str) -> Value {
    json!({
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}]
    })
}

/// Responses 协议标准 `output[type=reasoning]` item；Codex / ZCode 等客户端靠这一项
/// 渲染 <think> 标签。`summary[].text` 沿用 OpenAI Responses 的结构。
fn response_reasoning_item(reasoning_id: &str, text: &str) -> Value {
    json!({
        "id": reasoning_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}]
    })
}

fn response_function_item_id(tool_call: &ToolCall) -> String {
    format!("fc_{}", tool_call.id.trim_start_matches("call_"))
}

fn response_function_call_item(tool_call: &ToolCall) -> Value {
    json!({
        "id": response_function_item_id(tool_call),
        "type": "function_call",
        "status": "completed",
        "call_id": tool_call.id,
        "name": tool_call.function.name,
        "arguments": tool_call.function.arguments
    })
}

fn response_status(finish_reason: &str) -> (&'static str, Option<Value>) {
    if finish_reason == "length" { ("incomplete", Some(json!({"reason": "max_output_tokens"}))) } else { ("completed", None) }
}

fn response_usage(input_tokens: usize, output_tokens: usize) -> Value {
    json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens
    })
}

#[allow(clippy::too_many_arguments)]
fn response_envelope(request: &ResponsesRequest, response_id: &str, created: u64, status: &str, output: Vec<Value>, usage: Option<Value>, error: Option<Value>, incomplete_details: Option<Value>) -> Value {
    json!({
        "id": response_id,
        "object": "response",
        "created_at": created,
        "completed_at": if matches!(status, "completed" | "incomplete" | "failed") { Some(created) } else { None },
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "model": request.model,
        "output": output,
        "parallel_tool_calls": request.parallel_tool_calls.unwrap_or(true),
        "previous_response_id": request.previous_response_id,
        "reasoning": request.reasoning.clone().unwrap_or_else(|| json!({"effort": null, "summary": null})),
        "store": request.store.unwrap_or(true),
        "temperature": request.temperature.unwrap_or(1.0),
        "text": request.text.clone().unwrap_or_else(|| json!({"format": {"type": "text"}})),
        "tool_choice": request.tool_choice.clone().unwrap_or_else(|| Value::String("auto".to_owned())),
        "tools": request.tools.clone().unwrap_or_default(),
        "top_p": request.top_p.unwrap_or(1.0),
        "truncation": request.truncation.clone().unwrap_or_else(|| "disabled".to_owned()),
        "usage": usage,
        "metadata": request.metadata.clone().unwrap_or_else(|| json!({}))
    })
}

#[cfg(test)]
pub(super) fn responses_request(stream: bool) -> ResponsesRequest {
    ResponsesRequest {
        model: "glm-5.2".to_owned(),
        input: Value::String("你好".to_owned()),
        instructions: None,
        stream,
        max_output_tokens: Some(8),
        temperature: None,
        top_p: None,
        seed: None,
        reasoning: None,
        thinking_token_budget: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        previous_response_id: None,
        background: None,
        store: None,
        metadata: None,
        text: None,
        truncation: None,
        cache_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 续接持久化重启后静默恢复() {
        let root = std::env::temp_dir().join(format!("zllm-continuations-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let message = serde_json::from_value::<ChatMessage>(json!({"role": "user", "content": "历史消息", "reasoning_content": null, "name": null, "tool_call_id": null, "tool_calls": null})).unwrap();
        let continuation = ResponseContinuation { model: "deepseek-v4-flash".to_owned(), messages: vec![message], cache_id: "cache-1".to_owned(), instructions: Some("指令".to_owned()) };
        {
            let mut store = ResponseContinuations::open(&root);
            store.insert("resp-1".to_owned(), continuation);
        }
        // 模拟调度器重启:重新打开同目录,内存为空,按 key 从磁盘懒恢复。
        let mut recovered = ResponseContinuations::open(&root);
        let got = recovered.get("resp-1").expect("重启后应能恢复续接");
        assert_eq!(got.model, "deepseek-v4-flash");
        assert_eq!(got.cache_id, "cache-1");
        assert_eq!(got.messages.len(), 1);
        assert_eq!(got.instructions.as_deref(), Some("指令"));
        assert!(recovered.entries.contains_key("resp-1"), "恢复后应进入内存缓存");
        assert!(recovered.get("resp-missing").is_none());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn zcode_deepseek缺失reasoning时启用兼容默认值() {
        let mut headers = HeaderMap::new();
        headers.insert("user-agent", "ZCode/3.8.1".parse().unwrap());
        assert!(zcode_client(&headers));

        let mut request = responses_request(true);
        request.model = "DeepSeek-V4-Flash".to_owned();
        apply_zcode_reasoning_compat(true, &mut request);
        assert_eq!(request.reasoning, Some(json!({"effort": "max"})));
        let forwarded = response_chat_request(&request, None).unwrap();
        assert_eq!(forwarded.reasoning_effort.as_deref(), Some("max"));
        assert_eq!(forwarded.thinking, Some(json!({"type": "enabled"})));

        request.reasoning = Some(json!({"effort": "low"}));
        apply_zcode_reasoning_compat(true, &mut request);
        assert_eq!(request.reasoning, Some(json!({"effort": "low"})), "显式参数不能被兼容默认值覆盖");

        let mut other_client = responses_request(true);
        other_client.model = "deepseek-v4-flash".to_owned();
        apply_zcode_reasoning_compat(false, &mut other_client);
        assert!(other_client.reasoning.is_none());

        let mut other_model = responses_request(true);
        apply_zcode_reasoning_compat(true, &mut other_model);
        assert!(other_model.reasoning.is_none());
    }

    #[test]
    fn responses未指定输出上限时使用1m总上下文() {
        let mut request = responses_request(true);
        request.max_output_tokens = None;
        let forwarded = response_chat_request(&request, None).unwrap();
        assert_eq!(forwarded.max_completion_tokens, Some(1_048_576));

        request.max_output_tokens = Some(4096);
        let forwarded = response_chat_request(&request, None).unwrap();
        assert_eq!(forwarded.max_completion_tokens, Some(4096));
    }

    /// 与 chat 路径同款 `split_reasoning_delta` 行为:首次见到 <think> 段时全归 reasoning,
    /// 见 </think> 后切换到 text;跨多个 delta 调用要能正确切。
    #[test]
    fn split_reasoning_delta_keeps_think_then_text() {
        let mut in_reasoning = true;
        let (reasoning, text) = split_reasoning_delta("让我想想", &mut in_reasoning);
        assert_eq!(reasoning, "让我想想");
        assert_eq!(text, "");
        assert!(in_reasoning);

        let (reasoning, text) = split_reasoning_delta("</think>答案是42", &mut in_reasoning);
        assert_eq!(reasoning, "");
        assert_eq!(text, "答案是42");
        assert!(!in_reasoning);

        let (reasoning, text) = split_reasoning_delta("再补一句", &mut in_reasoning);
        assert_eq!(reasoning, "");
        assert_eq!(text, "再补一句");
    }

    /// request 没 opt-in reasoning 时 (in_reasoning=false) 整段都走 text,
    /// 不会误把内容吞进 reasoning_text。
    #[test]
    fn split_reasoning_delta_bypassed_when_request_did_not_opt_in() {
        let mut in_reasoning = false;
        let (reasoning, text) = split_reasoning_delta("<think>内容</think>答案", &mut in_reasoning);
        assert_eq!(reasoning, "");
        assert_eq!(text, "<think>内容</think>答案");
    }

    /// `response_reasoning_item` 必须严格按 OpenAI Responses 标准 shape,
    /// Codex / ZCode 靠 `type == "reasoning"` + `summary[].text` 渲染 <think> 标签。
    #[test]
    fn response_reasoning_item_matches_openai_responses_shape() {
        let item = response_reasoning_item("rs_abc", "链上思考");
        assert_eq!(item["type"], "reasoning");
        assert_eq!(item["id"], "rs_abc");
        let summary = item["summary"].as_array().expect("summary 必须是数组");
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0]["type"], "summary_text");
        assert_eq!(summary[0]["text"], "链上思考");
    }

    #[tokio::test]
    async fn inference_error_is_forwarded_before_response_failed() {
        let (event_tx, event_rx) = mpsc::channel(1);
        let cancellation = InferenceCancellation::new(scheduler::Scheduler::default(), "req_error".to_owned());
        let mut output = spawn_response_producer(
            Arc::new(Mutex::new(ResponseContinuations::default())),
            "req_error".to_owned(),
            "resp_error".to_owned(),
            "resp_error".to_owned(),
            None,
            responses_request(true),
            Vec::new(),
            json!({"model":"test", "messages":[]}),
            event_rx,
            cancellation,
        );
        event_tx.send(InferenceEvent::Error { message: "ROCm pipeline failed".to_owned() }).await.unwrap();
        drop(event_tx);

        let mut events = Vec::new();
        while let Some(event) = output.recv().await {
            events.push((event.kind, serde_json::from_str::<Value>(&event.data).unwrap()));
        }
        assert_eq!(events.iter().map(|(kind, _)| kind.as_str()).collect::<Vec<_>>(), ["response.created", "response.in_progress", "error", "response.failed"]);
        assert_eq!(events[2].1["error"]["message"], "ROCm pipeline failed");
        assert_eq!(events[3].1["response"]["error"]["message"], "ROCm pipeline failed");
    }
}
