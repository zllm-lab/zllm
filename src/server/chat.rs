//! Chat Completions 协议：流式与非流式推理事件的 SSE/JSON 组装，以及 XML
//! 工具调用文本兜底解析的协议侧组装。

use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::scheduler::{self, InferenceEvent, ToolCall};
use super::{
    ApiError, CancelOnDropStream, ChatCompletionRequest, InferenceCancellation, InferenceLease, ServerState, attach_cache_namespace, authorize, await_started, dispatch_error, request_cache_id, resolve_model_alias, scoped_cache_id,
    sse_json, sse_utf8, validate_chat_request, with_request_id,
};
use crate::runtime::tool::{ParsedToolCall, ToolCallStream, ToolDialect, ToolOutput, tool_call_id};

pub(super) async fn chat_completions(State(state): State<ServerState>, mut headers: HeaderMap, payload: Result<Json<ChatCompletionRequest>, JsonRejection>) -> Response {
    let request_id = state.request_id();
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
    if let Err(error) = validate_chat_request(&request, &state.config.models) {
        return error.into_response(&request_id);
    }
    // 别名请求重写为后台真实模型再调度（cache hash 与节点侧一致）；响应仍回显客户端请求的模型名。
    let display_model = request.model.clone();
    request.model = resolve_model_alias(&state.config, &request.model).to_owned();
    if request.cache_id.is_none() {
        request.cache_id = request_cache_id(&request);
    }
    if let Some(cache_id) = request.cache_id.take() {
        request.cache_id = Some(scoped_cache_id(namespace.as_deref(), &cache_id));
    }
    let mut request_value = match serde_json::to_value(&request) {
        Ok(value) => value,
        Err(error) => return ApiError { status: StatusCode::INTERNAL_SERVER_ERROR, message: format!("序列化调度请求失败: {error}"), kind: "server_error", param: None, code: "scheduler_error" }.into_response(&request_id),
    };
    attach_cache_namespace(&mut request_value, namespace.as_deref());
    let dispatch_model = request.model.clone();
    let mut lease = InferenceLease::new(state.scheduler.clone(), request_id.clone());
    let events = match state.scheduler.dispatch_wait(request_id.clone(), dispatch_model, request.cache_id.as_deref(), request_value).await {
        Ok(events) => events,
        Err(error) => {
            lease.disarm();
            return dispatch_error(error).into_response(&request_id);
        }
    };
    let tool_schemas = if request.tool_choice.as_ref().and_then(Value::as_str) == Some("none") { HashMap::new() } else { tool_argument_schemas(request.tools.as_deref().unwrap_or_default()) };
    let reasoning = request.reasoning_effort.is_some() || request.thinking_token_budget.is_some() || request.thinking.as_ref().and_then(Value::as_object).and_then(|value| value.get("type")).and_then(Value::as_str) == Some("enabled");
    if request.stream { stream_completion(state, request_id, display_model, tool_schemas, reasoning, events, lease).await } else { collect_completion(state, request_id, display_model, tool_schemas, reasoning, events, lease).await }
}

pub(super) fn tool_argument_schemas(tools: &[Value]) -> std::collections::HashMap<String, Value> {
    crate::runtime::tool::tool_argument_schemas(tools)
}

/// 文本兜底解析出的工具调用补上请求内稳定、跨请求唯一的 id。
pub(super) fn parsed_tool_call(scope: &str, index: usize, call: ParsedToolCall) -> ToolCall {
    ToolCall { id: tool_call_id(scope, index, &call.name), kind: "function".to_owned(), function: scheduler::ToolFunction { name: call.name, arguments: call.arguments } }
}

use crate::runtime::session::ReasoningStream;

/// 流式 SSE 状态机：直接在 hyper 连接任务上被 poll（stream::unfold），
/// 不再经独立 producer task 转发，每 token 少一跳 mpsc + 一次任务唤醒。
/// 热路径（content/reasoning chunk）用预拼好的头尾串接序列化后的文本，
/// 避免每 token 构建 Value 树。
struct SseState {
    events: mpsc::Receiver<InferenceEvent>,
    pending: std::collections::VecDeque<Result<Event, Infallible>>,
    cancellation: InferenceCancellation,
    content_tail: String,
    reasoning_tail: String,
    request_id: String,
    model: String,
    created: u64,
    closed: bool,
    completed: bool,
    done: bool,
    streamed_tool_calls: HashSet<usize>,
    tool_fallback: Option<ToolCallStream>,
    fallback_tool_index: usize,
    fallback_has_tools: bool,
    in_reasoning: ReasoningStream,
}

/// chunk 前缀与线上输出保持同一 key 序（serde_json Map 字典序）：
/// `{"choices":[{"delta":{"content":<text>,"reasoning_content":null},...`。
const CONTENT_CHUNK_HEAD: &str = "{\"choices\":[{\"delta\":{\"content\":";
const REASONING_CHUNK_HEAD: &str = "{\"choices\":[{\"delta\":{\"content\":null,\"reasoning_content\":";

impl SseState {
    fn new(request_id: String, model: String, created: u64, tool_schemas: HashMap<String, Value>, reasoning: bool, events: mpsc::Receiver<InferenceEvent>, cancellation: InferenceCancellation) -> Self {
        let json_string = |value: &str| serde_json::to_string(value).expect("字符串序列化不会失败");
        let chunk_ids = format!("\"created\":{created},\"id\":{},\"model\":{},\"object\":\"chat.completion.chunk\"}}", json_string(&request_id), json_string(&model));
        let content_tail = format!(",\"reasoning_content\":null}},\"finish_reason\":null,\"index\":0}}],{chunk_ids}");
        let reasoning_tail = format!("}},\"finish_reason\":null,\"index\":0}}],{chunk_ids}");
        let role = json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "", "reasoning_content": null}, "finish_reason": null}]
        });
        Self {
            events,
            pending: std::collections::VecDeque::from([Ok(Event::default().data(sse_json(&role)))]),
            cancellation,
            content_tail,
            reasoning_tail,
            request_id,
            model,
            created,
            closed: false,
            completed: false,
            done: false,
            streamed_tool_calls: HashSet::new(),
            tool_fallback: (!tool_schemas.is_empty()).then(|| ToolDialect::Auto.stream(tool_schemas)),
            fallback_tool_index: 0,
            fallback_has_tools: false,
            in_reasoning: ReasoningStream::new(reasoning),
        }
    }

    fn content_chunk(&self, text: &str) -> String {
        format!("{CONTENT_CHUNK_HEAD}{}{}", serde_json::to_string(text).expect("字符串序列化不会失败"), self.content_tail)
    }

    fn reasoning_chunk(&self, text: &str) -> String {
        format!("{REASONING_CHUNK_HEAD}{}{}", serde_json::to_string(text).expect("字符串序列化不会失败"), self.reasoning_tail)
    }

    async fn next_event(&mut self) -> Option<Result<Event, Infallible>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            if self.done {
                return None;
            }
            if self.closed {
                // 事件通道在无终态事件时关闭(节点掉线等):补发 error 与 [DONE],
                // 否则流式客户端表现为"流莫名结束"且无法区分失败原因。
                if !self.completed {
                    self.completed = true;
                    let error = json!({"error": {"message": "inference stream closed without completion", "type": "server_error", "code": "inference_error"}});
                    self.pending.push_back(Ok(Event::default().event("error").data(sse_json(&error))));
                    self.pending.push_back(Ok(Event::default().data("[DONE]")));
                    self.cancellation.cancel().await;
                }
                self.done = true;
                continue;
            }
            match self.events.recv().await {
                Some(event) => self.handle(event),
                None => self.closed = true,
            }
        }
    }

    fn emit_text(&mut self, reasoning: &str, text: &str) {
                if !reasoning.is_empty() {
                    self.pending.push_back(Ok(Event::default().data(self.reasoning_chunk(reasoning))));
                }
                if text.is_empty() {
                    return;
                }
                let output = match self.tool_fallback.as_mut() {
                    Some(fallback) => fallback.push(text),
                    None => vec![ToolOutput::Text(text.to_owned())],
                };
                for item in output {
                    let chunk = match item {
                        ToolOutput::Text(text) => self.content_chunk(&text),
                        ToolOutput::ToolCall(call) => {
                            while self.streamed_tool_calls.contains(&self.fallback_tool_index) {
                                self.fallback_tool_index += 1;
                            }
                            let index = self.fallback_tool_index;
                            self.fallback_tool_index += 1;
                            self.streamed_tool_calls.insert(index);
                            self.fallback_has_tools = true;
                            let tool_call = parsed_tool_call(&self.request_id, index, call);
                            let chunk = json!({
                                "id": self.request_id,
                                "object": "chat.completion.chunk",
                                "created": self.created,
                                "model": self.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"reasoning_content": null, "tool_calls": [{
                                        "index": index,
                                        "id": tool_call.id,
                                        "type": tool_call.kind,
                                        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                                    }]},
                                    "finish_reason": null
                                }]
                            });
                            sse_json(&chunk)
                        }
                    };
                    self.pending.push_back(Ok(Event::default().data(chunk)));
                }
    }

    fn handle(&mut self, event: InferenceEvent) {
        match event {
            InferenceEvent::Started => {}
            InferenceEvent::Token { text, .. } => {
                let (reasoning, text) = self.in_reasoning.push(&text);
                self.emit_text(&reasoning, &text);
            }
            InferenceEvent::ToolCallDelta { delta } => {
                self.streamed_tool_calls.insert(delta.index);
                let mut tool_delta = json!({
                    "index": delta.index,
                    "function": {"arguments": delta.arguments}
                });
                if let Some(id) = delta.id {
                    tool_delta["id"] = Value::String(id);
                    tool_delta["type"] = Value::String("function".to_owned());
                }
                if let Some(name) = delta.name {
                    tool_delta["function"]["name"] = Value::String(name);
                }
                let chunk = json!({
                    "id": self.request_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"reasoning_content": null, "tool_calls": [tool_delta]},
                        "finish_reason": null
                    }]
                });
                self.pending.push_back(Ok(Event::default().data(sse_json(&chunk))));
            }
            InferenceEvent::ToolCall { index, tool_call } => {
                if self.streamed_tool_calls.contains(&index) {
                    return;
                }
                let chunk = json!({
                    "id": self.request_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "reasoning_content": null,
                            "tool_calls": [{
                                "index": index,
                                "id": tool_call.id,
                                "type": tool_call.kind,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }]
                        },
                        "finish_reason": null
                    }]
                });
                self.pending.push_back(Ok(Event::default().data(sse_json(&chunk))));
            }
            InferenceEvent::Completed { finish_reason, prompt_tokens, completion_tokens } => {
                self.cancellation.disarm();
                let (reasoning, text) = self.in_reasoning.finish();
                self.emit_text(&reasoning, &text);
                if let Some(fallback) = self.tool_fallback.as_mut() {
                    for item in fallback.finish() {
                        let chunk = match item {
                            ToolOutput::Text(text) => self.content_chunk(&text),
                            ToolOutput::ToolCall(call) => {
                                while self.streamed_tool_calls.contains(&self.fallback_tool_index) {
                                    self.fallback_tool_index += 1;
                                }
                                let index = self.fallback_tool_index;
                                self.fallback_tool_index += 1;
                                self.streamed_tool_calls.insert(index);
                                self.fallback_has_tools = true;
                                let tool_call = parsed_tool_call(&self.request_id, index, call);
                                let chunk = json!({
                                    "id": self.request_id,
                                    "object": "chat.completion.chunk",
                                    "created": self.created,
                                    "model": self.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"reasoning_content": null, "tool_calls": [{
                                            "index": index,
                                            "id": tool_call.id,
                                            "type": tool_call.kind,
                                            "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                                        }]},
                                        "finish_reason": null
                                    }]
                                });
                                sse_json(&chunk)
                            }
                        };
                        self.pending.push_back(Ok(Event::default().data(chunk)));
                    }
                }
                let finish_reason = if self.fallback_has_tools { "tool_calls" } else { &finish_reason };
                let chunk = json!({
                    "id": self.request_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": null}, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                });
                self.pending.push_back(Ok(Event::default().data(sse_json(&chunk))));
                self.pending.push_back(Ok(Event::default().data("[DONE]")));
                self.completed = true;
                self.done = true;
            }
            InferenceEvent::Error { message } => {
                self.cancellation.disarm();
                let error = json!({"error": {"message": message, "type": "server_error", "code": "inference_error"}});
                self.pending.push_back(Ok(Event::default().event("error").data(sse_json(&error))));
                self.pending.push_back(Ok(Event::default().data("[DONE]")));
                self.completed = true;
                self.done = true;
            }
        }
    }
}

/// 把 SseState 的异步状态机包成 Stream：state 随 future 移动，避免自引用。
/// 直接在 hyper 连接任务上 poll，不需要独立 producer task。
struct SseStream {
    drive: Option<std::pin::Pin<Box<dyn std::future::Future<Output = (SseState, Option<Result<Event, Infallible>>)> + Send>>>,
}

impl SseStream {
    fn new(state: SseState) -> Self {
        Self { drive: Some(Box::pin(Self::step(state))) }
    }

    async fn step(mut state: SseState) -> (SseState, Option<Result<Event, Infallible>>) {
        let event = state.next_event().await;
        (state, event)
    }
}

impl tokio_stream::Stream for SseStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, context: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        let Some(drive) = self.drive.as_mut() else {
            return std::task::Poll::Ready(None);
        };
        match drive.as_mut().poll(context) {
            std::task::Poll::Pending => std::task::Poll::Pending,
            std::task::Poll::Ready((state, event)) => {
                self.drive = event.as_ref().map(|_| Box::pin(Self::step(state)) as _);
                std::task::Poll::Ready(event)
            }
        }
    }
}

pub(super) async fn stream_completion(_state: ServerState, request_id: String, model: String, tool_schemas: HashMap<String, Value>, reasoning: bool, mut events: mpsc::Receiver<InferenceEvent>, lease: InferenceLease) -> Response {
    if let Err(error) = await_started(&mut events).await {
        lease.cancel().await;
        return error.into_response(&request_id);
    }
    let cancellation = lease.into_cancellation();
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let state = SseState::new(request_id.clone(), model, created, tool_schemas, reasoning, events, cancellation.clone());
    let stream = CancelOnDropStream::new(SseStream::new(state), cancellation.clone());
    let mut response = sse_utf8(with_request_id(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)).text("keep-alive")).into_response(), &request_id));
    // Anthropic adapter 会再包一层 SSE；共享 handle 让最外层 body 也能直接取消。
    response.extensions_mut().insert(cancellation);
    response
}

pub(super) async fn collect_completion(_state: ServerState, request_id: String, model: String, tool_schemas: HashMap<String, Value>, reasoning: bool, mut events: mpsc::Receiver<InferenceEvent>, mut lease: InferenceLease) -> Response {
    if let Err(error) = await_started(&mut events).await {
        lease.cancel().await;
        return error.into_response(&request_id);
    }
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut in_reasoning = ReasoningStream::new(reasoning);
    let mut tool_calls = Vec::<ToolCall>::new();
    while let Some(event) = events.recv().await {
        match event {
            InferenceEvent::Token { text, .. } => {
                let (reasoning, text) = in_reasoning.push(&text);
                reasoning_content.push_str(&reasoning);
                content.push_str(&text);
            }
            InferenceEvent::ToolCallDelta { .. } => {}
            InferenceEvent::ToolCall { tool_call, .. } => tool_calls.push(tool_call),
            InferenceEvent::Completed { finish_reason, prompt_tokens, completion_tokens } => {
                lease.disarm();
                let (reasoning, text) = in_reasoning.finish();
                reasoning_content.push_str(&reasoning);
                content.push_str(&text);
                let mut finish_reason = finish_reason;
                if !tool_schemas.is_empty() {
                    let (text, parsed_tool_calls) = ToolDialect::Auto.split_output(&content, &tool_schemas);
                    content = text;
                    if !parsed_tool_calls.is_empty() {
                        finish_reason = "tool_calls".to_owned();
                        tool_calls.extend(parsed_tool_calls.into_iter().enumerate().map(|(index, call)| parsed_tool_call(&request_id, index, call)));
                    }
                }
                let message_content = if content.is_empty() && !tool_calls.is_empty() { Value::Null } else { Value::String(content) };
                let mut message = json!({"role": "assistant", "content": message_content});
                if !reasoning_content.is_empty() {
                    message["reasoning_content"] = Value::String(reasoning_content);
                }
                if !tool_calls.is_empty() {
                    message["tool_calls"] = json!(tool_calls);
                }
                return with_request_id(
                    Json(json!({
                        "id": request_id,
                        "object": "chat.completion",
                        "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                        "model": model,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    }))
                    .into_response(),
                    &request_id,
                );
            }
            InferenceEvent::Error { message } => {
                lease.disarm();
                return ApiError { status: StatusCode::BAD_GATEWAY, message, kind: "server_error", param: None, code: "inference_error" }.into_response(&request_id);
            }
            InferenceEvent::Started => {}
        }
    }
    lease.cancel().await;
    ApiError { status: StatusCode::BAD_GATEWAY, message: "节点在完成推理前关闭了事件流".to_owned(), kind: "server_error", param: None, code: "node_disconnected" }.into_response(&request_id)
}
