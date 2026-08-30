use super::*;

use axum::extract::rejection::JsonRejection;

#[derive(Clone, Debug, Deserialize)]
pub(super) struct AnthropicMessagesRequest {
    pub(super) model: String,
    pub(super) max_tokens: u64,
    pub(super) messages: Vec<Value>,
    #[serde(default)]
    pub(super) stream: bool,
    pub(super) system: Option<Value>,
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) top_k: Option<u64>,
    pub(super) stop_sequences: Option<Vec<String>>,
    pub(super) tools: Option<Vec<Value>>,
    pub(super) tool_choice: Option<Value>,
    pub(super) metadata: Option<Value>,
    pub(super) cache_id: Option<String>,
}

pub(super) async fn messages(State(state): State<ServerState>, headers: HeaderMap, Json(request): Json<AnthropicMessagesRequest>) -> Response {
    let stream = request.stream;
    let model = request.model.clone();
    let chat_request = match anthropic_to_chat_request(request) {
        Ok(request) => request,
        Err(message) => return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", message),
    };

    // Anthropic SDK 使用 x-api-key；授权改写在 `authorize()` 内统一处理（见 normalize_anthropic_api_key）。
    let response = chat_completions(State(state), headers, Ok(Json(chat_request))).await;
    if !response.status().is_success() {
        return anthropic_from_openai_error(response).await;
    }
    if stream { anthropic_stream_response(response, model) } else { anthropic_json_response(response).await }
}

/// count_tokens 请求与 Messages 同构,但 max_tokens 等输出参数不参与,只保留影响
/// 输入长度的字段;字段宽松缺省,Claude CLI 的计数调用不带完整输出配置。
#[derive(Debug, Deserialize)]
pub(super) struct AnthropicCountTokensRequest {
    #[serde(default)]
    pub(super) messages: Vec<Value>,
    #[serde(default)]
    pub(super) system: Option<Value>,
    #[serde(default)]
    pub(super) tools: Option<Vec<Value>>,
}

/// Claude CLI 每轮调用 /v1/messages/count_tokens 决策自动压缩时机。按内容估算输入
/// token(CJK 约 1 字 1 token,其余约 4 字符 1 token,图片按 base64 体积折算);误差
/// 对压缩阈值判断无害,却能让 CLI 正常推进,不再因接口缺失重试风暴。
pub(super) async fn count_tokens(State(state): State<ServerState>, mut headers: HeaderMap, payload: Result<Json<AnthropicCountTokensRequest>, JsonRejection>) -> Response {
    let request_id = state.request_id();
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("count_tokens 请求 JSON 无效: {}", error.body_text())),
    };
    let mut tokens = 0u64;
    if let Some(system) = request.system.as_ref() {
        tokens += anthropic_content_tokens(system);
    }
    for message in &request.messages {
        tokens += 5; // 每条消息的 role/封装开销
        tokens += anthropic_content_tokens(message.get("content").unwrap_or(&Value::Null));
    }
    for tool in request.tools.iter().flatten() {
        tokens += 10 + anthropic_estimate_tokens(&tool.to_string());
    }
    with_request_id(Json(json!({"input_tokens": tokens})).into_response(), &request_id)
}

fn anthropic_content_tokens(content: &Value) -> u64 {
    if let Some(text) = content.as_str() {
        return anthropic_estimate_tokens(text);
    }
    let Some(blocks) = content.as_array() else {
        return anthropic_estimate_tokens(&content.to_string());
    };
    let mut tokens = 0;
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => tokens += anthropic_estimate_tokens(block.get("text").and_then(Value::as_str).unwrap_or_default()),
            Some("image") => {
                // 官方按 (width*height)/750 计;不解码图片,用 base64 体积粗折算。
                if let Some(data) = block.pointer("/source/data").and_then(Value::as_str) {
                    tokens += (data.len() as u64) / 1500 + 1;
                }
            }
            Some("tool_use") => {
                tokens += 4 + anthropic_estimate_tokens(block.get("name").and_then(Value::as_str).unwrap_or_default()) + anthropic_estimate_tokens(&block.get("input").cloned().unwrap_or_else(|| json!({})).to_string());
            }
            Some("tool_result") => {
                tokens += 4 + anthropic_content_tokens(block.get("content").unwrap_or(&Value::Null));
            }
            Some("thinking") | Some("redacted_thinking") => {}
            _ => tokens += anthropic_estimate_tokens(&block.to_string()),
        }
    }
    tokens
}

pub(super) fn anthropic_estimate_tokens(text: &str) -> u64 {
    let mut cjk = 0u64;
    let mut other = 0u64;
    for character in text.chars() {
        if matches!(character as u32, 0x3000..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF | 0x20000..=0x2FA1F) {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    cjk + other / 4 + u64::from(!text.is_empty())
}

pub(super) fn anthropic_to_chat_request(request: AnthropicMessagesRequest) -> Result<ChatCompletionRequest, String> {
    if request.model.trim().is_empty() {
        return Err("model 不能为空".into());
    }
    if request.max_tokens == 0 {
        return Err("max_tokens 必须大于 0".into());
    }
    if request.messages.is_empty() {
        return Err("messages 不能为空".into());
    }

    let mut messages = Vec::new();
    if let Some(system) = request.system.as_ref() {
        let content = anthropic_text_content(system)?;
        if !content.is_empty() {
            messages.push(json!({"role": "system", "content": content}));
        }
    }
    for message in &request.messages {
        anthropic_convert_message(message, &mut messages)?;
    }

    let tools = request.tools.as_ref().map(|tools| tools.iter().map(anthropic_convert_tool).collect::<Result<Vec<_>, _>>()).transpose()?;
    let tool_choice = request.tool_choice.as_ref().map(anthropic_convert_tool_choice).transpose()?;

    let mut value = json!({
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
        "max_completion_tokens": request.max_tokens,
        "cache_id": request.cache_id,
    });
    let object = value.as_object_mut().expect("Anthropic 请求转换结果必须是对象");
    if let Some(temperature) = request.temperature {
        object.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = request.top_p {
        object.insert("top_p".into(), json!(top_p));
    }
    if let Some(stop_sequences) = request.stop_sequences {
        if stop_sequences.len() > 4 {
            return Err("当前后端最多支持 4 个 stop_sequences".into());
        }
        object.insert("stop".into(), json!(stop_sequences));
    }
    if let Some(tools) = tools {
        object.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tool_choice) = tool_choice {
        object.insert("tool_choice".into(), tool_choice);
    }
    if request.stream {
        object.insert("stream_options".into(), json!({"include_usage": true}));
    }

    // top_k 和 metadata 不参与当前采样/调度，但接受字段以兼容官方 SDK。
    let _ = request.top_k;
    let _ = request.metadata;
    serde_json::from_value(value).map_err(|error| format!("Anthropic 请求转换失败: {error}"))
}

fn anthropic_convert_message(message: &Value, output: &mut Vec<Value>) -> Result<(), String> {
    let object = message.as_object().ok_or_else(|| "messages 的元素必须是对象".to_string())?;
    let role = object.get("role").and_then(Value::as_str).ok_or_else(|| "message.role 必须是字符串".to_string())?;
    let content = object.get("content").ok_or_else(|| "message.content 不能为空".to_string())?;
    // Anthropic 规范把 system 放在顶层；Claude Code 前的兼容网关也可能把它
    // 下沉到 messages。两种形式统一成 OpenAI system message，避免二次适配失败。
    if role == "system" {
        let content = anthropic_text_content(content)?;
        if !content.is_empty() {
            output.push(json!({"role": "system", "content": content}));
        }
        return Ok(());
    }
    if role != "user" && role != "assistant" {
        return Err(format!("Anthropic message.role 不支持 {role}"));
    }
    if content.is_string() {
        output.push(json!({"role": role, "content": content}));
        return Ok(());
    }
    let blocks = content.as_array().ok_or_else(|| "message.content 必须是字符串或内容块数组".to_string())?;
    if role == "assistant" { anthropic_convert_assistant_blocks(blocks, output) } else { anthropic_convert_user_blocks(blocks, output) }
}

fn anthropic_convert_assistant_blocks(blocks: &[Value], output: &mut Vec<Value>) -> Result<(), String> {
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => content.push_str(block.get("text").and_then(Value::as_str).unwrap_or_default()),
            Some("tool_use") => {
                let id = anthropic_required_str(block, "id", "tool_use")?;
                let name = anthropic_required_str(block, "name", "tool_use")?;
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": input.to_string()},
                }));
            }
            Some("thinking") | Some("redacted_thinking") => {}
            Some(kind) => return Err(format!("assistant 内容块不支持 {kind}")),
            None => return Err("assistant 内容块缺少 type".into()),
        }
    }
    let content = if content.is_empty() { Value::Null } else { Value::String(content) };
    let mut message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        message.as_object_mut().expect("assistant message 必须是对象").insert("tool_calls".into(), Value::Array(tool_calls));
    }
    output.push(message);
    Ok(())
}

fn anthropic_convert_user_blocks(blocks: &[Value], output: &mut Vec<Value>) -> Result<(), String> {
    let mut content = Vec::new();
    let mut tool_results = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => content.push(json!({
                "type": "text",
                "text": block.get("text").and_then(Value::as_str).unwrap_or_default(),
            })),
            Some("image") => content.push(anthropic_convert_image(block)?),
            Some("tool_result") => {
                let tool_call_id = anthropic_required_str(block, "tool_use_id", "tool_result")?;
                let result = block.get("content").map(anthropic_text_content).transpose()?.unwrap_or_default();
                tool_results.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }));
            }
            Some(kind) => return Err(format!("user 内容块不支持 {kind}")),
            None => return Err("user 内容块缺少 type".into()),
        }
    }
    output.extend(tool_results);
    if !content.is_empty() {
        let text_only = content.iter().all(|part| part.get("type").and_then(Value::as_str) == Some("text"));
        let content = if text_only {
            let mut text = String::new();
            for part in &content {
                text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default());
            }
            Value::String(text)
        } else {
            Value::Array(content)
        };
        output.push(json!({"role": "user", "content": content}));
    }
    Ok(())
}

fn anthropic_convert_image(block: &Value) -> Result<Value, String> {
    let source = block.get("source").and_then(Value::as_object).ok_or_else(|| "image.source 必须是对象".to_string())?;
    let url = match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = source.get("media_type").and_then(Value::as_str).ok_or_else(|| "base64 image 缺少 media_type".to_string())?;
            let data = source.get("data").and_then(Value::as_str).ok_or_else(|| "base64 image 缺少 data".to_string())?;
            format!("data:{media_type};base64,{data}")
        }
        Some("url") => source.get("url").and_then(Value::as_str).ok_or_else(|| "url image 缺少 url".to_string())?.to_string(),
        Some(kind) => return Err(format!("image.source.type 不支持 {kind}")),
        None => return Err("image.source 缺少 type".into()),
    };
    Ok(json!({"type": "image_url", "image_url": {"url": url}}))
}

fn anthropic_text_content(value: &Value) -> Result<String, String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }
    let blocks = value.as_array().ok_or_else(|| "内容必须是字符串或内容块数组".to_string())?;
    let mut text = String::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(block.get("text").and_then(Value::as_str).unwrap_or_default());
            }
            Some(kind) => return Err(format!("文本内容块不支持 {kind}")),
            None => return Err("文本内容块缺少 type".into()),
        }
    }
    Ok(text)
}

fn anthropic_convert_tool(tool: &Value) -> Result<Value, String> {
    let name = anthropic_required_str(tool, "name", "tool")?;
    let input_schema = tool.get("input_schema").cloned().ok_or_else(|| format!("tool {name} 缺少 input_schema"))?;
    let mut function = json!({"name": name, "parameters": input_schema});
    if let Some(description) = tool.get("description").and_then(Value::as_str) {
        function.as_object_mut().expect("function 必须是对象").insert("description".into(), json!(description));
    }
    Ok(json!({"type": "function", "function": function}))
}

fn anthropic_convert_tool_choice(choice: &Value) -> Result<Value, String> {
    let kind = choice.get("type").and_then(Value::as_str).ok_or_else(|| "tool_choice.type 必须是字符串".to_string())?;
    match kind {
        "auto" => Ok(json!("auto")),
        "none" => Ok(json!("none")),
        "any" => Ok(json!("required")),
        "tool" => {
            let name = anthropic_required_str(choice, "name", "tool_choice")?;
            Ok(json!({"type": "function", "function": {"name": name}}))
        }
        _ => Err(format!("tool_choice.type 不支持 {kind}")),
    }
}

fn anthropic_required_str<'a>(value: &'a Value, key: &str, owner: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).ok_or_else(|| format!("{owner}.{key} 必须是字符串"))
}

async fn anthropic_json_response(response: Response) -> Response {
    let status = response.status();
    let body = match axum::body::to_bytes(response.into_body(), 32 * 1024 * 1024).await {
        Ok(body) => body,
        Err(error) => {
            return anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", format!("读取内部响应失败: {error}"));
        }
    };
    let value: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", format!("解析内部响应失败: {error}"));
        }
    };
    if !status.is_success() {
        return anthropic_openai_error_value(status, &value);
    }
    let choice = &value["choices"][0];
    let message = &choice["message"];
    let mut content = Vec::new();
    if let Some(text) = message.get("content").and_then(Value::as_str)
        && !text.is_empty()
    {
        content.push(json!({"type": "text", "text": text}));
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let arguments = call["function"]["arguments"].as_str().and_then(|arguments| serde_json::from_str::<Value>(arguments).ok()).unwrap_or_else(|| json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": call.get("id").and_then(Value::as_str).unwrap_or_default(),
                "name": call["function"].get("name").and_then(Value::as_str).unwrap_or_default(),
                "input": arguments,
            }));
        }
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    let has_tool = content.iter().any(|block| block["type"] == "tool_use");
    let stop_reason = if has_tool { "tool_use" } else { anthropic_stop_reason(choice.get("finish_reason").and_then(Value::as_str)) };
    let id = anthropic_message_id(value.get("id").and_then(Value::as_str));
    let usage = value.get("usage").cloned().unwrap_or_else(|| json!({}));
    Json(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": value.get("model").cloned().unwrap_or(Value::Null),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    }))
    .into_response()
}

#[derive(Default)]
struct AnthropicStreamTool {
    id: String,
    name: String,
    arguments: String,
    output_index: Option<usize>,
}

pub(super) fn anthropic_stream_response(response: Response, request_model: String) -> Response {
    // Chat 内层和 Anthropic 外层共享同一个 cancellation；最外层 body 被客户端
    // drop 时不必等待 adapter 再收到一个 token 才能传播断连。
    let cancellation = response.extensions().get::<InferenceCancellation>().cloned();
    let mut body = response.into_body().into_data_stream();
    let (tx, rx) = mpsc::channel::<Result<axum::body::Bytes, Infallible>>(32);
    tokio::spawn(async move {
        let mut buffer = String::new();
        let mut started = false;
        let mut model = request_model;
        let mut text_open = false;
        let mut next_index = 0usize;
        let mut tools = Vec::<AnthropicStreamTool>::new();
        let mut finish_reason = None::<String>;
        let mut input_tokens = 0u64;
        let mut output_tokens = 0u64;
        // 内层 Chat SSE 的注释心跳不会形成 Anthropic data event，必须由最外层
        // 直接写 socket；否则长 prefill 期间客户端先 idle timeout，服务端也要等
        // 首 token 才感知断连，导致旧 cache writer 持续占用且 retry 无法命中。
        let mut keep_alive = tokio::time::interval(Duration::from_secs(10));
        keep_alive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let chunk = tokio::select! {
                chunk = body.next() => {
                    let Some(chunk) = chunk else { break };
                    chunk
                }
                _ = keep_alive.tick() => {
                    if tx.send(Ok(axum::body::Bytes::from_static(b": keep-alive\n\n"))).await.is_err() {
                        return;
                    }
                    continue;
                }
            };
            // 客户端断连(rx drop):立即放弃内层 body,让 stream_completion 的
            // 断连探测生效并 cancel 推理;否则推理照跑到 EOF、调度端永不取消。
            if tx.is_closed() {
                return;
            }
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    anthropic_send_sse(
                        &tx,
                        "error",
                        json!({
                            "type": "error",
                            "error": {"type": "api_error", "message": error.to_string()}
                        }),
                    )
                    .await;
                    return;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(end) = buffer.find("\n\n") {
                let event = buffer[..end].to_string();
                buffer.drain(..end + 2);
                let data = event.lines().filter_map(|line| line.strip_prefix("data:")).map(str::trim_start).collect::<Vec<_>>().join("\n");
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    anthropic_finish_stream(&tx, &mut text_open, &mut next_index, &tools, finish_reason.as_deref(), input_tokens, output_tokens).await;
                    return;
                }
                let value: Value = match serde_json::from_str(&data) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                if value.get("error").is_some() {
                    let message = value["error"].get("message").and_then(Value::as_str).unwrap_or("内部推理失败");
                    anthropic_send_sse(
                        &tx,
                        "error",
                        json!({
                            "type": "error",
                            "error": {"type": "api_error", "message": message}
                        }),
                    )
                    .await;
                    return;
                }
                if !started {
                    let message_id = anthropic_message_id(value.get("id").and_then(Value::as_str));
                    if let Some(response_model) = value.get("model").and_then(Value::as_str) {
                        model = response_model.to_string();
                    }
                    anthropic_send_sse(
                        &tx,
                        "message_start",
                        json!({
                            "type": "message_start",
                            "message": {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "model": model,
                                "content": [],
                                "stop_reason": null,
                                "stop_sequence": null,
                                "usage": {"input_tokens": 0, "output_tokens": 0}
                            }
                        }),
                    )
                    .await;
                    started = true;
                }
                if let Some(usage) = value.get("usage") {
                    input_tokens = usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(input_tokens);
                    output_tokens = usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(output_tokens);
                }
                let choice = &value["choices"][0];
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    finish_reason = Some(reason.to_string());
                }
                let delta = &choice["delta"];
                if let Some(text) = delta.get("content").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    if !text_open {
                        anthropic_send_sse(
                            &tx,
                            "content_block_start",
                            json!({
                                "type": "content_block_start",
                                "index": next_index,
                                "content_block": {"type": "text", "text": ""}
                            }),
                        )
                        .await;
                        text_open = true;
                    }
                    anthropic_send_sse(
                        &tx,
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": next_index,
                            "delta": {"type": "text_delta", "text": text}
                        }),
                    )
                    .await;
                }
                if let Some(tool_deltas) = delta.get("tool_calls").and_then(Value::as_array) {
                    for tool_delta in tool_deltas {
                        let index = tool_delta.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                        if tools.len() <= index {
                            tools.resize_with(index + 1, AnthropicStreamTool::default);
                        }
                        let tool = &mut tools[index];
                        if let Some(id) = tool_delta.get("id").and_then(Value::as_str) {
                            tool.id.push_str(id);
                        }
                        if let Some(name) = tool_delta["function"].get("name").and_then(Value::as_str) {
                            tool.name.push_str(name);
                        }
                        if let Some(arguments) = tool_delta["function"].get("arguments").and_then(Value::as_str) {
                            tool.arguments.push_str(arguments);
                        }
                        if tool.output_index.is_none() && !tool.id.is_empty() && !tool.name.is_empty() {
                            if text_open {
                                anthropic_send_sse(&tx, "content_block_stop", json!({"type": "content_block_stop", "index": next_index})).await;
                                text_open = false;
                                next_index += 1;
                            }
                            let output_index = next_index;
                            next_index += 1;
                            anthropic_send_sse(
                                &tx,
                                "content_block_start",
                                json!({
                                    "type": "content_block_start",
                                    "index": output_index,
                                    "content_block": {"type": "tool_use", "id": tool.id, "name": tool.name, "input": {}}
                                }),
                            )
                            .await;
                            tool.output_index = Some(output_index);
                        }
                        if let Some(output_index) = tool.output_index
                            && let Some(arguments) = tool_delta["function"].get("arguments").and_then(Value::as_str).filter(|arguments| !arguments.is_empty())
                        {
                            anthropic_send_sse(
                                &tx,
                                "content_block_delta",
                                json!({
                                    "type": "content_block_delta",
                                    "index": output_index,
                                    "delta": {"type": "input_json_delta", "partial_json": arguments}
                                }),
                            )
                            .await;
                        }
                    }
                }
            }
        }
        if !started {
            let id = anthropic_message_id(None);
            anthropic_send_sse(
                &tx,
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": id, "type": "message", "role": "assistant", "model": model,
                        "content": [], "stop_reason": null, "stop_sequence": null,
                        "usage": {"input_tokens": 0, "output_tokens": 0}
                    }
                }),
            )
            .await;
        }
        anthropic_finish_stream(&tx, &mut text_open, &mut next_index, &tools, finish_reason.as_deref(), input_tokens, output_tokens).await;
    });
    let stream = ReceiverStream::new(rx);
    let body = match cancellation {
        Some(cancellation) => axum::body::Body::from_stream(CancelOnDropStream::new(stream, cancellation)),
        None => axum::body::Body::from_stream(stream),
    };
    sse_utf8(Response::builder().header("cache-control", "no-cache").body(body).expect("Anthropic SSE response 构建失败"))
}

async fn anthropic_finish_stream(tx: &mpsc::Sender<Result<axum::body::Bytes, Infallible>>, text_open: &mut bool, next_index: &mut usize, tools: &[AnthropicStreamTool], finish_reason: Option<&str>, input_tokens: u64, output_tokens: u64) {
    if *text_open {
        anthropic_send_sse(
            tx,
            "content_block_stop",
            json!({
                "type": "content_block_stop", "index": *next_index
            }),
        )
        .await;
        *text_open = false;
        *next_index += 1;
    }
    for tool in tools {
        if let Some(index) = tool.output_index {
            anthropic_send_sse(tx, "content_block_stop", json!({"type": "content_block_stop", "index": index})).await;
            continue;
        }
        let index = *next_index;
        anthropic_send_sse(
            tx,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": tool.id, "name": tool.name, "input": {}}
            }),
        )
        .await;
        anthropic_send_sse(
            tx,
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": tool.arguments}
            }),
        )
        .await;
        anthropic_send_sse(
            tx,
            "content_block_stop",
            json!({
                "type": "content_block_stop", "index": index
            }),
        )
        .await;
        *next_index += 1;
    }
    if *next_index == 0 {
        anthropic_send_sse(
            tx,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
        )
        .await;
        anthropic_send_sse(
            tx,
            "content_block_stop",
            json!({
                "type": "content_block_stop", "index": 0
            }),
        )
        .await;
    }
    let stop_reason = if tools.is_empty() { anthropic_stop_reason(finish_reason) } else { "tool_use" };
    anthropic_send_sse(
        tx,
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
        }),
    )
    .await;
    anthropic_send_sse(tx, "message_stop", json!({"type": "message_stop"})).await;
}

async fn anthropic_send_sse(tx: &mpsc::Sender<Result<axum::body::Bytes, Infallible>>, event: &str, value: Value) {
    let data = format!("event: {event}\ndata: {}\n\n", super::sse_json(&value));
    let _ = tx.send(Ok(axum::body::Bytes::from(data))).await;
}

async fn anthropic_from_openai_error(response: Response) -> Response {
    let status = response.status();
    match axum::body::to_bytes(response.into_body(), 32 * 1024 * 1024).await {
        Ok(body) => match serde_json::from_slice::<Value>(&body) {
            Ok(value) => anthropic_openai_error_value(status, &value),
            Err(_) => anthropic_error(status, "api_error", String::from_utf8_lossy(&body)),
        },
        Err(error) => anthropic_error(status, "api_error", error.to_string()),
    }
}

fn anthropic_openai_error_value(status: StatusCode, value: &Value) -> Response {
    let message = value.get("error").and_then(|error| error.get("message")).and_then(Value::as_str).unwrap_or("请求失败");
    let kind = if status == StatusCode::UNAUTHORIZED {
        "authentication_error"
    } else if status == StatusCode::TOO_MANY_REQUESTS {
        "rate_limit_error"
    } else if status.is_client_error() {
        "invalid_request_error"
    } else {
        "api_error"
    };
    anthropic_error(status, kind, message)
}

fn anthropic_error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "type": "error",
            "error": {"type": kind, "message": message.into()}
        })),
    )
        .into_response()
}

fn anthropic_stop_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") => "max_tokens",
        _ => "end_turn",
    }
}

fn anthropic_message_id(id: Option<&str>) -> String {
    match id {
        Some(id) if id.starts_with("chatcmpl-") => format!("msg_{}", &id[9..]),
        Some(id) if id.starts_with("req_") => format!("msg_{}", &id[4..]),
        Some(id) => format!("msg_{id}"),
        None => format!("msg_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()),
    }
}
