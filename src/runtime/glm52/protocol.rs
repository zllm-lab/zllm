//! GLM-5.2 对话协议：模板、思考围栏与工具调用流。

use std::collections::HashMap;

use serde_json::Value;

use crate::config::Glm52ReasoningEffort;
use crate::runtime::session::{ToolCall, ToolCallDelta, ToolFunction};
use crate::runtime::tool::{ToolDialect, XmlToolCallStream, XmlToolStreamEvent, tool_argument_schemas};

pub(super) fn request_reasoning_effort(request: &Value, maximum: Glm52ReasoningEffort) -> Result<Glm52ReasoningEffort, String> {
    // OpenAI 标准四档全收;low / medium 没对应 enum variant,落到 High(模型依然思考,
    // 不会让 Codex 这种默认 low 的客户端 400)。max 仍受 YAML maximum 约束。
    let requested = match request.get("reasoning_effort") {
        None | Some(Value::Null) => maximum,
        Some(Value::String(value)) if value.eq_ignore_ascii_case("low") => Glm52ReasoningEffort::High,
        Some(Value::String(value)) if value.eq_ignore_ascii_case("medium") => Glm52ReasoningEffort::High,
        Some(Value::String(value)) if value.eq_ignore_ascii_case("high") => Glm52ReasoningEffort::High,
        Some(Value::String(value)) if value.eq_ignore_ascii_case("max") => Glm52ReasoningEffort::Max,
        Some(_) => return Err("reasoning_effort 必须是 low / medium / high / max".to_owned()),
    };
    // YAML 约束本节点允许的最高档位；请求 max 不能越过 high 上限。
    Ok(match (maximum, requested) {
        (Glm52ReasoningEffort::High, Glm52ReasoningEffort::Max) => Glm52ReasoningEffort::High,
        _ => requested,
    })
}

pub(super) fn request_thinking_token_budget(request: &Value, default: Option<usize>) -> Result<Option<usize>, String> {
    match request.get("thinking_token_budget") {
        None | Some(Value::Null) => Ok(default),
        Some(value) => match value.as_i64() {
            Some(-1) => Ok(None),
            Some(value) if value >= 0 => usize::try_from(value).map(Some).map_err(|_| "thinking_token_budget 超出本机 usize 范围".to_owned()),
            _ => Err("thinking_token_budget 必须是 -1 或非负整数".to_owned()),
        },
    }
}

/// 思考预算只改变边界 token，不结束请求。返回 MTP 应保留的 verify rows。
pub(super) fn enforce_thinking_token_budget(tokens: &mut Vec<u32>, emitted: usize, budget: usize, end_token: u32) -> Option<usize> {
    let remaining = budget.saturating_sub(emitted);
    if tokens.iter().take(remaining.saturating_add(1)).any(|&token| token == end_token) || tokens.len() <= remaining {
        return None;
    }
    tokens.truncate(remaining.saturating_add(1));
    tokens[remaining] = end_token;
    Some(tokens.len())
}

#[cfg(test)]
pub(super) fn chat_prompt_glm52(request: &Value) -> Result<String, String> {
    chat_prompt_glm52_with_template(request, false, Glm52ReasoningEffort::Max)
}

pub(super) fn chat_prompt_glm52_with_template(request: &Value, official_template: bool, reasoning_effort: Glm52ReasoningEffort) -> Result<String, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    if messages.is_empty() {
        return Err("messages 不能为空".to_owned());
    }
    let mut prompt = if official_template { format!("[gMASK]<sop>\n<|system|>Reasoning Effort: {}", reasoning_effort.prompt_value()) } else { String::from("[gMASK]<sop>") };
    let tools = crate::runtime::tool::request_tools(request)?;
    if let Some(instructions) = ToolDialect::GlmXml.instructions(tools, request.get("tool_choice"))? {
        prompt.push_str("<|system|>");
        prompt.push_str(&instructions);
    }
    append_glm_messages(&mut prompt, messages)?;
    if official_template {
        prompt.push_str("<|assistant|><think>");
    } else {
        // 普通 Claude/Codex 不接收私有思维链。
        prompt.push_str("<|assistant|></think>");
    }
    Ok(prompt)
}

/// terminal cache 已经覆盖最近一条 assistant 的真实生成 token（包括私有推理），
/// resume 只编码它之后的新消息与下一轮生成头。
pub(super) fn chat_prompt_suffix_glm52(request: &Value, assistant: usize, official_template: bool) -> Result<String, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    if messages.len() <= assistant + 1 {
        return Err("GLM-5.2 resume 边界之后没有新消息".to_owned());
    }
    let mut prompt = String::new();
    append_glm_messages(&mut prompt, &messages[assistant + 1..])?;
    prompt.push_str(if official_template { "<|assistant|><think>" } else { "<|assistant|></think>" });
    Ok(prompt)
}

/// 边界 assistant 之后是否还有消息。没有时 resume 的语义是"继续上一轮",
/// 应从终点状态续写而不是报错。
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(super) fn boundary_has_followup(request: &Value, assistant: usize) -> bool {
    request.get("messages").and_then(Value::as_array).is_some_and(|messages| messages.len() > assistant + 1)
}

fn append_glm_messages(prompt: &mut String, messages: &[Value]) -> Result<(), String> {
    let mut previous_tool = false;
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        let content = glm_text_content(message.get("content"))?;
        match role {
            "system" | "developer" => prompt.push_str(&format!("<|system|>{content}")),
            "user" => prompt.push_str(&format!("<|user|>{content}")),
            "assistant" => {
                prompt.push_str("<|assistant|></think>");
                prompt.push_str(&content);
                append_glm_history_tool_calls(prompt, message.get("tool_calls"))?;
            }
            "tool" => {
                if !previous_tool {
                    prompt.push_str("<|observation|>");
                }
                prompt.push_str("<tool_response>");
                prompt.push_str(&content);
                prompt.push_str("</tool_response>");
            }
            _ => return Err(format!("GLM-5.2 不支持 message.role={role}")),
        }
        previous_tool = role == "tool";
    }
    Ok(())
}

fn append_glm_history_tool_calls(prompt: &mut String, value: Option<&Value>) -> Result<(), String> {
    prompt.push_str(&ToolDialect::GlmXml.render_history(value)?);
    Ok(())
}

fn glm_text_content(content: Option<&Value>) -> Result<String, String> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    return Err("GLM-5.2 当前只支持 text content part".to_owned());
                }
                text.push_str(part.get("text").and_then(Value::as_str).ok_or("text content part 缺少 text")?);
            }
            Ok(text)
        }
        None | Some(Value::Null) => Ok(String::new()),
        _ => Err("message.content 必须是字符串或 text part 数组".to_owned()),
    }
}

pub(super) struct GlmToolCallStream {
    inner: XmlToolCallStream,
    pub(super) calls: Vec<ToolCall>,
}

impl GlmToolCallStream {
    pub(super) fn new(request: &Value, scope: &str) -> Self {
        let schemas = if request.get("tool_choice").and_then(Value::as_str) == Some("none") { HashMap::new() } else { request.get("tools").and_then(Value::as_array).map_or_else(HashMap::new, |tools| tool_argument_schemas(tools)) };
        let stream_deltas = request.get("stream").and_then(Value::as_bool).unwrap_or(false);
        Self { inner: XmlToolCallStream::new(scope, schemas, stream_deltas), calls: Vec::new() }
    }

    pub(super) fn push<F, G>(&mut self, text: &str, mut emit_text: F, mut emit_tool: G) -> bool
    where
        F: FnMut(String) -> bool,
        G: FnMut(ToolCallDelta) -> bool,
    {
        for event in self.inner.push(text) {
            match event {
                XmlToolStreamEvent::Text(text) => {
                    if !emit_text(text) {
                        return false;
                    }
                }
                XmlToolStreamEvent::Delta { index, id, name, arguments } => {
                    if !emit_tool(ToolCallDelta { index, id, name, arguments }) {
                        return false;
                    }
                }
                XmlToolStreamEvent::ToolCall { id, call, .. } => self.calls.push(ToolCall { id, kind: "function".to_owned(), function: ToolFunction { name: call.name, arguments: call.arguments } }),
            }
        }
        true
    }

    pub(super) fn finish<F>(&mut self, mut emit: F) -> bool
    where
        F: FnMut(String) -> bool,
    {
        self.inner.finish().into_iter().all(|event| match event {
            XmlToolStreamEvent::Text(text) => emit(text),
            XmlToolStreamEvent::Delta { .. } | XmlToolStreamEvent::ToolCall { .. } => true,
        })
    }
}

/// thinking 关闭后,答案里再出现的 <think>/</think> 是协议残留(模型退化复读或
/// 围栏强制收口后的复发),必须在外发前剥掉:漏给客户端会在 Codex 渲染成大量
/// 未配对标签,写进 response_text 还会随多轮历史回灌放大。拼写的标签可能
/// 跨 token 边界,尾部疑似半个标签的字节留在 pending 等下一个 token 拼接。
#[derive(Default)]
pub(super) struct ThinkTagFilter {
    pending: String,
}

const THINK_TAGS: [&str; 2] = ["<think>", "</think>"];

impl ThinkTagFilter {
    pub(super) fn push(&mut self, text: &str, thinking_open: bool) -> String {
        if thinking_open && self.pending.is_empty() {
            return text.to_owned();
        }
        let buffer = std::mem::take(&mut self.pending) + text;
        let mut out = buffer.replace(THINK_TAGS[0], "").replace(THINK_TAGS[1], "");
        // 从长到短检查尾部是否为某个标签的真前缀;持有最长前缀,否则提前放行
        // 会把 "</thi" 撕碎导致下一个 token 拼不出完整标签。
        for len in (1..THINK_TAGS[1].len()).rev() {
            if THINK_TAGS.iter().any(|tag| out.ends_with(&tag[..len])) {
                let split = out.len() - len;
                self.pending = out[split..].to_owned();
                out.truncate(split);
                break;
            }
        }
        out
    }

    /// 流结束时放行 held 的半个标签;此时不会再有新文本,不可能拼成完整标签。
    pub(super) fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(super) fn emit_glm_tool_aware_chunk<F, G>(stream: &mut GlmToolCallStream, token_id: u32, chunk: &str, response: &mut String, on_token: &mut F, on_tool_call_delta: &mut G) -> bool
where
    F: FnMut(u32, String) -> bool + ?Sized,
    G: FnMut(ToolCallDelta) -> bool + ?Sized,
{
    stream.push(
        chunk,
        |visible| {
            if !on_token(token_id, visible.clone()) {
                return false;
            }
            response.push_str(&visible);
            true
        },
        on_tool_call_delta,
    )
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(super) fn finish_glm_tool_aware_stream<F>(stream: &mut GlmToolCallStream, response: &mut String, on_token: &mut F) -> bool
where
    F: FnMut(u32, String) -> bool + ?Sized,
{
    stream.finish(|visible| {
        if !on_token(0, visible.clone()) {
            return false;
        }
        response.push_str(&visible);
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_prompt_history_is_append_only() {
        let first = json!({"messages": [{"role": "user", "content": "你好"}]});
        let continued = json!({"messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "世界"},
            {"role": "user", "content": "继续"}
        ]});
        let first_prompt = chat_prompt_glm52(&first).unwrap();
        let continued_prompt = chat_prompt_glm52(&continued).unwrap();
        assert!(continued_prompt.starts_with(&(first_prompt + "世界")));
    }

    #[test]
    fn resume_suffix只包含assistant边界后的消息() {
        let request = json!({"messages": [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "可见答案"},
            {"role": "tool", "content": "工具结果"},
            {"role": "user", "content": "继续"}
        ]});
        let suffix = chat_prompt_suffix_glm52(&request, 1, true).unwrap();
        assert_eq!(suffix, "<|observation|><tool_response>工具结果</tool_response><|user|>继续<|assistant|><think>");
        assert!(!suffix.contains("旧问题"));
        assert!(!suffix.contains("可见答案"));
    }

    #[test]
    fn chat_prompt_and_stream_follow_glm_tool_format() {
        let request = json!({
            "stream": true,
            "messages": [{"role": "user", "content": "读文件"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取文件",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "line": {"type": "integer"}}
                    }
                }
            }]
        });
        let prompt = chat_prompt_glm52(&request).unwrap();
        assert!(prompt.starts_with("[gMASK]<sop><|system|># Tools"));
        assert!(prompt.ends_with("<|user|>读文件<|assistant|></think>"));

        let mut stream = GlmToolCallStream::new(&request, "test");
        let mut visible = String::new();
        let mut deltas = Vec::new();
        assert!(stream.push(
            "先检查<tool_ca",
            |text| {
                visible.push_str(&text);
                true
            },
            |delta| {
                deltas.push(delta);
                true
            },
        ));
        assert!(stream.push(
            "ll>read_file<arg_key>path</arg_key><arg_value>/tmp/a</arg_value><arg_key>line</arg_key><arg_value>3</arg_value></tool_call>",
            |text| {
                visible.push_str(&text);
                true
            },
            |delta| {
                deltas.push(delta);
                true
            },
        ));
        assert!(stream.finish(|text| {
            visible.push_str(&text);
            true
        }));
        assert_eq!(visible, "先检查");
        assert_eq!(stream.calls.len(), 1);
        assert_eq!(stream.calls[0].function.name, "read_file");
        assert_eq!(serde_json::from_str::<Value>(&stream.calls[0].function.arguments).unwrap(), json!({"path": "/tmp/a", "line": 3}));
        assert_eq!(deltas.first().and_then(|delta| delta.name.as_deref()), Some("read_file"));
        assert_eq!(serde_json::from_str::<Value>(&deltas.iter().map(|delta| delta.arguments.as_str()).collect::<String>()).unwrap(), json!({"path": "/tmp/a", "line": 3}));
        assert_eq!(deltas.first().and_then(|delta| delta.id.as_deref()), Some(stream.calls[0].id.as_str()));
    }

    #[test]
    fn zcode_bash_arguments_stream_before_xml_closes() {
        let request = json!({
            "stream": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}
                }
            }]
        });
        let mut stream = GlmToolCallStream::new(&request, "test");
        let mut visible = String::new();
        let mut deltas = Vec::new();
        assert!(stream.push(
            "。<tool_call>Bash<arg_key>command</arg_key><arg_value>python3 - <<'PYEOF'\nimport re",
            |text| {
                visible.push_str(&text);
                true
            },
            |delta| {
                deltas.push(delta);
                true
            },
        ));
        assert!(stream.push(
            "\nfiles = { 'src/kernel/rocm/hip/linear.rs': ['launch']",
            |text| {
                visible.push_str(&text);
                true
            },
            |delta| {
                deltas.push(delta);
                true
            },
        ));
        assert_eq!(visible, "。");
        assert_eq!(deltas.first().and_then(|delta| delta.name.as_deref()), Some("Bash"));
        let arguments = deltas.iter().map(|delta| delta.arguments.as_str()).collect::<String>();
        assert!(arguments.starts_with(r#"{"command":"python3 - <<'PYEOF'\nimport re"#));
        assert!(arguments.contains("src/kernel/rocm/hip/linear.rs"));
        assert!(!arguments.ends_with('}'), "未闭合 XML 不能伪造完整 JSON 参数");
    }

    #[test]
    fn chat_prompt_renders_tool_round_trip() {
        let request = json!({
            "messages": [
                {"role": "user", "content": "读文件"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp/a\",\"line\":3}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
            ]
        });
        let prompt = chat_prompt_glm52(&request).unwrap();
        // serde_json 默认 BTreeMap 按 key 排序迭代,line 排在 path 前。
        assert!(prompt.contains("<|assistant|></think><tool_call>read_file<arg_key>line</arg_key><arg_value>3</arg_value><arg_key>path</arg_key><arg_value>/tmp/a</arg_value></tool_call>"));
        assert!(prompt.contains("<|observation|><tool_response>ok</tool_response>"));
    }

    #[test]
    fn official_chat_prompt_uses_default_max_reasoning() {
        let request = json!({"messages": [{"role": "user", "content": "solve"}]});
        let prompt = chat_prompt_glm52_with_template(&request, true, Glm52ReasoningEffort::Max).unwrap();
        assert_eq!(prompt, "[gMASK]<sop>\n<|system|>Reasoning Effort: Max<|user|>solve<|assistant|><think>");
    }

    #[test]
    fn official_chat_prompt_uses_configured_reasoning_effort() {
        let request = json!({"messages": [{"role": "user", "content": "solve"}]});
        let prompt = chat_prompt_glm52_with_template(&request, true, Glm52ReasoningEffort::High).unwrap();
        assert_eq!(prompt, "[gMASK]<sop>\n<|system|>Reasoning Effort: High<|user|>solve<|assistant|><think>");
    }

    #[test]
    fn thinking_budget_inserts_end_at_boundary() {
        let mut tokens = vec![11, 12, 13, 14];
        assert_eq!(enforce_thinking_token_budget(&mut tokens, 2, 4, 99), Some(3));
        assert_eq!(tokens, [11, 12, 99]);

        let mut immediate = vec![11, 12];
        assert_eq!(enforce_thinking_token_budget(&mut immediate, 0, 0, 99), Some(1));
        assert_eq!(immediate, [99]);

        let mut natural = vec![11, 99, 12];
        assert_eq!(enforce_thinking_token_budget(&mut natural, 2, 4, 99), None);
        assert_eq!(natural, [11, 99, 12]);
    }

    #[test]
    fn request_reasoning_controls_override_yaml_defaults() {
        let request = json!({"reasoning_effort": "high", "thinking_token_budget": 32});
        assert_eq!(request_reasoning_effort(&request, Glm52ReasoningEffort::Max).unwrap(), Glm52ReasoningEffort::High);
        assert_eq!(request_reasoning_effort(&json!({"reasoning_effort": "max"}), Glm52ReasoningEffort::High).unwrap(), Glm52ReasoningEffort::High);
        assert_eq!(request_thinking_token_budget(&request, Some(64)).unwrap(), Some(32));
        assert_eq!(request_thinking_token_budget(&json!({"thinking_token_budget": -1}), Some(64)).unwrap(), None);
    }

    #[test]
    fn think_filter_passes_boundary_through_then_strips_stray_tags() {
        let mut filter = ThinkTagFilter::default();
        // thinking 未关闭:边界标签原样通过
        assert_eq!(filter.push("</think>", true), "</think>");
        // thinking 已关闭:答案里的残留标签剥掉,正常文本保留
        assert_eq!(filter.push("答案</think>正文<think>尾", false), "答案正文尾");
    }

    #[test]
    fn think_filter_buffers_cross_token_tag() {
        let mut filter = ThinkTagFilter::default();
        assert_eq!(filter.push("</think>", true), "</think>");
        // 拼写成普通 token 的 "</" "think" ">" 分三个 delta 到达
        assert_eq!(filter.push("a</", false), "a");
        assert_eq!(filter.push("think", false), "");
        assert_eq!(filter.push(">b", false), "b");
    }

    #[test]
    fn think_filter_finish_releases_held_partial_tag() {
        let mut filter = ThinkTagFilter::default();
        assert_eq!(filter.push("</think>", true), "</think>");
        assert_eq!(filter.push("a <", false), "a ");
        assert_eq!(filter.finish(), "<");
        assert_eq!(filter.finish(), "");
    }
}
