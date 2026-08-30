//! Ornith 节点的请求层协议：chat prompt 拼接、工具说明注入与 tool_call 流式解析。
//!
//! 纯 host 逻辑，不依赖任何 backend；Metal 与 ROCm 节点引擎共用。

use serde_json::Value;

use crate::runtime::session::text_content;

pub fn chat_prompt_suffix(request: &Value, assistant: usize) -> Result<String, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    let mut prompt = "<|im_end|>\n".to_owned();
    append_chat_messages(&mut prompt, &messages[assistant + 1..])?;
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

pub fn chat_prompt(request: &Value) -> Result<String, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    if messages.is_empty() {
        return Err("messages 不能为空".to_owned());
    }
    let mut prompt = String::new();
    if let Some(instructions) = tool_instructions(request)? {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(&instructions);
        prompt.push_str("<|im_end|>\n");
    }
    append_chat_messages(&mut prompt, messages)?;
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

fn append_chat_messages(prompt: &mut String, messages: &[Value]) -> Result<(), String> {
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        if role == "tool" {
            prompt.push_str("<|im_start|>user\n");
            while index < messages.len() && messages[index].get("role").and_then(Value::as_str) == Some("tool") {
                let content = text_content(messages[index].get("content"))?;
                prompt.push_str("<tool_response>\n");
                prompt.push_str(&content);
                prompt.push_str("\n</tool_response>\n");
                index += 1;
            }
            prompt.push_str("<|im_end|>\n");
            continue;
        }
        let content = text_content(message.get("content"))?;
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&content);
        if role == "assistant" {
            append_history_tool_calls(prompt, message.get("tool_calls"))?;
        }
        prompt.push_str("<|im_end|>\n");
        index += 1;
    }
    Ok(())
}

fn append_history_tool_calls(prompt: &mut String, value: Option<&Value>) -> Result<(), String> {
    prompt.push_str(&crate::runtime::tool::ToolDialect::ChatmlJson.render_history(value)?);
    Ok(())
}

fn tool_instructions(request: &Value) -> Result<Option<String>, String> {
    let tools = crate::runtime::tool::request_tools(request)?;
    crate::runtime::tool::ToolDialect::ChatmlJson.instructions(tools, request.get("tool_choice"))
}
