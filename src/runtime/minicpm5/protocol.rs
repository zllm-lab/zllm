//! MiniCPM5 ChatML 输入协议与思考围栏。

use serde_json::Value;

use crate::runtime::session::text_content;

pub fn chat_prompt(request: &Value, thinking: bool) -> Result<String, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    let mut prompt = String::from("<s>");
    let mut has_user = false;
    append_messages(&mut prompt, messages, &mut has_user, thinking)?;
    if !has_user {
        return Err("MiniCPM5 请求缺少 messages.user.content".to_owned());
    }
    append_generation_head(&mut prompt, thinking);
    Ok(prompt)
}

fn append_messages(prompt: &mut String, messages: &[Value], has_user: &mut bool, thinking: bool) -> Result<(), String> {
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        let content = text_content(message.get("content"))?;
        match role {
            "system" => prompt.push_str(&format!("<|im_start|>system\n{content}<|im_end|>\n")),
            "user" => {
                *has_user = true;
                prompt.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n"));
            }
            // 重放历史 assistant 必须注入与生成时相同的 think 围栏,否则多轮全量
            // 渲染与首轮生成序列在围栏处错位,terminal cache 的 token 级对齐永远失败。
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                append_generation_head_fence(prompt, thinking);
                prompt.push_str(&format!("{content}<|im_end|>\n"));
            }
            other => return Err(format!("MiniCPM5 节点不支持 role '{other}'")),
        }
    }
    Ok(())
}

fn append_generation_head(prompt: &mut String, thinking: bool) {
    prompt.push_str("<|im_start|>assistant\n");
    append_generation_head_fence(prompt, thinking);
}

fn append_generation_head_fence(prompt: &mut String, thinking: bool) {
    prompt.push_str(if thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" });
}

pub fn chat_prompt_suffix(request: &Value, assistant: usize, thinking: bool) -> Result<String, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    if messages.len() <= assistant {
        return Err("MiniCPM5 resume 边界之后没有新消息".to_owned());
    }
    let mut prompt = String::from("<|im_end|>\n");
    let mut has_user = false;
    for message in &messages[assistant + 1..] {
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        let content = text_content(message.get("content"))?;
        match role {
            "system" => prompt.push_str(&format!("<|im_start|>system\n{content}<|im_end|>\n")),
            "user" => {
                has_user = true;
                prompt.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n"));
            }
            // 与 append_messages 一致:重放 assistant 注入 think 围栏。
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                append_generation_head_fence(&mut prompt, thinking);
                prompt.push_str(&format!("{content}<|im_end|>\n"));
            }
            other => return Err(format!("MiniCPM5 节点不支持 role '{other}'")),
        }
    }
    let _ = has_user;
    append_generation_head(&mut prompt, thinking);
    Ok(prompt)
}
