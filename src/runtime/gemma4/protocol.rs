//! Gemma 4 输入协议与 resume token 边界。

use serde_json::Value;

use crate::runtime::session::{ContentPiece, content_pieces, text_content};

/// 把 OpenAI 风格 messages 拼成 Gemma4 模板的保序片段。
/// user 消息接受 text/image_url 混排，其余角色只接受文本。
pub fn chat_segments(request: &Value) -> Result<Vec<ContentPiece>, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    if messages.is_empty() {
        return Err("messages 不能为空".to_owned());
    }
    let mut segments = vec![ContentPiece::Text("<bos>".to_owned())];
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        match role {
            "assistant" => push_text_turn(&mut segments, "<|turn>model\n", message)?,
            "user" => {
                let pieces = content_pieces(message.get("content"))?;
                if pieces.iter().all(|piece| matches!(piece, ContentPiece::Text(text) if text.is_empty())) {
                    return Err("message.content 不能为空".to_owned());
                }
                segments.push(ContentPiece::Text("<|turn>user\n".to_owned()));
                segments.extend(pieces);
                segments.push(ContentPiece::Text("<turn|>\n".to_owned()));
            }
            "system" | "developer" | "tool" => push_text_turn(&mut segments, "<|turn>user\n", message)?,
            other => return Err(format!("Gemma4 节点不支持 role '{other}'")),
        }
    }
    let generation_prompt = if thinking_enabled(request) { "<|turn>model\n<|channel>thought\n<channel|>" } else { "<|turn>model\n" };
    segments.push(ContentPiece::Text(generation_prompt.to_owned()));
    Ok(segments)
}

fn thinking_enabled(request: &Value) -> bool {
    request.get("enable_thinking").and_then(Value::as_bool) == Some(true) || request.get("thinking").and_then(Value::as_object).and_then(|thinking| thinking.get("type")).and_then(Value::as_str) == Some("enabled")
}

fn push_text_turn(segments: &mut Vec<ContentPiece>, prefix: &str, message: &Value) -> Result<(), String> {
    let content = text_content(message.get("content"))?;
    if content.is_empty() {
        return Err("message.content 不能为空".to_owned());
    }
    segments.push(ContentPiece::Text(prefix.to_owned()));
    segments.push(ContentPiece::Text(content));
    segments.push(ContentPiece::Text("<turn|>\n".to_owned()));
    Ok(())
}

/// 纯文本请求的模板拼接；调用方必须先确认没有图像段。
pub fn segments_text(segments: &[ContentPiece]) -> String {
    segments
        .iter()
        .map(|segment| match segment {
            ContentPiece::Text(text) => text.as_str(),
            ContentPiece::Image { .. } => unreachable!("纯文本渲染不应出现图像段"),
        })
        .collect()
}

pub fn chat_prompt_suffix(request: &Value, assistant: usize) -> Result<String, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    if messages.len() <= assistant {
        return Err("Gemma4 resume 边界之后没有新消息".to_owned());
    }
    let mut prompt = String::from("<turn|>\n");
    for message in &messages[assistant + 1..] {
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        let content = text_content(message.get("content"))?;
        if content.is_empty() {
            return Err("message.content 不能为空".to_owned());
        }
        match role {
            "assistant" => prompt.push_str(&format!("<|turn>model\n{content}<turn|>\n")),
            "user" | "system" | "developer" | "tool" => prompt.push_str(&format!("<|turn>user\n{content}<turn|>\n")),
            other => return Err(format!("Gemma4 节点不支持 role '{other}'")),
        }
    }
    prompt.push_str("<|turn>model\n");
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 纯文本模板与旧_chat_prompt_逐字节一致() {
        let request = serde_json::json!({"messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [{"type": "text", "text": "continue"}]}
        ]});
        let expected = "<bos><|turn>user\nsystem<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\nanswer<turn|>\n<|turn>user\ncontinue<turn|>\n<|turn>model\n";
        assert_eq!(segments_text(&chat_segments(&request).unwrap()), expected);
    }

    #[test]
    fn user_图像段保序内联() {
        let request = serde_json::json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "看图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "text", "text": "如何"}
        ]}]});
        let rendered = chat_segments(&request)
            .unwrap()
            .iter()
            .map(|segment| match segment {
                ContentPiece::Text(text) => text.clone(),
                ContentPiece::Image { url } => format!("[image:{url}]"),
            })
            .collect::<String>();
        assert_eq!(rendered, "<bos><|turn>user\n看图[image:data:image/png;base64,AAA]如何<turn|>\n<|turn>model\n");
    }

    #[test]
    fn 显式开启思考才追加_thought_通道() {
        let request = serde_json::json!({"enable_thinking": true, "messages": [{"role": "user", "content": "hello"}]});
        assert!(segments_text(&chat_segments(&request).unwrap()).ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    #[test]
    fn 非_user_角色拒绝图像() {
        let request = serde_json::json!({"messages": [{"role": "system", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]});
        assert!(chat_segments(&request).is_err());
    }

    #[test]
    fn 空内容仍然报错() {
        assert!(chat_segments(&serde_json::json!({"messages": [{"role": "user", "content": ""}]})).is_err());
        assert!(chat_segments(&serde_json::json!({"messages": [{"role": "user", "content": []}]})).is_err());
    }
}
