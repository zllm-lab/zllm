//! GLM-5.3-Flash 输入协议与多模态消息分段。

use serde_json::Value;

use crate::runtime::session::{ContentPiece, content_pieces, text_content};

/// user 消息支持图文混排；其余角色只接受文本。
/// 纯文本拼接结果保持与官方模板逐字节一致。
pub fn chat_segments(request: &Value) -> Result<Vec<ContentPiece>, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    let mut segments = vec![ContentPiece::Text("[gMASK]<sop>\n".to_owned())];
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        match role {
            "developer" | "system" => {
                segments.push(ContentPiece::Text("<|system|>".to_owned()));
                segments.push(ContentPiece::Text(text_content(message.get("content"))?));
                segments.push(ContentPiece::Text("\n".to_owned()));
            }
            "user" => {
                segments.push(ContentPiece::Text("<|user|>".to_owned()));
                segments.extend(content_pieces(message.get("content"))?);
                segments.push(ContentPiece::Text("\n".to_owned()));
            }
            "assistant" => {
                segments.push(ContentPiece::Text("<|assistant|>\n<think></think>\n".to_owned()));
                segments.push(ContentPiece::Text(text_content(message.get("content"))?.trim().to_owned()));
                segments.push(ContentPiece::Text("\n".to_owned()));
            }
            "tool" => {
                segments.push(ContentPiece::Text("<|observation|><tool_response>".to_owned()));
                segments.push(ContentPiece::Text(text_content(message.get("content"))?));
                segments.push(ContentPiece::Text("</tool_response>\n".to_owned()));
            }
            other => return Err(format!("GLM-5.3-Flash 不支持 message.role={other}")),
        }
    }
    segments.push(ContentPiece::Text("<|assistant|><think>".to_owned()));
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_text(segments: &[ContentPiece]) -> String {
        segments
            .iter()
            .map(|segment| match segment {
                ContentPiece::Text(text) => text.as_str(),
                ContentPiece::Image { .. } => panic!("纯文本渲染不应出现图像段"),
            })
            .collect()
    }

    #[test]
    fn chat_segments_matches_official_text_template() {
        let request = serde_json::json!({"messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}]});
        assert_eq!(render_text(&chat_segments(&request).unwrap()), "[gMASK]<sop>\n<|system|>system\n<|user|>hello\n<|assistant|><think>");
    }

    #[test]
    fn chat_segments_renders_assistant_and_tool_history() {
        let request = serde_json::json!({"messages": [
            {"role": "assistant", "content": "answer"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "continue"}
        ]});
        assert_eq!(render_text(&chat_segments(&request).unwrap()), "[gMASK]<sop>\n<|assistant|>\n<think></think>\nanswer\n<|observation|><tool_response>result</tool_response>\n<|user|>continue\n<|assistant|><think>");
    }

    #[test]
    fn user_image_parts_inline_in_order() {
        let request = serde_json::json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "看这两张图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "http://example.com/2.png"}},
            {"type": "text", "text": "说说区别"}
        ]}]});
        let segments = chat_segments(&request).unwrap();
        let urls = segments
            .iter()
            .filter_map(|segment| match segment {
                ContentPiece::Image { url } => Some(url.as_str()),
                ContentPiece::Text(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(urls, ["data:image/png;base64,AAA", "http://example.com/2.png"]);
        let texts = segments
            .iter()
            .filter_map(|segment| match segment {
                ContentPiece::Text(text) => Some(text.as_str()),
                ContentPiece::Image { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(texts.concat(), "[gMASK]<sop>\n<|user|>看这两张图说说区别\n<|assistant|><think>");
    }

    #[test]
    fn non_user_roles_reject_image_parts() {
        let request = serde_json::json!({"messages": [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]});
        assert!(chat_segments(&request).is_err());
    }
}
