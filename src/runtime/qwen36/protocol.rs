//! Qwen3.6/3.8 ChatML 输入协议与 resume 边界。

use serde_json::Value;

use crate::runtime::session::text_content;
use crate::runtime::session::{ContentPiece, content_pieces};

/// 解析 OpenAI 风格 messages:user 消息接受 text/image_url 混排 content,
/// 其余角色沿用 text_content(出现图像直接报错)。
pub fn parse_messages(request: &Value) -> Result<Vec<(String, Vec<ContentPiece>)>, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    if messages.is_empty() {
        return Err("messages 不能为空".to_owned());
    }
    messages
        .iter()
        .map(|message| {
            let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
            let content = message.get("content");
            let pieces = content_pieces(content)?;
            if role != "user" && pieces.iter().any(|piece| matches!(piece, ContentPiece::Image { .. })) {
                return Err(format!("Qwen 节点不支持 {role} 消息中的图像"));
            }
            Ok((role.to_owned(), pieces))
        })
        .collect()
}

fn push_piece(prompt: &mut String, piece: &ContentPiece, blocks: &[String], next_block: &mut usize) -> Result<(), String> {
    match piece {
        ContentPiece::Text(text) => prompt.push_str(text),
        ContentPiece::Image { .. } => {
            let block = blocks.get(*next_block).ok_or("Qwen 图像占位符与 content 顺序不匹配")?;
            *next_block += 1;
            prompt.push_str(block);
        }
    }
    Ok(())
}

/// 拼装 Qwen im_start 模板;blocks 与 user 消息中的图像按出现顺序一一对应。
/// `thinking=false` 时生成头注入官方空围栏(Qwen3 ChatML 的 enable_thinking 语义),
/// assistant 历史重放同样注入,保持与生成序列同构。
pub fn render_prompt(turns: &[(String, Vec<ContentPiece>)], blocks: &[String], thinking: bool) -> Result<String, String> {
    render_prompt_with_instructions(turns, blocks, thinking, None, None)
}

/// 请求级入口额外注入工具 schema；图像预处理后的 blocks 仍由模型 session 提供。
pub fn render_request_prompt(request: &Value, turns: &[(String, Vec<ContentPiece>)], blocks: &[String], thinking: bool) -> Result<String, String> {
    let tools = crate::runtime::tool::request_tools(request)?;
    let instructions = crate::runtime::tool::ToolDialect::ChatmlJson.instructions(tools, request.get("tool_choice"))?;
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    render_prompt_with_instructions(turns, blocks, thinking, instructions.as_deref(), Some(messages))
}

fn render_prompt_with_instructions(turns: &[(String, Vec<ContentPiece>)], blocks: &[String], thinking: bool, instructions: Option<&str>, messages: Option<&[Value]>) -> Result<String, String> {
    let mut next_block = 0usize;
    let mut prompt = String::new();
    if let Some(instructions) = instructions {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(instructions);
        prompt.push_str("<|im_end|>\n");
    }
    for (index, (role, pieces)) in turns.iter().enumerate() {
        let (open, close) = match role.as_str() {
            "system" | "developer" => ("<|im_start|>system\n", "<|im_end|>\n"),
            "user" => ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant" => ("<|im_start|>assistant\n", "<|im_end|>\n"),
            "tool" => ("<|im_start|>user\n<tool_response>\n", "\n</tool_response><|im_end|>\n"),
            other => return Err(format!("Qwen 节点不支持 role '{other}'")),
        };
        prompt.push_str(open);
        if !thinking && role == "assistant" {
            push_think_fence(&mut prompt, false);
        }
        for piece in pieces {
            push_piece(&mut prompt, piece, blocks, &mut next_block)?;
        }
        if role == "assistant"
            && let Some(message) = messages.and_then(|messages| messages.get(index))
        {
            prompt.push_str(&crate::runtime::tool::ToolDialect::ChatmlJson.render_history(message.get("tool_calls"))?);
        }
        prompt.push_str(close);
    }
    if next_block != blocks.len() {
        return Err("Qwen 图像占位符未被完整消费".to_owned());
    }
    prompt.push_str("<|im_start|>assistant\n");
    if !thinking {
        push_think_fence(&mut prompt, false);
    }
    Ok(prompt)
}

/// Qwen3 官方模板的思考围栏:关闭思考时输出空块,模型直接作答。
fn push_think_fence(prompt: &mut String, _open: bool) {
    prompt.push_str("<think>\n\n</think>\n\n");
}

pub fn chat_prompt_suffix(request: &Value, assistant: usize, thinking: bool) -> Result<String, String> {
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    if messages.len() <= assistant {
        return Err("Qwen resume 边界之后没有新消息".to_owned());
    }
    let mut prompt = String::from("<|im_end|>\n");
    for message in &messages[assistant + 1..] {
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        let content = text_content(message.get("content"))?;
        match role {
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                if !thinking {
                    push_think_fence(&mut prompt, false);
                }
                prompt.push_str(&content);
                prompt.push_str(&crate::runtime::tool::ToolDialect::ChatmlJson.render_history(message.get("tool_calls"))?);
                prompt.push_str("<|im_end|>\n");
            }
            "user" => prompt.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n")),
            "system" | "developer" => prompt.push_str(&format!("<|im_start|>system\n{content}<|im_end|>\n")),
            "tool" => prompt.push_str(&format!("<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")),
            other => return Err(format!("Qwen 节点不支持 role '{other}'")),
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    if !thinking {
        push_think_fence(&mut prompt, false);
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_prompt_rendering_matches_legacy_template() {
        let request = json!({"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "tool says"},
            {"role": "user", "content": [{"type": "text", "text": "again"}]},
        ]});
        let prompt = render_prompt(&parse_messages(&request).unwrap(), &[], true).unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nhi<|im_end|>\n<|im_start|>user\n<tool_response>\ntool says\n</tool_response><|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn multimodal_history_resume只渲染图片轮后的文本后缀() {
        let request = json!({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "描述图片"},
                {"type": "image_url", "image_url": {"url": "/tmp/history.png"}}
            ]},
            {"role": "assistant", "content": "图片里有一只猫"},
            {"role": "user", "content": "它是什么颜色？"}
        ]});
        let suffix = chat_prompt_suffix(&request, 1, true).unwrap();
        assert_eq!(suffix, "<|im_end|>\n<|im_start|>user\n它是什么颜色？<|im_end|>\n<|im_start|>assistant\n");
        assert!(!suffix.contains("history.png"));
    }

    #[test]
    fn user_image_parts_inline_placeholders_in_order() {
        let request = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "看这两张图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "http://example.com/2.png"}},
            {"type": "text", "text": "说说区别"},
        ]}]});
        let blocks = vec!["<BLOCK1>".to_owned(), "<BLOCK2>".to_owned()];
        let prompt = render_prompt(&parse_messages(&request).unwrap(), &blocks, true).unwrap();
        assert_eq!(prompt, "<|im_start|>user\n看这两张图<BLOCK1><BLOCK2>说说区别<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn non_user_roles_reject_images_and_missing_url() {
        let assistant_image = json!({"messages": [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]});
        assert!(parse_messages(&assistant_image).is_err());
        let missing_url = json!({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]});
        assert!(parse_messages(&missing_url).is_err());
    }

    #[test]
    fn request_prompt注入工具说明() {
        let request = json!({
            "messages": [
                {"role": "user", "content": "weather"},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Shanghai\"}"}}]},
                {"role": "tool", "content": "sunny"},
                {"role": "user", "content": "thanks"}
            ],
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
        });
        let turns = parse_messages(&request).unwrap();
        let prompt = render_request_prompt(&request, &turns, &[], true).unwrap();
        assert!(prompt.contains("Available function schemas"));
        assert!(prompt.contains("weather"));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("<tool_response>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
