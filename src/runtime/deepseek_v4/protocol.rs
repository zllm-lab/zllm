//! DeepSeek-V4 消息编码、reasoning 控制与 terminal resume 协议。

pub(crate) const BOS: &str = "<｜begin▁of▁sentence｜>";
pub(crate) const EOS: &str = "<｜end▁of▁sentence｜>";
pub(crate) const USER: &str = "<｜User｜>";
pub(crate) const ASSISTANT: &str = "<｜Assistant｜>";
pub(crate) const HIGH_REASONING_PREFIX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const MAX_REASONING_PREFIX: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n\
You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n\
Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

#[derive(Clone, Copy)]
pub struct DeepSeekV4ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
    pub reasoning_content: Option<&'a str>,
    pub tool_calls: &'a str,
    pub tool_call_id: Option<&'a str>,
    pub tool_call_ids: &'a [String],
}

pub fn chat_prompt<'a>(messages: impl IntoIterator<Item = DeepSeekV4ChatMessage<'a>>, tool_instructions: Option<&str>, thinking: Option<&str>, reasoning_effort: Option<&str>) -> Result<String, String> {
    chat_prompt_with_public_prefix(messages, tool_instructions, thinking, reasoning_effort).map(|(prompt, _)| prompt)
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
pub(crate) fn resume_suffix(full_prompt: &str, assistant_prompt: &str) -> Result<String, String> {
    let prefix = assistant_prompt.strip_suffix(EOS).ok_or("DeepSeek-V4 assistant prompt 缺少 EOS")?;
    full_prompt.strip_prefix(prefix).map(str::to_owned).ok_or("DeepSeek-V4 terminal resume prompt 前缀不一致".to_owned())
}

pub fn chat_prompt_with_public_prefix<'a>(messages: impl IntoIterator<Item = DeepSeekV4ChatMessage<'a>>, tool_instructions: Option<&str>, thinking: Option<&str>, reasoning_effort: Option<&str>) -> Result<(String, Option<usize>), String> {
    let thinking = match thinking {
        None => false,
        Some("enabled") => true,
        Some("disabled") => false,
        Some(_) => return Err("thinking.type 必须是 enabled / disabled".to_owned()),
    };
    let reasoning_prefix = match reasoning_effort {
        None | Some("low" | "medium") => "",
        Some("high") if thinking => HIGH_REASONING_PREFIX,
        Some("max" | "xhigh") if thinking => MAX_REASONING_PREFIX,
        Some("high" | "max" | "xhigh") => "",
        Some(_) => return Err("DeepSeek-V4 reasoning_effort 必须是 low / medium / high / max / xhigh".to_owned()),
    };

    let messages = messages.into_iter().collect::<Vec<_>>();
    for message in &messages {
        if !matches!(message.role, "developer" | "system" | "user" | "assistant" | "tool") {
            return Err(format!("DeepSeek-V4 当前不支持 message.role={}", message.role));
        }
    }
    let original_last_user = messages.iter().rposition(|message| matches!(message.role, "developer" | "user" | "tool")).ok_or("DeepSeek-V4 请求缺少 user/developer/tool message")?;
    let drop_thinking = thinking && tool_instructions.is_none();
    let kept = messages.iter().enumerate().filter_map(|(index, message)| (!(drop_thinking && message.role == "developer" && index < original_last_user)).then_some(index)).collect::<Vec<_>>();
    let last_user = kept.iter().rposition(|&index| matches!(messages[index].role, "developer" | "user" | "tool")).ok_or("DeepSeek-V4 请求缺少保留的 user/developer/tool message")?;
    let tool_owner = tool_instructions.and_then(|_| kept.iter().copied().find(|&index| messages[index].role == "system").or_else(|| kept.iter().copied().find(|&index| messages[index].role == "developer")));
    let mut prompt = String::from(BOS);
    let mut public_prefix_bytes = None;
    prompt.push_str(reasoning_prefix);
    if tool_owner.is_none()
        && let Some(instructions) = tool_instructions
    {
        prompt.push_str("\n\n");
        prompt.push_str(instructions);
        public_prefix_bytes = Some(prompt.len());
    }
    let mut cursor = 0usize;
    let mut last_tool_order = std::collections::HashMap::<&str, usize>::new();
    while cursor < kept.len() {
        let source_index = kept[cursor];
        let message = &messages[source_index];
        match message.role {
            "system" => {
                prompt.push_str(message.content);
                if tool_owner == Some(source_index) {
                    prompt.push_str("\n\n");
                    prompt.push_str(tool_instructions.expect("tool owner 仅在有 instructions 时存在"));
                    public_prefix_bytes = Some(prompt.len());
                }
                cursor += 1;
            }
            "developer" => {
                prompt.push_str(USER);
                prompt.push_str(message.content);
                if tool_owner == Some(source_index) {
                    prompt.push_str("\n\n");
                    prompt.push_str(tool_instructions.expect("tool owner 仅在有 instructions 时存在"));
                    public_prefix_bytes = Some(prompt.len());
                }
                cursor += 1;
                if cursor == kept.len() || messages[kept[cursor]].role == "assistant" {
                    prompt.push_str(ASSISTANT);
                    prompt.push_str(if thinking { "<think>" } else { "</think>" });
                }
            }
            "assistant" => {
                if thinking && (!drop_thinking || cursor > last_user) {
                    prompt.push_str(message.reasoning_content.unwrap_or_default());
                    prompt.push_str("</think>");
                }
                prompt.push_str(message.content);
                prompt.push_str(message.tool_calls);
                prompt.push_str(EOS);
                last_tool_order.clear();
                last_tool_order.extend(message.tool_call_ids.iter().enumerate().map(|(index, id)| (id.as_str(), index)));
                cursor += 1;
            }
            "user" | "tool" => {
                let start = cursor;
                while cursor < kept.len() && matches!(messages[kept[cursor]].role, "user" | "tool") {
                    cursor += 1;
                }
                let mut tool_results = kept[start..cursor].iter().copied().filter(|&index| messages[index].role == "tool").collect::<Vec<_>>();
                tool_results.sort_by_key(|&index| messages[index].tool_call_id.and_then(|id| last_tool_order.get(id)).copied().unwrap_or(0));
                let mut sorted_tool = 0usize;
                prompt.push_str(USER);
                for (offset, &index) in kept[start..cursor].iter().enumerate() {
                    if offset != 0 {
                        prompt.push_str("\n\n");
                    }
                    if messages[index].role == "tool" {
                        let result = &messages[tool_results[sorted_tool]];
                        sorted_tool += 1;
                        prompt.push_str("<tool_result>");
                        prompt.push_str(result.content);
                        prompt.push_str("</tool_result>");
                    } else {
                        prompt.push_str(messages[index].content);
                    }
                }
                if cursor == kept.len() || messages[kept[cursor]].role == "assistant" {
                    prompt.push_str(ASSISTANT);
                    prompt.push_str(if thinking { "<think>" } else { "</think>" });
                }
            }
            _ => unreachable!("role 已预先检查"),
        }
    }
    Ok((prompt, public_prefix_bytes))
}
