//! Qwen3 家族输入输出协议。

/// 经典 dense Qwen3 的 ChatML 单轮 prompt。
pub fn chat_prompt(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}
