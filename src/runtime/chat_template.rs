//! Chat template 渲染：GGUF 模型把 chat 模板（完整 Jinja）存在 `tokenizer.chat_template`，
//! 本模块用 minijinja 编译并渲染 /v1/chat/completions 请求 JSON，产出 prompt 字符串。
//! 渲染上下文对齐 llama.cpp：`messages` / `tools` / `add_generation_prompt` /
//! `bos_token` / `eos_token`，请求里的其余字段（如 `enable_thinking`）整体透传。

use minijinja::{Environment, Error, ErrorKind, Value};

/// 编译好的 chat template：模板在 `new` 时编译一次，`render` 可重复调用。
pub struct ChatTemplate {
    env: Environment<'static>,
}

/// 模板在 env 里的固定名字，只为让编译/渲染错误信息可定位。
const TEMPLATE_NAME: &str = "chat";

impl ChatTemplate {
    /// 编译模板。注册 `raise_exception`（模板主动报"请求非法"用，转为渲染错误而非 panic），
    /// 以及 bos/eos 的默认 globals（接线时用 `set_special_tokens` 覆盖为模型真实 token 串）。
    pub fn new(template: &str) -> Result<Self, String> {
        let mut env = Environment::new();
        // chat 模板按 Jinja2/Python 语义写（dict.get / str.split 等）,minijinja 原生没有
        // 这些方法;pycompat 回调按原语义补齐,避免改模板。
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |message: String| -> Result<(), Error> { Err(Error::new(ErrorKind::InvalidOperation, message)) });
        env.add_global("bos_token", "<bos>");
        env.add_global("eos_token", "<eos>");
        // 源码转 owned，让 env 持有 'static 模板，ChatTemplate 可自由移动/共享。
        env.add_template_owned(TEMPLATE_NAME, template.to_owned()).map_err(|e| format!("chat template 编译失败({TEMPLATE_NAME}): {e}"))?;
        Ok(Self { env })
    }

    /// 用模型真实的 BOS/EOS token 串覆盖默认值（来自 GGUF metadata / tokenizer 配置）。
    pub fn set_special_tokens(&mut self, bos_token: impl Into<String>, eos_token: impl Into<String>) {
        self.env.add_global("bos_token", Value::from(bos_token.into()));
        self.env.add_global("eos_token", Value::from(eos_token.into()));
    }

    /// 渲染 OpenAI chat completions 请求 JSON 为 prompt。
    ///
    /// 请求 object 整体进入模板上下文（模板自取 messages/tools 等所需字段），
    /// 缺省补 `add_generation_prompt=true`；bos_token/eos_token 由 env globals 兜底，
    /// 请求里显式给出时覆盖。
    pub fn render(&self, request: &serde_json::Value) -> Result<String, String> {
        let mut context = request.as_object().cloned().ok_or_else(|| "chat completions 请求必须是 JSON object".to_string())?;
        context.entry("add_generation_prompt").or_insert(serde_json::Value::Bool(true));
        let template = self.env.get_template(TEMPLATE_NAME).map_err(|e| format!("chat template 缺失({TEMPLATE_NAME}): {e}"))?;
        template.render(Value::from_serialize(&context)).map_err(|e| format!("chat template 渲染失败({TEMPLATE_NAME}): {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_messages() {
        let template = ChatTemplate::new("{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}").unwrap();
        let request = serde_json::json!({"messages": [{"role": "user", "content": "你好"}]});
        assert_eq!(template.render(&request).unwrap(), "<user>你好<assistant>");
    }

    #[test]
    fn render_tools_branch() {
        let template = ChatTemplate::new("{% if tools %}[tools:{% for t in tools %}{{ t.function.name }}{% endfor %}]{% endif %}{{ messages[0].content }}").unwrap();
        let with_tools = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        });
        assert_eq!(template.render(&with_tools).unwrap(), "[tools:get_weather]hi");
        let without_tools = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(template.render(&without_tools).unwrap(), "hi");
    }

    #[test]
    fn raise_exception_fails_render() {
        let template = ChatTemplate::new("{% if messages[0].role != 'system' %}{{ raise_exception('第一条消息必须是 system') }}{% endif %}ok").unwrap();
        let bad = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        let error = template.render(&bad).unwrap_err();
        assert!(error.contains("第一条消息必须是 system"), "错误信息应带模板原文: {error}");
        let good = serde_json::json!({"messages": [{"role": "system", "content": "hi"}]});
        assert_eq!(template.render(&good).unwrap(), "ok");
    }

    /// 真实 gemma4 GGUF 模板验证：先按 GGUF v3 kv 布局提取 tokenizer.chat_template 到
    /// /tmp/gemma4_chat_template.jinja，再 `cargo test --lib chat_template -- --include-ignored`。
    #[test]
    #[ignore = "依赖本地提取的 /tmp/gemma4_chat_template.jinja"]
    fn render_gemma4_real_template() {
        let source = std::fs::read_to_string("/tmp/gemma4_chat_template.jinja").expect("先从 GGUF 提取模板到 /tmp/gemma4_chat_template.jinja");
        let template = ChatTemplate::new(&source).unwrap();

        // 纯文本多轮：system + user/assistant 交替。
        let plain = serde_json::json!({"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
            {"role": "user", "content": "1+1=?"},
        ]});
        let prompt = template.render(&plain).unwrap();
        println!("---- 纯文本多轮 ----\n{prompt}");
        assert!(prompt.starts_with("<bos>"), "应以 bos_token 开头");
        assert!(prompt.contains("<|turn>system\n"));
        assert!(prompt.contains("<|turn>user\n你好<turn|>"));
        assert!(prompt.contains("<|turn>model\n"));
        assert!(prompt.ends_with("<|turn>model\n<|channel>thought\n<channel|>"), "add_generation_prompt 应收尾");

        // 带 tools + tool_calls 一轮完整调用。
        let with_tools = serde_json::json!({
            "messages": [
                {"role": "user", "content": "北京天气怎么样？"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "北京"}}},
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "晴，25°C"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询城市天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "城市名"}},
                        "required": ["city"],
                    },
                },
            }],
        });
        let prompt = template.render(&with_tools).unwrap();
        println!("---- 带 tools ----\n{prompt}");
        assert!(prompt.contains("<|tool>declaration:get_weather"), "tools 应渲染为声明块");
        assert!(prompt.contains("<|tool_call>call:get_weather{city:<|\"|>北京<|\"|>}<tool_call|>"));
        assert!(prompt.contains("<|tool_response>response:get_weather{value:"));
    }
}
