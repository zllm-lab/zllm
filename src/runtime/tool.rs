//! 模型无关的工具协议：工具 schema 注入、原生 dialect 文本编解码与流式解析。
//!
//! API 层统一使用 OpenAI function tools / JSON Schema；模型只选择训练时使用的
//! 文本 dialect。backend 不感知工具名、XML/DSML 或 JSON Schema。

use std::collections::HashMap;

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
use std::sync::Arc;

use serde_json::{Value, json};

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
use crate::{
    backend::TokenFence,
    runtime::{
        generation_guard::TokenFenceProgram,
        json_fence::{JsonSchemaFence, JsonTokenTable},
    },
    tokenizer::Tokenizer,
};

/// 输出协议解析前不能丢弃 special token：工具和思考边界也可能在词表中
/// 标为 special。所有模型共用此入口，仅放行已支持协议的标记，角色/EOS
/// 等控制 token 仍隐藏；JSON 文法候选等非输出用途继续用原始 decoder API。
pub fn decode_output_token(detokenizer: &crate::tokenizer::Detokenizer, token: u32) -> std::io::Result<Vec<u8>> {
    let visible = detokenizer.decode_bytes(&[token], true)?;
    if !visible.is_empty() {
        return Ok(visible);
    }
    let raw = detokenizer.decode_bytes(&[token], false)?;
    let protocol = std::str::from_utf8(&raw).is_ok_and(|text| {
        matches!(text, "<tool_call>" | "</tool_call>" | "<arg_key>" | "</arg_key>" | "<arg_value>" | "</arg_value>" | "<think>" | "</think>")
            || text.starts_with("<｜DSML｜") || text.starts_with("</｜DSML｜")
    });
    Ok(if protocol { raw } else { visible })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolDialect {
    GlmXml,
    ChatmlJson,
    DeepseekDsml,
    /// HTTP 输出兜底同时识别已支持的原生 dialect。
    Auto,
}

impl ToolDialect {
    /// 同一种原生协议统一描述输入说明、历史重放、流式解析和完整输出拆分。
    pub fn instructions(self, tools: &[Value], tool_choice: Option<&Value>) -> Result<Option<String>, String> {
        let functions = validated_tool_functions(tools, tool_choice)?;
        if tools.is_empty() || tool_choice.and_then(Value::as_str) == Some("none") {
            return Ok(None);
        }
        let schemas = functions.into_iter().map(python_json).collect::<Result<Vec<_>, _>>()?.join("\n");
        let mut instructions = match self {
            Self::GlmXml | Self::Auto => format!(
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nAvailable function schemas:\n<tools>\n{schemas}\n</tools>\n\nFor each function call, output exactly:\n<tool_call>{{function-name}}<arg_key>{{argument-name}}</arg_key><arg_value>{{argument-value}}</arg_value>...</tool_call>\nString values are emitted as plain text; numbers, booleans, arrays and objects are emitted as JSON."
            ),
            Self::ChatmlJson => {
                format!(
                    "# Tools\n\nYou may call one or more functions. Available function schemas:\n<tools>\n{schemas}\n</tools>\n\nReturn each call as JSON inside <tool_call></tool_call>:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
                )
            }
            Self::DeepseekDsml => format!(
                "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n### Available Tool Schemas\n\n{schemas}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
            ),
        };
        match tool_choice {
            Some(Value::String(choice)) if choice == "required" => instructions.push_str("\nYou must call at least one provided function."),
            Some(Value::Object(choice)) => {
                if let Some(name) = choice.get("function").and_then(|function| function.get("name")).and_then(Value::as_str) {
                    instructions.push_str("\nYou must call the function `");
                    instructions.push_str(name);
                    instructions.push_str("`.");
                }
            }
            _ => {}
        }
        Ok(Some(instructions))
    }

    pub fn render_history(self, value: Option<&Value>) -> Result<String, String> {
        let calls = match value {
            None | Some(Value::Null) => return Ok(String::new()),
            Some(Value::Array(calls)) => calls,
            Some(_) => return Err("assistant.tool_calls 必须是数组".to_owned()),
        };
        match self {
            Self::GlmXml | Self::Auto => render_xml_history(calls),
            Self::ChatmlJson => render_json_history(calls),
            Self::DeepseekDsml => render_dsml_history(calls),
        }
    }

    pub fn render_result(self, content: &str) -> String {
        match self {
            Self::GlmXml | Self::ChatmlJson | Self::Auto => format!("<tool_response>{content}</tool_response>"),
            Self::DeepseekDsml => format!("<tool_result>{content}</tool_result>"),
        }
    }

    pub fn stream(self, schemas: HashMap<String, Value>) -> ToolCallStream {
        ToolCallStream::new(self, schemas)
    }

    pub fn split_output(self, text: &str, schemas: &HashMap<String, Value>) -> (String, Vec<ParsedToolCall>) {
        let mut stream = self.stream(schemas.clone());
        let mut visible = String::new();
        let mut calls = Vec::new();
        for output in stream.push(text).into_iter().chain(stream.finish()) {
            match output {
                ToolOutput::Text(text) => visible.push_str(&text),
                ToolOutput::ToolCall(call) => calls.push(call),
            }
        }
        (visible, calls)
    }
}

/// transport validator 之外的统一请求边界：库模式和直接 runtime 调用也必须拒绝
/// 非数组 tools，不能把坏输入静默解释成“未启用工具”。
pub fn request_tools(request: &Value) -> Result<&[Value], String> {
    match request.get("tools") {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(tools)) => Ok(tools),
        Some(_) => Err("tools 必须是数组".to_owned()),
    }
}

/// embedded 调用不经过 HTTP validator，工具定义和选择语义必须在 runtime 边界
/// 再守一次；否则坏 schema 会被 filter_map 静默丢掉，named choice 也可能指向不存在的工具。
fn validated_tool_functions<'a>(tools: &'a [Value], tool_choice: Option<&Value>) -> Result<Vec<&'a Value>, String> {
    let mut functions = Vec::with_capacity(tools.len());
    let mut names = HashMap::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(format!("tools[{index}].type 必须是 function"));
        }
        let function = tool.get("function").filter(|value| value.is_object()).ok_or_else(|| format!("tools[{index}].function 必须是对象"))?;
        let name = function.get("name").and_then(Value::as_str).filter(|name| !name.trim().is_empty()).ok_or_else(|| format!("tools[{index}].function.name 不能为空"))?;
        if let Some(previous) = names.insert(name, index) {
            return Err(format!("tools[{index}].function.name 与 tools[{previous}] 重复: {name}"));
        }
        functions.push(function);
    }
    match tool_choice {
        None | Some(Value::Null) => {}
        Some(Value::String(choice)) if matches!(choice.as_str(), "none" | "auto") => {}
        Some(Value::String(choice)) if choice == "required" && !functions.is_empty() => {}
        Some(Value::String(choice)) if choice == "required" => return Err("tool_choice=required 时 tools 不能为空".to_owned()),
        Some(Value::Object(choice)) if choice.get("type").and_then(Value::as_str) == Some("function") => {
            let name = choice.get("function").and_then(Value::as_object).and_then(|function| function.get("name")).and_then(Value::as_str).ok_or("tool_choice.function.name 必须是字符串")?;
            if !names.contains_key(name) {
                return Err(format!("tool_choice 指定的 function {name} 不在 tools 中"));
            }
        }
        _ => return Err("tool_choice 必须是 none/auto/required 或指定 function 的对象".to_owned()),
    }
    Ok(functions)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

/// 工具调用 id 在一次响应内保持稳定，并通过请求级 scope 避免跨轮复用。
pub fn tool_call_id(scope: &str, index: usize, name: &str) -> String {
    let identity = format!("{scope}\0{index}\0{name}");
    let digest = blake3::hash(identity.as_bytes()).to_hex().to_string();
    format!("call_{}", &digest[..24])
}

/// 没有 transport request_id 的嵌入式调用按模型与完整请求生成稳定 scope。
pub fn request_tool_scope(model: &str, request: &Value) -> String {
    let identity = format!("{model}\0{request}");
    let digest = blake3::hash(identity.as_bytes()).to_hex().to_string();
    format!("{model}_{}", &digest[..24])
}

pub enum ToolOutput {
    Text(String),
    ToolCall(ParsedToolCall),
}

/// 请求级工具流：在 dialect parser 之上补齐稳定调用 id 与公共 API 结果。
/// 这层只依赖结构化 request，不属于任何具体模型或 backend。
pub struct RequestToolCallStream {
    inner: ToolCallStream,
    scope: String,
    pub calls: Vec<super::session::ToolCall>,
}

impl RequestToolCallStream {
    pub fn new(request: &Value, scope: &str, dialect: ToolDialect) -> Self {
        let schemas = if request.get("tool_choice").and_then(Value::as_str) == Some("none") { HashMap::new() } else { request.get("tools").and_then(Value::as_array).map_or_else(HashMap::new, |tools| tool_argument_schemas(tools)) };
        Self { inner: dialect.stream(schemas), scope: scope.to_owned(), calls: Vec::new() }
    }

    pub fn push(&mut self, text: &str, mut emit: impl FnMut(String) -> bool) -> bool {
        for output in self.inner.push(text) {
            match output {
                ToolOutput::Text(text) => {
                    if !emit(text) {
                        return false;
                    }
                }
                ToolOutput::ToolCall(call) => self.push_call(call),
            }
        }
        true
    }

    pub fn finish(&mut self, mut emit: impl FnMut(String) -> bool) -> bool {
        for output in self.inner.finish() {
            match output {
                ToolOutput::Text(text) => {
                    if !emit(text) {
                        return false;
                    }
                }
                ToolOutput::ToolCall(call) => self.push_call(call),
            }
        }
        true
    }

    fn push_call(&mut self, call: ParsedToolCall) {
        let id = tool_call_id(&self.scope, self.calls.len(), &call.name);
        self.calls.push(super::session::ToolCall { id, kind: "function".to_owned(), function: super::session::ToolFunction { name: call.name, arguments: call.arguments } });
    }
}

pub fn emit_request_tool_chunk<T: Copy>(stream: &mut RequestToolCallStream, token_id: T, chunk: &str, response: &mut String, on_token: &mut (impl FnMut(T, String) -> bool + ?Sized)) -> bool {
    stream.push(chunk, |visible| {
        if !on_token(token_id, visible.clone()) {
            return false;
        }
        response.push_str(&visible);
        true
    })
}

pub fn finish_request_tool_stream<T: Copy>(stream: &mut RequestToolCallStream, token_id: T, response: &mut String, on_token: &mut (impl FnMut(T, String) -> bool + ?Sized)) -> bool {
    stream.finish(|visible| {
        if !on_token(token_id, visible.clone()) {
            return false;
        }
        response.push_str(&visible);
        true
    })
}

pub fn tool_argument_schemas(tools: &[Value]) -> HashMap<String, Value> {
    tools
        .iter()
        .filter_map(|tool| {
            let function = tool.get("function")?;
            Some((function.get("name")?.as_str()?.to_owned(), function.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object"}))))
        })
        .collect()
}

/// 指定 function 时可在请求边界把 JSON Schema 编译成原生 DSML 围栏。auto
/// 仍保留模型选择能力并由流式 parser 处理，避免协议层擅自替模型选择工具。
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsmlNamedToolSpec {
    name: String,
    schema: Value,
    parameters: Vec<DsmlParameterSpec>,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct DsmlParameterSpec {
    name: String,
    string: bool,
    required: bool,
    schema: Value,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DsmlToolSpec {
    Named(DsmlNamedToolSpec),
    Auto(Vec<DsmlNamedToolSpec>),
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
pub fn dsml_tool_spec(tools: &[Value], tool_choice: Option<&Value>) -> Result<Option<DsmlToolSpec>, String> {
    if tools.is_empty() || tool_choice.and_then(Value::as_str) == Some("none") {
        return Ok(None);
    }
    let named = tool_choice.and_then(Value::as_object).and_then(|choice| choice.get("function")).and_then(Value::as_object).and_then(|function| function.get("name")).and_then(Value::as_str);
    if let Some(name) = named {
        return dsml_named_spec(tools, name).map(|spec| Some(DsmlToolSpec::Named(spec)));
    }
    let candidates = tools.iter().filter_map(|tool| tool.get("function").and_then(|function| function.get("name")).and_then(Value::as_str)).map(|name| dsml_named_spec(tools, name)).collect::<Result<Vec<_>, _>>()?;
    Ok((!candidates.is_empty()).then_some(DsmlToolSpec::Auto(candidates)))
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
fn dsml_named_spec(tools: &[Value], name: &str) -> Result<DsmlNamedToolSpec, String> {
    let function = tools
        .iter()
        .filter_map(|tool| tool.get("function"))
        .find(|function| function.get("name").and_then(Value::as_str).is_some_and(|candidate| candidate.eq_ignore_ascii_case(name)))
        .ok_or_else(|| format!("tool_choice 指定的 function {name} 不在 tools 中"))?;
    let name = function.get("name").and_then(Value::as_str).expect("已匹配的 function 必须有 name");
    if name.contains(['"', '<', '>']) {
        return Err(format!("function name {name:?} 不能安全编码为 DSML 属性"));
    }
    let schema = function.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object"}));
    let schema_object = schema.as_object();
    let properties = schema_object.and_then(|schema| schema.get("properties")).and_then(Value::as_object);
    let required = schema_object.and_then(|schema| schema.get("required")).and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
    for key in required {
        let key = key.as_str().ok_or_else(|| format!("function {name} required 项必须是字符串"))?;
        if properties.is_none_or(|properties| !properties.contains_key(key)) {
            return Err(format!("function {name} required parameter {key} 缺少 properties 定义"));
        }
    }
    let parameters = properties
        .into_iter()
        .flatten()
        .map(|(key, property)| {
            if key.contains(['"', '<', '>']) {
                return Err(format!("function {name} parameter {key:?} 不能安全编码为 DSML 属性"));
            }
            Ok(DsmlParameterSpec { name: key.to_owned(), string: property.get("type").and_then(Value::as_str) == Some("string"), required: required.iter().any(|required| required.as_str() == Some(key)), schema: property.clone() })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(DsmlNamedToolSpec { name: name.to_owned(), schema, parameters })
}

/// DSML named-tool 的 token 文法。固定标签、工具名和参数名都在采样前收窄
/// 候选 token；参数只能出现一次，required 齐全后才允许闭合 invoke。
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Clone)]
pub struct DsmlNamedToolFence {
    parameter_opens: Vec<Vec<u32>>,
    parameter_tail: Vec<u32>,
    final_close: Vec<u32>,
    values: Vec<Option<JsonSchemaFence>>,
    string_parameters: Vec<bool>,
    required: Vec<bool>,
    seen: Vec<bool>,
    close_token: u32,
    vocab_size: u32,
    terminal_tokens: Arc<[u32]>,
    markup_tokens: Arc<[u32]>,
    state: DsmlFenceState,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Clone, Debug, PartialEq, Eq)]
enum DsmlLiteralNext {
    ChooseParameter,
    Value(usize),
    Done,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Clone)]
struct DsmlLiteral {
    tokens: Vec<u32>,
    next: DsmlLiteralNext,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Clone)]
enum DsmlFenceState {
    Literal { candidates: Vec<DsmlLiteral>, offset: usize },
    Value(usize),
    Done,
    Broken,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Clone)]
pub enum DsmlToolFence {
    Named(DsmlNamedToolFence),
    Auto { trigger: Vec<u32>, trigger_offset: usize, candidates: Vec<DsmlNamedToolFence>, vocab_size: u32, active: bool },
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
impl DsmlToolFence {
    pub fn new(tokenizer: &Tokenizer, json_tokens: Arc<JsonTokenTable>, vocab_size: usize, terminal_tokens: &[u32], spec: DsmlToolSpec) -> Result<Self, String> {
        let markup_tokens: Arc<[u32]> = json_tokens.tokens_containing(b'<').into();
        match spec {
            DsmlToolSpec::Named(spec) => DsmlNamedToolFence::new(tokenizer, json_tokens, vocab_size, terminal_tokens, markup_tokens, spec).map(Self::Named),
            DsmlToolSpec::Auto(specs) => {
                let vocab_size = u32::try_from(vocab_size).map_err(|_| format!("DSML vocab_size={vocab_size} 超出 u32"))?;
                let trigger = tokenizer.tokenize("<｜DSML｜".as_bytes());
                if trigger.is_empty() {
                    return Err("DSML auto 围栏 trigger 不能为空".to_owned());
                }
                let candidates = specs.into_iter().map(|spec| DsmlNamedToolFence::new(tokenizer, json_tokens.clone(), vocab_size as usize, terminal_tokens, markup_tokens.clone(), spec)).collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Auto { trigger, trigger_offset: 0, candidates, vocab_size, active: false })
            }
        }
    }
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
impl TokenFenceProgram for DsmlToolFence {
    fn fence(&self) -> TokenFence {
        match self {
            Self::Named(fence) => fence.fence(),
            Self::Auto { candidates, vocab_size, active: true, .. } => {
                if candidates.len() == 1 {
                    return candidates[0].fence();
                }
                let fences = candidates.iter().map(TokenFenceProgram::fence).collect::<Vec<_>>();
                if fences.iter().any(TokenFence::is_open) {
                    return TokenFence::default();
                }
                let mut allowed = candidates.iter().filter_map(|candidate| candidate.fence().forced()).collect::<Vec<_>>();
                if allowed.len() != candidates.len() {
                    return TokenFence::excluding((0..*vocab_size).filter(|token| fences.iter().all(|fence| fence.forced().is_some_and(|forced| forced != *token) || fence.excluded().binary_search(token).is_ok())));
                }
                allowed.sort_unstable();
                allowed.dedup();
                if allowed.len() == 1 {
                    return TokenFence::forcing(allowed[0]);
                }
                TokenFence::excluding((0..*vocab_size).filter(|token| allowed.binary_search(token).is_err()))
            }
            Self::Auto { .. } => TokenFence::default(),
        }
    }

    fn advance(&mut self, token: u32) {
        match self {
            Self::Named(fence) => fence.advance(token),
            Self::Auto { trigger, trigger_offset, candidates, active, .. } if !*active => {
                if token == trigger[*trigger_offset] {
                    *trigger_offset += 1;
                    if *trigger_offset == trigger.len() {
                        for candidate in candidates.iter_mut() {
                            for &trigger_token in trigger.iter() {
                                candidate.advance(trigger_token);
                            }
                        }
                        *active = true;
                    }
                } else {
                    *trigger_offset = usize::from(token == trigger[0]);
                }
            }
            Self::Auto { candidates, active: true, .. } => {
                candidates.retain_mut(|candidate| {
                    let fence = candidate.fence();
                    if fence.forced().is_some_and(|expected| expected != token) || fence.excluded().binary_search(&token).is_ok() {
                        return false;
                    }
                    candidate.advance(token);
                    true
                });
            }
            Self::Auto { .. } => {}
        }
    }
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
impl DsmlNamedToolFence {
    pub fn new(tokenizer: &Tokenizer, json_tokens: Arc<JsonTokenTable>, vocab_size: usize, terminal_tokens: &[u32], markup_tokens: Arc<[u32]>, spec: DsmlNamedToolSpec) -> Result<Self, String> {
        let vocab_size = u32::try_from(vocab_size).map_err(|_| format!("DSML vocab_size={vocab_size} 超出 u32"))?;
        let close_token = tokenizer.tokenize(b"</").into_iter().next().ok_or("DSML tokenizer 无法编码 </")?;
        let prefix = tokenizer.tokenize(format!("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"{}\">\n", spec.name).as_bytes());
        let parameter_opens = spec.parameters.iter().map(|parameter| tokenizer.tokenize(format!("<｜DSML｜parameter name=\"{}\" string=\"{}\">", parameter.name, parameter.string).as_bytes())).collect::<Vec<_>>();
        let parameter_tail = dsml_suffix_tokens(tokenizer, "</｜DSML｜parameter>\n", close_token, 0)?;
        let final_close = tokenizer.tokenize("</｜DSML｜invoke>\n</｜DSML｜tool_calls>".as_bytes());
        if prefix.is_empty() || parameter_tail.is_empty() || final_close.is_empty() || parameter_opens.iter().any(Vec::is_empty) {
            return Err("DSML named-tool 围栏包含空 literal".to_owned());
        }
        let values = spec
            .parameters
            .iter()
            .map(|parameter| {
                if parameter.string { JsonSchemaFence::new_raw_string(&parameter.schema, &spec.schema, json_tokens.clone()) } else { JsonSchemaFence::new_with_root(&parameter.schema, &spec.schema, json_tokens.clone()) }.map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let string_parameters = spec.parameters.iter().map(|parameter| parameter.string).collect::<Vec<_>>();
        let required = spec.parameters.iter().map(|parameter| parameter.required).collect::<Vec<_>>();
        let seen = vec![false; spec.parameters.len()];
        Ok(Self {
            parameter_opens,
            parameter_tail,
            final_close,
            values,
            string_parameters,
            required,
            seen,
            close_token,
            vocab_size,
            terminal_tokens: terminal_tokens.to_vec().into(),
            markup_tokens,
            state: DsmlFenceState::Literal { candidates: vec![DsmlLiteral { tokens: prefix, next: DsmlLiteralNext::ChooseParameter }], offset: 0 },
        })
    }

    fn choose_parameter(&mut self) {
        let mut candidates = self.parameter_opens.iter().enumerate().filter(|(index, _)| !self.seen[*index]).map(|(index, tokens)| DsmlLiteral { tokens: tokens.clone(), next: DsmlLiteralNext::Value(index) }).collect::<Vec<_>>();
        if self.required.iter().zip(&self.seen).all(|(required, seen)| !required || *seen) {
            candidates.push(DsmlLiteral { tokens: self.final_close.clone(), next: DsmlLiteralNext::Done });
        }
        self.state = if candidates.is_empty() { DsmlFenceState::Broken } else { DsmlFenceState::Literal { candidates, offset: 0 } };
    }

    fn enter_next(&mut self, next: DsmlLiteralNext) {
        match next {
            DsmlLiteralNext::ChooseParameter => self.choose_parameter(),
            DsmlLiteralNext::Value(parameter) => {
                self.seen[parameter] = true;
                self.state = DsmlFenceState::Value(parameter);
            }
            DsmlLiteralNext::Done => self.state = DsmlFenceState::Done,
        }
    }
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
fn dsml_suffix_tokens(tokenizer: &Tokenizer, text: &str, close_token: u32, segment: usize) -> Result<Vec<u32>, String> {
    let mut tokens = tokenizer.tokenize(text.as_bytes());
    if tokens.first().copied() != Some(close_token) {
        return Err(format!("DSML segment={segment} close token 与 tokenizer 边界不一致"));
    }
    tokens.remove(0);
    Ok(tokens)
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
impl TokenFenceProgram for DsmlNamedToolFence {
    fn fence(&self) -> TokenFence {
        let fence = match &self.state {
            DsmlFenceState::Done | DsmlFenceState::Broken => return TokenFence::default(),
            DsmlFenceState::Value(parameter) => {
                self.values[*parameter].as_ref().map(|json| json.fence(self.close_token)).unwrap_or_else(|| TokenFence::excluding(self.markup_tokens.iter().copied().filter(|token| *token != self.close_token)))
            }
            DsmlFenceState::Literal { candidates, offset } => {
                let mut allowed = candidates.iter().filter_map(|candidate| candidate.tokens.get(*offset)).copied().collect::<Vec<_>>();
                allowed.sort_unstable();
                allowed.dedup();
                if allowed.len() == 1 { TokenFence::forcing(allowed[0]) } else { TokenFence::excluding((0..self.vocab_size).filter(|token| allowed.binary_search(token).is_err())) }
            }
        };
        let string_parameter = match &self.state {
            DsmlFenceState::Value(parameter) => self.string_parameters[*parameter],
            _ => false,
        };
        TokenFence::constrained(
            fence.forced(),
            fence.excluded().iter().copied().chain(self.terminal_tokens.iter().copied()).chain(string_parameter.then_some(()).into_iter().flat_map(|_| self.markup_tokens.iter().copied().filter(|token| *token != self.close_token))),
        )
    }

    fn advance(&mut self, token: u32) {
        match &mut self.state {
            DsmlFenceState::Done | DsmlFenceState::Broken => {}
            DsmlFenceState::Value(parameter) => {
                let parameter = *parameter;
                let complete = self.values[parameter].as_ref().is_none_or(JsonSchemaFence::complete);
                if token == self.close_token && complete {
                    self.state = DsmlFenceState::Literal { candidates: vec![DsmlLiteral { tokens: self.parameter_tail.clone(), next: DsmlLiteralNext::ChooseParameter }], offset: 0 };
                } else if let Some(json) = &mut self.values[parameter] {
                    json.advance(token);
                }
            }
            DsmlFenceState::Literal { candidates, offset } => {
                candidates.retain(|candidate| candidate.tokens.get(*offset).copied() == Some(token));
                if candidates.is_empty() {
                    self.state = DsmlFenceState::Broken;
                    return;
                }
                *offset += 1;
                let complete = candidates.iter().filter(|candidate| candidate.tokens.len() == *offset).map(|candidate| candidate.next.clone()).collect::<Vec<_>>();
                if !complete.is_empty() {
                    if complete.len() != candidates.len() || complete.iter().any(|next| next != &complete[0]) {
                        self.state = DsmlFenceState::Broken;
                    } else {
                        self.enter_next(complete[0].clone());
                    }
                }
            }
        }
    }
}

/// DeepSeek 官方 encoder 使用 Python `json.dumps(..., ensure_ascii=False)`；其
/// 逗号和冒号后的空格也是 prompt 的一部分，不能换成 serde_json 的紧凑格式。
fn python_json(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("null".to_owned()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value).map_err(|error| format!("序列化 tool 字符串: {error}")),
        Value::Array(values) => values.iter().map(python_json).collect::<Result<Vec<_>, _>>().map(|values| format!("[{}]", values.join(", "))),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| Ok(format!("{}: {}", serde_json::to_string(key).map_err(|error| format!("序列化 tool key: {error}"))?, python_json(value)?)))
            .collect::<Result<Vec<_>, String>>()
            .map(|values| format!("{{{}}}", values.join(", "))),
    }
}

fn render_xml_history(calls: &[Value]) -> Result<String, String> {
    let mut output = String::new();
    for call in calls {
        let function = call.get("function").and_then(Value::as_object).ok_or("assistant.tool_calls.function 必须是对象")?;
        let name = function.get("name").and_then(Value::as_str).ok_or("assistant.tool_calls.function.name 必须是字符串")?;
        let arguments = parse_arguments(function.get("arguments"))?;
        output.push_str("<tool_call>");
        output.push_str(name);
        for (key, value) in arguments {
            output.push_str("<arg_key>");
            output.push_str(&key);
            output.push_str("</arg_key><arg_value>");
            match value {
                Value::String(value) => output.push_str(&value),
                value => output.push_str(&serde_json::to_string(&value).map_err(|error| format!("序列化 tool argument {key}: {error}"))?),
            }
            output.push_str("</arg_value>");
        }
        output.push_str("</tool_call>");
    }
    Ok(output)
}

fn render_json_history(calls: &[Value]) -> Result<String, String> {
    let mut output = String::new();
    for call in calls {
        let function = call.get("function").and_then(Value::as_object).ok_or("assistant.tool_calls.function 必须是对象")?;
        let name = function.get("name").and_then(Value::as_str).ok_or("assistant.tool_calls.function.name 必须是字符串")?;
        let arguments = match function.get("arguments") {
            Some(Value::String(arguments)) => serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.clone())),
            Some(arguments) => arguments.clone(),
            None => json!({}),
        };
        output.push_str("\n<tool_call>\n");
        output.push_str(&serde_json::to_string(&json!({"name": name, "arguments": arguments})).map_err(|error| format!("序列化历史 tool_call: {error}"))?);
        output.push_str("\n</tool_call>");
    }
    Ok(output)
}

fn render_dsml_history(calls: &[Value]) -> Result<String, String> {
    let mut invokes = Vec::with_capacity(calls.len());
    for call in calls {
        let function = call.get("function").and_then(Value::as_object).ok_or("assistant.tool_calls.function 必须是对象")?;
        let name = function.get("name").and_then(Value::as_str).ok_or("assistant.tool_calls.function.name 必须是字符串")?;
        let arguments = parse_arguments(function.get("arguments"))?;
        let mut invoke = format!("<｜DSML｜invoke name=\"{name}\">\n");
        for (key, value) in arguments {
            let string = value.is_string();
            let value = match value {
                Value::String(value) => value,
                value => serde_json::to_string(&value).map_err(|error| format!("序列化 tool argument {key}: {error}"))?,
            };
            invoke.push_str(&format!("<｜DSML｜parameter name=\"{key}\" string=\"{string}\">{value}</｜DSML｜parameter>\n"));
        }
        invoke.push_str("</｜DSML｜invoke>");
        invokes.push(invoke);
    }
    Ok(format!("<｜DSML｜tool_calls>\n{}\n</｜DSML｜tool_calls>", invokes.join("\n")))
}

fn parse_arguments(value: Option<&Value>) -> Result<Vec<(String, Value)>, String> {
    let value = match value {
        Some(Value::String(arguments)) => serde_json::from_str(arguments).map_err(|error| format!("assistant tool arguments 不是 JSON: {error}"))?,
        Some(value) => value.clone(),
        None => json!({}),
    };
    value.as_object().map(|arguments| arguments.iter().map(|(key, value)| (key.clone(), value.clone())).collect()).ok_or("assistant tool arguments 必须是 JSON 对象".to_owned())
}

const XML_TOOL_CALL_OPEN: &str = "<tool_call>";
const XML_TOOL_CALL_CLOSE: &str = "</tool_call>";
const XML_ARG_KEY_OPEN: &str = "<arg_key>";
const XML_ARG_KEY_CLOSE: &str = "</arg_key>";
const XML_ARG_VALUE_OPEN: &str = "<arg_value>";
const XML_ARG_VALUE_CLOSE: &str = "</arg_value>";
const DSML_TOOL_CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
const DSML_TOOL_CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";
const DSML_TAG_PREFIX: &str = "<｜DSML｜";
const DSML_CLOSE_TAG_PREFIX: &str = "</｜DSML｜";
const DSML_ESCAPED_TAG_PREFIX: &str = "\\<｜DSML｜";
const DSML_ESCAPED_CLOSE_TAG_PREFIX: &str = "\\</｜DSML｜";
const DSML_INVOKE_OPEN: &str = "<｜DSML｜invoke";
const DSML_INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const DSML_PARAMETER_OPEN: &str = "<｜DSML｜parameter";
const DSML_PARAMETER_CLOSE: &str = "</｜DSML｜parameter>";

fn find_ascii_case_insensitive(text: &str, pattern: &str) -> Option<usize> {
    text.as_bytes().windows(pattern.len()).position(|window| window.eq_ignore_ascii_case(pattern.as_bytes()))
}

fn find_dsml_tag_prefix(text: &str) -> Option<usize> {
    [DSML_TAG_PREFIX, DSML_CLOSE_TAG_PREFIX].into_iter().filter_map(|prefix| find_ascii_case_insensitive(text, prefix)).map(|start| start.checked_sub(1).filter(|&escaped| text.as_bytes()[escaped] == b'\\').unwrap_or(start)).min()
}

fn strip_prefix_ascii_case_insensitive<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.get(prefix.len()..).filter(|_| text.as_bytes().get(..prefix.len()).is_some_and(|value| value.eq_ignore_ascii_case(prefix.as_bytes())))
}

fn strip_suffix_ascii_case_insensitive<'a>(text: &'a str, suffix: &str) -> Option<&'a str> {
    let start = text.len().checked_sub(suffix.len())?;
    text.get(..start).filter(|_| text.as_bytes().get(start..).is_some_and(|value| value.eq_ignore_ascii_case(suffix.as_bytes())))
}

fn is_ascii_case_insensitive_prefix(text: &str, value: &str) -> bool {
    text.len() < value.len() && value.as_bytes().get(..text.len()).is_some_and(|prefix| text.as_bytes().eq_ignore_ascii_case(prefix))
}

fn strip_dsml_close_tag_prefix(text: &str) -> Option<(usize, &str)> {
    let (escaped, text) = text.strip_prefix('\\').map_or((0, text), |text| (1, text));
    strip_prefix_ascii_case_insensitive(text, DSML_CLOSE_TAG_PREFIX).map(|rest| (escaped + DSML_CLOSE_TAG_PREFIX.len(), rest))
}

fn dsml_close_tag_end(text: &str) -> Option<usize> {
    let (prefix_len, rest) = strip_dsml_close_tag_prefix(text)?;
    let name_end = rest.find('>')?;
    let name = &rest[..name_end];
    (!name.is_empty() && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')).then_some(prefix_len + name_end + 1)
}

fn is_dsml_tag_fragment(text: &str) -> bool {
    let text = text.strip_prefix('\\').unwrap_or(text);
    [DSML_TAG_PREFIX, DSML_CLOSE_TAG_PREFIX].into_iter().any(|prefix| is_ascii_case_insensitive_prefix(text, prefix) || strip_prefix_ascii_case_insensitive(text, prefix).is_some())
}

fn is_dsml_close_tag_fragment(text: &str) -> bool {
    let text = text.strip_prefix('\\').unwrap_or(text);
    is_ascii_case_insensitive_prefix(text, DSML_CLOSE_TAG_PREFIX) || strip_prefix_ascii_case_insensitive(text, DSML_CLOSE_TAG_PREFIX).is_some()
}

fn strip_dsml_invoke_open(text: &str) -> Option<&str> {
    strip_prefix_ascii_case_insensitive(text, DSML_INVOKE_OPEN)
}

fn find_dsml_invoke_open(text: &str) -> Option<usize> {
    find_ascii_case_insensitive(text, DSML_INVOKE_OPEN)
}

fn is_dsml_invoke_prefix(text: &str) -> bool {
    is_ascii_case_insensitive_prefix(text, DSML_INVOKE_OPEN)
}

fn encoded_argument(parameters: Option<&Value>, key: &str, raw: &str) -> Option<String> {
    let declared = parameter_schema(parameters, key).and_then(|schema| schema.get("type")).and_then(Value::as_str);
    if declared.is_some_and(|kind| kind != "string") && serde_json::from_str::<Value>(raw).is_ok() {
        return Some(raw.to_owned());
    }
    serde_json::to_string(raw).ok()
}

fn schema_entry<'a>(schemas: &'a HashMap<String, Value>, name: &str) -> Option<(&'a str, &'a Value)> {
    schemas.iter().find(|(candidate, _)| candidate.eq_ignore_ascii_case(name)).map(|(name, schema)| (name.as_str(), schema))
}

fn parameter_schema<'a>(parameters: Option<&'a Value>, key: &str) -> Option<&'a Value> {
    parameters.and_then(|parameters| parameters.get("properties")).and_then(Value::as_object).and_then(|properties| properties.iter().find(|(candidate, _)| candidate.eq_ignore_ascii_case(key)).map(|(_, schema)| schema))
}

fn canonical_argument_name(parameters: Option<&Value>, key: &str) -> String {
    parameters.and_then(|parameters| parameters.get("properties")).and_then(Value::as_object).and_then(|properties| properties.keys().find(|candidate| candidate.eq_ignore_ascii_case(key))).cloned().unwrap_or_else(|| key.to_owned())
}

fn canonical_arguments(parameters: Option<&Value>, arguments: &serde_json::Map<String, Value>) -> Option<serde_json::Map<String, Value>> {
    let mut canonical = serde_json::Map::new();
    for (key, value) in arguments {
        let key = canonical_argument_name(parameters, key);
        if canonical.insert(key, value.clone()).is_some() {
            return None;
        }
    }
    Some(canonical)
}

fn arguments_match_schema(parameters: Option<&Value>, arguments: &serde_json::Map<String, Value>) -> bool {
    let Some(parameters) = parameters else { return true };
    if parameters.get("required").and_then(Value::as_array).is_some_and(|required| required.iter().filter_map(Value::as_str).any(|key| !arguments.contains_key(key))) {
        return false;
    }
    let properties = parameters.get("properties").and_then(Value::as_object);
    for (key, value) in arguments {
        let Some(schema) = properties.and_then(|properties| properties.get(key)) else {
            if parameters.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                return false;
            }
            continue;
        };
        let valid_type = match schema.get("type").and_then(Value::as_str) {
            Some("string") => value.is_string(),
            Some("integer") => value.as_i64().is_some() || value.as_u64().is_some(),
            Some("number") => value.is_number(),
            Some("boolean") => value.is_boolean(),
            Some("array") => value.is_array(),
            Some("object") => value.is_object(),
            Some("null") => value.is_null(),
            Some(_) | None => true,
        };
        if !valid_type || schema.get("enum").and_then(Value::as_array).is_some_and(|values| !values.contains(value)) {
            return false;
        }
    }
    true
}

pub fn parse_xml_tool_call(text: &str, schemas: &HashMap<String, Value>) -> Option<ParsedToolCall> {
    let payload = strip_suffix_ascii_case_insensitive(strip_prefix_ascii_case_insensitive(text, XML_TOOL_CALL_OPEN)?, XML_TOOL_CALL_CLOSE)?;
    let name_end = find_ascii_case_insensitive(payload, XML_ARG_KEY_OPEN).unwrap_or(payload.len());
    let parsed_name = payload[..name_end].trim();
    if parsed_name.is_empty() {
        return None;
    }
    let (name, parameters) = if schemas.is_empty() { (parsed_name, None) } else { schema_entry(schemas, parsed_name).map(|(name, schema)| (name, Some(schema)))? };
    let mut arguments = serde_json::Map::new();
    let mut rest = &payload[name_end..];
    while !rest.is_empty() {
        let key = strip_prefix_ascii_case_insensitive(rest, XML_ARG_KEY_OPEN)?;
        let key_end = find_ascii_case_insensitive(key, XML_ARG_KEY_CLOSE)?;
        let key_name = canonical_argument_name(parameters, key[..key_end].trim());
        rest = &key[key_end + XML_ARG_KEY_CLOSE.len()..];
        let value = strip_prefix_ascii_case_insensitive(rest, XML_ARG_VALUE_OPEN)?;
        let value_end = find_ascii_case_insensitive(value, XML_ARG_VALUE_CLOSE)?;
        let raw = value[..value_end].trim();
        rest = &value[value_end + XML_ARG_VALUE_CLOSE.len()..];
        if key_name.is_empty() || arguments.contains_key(&key_name) {
            return None;
        }
        arguments.insert(key_name.clone(), serde_json::from_str(&encoded_argument(parameters, &key_name, raw)?).ok()?);
    }
    arguments_match_schema(parameters, &arguments).then(|| ParsedToolCall { name: name.to_owned(), arguments: Value::Object(arguments).to_string() })
}

fn parse_json_tool_call(text: &str, schemas: &HashMap<String, Value>) -> Option<ParsedToolCall> {
    let payload = text.strip_prefix(XML_TOOL_CALL_OPEN)?.strip_suffix(XML_TOOL_CALL_CLOSE)?.trim();
    let value = serde_json::from_str::<Value>(payload).ok()?;
    let object = value.as_object()?;
    let function = object.get("function").and_then(Value::as_object).unwrap_or(object);
    let name = function.get("name").and_then(Value::as_str)?.trim();
    let schema = schemas.get(name)?;
    let arguments = match function.get("arguments").or_else(|| function.get("rguments")) {
        Some(Value::String(arguments)) => serde_json::from_str::<Value>(arguments).ok()?,
        Some(arguments) => arguments.clone(),
        None => json!({}),
    };
    let arguments = arguments.as_object()?;
    arguments_match_schema(schema.get("parameters"), arguments).then(|| ParsedToolCall { name: name.to_owned(), arguments: Value::Object(arguments.clone()).to_string() })
}

fn parse_dsml_invoke(text: &str, schemas: &HashMap<String, Value>) -> Option<ParsedToolCall> {
    let (parsed_name, body) = dsml_invoke_parts(text)?;
    if parsed_name.is_empty() {
        return None;
    }
    let (name, parameters) = if schemas.is_empty() { (parsed_name, None) } else { schema_entry(schemas, parsed_name).map(|(name, schema)| (name, Some(schema)))? };
    let body = strip_suffix_ascii_case_insensitive(body, DSML_INVOKE_CLOSE)?.trim();
    if body.starts_with('{') {
        let arguments = serde_json::from_str::<Value>(body).ok()?;
        let arguments = canonical_arguments(parameters, arguments.as_object()?)?;
        return arguments_match_schema(parameters, &arguments).then(|| ParsedToolCall { name: name.to_owned(), arguments: Value::Object(arguments).to_string() });
    }
    parse_dsml_parameters(name, parameters, body)
}

fn dsml_invoke_parts(text: &str) -> Option<(&str, &str)> {
    let open_end = text.find('>')?;
    let header = text.get(..=open_end)?;
    let name = strip_dsml_invoke_open(header)?;
    let name = strip_prefix_ascii_case_insensitive(name.trim_start(), "name=\"")?.strip_suffix("\">")?;
    let body = text.get(open_end + 1..)?;
    Some((name, body))
}

fn parse_dsml_parameters(name: &str, parameters: Option<&Value>, body: &str) -> Option<ParsedToolCall> {
    let mut arguments = serde_json::Map::new();
    let mut rest = body;
    while !rest.is_empty() {
        rest = rest.trim_start();
        let after_open = strip_prefix_ascii_case_insensitive(rest, DSML_PARAMETER_OPEN)?.trim_start();
        let header_end = after_open.find('>')?;
        let attributes = &after_open[..header_end];
        let name_start = strip_prefix_ascii_case_insensitive(attributes, "name=\"")?;
        let name_end = name_start.find('"')?;
        let key = canonical_argument_name(parameters, &name_start[..name_end]);
        let string = strip_prefix_ascii_case_insensitive(name_start[name_end + 1..].trim(), "string=\"")?.strip_suffix('"')?;
        let value = &after_open[header_end + 1..];
        let value_end = find_ascii_case_insensitive(value, DSML_PARAMETER_CLOSE)?;
        let raw = &value[..value_end];
        rest = &value[value_end + DSML_PARAMETER_CLOSE.len()..];
        if key.is_empty() || arguments.contains_key(&key) {
            return None;
        }
        let value = if string.eq_ignore_ascii_case("true") { Value::String(raw.to_owned()) } else { serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned())) };
        arguments.insert(key, value);
    }
    arguments_match_schema(parameters, &arguments).then(|| ParsedToolCall { name: name.to_owned(), arguments: Value::Object(arguments).to_string() })
}

fn dsml_invoke_end(text: &str) -> Option<usize> {
    find_ascii_case_insensitive(text, DSML_INVOKE_CLOSE).map(|start| start + DSML_INVOKE_CLOSE.len())
}

pub struct ToolCallStream {
    pending: String,
    schemas: HashMap<String, Value>,
    dialect: ToolDialect,
    dsml_block_open: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XmlToolStreamEvent {
    Text(String),
    Delta { index: usize, id: Option<String>, name: Option<String>, arguments: String },
    ToolCall { index: usize, id: String, call: ParsedToolCall },
}

/// XML 工具协议的增量输出状态。解析、schema 类型转换和 marker buffering
/// 属于 dialect；server 只把事件映射成 API delta。
pub struct XmlToolCallStream {
    pending: String,
    schemas: HashMap<String, Value>,
    scope: String,
    calls: usize,
    streamed: Option<(String, usize)>,
    stream_deltas: bool,
}

impl XmlToolCallStream {
    pub fn new(scope: impl Into<String>, schemas: HashMap<String, Value>, stream_deltas: bool) -> Self {
        Self { pending: String::new(), schemas, scope: scope.into(), calls: 0, streamed: None, stream_deltas }
    }

    pub fn push(&mut self, text: &str) -> Vec<XmlToolStreamEvent> {
        self.pending.push_str(text);
        if self.schemas.is_empty() {
            return vec![XmlToolStreamEvent::Text(std::mem::take(&mut self.pending))];
        }
        let mut events = Vec::new();
        loop {
            if let Some(start) = self.pending.find(XML_TOOL_CALL_OPEN) {
                if start > 0 {
                    events.push(XmlToolStreamEvent::Text(self.pending[..start].to_owned()));
                    self.pending.drain(..start);
                    continue;
                }
                let Some(close) = self.pending[XML_TOOL_CALL_OPEN.len()..].find(XML_TOOL_CALL_CLOSE) else {
                    self.push_delta(&mut events);
                    break;
                };
                let close = XML_TOOL_CALL_OPEN.len() + close;
                let end = close + XML_TOOL_CALL_CLOSE.len();
                self.push_delta(&mut events);
                let raw = self.pending[..end].to_owned();
                self.pending.drain(..end);
                if let Some(call) = parse_xml_tool_call(&raw, &self.schemas) {
                    let index = self.calls;
                    let id = tool_call_id(&self.scope, index, &call.name);
                    self.calls += 1;
                    events.push(XmlToolStreamEvent::ToolCall { index, id, call });
                } else if self.streamed.is_none() {
                    events.push(XmlToolStreamEvent::Text(raw));
                }
                self.streamed = None;
                continue;
            }
            let keep = marker_prefix_len(&self.pending, ToolDialect::GlmXml);
            let visible = self.pending.len() - keep;
            if visible != 0 {
                events.push(XmlToolStreamEvent::Text(self.pending[..visible].to_owned()));
                self.pending.drain(..visible);
            }
            break;
        }
        events
    }

    pub fn finish(&mut self) -> Vec<XmlToolStreamEvent> {
        if self.pending.is_empty() || self.streamed.is_some() {
            self.pending.clear();
            return Vec::new();
        }
        vec![XmlToolStreamEvent::Text(std::mem::take(&mut self.pending))]
    }

    fn push_delta(&mut self, events: &mut Vec<XmlToolStreamEvent>) {
        if !self.stream_deltas || !self.pending.starts_with(XML_TOOL_CALL_OPEN) {
            return;
        }
        let payload = &self.pending[XML_TOOL_CALL_OPEN.len()..];
        let Some((name, arguments)) = partial_xml_tool_call(payload, &self.schemas) else { return };
        let index = self.calls;
        let (id, name_delta, arguments) = match self.streamed.as_mut() {
            Some((streamed_name, length)) => {
                if streamed_name != &name || arguments.len() < *length {
                    return;
                }
                let delta = arguments[*length..].to_owned();
                *length = arguments.len();
                (None, None, delta)
            }
            None => {
                self.streamed = Some((name.clone(), arguments.len()));
                (Some(tool_call_id(&self.scope, index, &name)), Some(name), arguments)
            }
        };
        if id.is_some() || !arguments.is_empty() {
            events.push(XmlToolStreamEvent::Delta { index, id, name: name_delta, arguments });
        }
    }
}

fn partial_xml_tool_call(payload: &str, schemas: &HashMap<String, Value>) -> Option<(String, String)> {
    let first_arg = payload.find(XML_ARG_KEY_OPEN)?;
    let name = payload[..first_arg].trim();
    let schema = schemas.get(name)?;
    let mut rest = &payload[first_arg..];
    let mut arguments = String::from("{");
    let mut count = 0usize;
    loop {
        let Some(key_start) = rest.strip_prefix(XML_ARG_KEY_OPEN) else {
            if rest.starts_with(XML_TOOL_CALL_CLOSE) {
                arguments.push('}');
            }
            break;
        };
        let Some(key_end) = key_start.find(XML_ARG_KEY_CLOSE) else { break };
        let key = key_start[..key_end].trim();
        let after_key = &key_start[key_end + XML_ARG_KEY_CLOSE.len()..];
        let Some(value_start) = after_key.strip_prefix(XML_ARG_VALUE_OPEN) else { break };
        if count > 0 {
            arguments.push(',');
        }
        arguments.push_str(&serde_json::to_string(key).ok()?);
        arguments.push(':');
        let string = schema.pointer(&format!("/properties/{}/type", json_pointer_escape(key))).and_then(Value::as_str) == Some("string");
        if let Some(value_end) = value_start.find(XML_ARG_VALUE_CLOSE) {
            let raw = value_start[..value_end].trim();
            if string {
                arguments.push_str(&serde_json::to_string(raw).ok()?);
            } else if serde_json::from_str::<Value>(raw).is_ok() {
                arguments.push_str(raw);
            } else {
                arguments.push_str(&serde_json::to_string(raw).ok()?);
            }
            rest = &value_start[value_end + XML_ARG_VALUE_CLOSE.len()..];
            count += 1;
            continue;
        }
        if string {
            let keep = partial_marker_suffix(value_start, XML_ARG_VALUE_CLOSE);
            let partial = value_start[..value_start.len() - keep].trim();
            let mut encoded = serde_json::to_string(partial).ok()?;
            encoded.pop();
            arguments.push_str(&encoded);
        }
        break;
    }
    Some((name.to_owned(), arguments))
}

fn partial_marker_suffix(value: &str, marker: &str) -> usize {
    (1..marker.len()).rev().find(|&length| value.ends_with(&marker[..length])).unwrap_or(0)
}

fn json_pointer_escape(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

impl ToolCallStream {
    pub fn new(dialect: ToolDialect, schemas: HashMap<String, Value>) -> Self {
        Self { pending: String::new(), schemas, dialect, dsml_block_open: false }
    }

    pub fn push(&mut self, text: &str) -> Vec<ToolOutput> {
        self.pending.push_str(text);
        let mut output = Vec::new();
        loop {
            if self.dsml_block_open {
                let trimmed = self.pending.trim_start();
                let whitespace = self.pending.len() - trimmed.len();
                if trimmed.is_empty() {
                    break;
                }
                if strip_prefix_ascii_case_insensitive(trimmed, DSML_TOOL_CALLS_CLOSE).is_some() {
                    self.pending.drain(..whitespace + DSML_TOOL_CALLS_CLOSE.len());
                    self.dsml_block_open = false;
                    continue;
                }
                if let Some(end) = dsml_close_tag_end(trimmed) {
                    self.pending.drain(..whitespace + end);
                    continue;
                }
                if is_dsml_close_tag_fragment(trimmed) {
                    break;
                }
                if strip_dsml_invoke_open(trimmed).is_some() {
                    let Some(end) = dsml_invoke_end(trimmed) else { break };
                    let block = trimmed[..end].to_owned();
                    self.pending.drain(..whitespace + end);
                    if let Some(call) = parse_dsml_invoke(&block, &self.schemas) {
                        output.push(ToolOutput::ToolCall(call));
                    }
                    // 已进入协议块的内容不能退化成正文，否则客户端会把原始 DSML 当成损坏的工具调用。
                    continue;
                }
                if is_ascii_case_insensitive_prefix(trimmed, DSML_TOOL_CALLS_CLOSE) || is_dsml_invoke_prefix(trimmed) {
                    break;
                }
                if let Some(close) = find_ascii_case_insensitive(trimmed, DSML_TOOL_CALLS_CLOSE) {
                    self.pending.drain(..whitespace + close + DSML_TOOL_CALLS_CLOSE.len());
                    self.dsml_block_open = false;
                    continue;
                }
                break;
            }
            if matches!(self.dialect, ToolDialect::DeepseekDsml | ToolDialect::Auto) {
                let trimmed = self.pending.trim_start();
                let whitespace = self.pending.len() - trimmed.len();
                if let Some(end) = dsml_close_tag_end(trimmed) {
                    self.pending.drain(..whitespace + end);
                    continue;
                }
                if is_dsml_close_tag_fragment(trimmed) {
                    break;
                }
            }
            let Some((start, dialect, bare_dsml)) = next_marker(&self.pending, self.dialect) else {
                if matches!(self.dialect, ToolDialect::DeepseekDsml | ToolDialect::Auto) && self.pending.trim().is_empty() {
                    break;
                }
                let marker_prefix = marker_prefix_len(&self.pending, self.dialect);
                let malformed_dsml = matches!(self.dialect, ToolDialect::DeepseekDsml | ToolDialect::Auto).then(|| find_dsml_tag_prefix(&self.pending).map(|start| self.pending.len() - start)).flatten().unwrap_or(0);
                let keep = marker_prefix.max(malformed_dsml);
                let visible = self.pending.len() - keep;
                if visible > 0 {
                    let text = &self.pending[..visible];
                    if !matches!(self.dialect, ToolDialect::DeepseekDsml | ToolDialect::Auto) || !text.trim().is_empty() {
                        output.push(ToolOutput::Text(text.to_owned()));
                    }
                    self.pending.drain(..visible);
                }
                break;
            };
            if start > 0 {
                let prefix = &self.pending[..start];
                let visible = if dialect == ToolDialect::DeepseekDsml { find_dsml_tag_prefix(prefix).unwrap_or(start) } else { start };
                if visible > 0 {
                    let text = &self.pending[..visible];
                    // 工具块前的空白属于模型原始输出，历史重放必须逐字保留，否则 KV cache 前缀会分叉。
                    output.push(ToolOutput::Text(text.to_owned()));
                }
                self.pending.drain(..start);
                continue;
            }
            if dialect == ToolDialect::DeepseekDsml && !bare_dsml {
                self.pending.drain(..DSML_TOOL_CALLS_OPEN.len());
                self.dsml_block_open = true;
                continue;
            }
            let end = match (dialect, bare_dsml) {
                (ToolDialect::GlmXml, _) => find_ascii_case_insensitive(&self.pending, XML_TOOL_CALL_CLOSE).map(|start| start + XML_TOOL_CALL_CLOSE.len()),
                (ToolDialect::ChatmlJson, _) => find_ascii_case_insensitive(&self.pending, XML_TOOL_CALL_CLOSE).map(|start| start + XML_TOOL_CALL_CLOSE.len()),
                (ToolDialect::DeepseekDsml, false) => unreachable!(),
                (ToolDialect::DeepseekDsml, true) => dsml_invoke_end(&self.pending),
                (ToolDialect::Auto, _) => unreachable!(),
            };
            let Some(end) = end else { break };
            let block = self.pending[..end].to_owned();
            self.pending.drain(..end);
            let calls = match (dialect, bare_dsml) {
                (ToolDialect::GlmXml, _) => parse_xml_tool_call(&block, &self.schemas).map(|call| vec![call]),
                (ToolDialect::ChatmlJson, _) => parse_json_tool_call(&block, &self.schemas).map(|call| vec![call]),
                (ToolDialect::DeepseekDsml, false) => unreachable!(),
                (ToolDialect::DeepseekDsml, true) => parse_dsml_invoke(&block, &self.schemas).map(|call| vec![call]),
                (ToolDialect::Auto, _) => unreachable!(),
            };
            match calls {
                Some(calls) => output.extend(calls.into_iter().map(ToolOutput::ToolCall)),
                // 裸 invoke 仍属于协议，解析失败时丢弃，不能泄漏给不理解 DSML 的客户端。
                None if dialect == ToolDialect::DeepseekDsml => {}
                None => output.push(ToolOutput::Text(block)),
            }
        }
        output
    }

    pub fn finish(&mut self) -> Vec<ToolOutput> {
        if self.dsml_block_open {
            self.pending.clear();
            self.dsml_block_open = false;
            return Vec::new();
        }
        if self.pending.is_empty() {
            return Vec::new();
        }
        let pending = std::mem::take(&mut self.pending);
        if matches!(self.dialect, ToolDialect::DeepseekDsml | ToolDialect::Auto) && is_dsml_tag_fragment(pending.trim()) {
            return Vec::new();
        }
        vec![ToolOutput::Text(pending)]
    }
}

fn next_marker(text: &str, dialect: ToolDialect) -> Option<(usize, ToolDialect, bool)> {
    let mut markers = Vec::with_capacity(3);
    if matches!(dialect, ToolDialect::GlmXml | ToolDialect::Auto) {
        markers.push((find_ascii_case_insensitive(text, XML_TOOL_CALL_OPEN), ToolDialect::GlmXml, false));
    }
    if dialect == ToolDialect::ChatmlJson {
        markers.push((find_ascii_case_insensitive(text, XML_TOOL_CALL_OPEN), ToolDialect::ChatmlJson, false));
    }
    if matches!(dialect, ToolDialect::DeepseekDsml | ToolDialect::Auto) {
        markers.push((find_ascii_case_insensitive(text, DSML_TOOL_CALLS_OPEN), ToolDialect::DeepseekDsml, false));
        markers.push((find_dsml_invoke_open(text), ToolDialect::DeepseekDsml, true));
    }
    markers.into_iter().filter_map(|(index, dialect, bare)| index.map(|index| (index, dialect, bare))).min_by_key(|(index, _, bare)| (*index, *bare))
}

fn marker_prefix_len(text: &str, dialect: ToolDialect) -> usize {
    let markers: &[&str] = match dialect {
        ToolDialect::GlmXml => &[XML_TOOL_CALL_OPEN],
        ToolDialect::ChatmlJson => &[XML_TOOL_CALL_OPEN],
        ToolDialect::DeepseekDsml => &[DSML_TOOL_CALLS_OPEN, DSML_INVOKE_OPEN, DSML_CLOSE_TAG_PREFIX, DSML_ESCAPED_TAG_PREFIX, DSML_ESCAPED_CLOSE_TAG_PREFIX],
        ToolDialect::Auto => &[XML_TOOL_CALL_OPEN, DSML_TOOL_CALLS_OPEN, DSML_INVOKE_OPEN, DSML_CLOSE_TAG_PREFIX, DSML_ESCAPED_TAG_PREFIX, DSML_ESCAPED_CLOSE_TAG_PREFIX],
    };
    (1..markers.iter().map(|marker| marker.len()).max().unwrap_or(1))
        .rev()
        .find(|&length| {
            text.len() >= length
                && text.is_char_boundary(text.len() - length)
                && markers.iter().any(|marker| marker.as_bytes().get(..length).is_some_and(|prefix| text.as_bytes().get(text.len() - length..).is_some_and(|suffix| suffix.eq_ignore_ascii_case(prefix))))
        })
        .unwrap_or(0)
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Clone, Copy)]
pub struct XmlToolTokens {
    pub call_open: u32,
    pub call_close: u32,
    pub key_open: u32,
    pub key_close: u32,
    pub value_open: u32,
    pub value_close: u32,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
impl XmlToolTokens {
    fn all(self) -> [u32; 6] {
        [self.call_open, self.call_close, self.key_open, self.key_close, self.value_open, self.value_close]
    }
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XmlToolState {
    Text,
    ToolName,
    ArgKey,
    AfterArgKey,
    ArgValue,
    AfterArgValue,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[derive(Clone)]
pub struct XmlToolFence {
    enabled: bool,
    state: XmlToolState,
    tokens: XmlToolTokens,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
impl XmlToolFence {
    pub fn new(enabled: bool, tokens: XmlToolTokens) -> Self {
        Self { enabled, state: XmlToolState::Text, tokens }
    }

    pub fn state(&self) -> XmlToolState {
        self.state
    }
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
impl TokenFenceProgram for XmlToolFence {
    fn fence(&self) -> TokenFence {
        if !self.enabled {
            return TokenFence::excluding(self.tokens.all());
        }
        let t = self.tokens;
        let excluded: &[u32] = match self.state {
            XmlToolState::Text => &[t.call_close, t.key_open, t.key_close, t.value_open, t.value_close],
            XmlToolState::ToolName => &[t.call_open, t.key_close, t.value_open, t.value_close],
            XmlToolState::ArgKey => &[t.call_open, t.call_close, t.key_open, t.value_open, t.value_close],
            XmlToolState::AfterArgKey => &[t.call_open, t.call_close, t.key_open, t.key_close, t.value_close],
            XmlToolState::ArgValue => &[t.call_open, t.call_close, t.key_open, t.key_close, t.value_open],
            XmlToolState::AfterArgValue => &[t.call_open, t.key_close, t.value_open, t.value_close],
        };
        TokenFence::excluding(excluded.iter().copied())
    }

    fn advance(&mut self, token: u32) {
        if !self.enabled {
            return;
        }
        let t = self.tokens;
        self.state = match (self.state, token) {
            (XmlToolState::Text, token) if token == t.call_open => XmlToolState::ToolName,
            (XmlToolState::ToolName, token) if token == t.key_open => XmlToolState::ArgKey,
            (XmlToolState::ToolName, token) if token == t.call_close => XmlToolState::Text,
            (XmlToolState::ArgKey, token) if token == t.key_close => XmlToolState::AfterArgKey,
            (XmlToolState::AfterArgKey, token) if token == t.value_open => XmlToolState::ArgValue,
            (XmlToolState::ArgValue, token) if token == t.value_close => XmlToolState::AfterArgValue,
            (XmlToolState::AfterArgValue, token) if token == t.key_open => XmlToolState::ArgKey,
            (XmlToolState::AfterArgValue, token) if token == t.call_close => XmlToolState::Text,
            (state, _) => state,
        };
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn special_output_tokens_preserve_xml_json_and_dsml_protocols() {
        use super::*;
        let schemas = HashMap::from([("Read".to_owned(), json!({"type": "object", "properties": {"file_path": {"type": "string"}}}))]);
        for (dialect, pieces) in [ToolDialect::GlmXml, ToolDialect::ChatmlJson, ToolDialect::DeepseekDsml].into_iter().zip([
            vec![("<tool_call>", true), ("Read", false), ("<arg_key>", true), ("file_path", false), ("</arg_key>", true), ("<arg_value>", true), ("/tmp/test.txt", false), ("</arg_value>", true), ("</tool_call>", true)],
            vec![("<tool_call>", true), (r#"{"name":"Read","arguments":{"file_path":"/tmp/test.txt"}}"#, false), ("</tool_call>", true)],
            vec![("<｜DSML｜", true), ("tool_calls>", false), ("<｜DSML｜", true), (r#"invoke name="Read">"#, false), ("<｜DSML｜", true), (r#"parameter name="file_path" string="true">"#, false), ("/tmp/test.txt", false), ("</｜DSML｜parameter>", true), ("</｜DSML｜invoke>", true), ("</｜DSML｜tool_calls>", true)],
        ]) {
            let words = pieces.iter().map(|(text, special)| if *special { text.to_string() } else { text.replace(' ', "Ġ") }).collect::<Vec<_>>();
            let special = pieces.iter().map(|(_, special)| *special).collect::<Vec<_>>();
            let decoder = crate::tokenizer::Detokenizer::from_bpe_tokens(&words, &special).unwrap();
            let mut stream = dialect.stream(schemas.clone());
            let mut visible = String::new();
            let mut calls = Vec::new();
            for token in 0..pieces.len() as u32 {
                let decoded = decode_output_token(&decoder, token).unwrap();
                for event in stream.push(std::str::from_utf8(&decoded).unwrap()) {
                    match event { ToolOutput::Text(text) => visible.push_str(&text), ToolOutput::ToolCall(call) => calls.push(call) }
                }
            }
            for event in stream.finish() {
                match event { ToolOutput::Text(text) => visible.push_str(&text), ToolOutput::ToolCall(call) => calls.push(call) }
            }
            assert!(visible.trim().is_empty(), "{visible}");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "Read");
            assert_eq!(serde_json::from_str::<Value>(&calls[0].arguments).unwrap(), json!({"file_path": "/tmp/test.txt"}));
        }
        let words = ["<|assistant|>", "<|endoftext|>", "[gMASK]", "<think>", "</think>"];
        let decoder = crate::tokenizer::Detokenizer::from_bpe_tokens(&words.map(str::to_owned), &[true; 5]).unwrap();
        for token in 0..3 { assert!(decode_output_token(&decoder, token).unwrap().is_empty()); }
        for token in 3..5 { assert_eq!(decode_output_token(&decoder, token).unwrap(), words[token as usize].as_bytes()); }
    }
    use super::*;

    fn schemas() -> HashMap<String, Value> {
        HashMap::from([
            ("read_file".to_owned(), json!({"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}}, "required": ["path"], "additionalProperties": false})),
            ("Read".to_owned(), json!({"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"], "additionalProperties": false})),
            ("ping".to_owned(), json!({"type": "object", "properties": {}})),
        ])
    }

    #[test]
    fn dsml_named_tool从schema提取全部参数与required标记() {
        let tools = vec![json!({"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"offset":{"type":"integer"}},"required":["file_path","offset"]}}})];
        let choice = json!({"type":"function","function":{"name":"Read"}});
        let DsmlToolSpec::Named(spec) = dsml_tool_spec(&tools, Some(&choice)).unwrap().unwrap() else { panic!("指定 function 应生成 named spec") };
        assert_eq!(spec.name, "Read");
        assert_eq!(spec.parameters.iter().map(|parameter| (&*parameter.name, parameter.string, parameter.required)).collect::<Vec<_>>(), [("file_path", true, true), ("offset", false, true)]);
    }

    #[test]
    fn deepseek工具提示逐字匹配官方encoder() {
        let tools = vec![json!({"type":"function","function":{"name":"Read","parameters":{"type":"object","required":["file_path"]}}})];
        let instructions = ToolDialect::DeepseekDsml.instructions(&tools, Some(&Value::String("auto".to_owned()))).unwrap().unwrap();
        assert!(instructions.contains(r#"{"name": "Read", "parameters": {"required": ["file_path"], "type": "object"}}"#));
        assert!(instructions.contains("</｜DSML｜tool_calls>\nString parameters"));
        assert!(instructions.contains("final response.\n### Available Tool Schemas"));
        assert!(!instructions.contains("</｜DSML｜tool_calls>\n\nString parameters"));
    }

    #[test]
    fn dsml_named_tool围栏在候选token中限定参数分支() {
        let mut fence = DsmlNamedToolFence {
            parameter_opens: vec![vec![20], vec![21]],
            parameter_tail: vec![13, 14],
            final_close: vec![30],
            values: vec![None, None],
            string_parameters: vec![true, true],
            required: vec![true, false],
            seen: vec![false, false],
            close_token: 12,
            vocab_size: 32,
            terminal_tokens: Arc::from([31]),
            markup_tokens: Arc::from([5, 12]),
            state: DsmlFenceState::Literal { candidates: vec![DsmlLiteral { tokens: vec![10, 11], next: DsmlLiteralNext::ChooseParameter }], offset: 0 },
        };
        assert_eq!(fence.fence().forced(), Some(10));
        fence.advance(10);
        assert_eq!(fence.fence().forced(), Some(11));
        fence.advance(11);
        assert_eq!(fence.fence().excluded(), (0..32).filter(|token| ![20, 21].contains(token)).collect::<Vec<_>>());
        fence.advance(21);
        assert_eq!(fence.fence().excluded(), &[5, 31]);
        fence.advance(29);
        assert_eq!(fence.fence().excluded(), &[5, 31]);
        fence.advance(12);
        assert_eq!(fence.fence().forced(), Some(13));
        fence.advance(13);
        fence.advance(14);
        assert_eq!(fence.fence().forced(), Some(20));
        fence.advance(20);
        assert_eq!(fence.fence().excluded(), &[5, 31]);
        fence.advance(12);
        fence.advance(13);
        fence.advance(14);
        assert_eq!(fence.fence().forced(), Some(30));
        fence.advance(30);
        assert!(fence.fence().is_open());
    }

    #[test]
    fn dsml_auto检测区域后按工具候选收敛() {
        let candidate = |tail| DsmlNamedToolFence {
            parameter_opens: Vec::new(),
            parameter_tail: vec![30],
            final_close: vec![31],
            values: Vec::new(),
            string_parameters: Vec::new(),
            required: Vec::new(),
            seen: Vec::new(),
            close_token: 99,
            vocab_size: 32,
            terminal_tokens: Arc::from([]),
            markup_tokens: Arc::from([]),
            state: DsmlFenceState::Literal { candidates: vec![DsmlLiteral { tokens: vec![1, 2, 3, tail, tail + 10], next: DsmlLiteralNext::Done }], offset: 0 },
        };
        let mut fence = DsmlToolFence::Auto { trigger: vec![1, 2], trigger_offset: 0, candidates: vec![candidate(10), candidate(11)], vocab_size: 32, active: false };
        fence.advance(7);
        assert!(fence.fence().is_open());
        fence.advance(1);
        fence.advance(2);
        assert_eq!(fence.fence().forced(), Some(3));
        fence.advance(3);
        assert_eq!(fence.fence().excluded(), (0..32).filter(|token| ![10, 11].contains(token)).collect::<Vec<_>>());
        fence.advance(10);
        assert_eq!(fence.fence().forced(), Some(20));
    }

    #[test]
    fn glm_xml流式解析并按schema保留类型() {
        let mut stream = ToolCallStream::new(ToolDialect::GlmXml, schemas());
        let mut output = stream.push("先查<tool_ca");
        output.extend(stream.push("ll>read_file<arg_key>path</arg_key><arg_value>/tmp/a</arg_value><arg_key>offset</arg_key><arg_value>3</arg_value></tool_call>"));
        assert!(matches!(&output[0], ToolOutput::Text(text) if text == "先查"));
        let ToolOutput::ToolCall(call) = &output[1] else { panic!("XML 工具调用未转换") };
        assert_eq!(call.name, "read_file");
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap()["offset"], 3);
        assert!(stream.finish().is_empty());

        let mixed_case = "<TOOL_CALL>read_file<ARG_KEY>path</ARG_KEY><Arg_Value>/tmp/case</aRg_VaLuE></tOoL_cAlL>";
        let output = stream.push(mixed_case);
        let ToolOutput::ToolCall(call) = &output[0] else { panic!("XML 标签大小写不敏感解析失败") };
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap()["path"], "/tmp/case");

        let output = stream.push("<TOOL_CALL>READ_FILE<ARG_KEY>PATH</ARG_KEY><ARG_VALUE>/tmp/name</ARG_VALUE></TOOL_CALL>");
        let ToolOutput::ToolCall(call) = &output[0] else { panic!("XML 工具名与参数名大小写不敏感解析失败") };
        assert_eq!(call.name, "read_file");
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap(), json!({"path": "/tmp/name"}));
    }

    #[test]
    fn glm_xml增量事件与最终调用共享同一协议状态() {
        let mut stream = XmlToolCallStream::new("req-a", schemas(), true);
        let first = stream.push("正文<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/");
        assert!(matches!(&first[0], XmlToolStreamEvent::Text(text) if text == "正文"));
        let XmlToolStreamEvent::Delta { index, id: Some(id), name: Some(name), arguments } = &first[1] else { panic!("首个工具增量缺少 identity") };
        assert_eq!((*index, name.as_str()), (0, "read_file"));
        assert!(id.starts_with("call_"));
        assert_eq!(arguments, r#"{"path":"/tmp/"#);

        let second = stream.push("a</arg_value><arg_key>offset</arg_key><arg_value>3</arg_value></tool_call>");
        assert!(matches!(&second[0], XmlToolStreamEvent::Delta { index: 0, id: None, name: None, arguments } if arguments == r#"a","offset":3}"#));
        let XmlToolStreamEvent::ToolCall { index, id: final_id, call } = &second[1] else { panic!("完整 XML 未形成最终工具调用") };
        assert_eq!(*index, 0);
        assert_eq!(final_id, id);
        assert_eq!(call.name, "read_file");
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap(), json!({"path": "/tmp/a", "offset": 3}));
        assert!(stream.finish().is_empty());
    }

    #[test]
    fn deepseek_dsml支持标签参数与直接json() {
        let tagged = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a</｜DSML｜parameter>\n<｜DSML｜parameter name=\"offset\" string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let mut stream = ToolCallStream::new(ToolDialect::DeepseekDsml, schemas());
        let output = stream.push(tagged);
        assert!(matches!(&output[0], ToolOutput::Text(text) if text == "\n\n"));
        let ToolOutput::ToolCall(call) = &output[1] else { panic!("DSML 工具调用未转换") };
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap(), json!({"path": "/tmp/a", "offset": 3}));

        let direct = "<｜DSML｜invoke name=\"read_file\">{\"path\":\"/tmp/b\"}</｜DSML｜invoke>";
        let output = stream.push(direct);
        let ToolOutput::ToolCall(call) = &output[0] else { panic!("DSML JSON 工具调用未转换") };
        assert_eq!(call.name, "read_file");
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap()["path"], "/tmp/b");

        let mixed_case = "<｜dsml｜INVOKE name=\"READ\"><｜DSML｜PARAMETER name=\"FILE_PATH\" string=\"TRUE\">/tmp/case</｜dsml｜PARAMETER></｜DSML｜INVOKE>";
        let output = stream.push(mixed_case);
        let ToolOutput::ToolCall(call) = &output[0] else { panic!("DSML 工具名与参数名大小写不敏感解析失败") };
        assert_eq!(call.name, "Read");
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap(), json!({"file_path": "/tmp/case"}));

        let mixed_case = "<｜dsml｜TOOL_CALLS>\n<｜DsMl｜InVoKe NAME=\"read_file\">\n<｜DSML｜PARAMETER NAME=\"path\" STRING=\"TRUE\">/tmp/case</｜dsml｜PaRaMeTeR>\n</｜DsMl｜InVoKe>\n</｜DSML｜Tool_Calls>";
        let output = stream.push(mixed_case);
        let ToolOutput::ToolCall(call) = &output[0] else { panic!("DSML 标签大小写不敏感解析失败") };
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap()["path"], "/tmp/case");
    }

    #[test]
    fn deepseek_dsml解析后历史重放保持原文() {
        let raw = "好的，让我继续。\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let mut stream = ToolCallStream::new(ToolDialect::DeepseekDsml, schemas());
        let output = stream.push(raw);
        let [ToolOutput::Text(text), ToolOutput::ToolCall(call)] = output.as_slice() else { panic!("DSML 输出未拆成正文和工具调用") };
        let calls = json!([{"function": {"name": call.name, "arguments": call.arguments}}]);
        let replay = format!("{text}{}", ToolDialect::DeepseekDsml.render_history(Some(&calls)).unwrap());
        assert_eq!(replay, raw);
    }

    #[test]
    fn deepseek_dsml合法调用前不泄漏畸形结束标签() {
        let mut stream = ToolCallStream::new(ToolDialect::DeepseekDsml, schemas());
        let first = stream.push("Good, that works now.\n\n</｜dSm");
        let [ToolOutput::Text(text)] = first.as_slice() else { panic!("普通正文未按预期输出") };
        assert_eq!(text, "Good, that works now.\n\n");

        let second = stream.push("l｜tool_c><｜DSML｜tool_calls>\n<｜DsMl｜InVoKe NAME=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>");
        let [ToolOutput::ToolCall(call)] = second.as_slice() else { panic!("合法 DSML 工具调用未转换") };
        assert_eq!(call.name, "read_file");
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap(), json!({"path": "/tmp/a"}));
        assert!(stream.finish().is_empty());
    }

    #[test]
    fn deepseek_dsml不泄漏孤立或截断协议标签() {
        let mut stream = ToolCallStream::new(ToolDialect::DeepseekDsml, schemas());
        let output = stream.push("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜parameter>");
        let [ToolOutput::ToolCall(call)] = output.as_slice() else { panic!("孤立结束标签不应污染工具调用") };
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap(), json!({"path": "/tmp/a"}));
        assert!(stream.finish().is_empty());

        let mut stream = ToolCallStream::new(ToolDialect::DeepseekDsml, schemas());
        let output = stream.push("<｜DSML｜invoke name=\"read_file\">{\"path\":\"/tmp/b\"}</｜DSML｜invoke>\\</｜DsMl｜PaRaMeTeR>");
        let [ToolOutput::ToolCall(call)] = output.as_slice() else { panic!("转义结束标签不应进入正文") };
        assert_eq!(serde_json::from_str::<Value>(&call.arguments).unwrap(), json!({"path": "/tmp/b"}));
        assert!(stream.finish().is_empty());

        let mut stream = ToolCallStream::new(ToolDialect::DeepseekDsml, schemas());
        assert!(stream.push("\\</｜dS").is_empty());
        assert!(stream.push("mL｜parameter>").is_empty());
        assert!(stream.push("<｜DSML｜tool_c").is_empty());
        assert!(stream.finish().is_empty());
    }

    #[test]
    fn deepseek_dsml不把畸形标签修补成工具调用() {
        let malformed = r#"Let me先看一下。<｜DSML｜tool\_c优化化 <｜DSML｜tool\_calls>
<｜DSML｜invrule name="Read">
<｜DSML｜parameter name="file\_path" string="true">/path/to/example.md\</｜DSML｜parameter>
\</｜DSML｜inv>
\</｜DSML｜tool\_calls>"#;
        let (visible, calls) = ToolDialect::DeepseekDsml.split_output(malformed, &schemas());
        assert!(calls.is_empty());
        assert_eq!(visible, "Let me先看一下。");

        let short_close = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/c</｜DSML｜parameter>\n</｜DSML｜inv>\n</｜DSML｜tool_calls>";
        let (_, calls) = ToolDialect::DeepseekDsml.split_output(short_close, &schemas());
        assert!(calls.is_empty());
    }

    #[test]
    fn deepseek_dsml不泄漏未完成或无效的工具块() {
        let incomplete = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">ssh root@host very-long-command";
        let (visible, calls) = ToolDialect::DeepseekDsml.split_output(incomplete, &schemas());
        assert!(visible.is_empty());
        assert!(calls.is_empty());

        let invalid = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">echo test</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>后续正文";
        let (visible, calls) = ToolDialect::DeepseekDsml.split_output(invalid, &schemas());
        assert_eq!(visible, "后续正文");
        assert!(calls.is_empty());

        let bare = "<｜DSML｜invoke name=\"Bash\">{\"command\":\"echo test\"}</｜DSML｜invoke>";
        let (visible, calls) = ToolDialect::DeepseekDsml.split_output(bare, &schemas());
        assert!(visible.is_empty());
        assert!(calls.is_empty());
    }

    #[test]
    fn 无参数工具与未知工具边界明确() {
        assert_eq!(parse_xml_tool_call("<tool_call>ping</tool_call>", &schemas()).unwrap().arguments, "{}");
        assert!(parse_xml_tool_call("<tool_call>missing</tool_call>", &schemas()).is_none());
        assert!(parse_xml_tool_call("<tool_call>read_file<arg_key>offset</arg_key><arg_value>3</arg_value></tool_call>", &schemas()).is_none());
        assert!(parse_xml_tool_call("<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/a</arg_value><arg_key>guess</arg_key><arg_value>x</arg_value></tool_call>", &schemas()).is_none());
    }

    #[test]
    fn xml围栏token由dialect提供() {
        let tokens = XmlToolTokens { call_open: 1, call_close: 2, key_open: 3, key_close: 4, value_open: 5, value_close: 6 };
        let mut fence = XmlToolFence::new(true, tokens);
        assert!(fence.fence().excluded().contains(&6));
        for token in [1, 3, 4, 5] {
            fence.advance(token);
        }
        assert!(!fence.fence().excluded().contains(&6));
    }

    #[test]
    fn 工具调用id请求内稳定且跨请求唯一() {
        assert_eq!(tool_call_id("req-a", 0, "Read"), tool_call_id("req-a", 0, "Read"));
        assert_ne!(tool_call_id("req-a", 0, "Read"), tool_call_id("req-b", 0, "Read"));
        assert_ne!(tool_call_id("req-a", 0, "Read"), tool_call_id("req-a", 1, "Read"));
        assert_eq!(request_tool_scope("qwen", &json!({"messages": []})), request_tool_scope("qwen", &json!({"messages": []})));
        assert_ne!(request_tool_scope("qwen", &json!({"messages": []})), request_tool_scope("ornith", &json!({"messages": []})));
    }

    #[test]
    fn 请求级工具流统一隐藏协议并产出结构化调用() {
        let request = json!({
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": schemas()["read_file"]}}]
        });
        let mut stream = RequestToolCallStream::new(&request, "req-a", ToolDialect::ChatmlJson);
        let mut visible = String::new();
        assert!(stream.push("before<tool_ca", |text| {
            visible.push_str(&text);
            true
        }));
        assert!(stream.push("ll>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"/tmp/a\"}}\n</tool_call>", |text| {
            visible.push_str(&text);
            true
        }));
        assert!(stream.finish(|text| {
            visible.push_str(&text);
            true
        }));
        assert_eq!(visible, "before");
        assert_eq!(stream.calls.len(), 1);
        assert_eq!(stream.calls[0].function.name, "read_file");
        assert_eq!(serde_json::from_str::<Value>(&stream.calls[0].function.arguments).unwrap(), json!({"path": "/tmp/a"}));
    }

    #[test]
    fn 工具说明在runtime边界拒绝坏定义与无效choice() {
        let dialect = ToolDialect::ChatmlJson;
        assert!(dialect.instructions(&[json!({"type": "function"})], None).unwrap_err().contains("function 必须是对象"));
        let tools = [json!({"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}), json!({"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}})];
        assert!(dialect.instructions(&tools, None).unwrap_err().contains("重复"));
        let tools = &tools[..1];
        let choice = json!({"type": "function", "function": {"name": "missing"}});
        assert!(dialect.instructions(tools, Some(&choice)).unwrap_err().contains("不在 tools 中"));
        assert!(dialect.instructions(&[], Some(&Value::String("required".to_owned()))).unwrap_err().contains("不能为空"));
        assert!(request_tools(&json!({"tools": {}})).unwrap_err().contains("必须是数组"));
        assert!(dialect.render_history(Some(&json!({}))).unwrap_err().contains("必须是数组"));
    }
}
