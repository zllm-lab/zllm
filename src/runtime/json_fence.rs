use crate::{backend::TokenFence, tokenizer::Detokenizer};
use regex_automata::meta::Regex;
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone)]
enum Schema {
    String { choices: Option<Vec<Vec<u8>>>, min_length: usize, max_length: Option<usize>, pattern: Option<Arc<Regex>>, format: Option<StringFormat> },
    Integer { minimum: Option<i128>, maximum: Option<i128> },
    Number { minimum: Option<(f64, bool)>, maximum: Option<(f64, bool)> },
    Boolean,
    Null,
    Array { item: usize, min_items: usize, max_items: Option<usize> },
    Object { properties: Vec<Property>, additional: Option<usize> },
    Literal { bytes: Vec<u8> },
    Union { variants: Vec<usize> },
    Intersection { variants: Vec<usize> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StringFormat {
    Uri,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Property {
    name: Vec<u8>,
    schema: usize,
    required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Return {
    Array { schema: usize, count: usize },
    Object { schema: usize, seen: Vec<bool>, property: Option<usize> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum NumberState {
    Minus,
    Zero,
    Integer,
    Dot,
    Fraction,
    Exponent,
    ExponentSign,
    ExponentDigits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Utf8State {
    remaining: u8,
    min: u8,
    max: u8,
}

impl Default for Utf8State {
    fn default() -> Self {
        Self { remaining: 0, min: 0x80, max: 0xbf }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Mode {
    Value(usize),
    ArrayStart { schema: usize, count: usize, after_comma: bool },
    ArrayAfterValue { schema: usize, count: usize },
    ObjectStart { schema: usize, seen: Vec<bool> },
    ObjectKey { schema: usize, seen: Vec<bool>, candidates: Vec<usize>, offset: usize, utf8: Utf8State },
    ObjectColon { schema: usize, seen: Vec<bool>, property: Option<usize> },
    ObjectAfterValue { schema: usize, seen: Vec<bool> },
    String { schema: usize, escape: u8, utf8: Utf8State, has_content: bool, characters: usize, raw: Option<Vec<u8>> },
    RawString { schema: usize, utf8: Utf8State, characters: usize, raw: Option<Vec<u8>> },
    StringChoice { schema: usize, candidates: Vec<usize>, offset: usize },
    Number { schema: usize, state: NumberState, negative: bool, magnitude: u128, overflow: bool, raw: Vec<u8> },
    Literal { bytes: &'static [u8], offset: usize },
    Fixed { schema: usize, offset: usize },
    Union { alternatives: Vec<State> },
    Intersection { alternatives: Vec<State> },
    Complete,
    Dead,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct State {
    mode: Mode,
    returns: Vec<Return>,
}

/// 模型启动期构造一次的 token 字节表。请求内 grammar matcher 共享它，避免在
/// decode 热路径反复 detokenize 整个词表。
#[derive(Debug)]
pub struct JsonTokenTable {
    bytes: Arc<[Option<Box<[u8]>>]>,
    vocab_size: u32,
    max_bytes: usize,
}

impl JsonTokenTable {
    pub fn new(detokenizer: &Detokenizer, vocab_size: usize) -> Result<Arc<Self>, String> {
        let vocab_size_u32 = u32::try_from(vocab_size).map_err(|_| format!("JSON fence vocab_size={vocab_size} 超出 u32"))?;
        let bytes = (0..vocab_size_u32).map(|token| detokenizer.decode_bytes(&[token], true).ok().filter(|bytes| !bytes.is_empty()).map(Vec::into_boxed_slice)).collect::<Vec<_>>();
        let max_bytes = bytes.iter().filter_map(|bytes| bytes.as_ref().map(|bytes| bytes.len())).max().unwrap_or(0);
        Ok(Arc::new(Self { bytes: bytes.into(), vocab_size: vocab_size_u32, max_bytes }))
    }

    /// 结构化协议可以复用同一份 token 字节表屏蔽分隔符。这里只提供
    /// 模型无关的字节查询，不感知 JSON、XML 或工具协议。
    pub(crate) fn tokens_containing(&self, byte: u8) -> Vec<u32> {
        self.bytes.iter().enumerate().filter(|(_, bytes)| bytes.as_deref().is_some_and(|bytes| bytes.contains(&byte))).map(|(token, _)| u32::try_from(token).expect("token table 已验证为 u32")).collect()
    }
}

#[derive(Clone)]
pub struct JsonSchemaFence {
    schemas: Arc<[Schema]>,
    tokens: Arc<JsonTokenTable>,
    cache: Arc<Mutex<HashMap<State, TokenFence>>>,
    state: State,
}

impl JsonSchemaFence {
    #[cfg(test)]
    pub fn new(schema: &Value, tokens: Arc<JsonTokenTable>) -> Result<Self, String> {
        Self::new_with_root(schema, schema, tokens)
    }

    pub(crate) fn new_with_root(schema: &Value, root: &Value, tokens: Arc<JsonTokenTable>) -> Result<Self, String> {
        let mut schemas = Vec::new();
        let mut resolving = HashSet::new();
        let schema = compile_schema_inner(schema, root, &mut schemas, &mut resolving)?;
        Ok(Self { schemas: schemas.into(), tokens, cache: Arc::new(Mutex::new(HashMap::new())), state: State { mode: Mode::Value(schema), returns: Vec::new() } })
    }

    pub(crate) fn new_raw_string(schema: &Value, root: &Value, tokens: Arc<JsonTokenTable>) -> Result<Self, String> {
        let mut fence = Self::new_with_root(schema, root, tokens)?;
        let Mode::Value(schema) = &fence.state.mode else { unreachable!() };
        let schema = *schema;
        if !schema_supports_raw_string(&fence.schemas, schema) {
            return Err("DSML string=true parameter 的 schema 必须约束为 string type".to_owned());
        }
        let raw = string_requires_raw(&fence.schemas, schema).then(Vec::new);
        fence.state.mode = Mode::RawString { schema, utf8: Utf8State::default(), characters: 0, raw };
        Ok(fence)
    }

    pub fn complete(&self) -> bool {
        state_complete(&self.schemas, &self.state)
    }

    pub fn fence(&self, close_token: u32) -> TokenFence {
        if self.state.mode == Mode::Complete {
            return TokenFence::forcing(close_token);
        }
        let cache_state = normalized_cache_state(&self.schemas, &self.state, self.tokens.max_bytes);
        if let Some(cache_state) = cache_state.as_ref()
            && let Ok(cache) = self.cache.lock()
            && let Some(fence) = cache.get(cache_state)
        {
            return fence.clone();
        }
        let can_end = self.complete();
        let excluded = (0..self.tokens.vocab_size).filter(|&token| {
            if token == close_token {
                return !can_end;
            }
            let Some(bytes) = self.tokens.bytes.get(token as usize).and_then(Option::as_deref) else { return true };
            let mut state = self.state.clone();
            bytes.iter().any(|&byte| !advance_byte(&self.schemas, &mut state, byte))
        });
        let fence = TokenFence::excluding(excluded);
        if let Some(cache_state) = cache_state
            && let Ok(mut cache) = self.cache.lock()
        {
            cache.insert(cache_state, fence.clone());
        }
        fence
    }

    pub fn advance(&mut self, token: u32) {
        let Some(bytes) = self.tokens.bytes.get(token as usize).and_then(Option::as_deref) else {
            self.state.mode = Mode::Dead;
            return;
        };
        for &byte in bytes {
            if !advance_byte(&self.schemas, &mut self.state, byte) {
                self.state.mode = Mode::Dead;
                return;
            }
        }
    }
}

#[cfg(test)]
fn compile_schema(value: &Value, schemas: &mut Vec<Schema>) -> Result<usize, String> {
    let mut resolving = HashSet::new();
    compile_schema_inner(value, value, schemas, &mut resolving)
}

fn compile_schema_inner(value: &Value, root: &Value, schemas: &mut Vec<Schema>, resolving: &mut HashSet<String>) -> Result<usize, String> {
    if value == &Value::Bool(true) {
        return push_schema(schemas, Schema::Literal { bytes: b"null".to_vec() });
    }
    if value == &Value::Bool(false) {
        return Err("JSON Schema false 不允许任何输出".to_owned());
    }
    let object = value.as_object().ok_or("JSON Schema 节点必须是 object")?;
    // 空 schema 与仅含 annotation 的 schema 等价于 true。沿用 true schema
    // 的 canonical null，生成合法子集而不引入无约束 JSON 状态机。
    if object.keys().all(|keyword| annotation_keyword(keyword)) {
        return push_schema(schemas, Schema::Literal { bytes: b"null".to_vec() });
    }
    if let Some(reference) = object.get("$ref") {
        let reference = reference.as_str().ok_or("JSON Schema $ref 必须是 string")?;
        let pointer = reference.strip_prefix('#').ok_or_else(|| format!("JSON Schema 仅支持本地 $ref，实际为 {reference:?}"))?;
        if !resolving.insert(reference.to_owned()) {
            return Err(format!("JSON Schema 暂不支持递归 $ref {reference:?}"));
        }
        let target = root.pointer(pointer).ok_or_else(|| format!("JSON Schema $ref {reference:?} 无法解析"))?;
        let result = compile_schema_inner(target, root, schemas, resolving);
        resolving.remove(reference);
        return intersect_sibling_constraints(object, "$ref", result?, root, schemas, resolving);
    }
    let one_of = object.get("oneOf");
    let any_of = object.get("anyOf");
    if one_of.is_some() && any_of.is_some() {
        return Err("JSON Schema 节点不能同时声明 oneOf 与 anyOf".to_owned());
    }
    if let Some((keyword, alternatives)) = one_of.map(|value| ("oneOf", value)).or_else(|| any_of.map(|value| ("anyOf", value))) {
        let alternatives = alternatives.as_array().ok_or_else(|| format!("JSON Schema {keyword} 必须是 array"))?;
        if alternatives.is_empty() {
            return Err(format!("JSON Schema {keyword} 不能为空"));
        }
        let variants = alternatives.iter().map(|alternative| compile_schema_inner(alternative, root, schemas, resolving)).collect::<Result<Vec<_>, _>>()?;
        if keyword == "oneOf" && variants.iter().enumerate().any(|(left, &left_schema)| variants.iter().skip(left + 1).any(|&right_schema| schemas_overlap(schemas, left_schema, right_schema))) {
            return Err("JSON Schema oneOf 分支可能同时匹配，当前不能保证 exactly-one 语义".to_owned());
        }
        let union = push_schema(schemas, Schema::Union { variants })?;
        return intersect_sibling_constraints(object, keyword, union, root, schemas, resolving);
    }
    if let Some(alternatives) = object.get("allOf") {
        let alternatives = alternatives.as_array().ok_or("JSON Schema allOf 必须是 array")?;
        if alternatives.is_empty() {
            return Err("JSON Schema allOf 不能为空".to_owned());
        }
        let variants = alternatives.iter().map(|alternative| compile_schema_inner(alternative, root, schemas, resolving)).collect::<Result<Vec<_>, _>>()?;
        let intersection = push_schema(schemas, Schema::Intersection { variants })?;
        return intersect_sibling_constraints(object, "allOf", intersection, root, schemas, resolving);
    }
    if let Some(types) = object.get("type").and_then(Value::as_array) {
        if types.is_empty() {
            return Err("JSON Schema type array 不能为空".to_owned());
        }
        let mut variants = Vec::with_capacity(types.len());
        for kind in types {
            let kind = kind.as_str().ok_or("JSON Schema type array 项必须是 string")?;
            let mut alternative = object.clone();
            alternative.insert("type".to_owned(), Value::String(kind.to_owned()));
            variants.push(compile_schema_inner(&Value::Object(alternative), root, schemas, resolving)?);
        }
        return push_schema(schemas, Schema::Union { variants });
    }
    if let Some(constant) = object.get("const") {
        validate_keywords(object, &["const", "type"])?;
        if let Some(kind) = object.get("type").and_then(Value::as_str)
            && !literal_matches_type(constant, kind)
        {
            return Err(format!("JSON Schema const 与 type={kind:?} 不一致"));
        }
        return push_schema(schemas, Schema::Literal { bytes: serde_json::to_vec(constant).map_err(|error| format!("序列化 JSON Schema const 失败: {error}"))? });
    }
    if object.get("type").is_none() && object.contains_key("enum") {
        validate_keywords(object, &["enum"])?;
        let values = object["enum"].as_array().ok_or("JSON Schema enum 必须是 array")?;
        if values.is_empty() {
            return Err("JSON Schema enum 不能为空".to_owned());
        }
        let variants = values.iter().map(|value| push_schema(schemas, Schema::Literal { bytes: serde_json::to_vec(value).map_err(|error| format!("序列化 JSON Schema enum 失败: {error}"))? })).collect::<Result<Vec<_>, _>>()?;
        return push_schema(schemas, Schema::Union { variants });
    }
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| object.contains_key("properties").then_some("object"))
        .or_else(|| object.contains_key("items").then_some("array"))
        .or_else(|| (object.contains_key("pattern") || object.contains_key("format")).then_some("string"))
        .ok_or("JSON Schema 节点缺少可生成的 type/组合约束")?;
    let schema = match kind {
        "string" => {
            validate_keywords(object, &["type", "enum", "minLength", "maxLength", "pattern", "format"])?;
            let choices = object
                .get("enum")
                .map(|values| {
                    values
                        .as_array()
                        .ok_or("JSON Schema string enum 必须是 array")?
                        .iter()
                        .map(|value| {
                            let value = value.as_str().ok_or("JSON Schema string enum 项必须是 string")?;
                            let encoded = serde_json::to_vec(value).map_err(|_| "JSON Schema string enum 序列化失败")?;
                            Ok(encoded[1..encoded.len() - 1].to_vec())
                        })
                        .collect::<Result<Vec<_>, &str>>()
                })
                .transpose()?;
            let min_length = schema_usize(object.get("minLength"), "string minLength")?.unwrap_or(0);
            let max_length = schema_usize(object.get("maxLength"), "string maxLength")?;
            if max_length.is_some_and(|maximum| min_length > maximum) {
                return Err(format!("JSON Schema string 长度范围为空: minLength={min_length} maxLength={max_length:?}"));
            }
            let pattern = object
                .get("pattern")
                .map(|value| {
                    let pattern = value.as_str().ok_or("JSON Schema string pattern 必须是 string".to_owned())?;
                    Regex::new(pattern).map(Arc::new).map_err(|error| format!("JSON Schema string pattern 无法编译: {error}"))
                })
                .transpose()?;
            let format = object
                .get("format")
                .map(|value| match value.as_str() {
                    Some("uri") => Ok(StringFormat::Uri),
                    Some(format) => Err(format!("JSON Schema string format={format:?} 尚不支持严格生成")),
                    None => Err("JSON Schema string format 必须是 string".to_owned()),
                })
                .transpose()?;
            Schema::String { choices, min_length, max_length, pattern, format }
        }
        "integer" => {
            validate_keywords(object, &["type", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"])?;
            let minimum = integer_lower_bound(object)?;
            let maximum = integer_upper_bound(object)?;
            if minimum.zip(maximum).is_some_and(|(minimum, maximum)| minimum > maximum) {
                return Err(format!("JSON Schema integer 范围为空: minimum={minimum:?} maximum={maximum:?}"));
            }
            Schema::Integer { minimum, maximum }
        }
        "number" => {
            validate_keywords(object, &["type", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"])?;
            let minimum = number_bound(object, "minimum", "exclusiveMinimum")?;
            let maximum = number_bound(object, "maximum", "exclusiveMaximum")?;
            if minimum.zip(maximum).is_some_and(|((minimum, min_inclusive), (maximum, max_inclusive))| minimum > maximum || minimum == maximum && (!min_inclusive || !max_inclusive)) {
                return Err(format!("JSON Schema number 范围为空: minimum={minimum:?} maximum={maximum:?}"));
            }
            Schema::Number { minimum, maximum }
        }
        "boolean" => {
            validate_keywords(object, &["type"])?;
            Schema::Boolean
        }
        "null" => {
            validate_keywords(object, &["type"])?;
            Schema::Null
        }
        "array" => {
            validate_keywords(object, &["type", "items", "minItems", "maxItems"])?;
            let item = compile_schema_inner(object.get("items").ok_or("JSON Schema array 缺少 items")?, root, schemas, resolving)?;
            let min_items = schema_usize(object.get("minItems"), "array minItems")?.unwrap_or(0);
            let max_items = schema_usize(object.get("maxItems"), "array maxItems")?;
            if max_items.is_some_and(|maximum| min_items > maximum) {
                return Err(format!("JSON Schema array 长度范围为空: minItems={min_items} maxItems={max_items:?}"));
            }
            Schema::Array { item, min_items, max_items }
        }
        "object" => {
            validate_keywords(object, &["type", "properties", "required", "additionalProperties", "propertyNames"])?;
            if let Some(property_names) = object.get("propertyNames") {
                let property_names = property_names.as_object().ok_or("JSON Schema object propertyNames 必须是 schema object")?;
                validate_keywords(property_names, &["type"])?;
                if property_names.get("type").and_then(Value::as_str) != Some("string") {
                    return Err("JSON Schema object 仅支持无附加约束的 propertyNames string".to_owned());
                }
            }
            let properties = object.get("properties").map(|value| value.as_object().ok_or("JSON Schema object properties 必须是 object")).transpose()?;
            let required = object.get("required").map(|value| value.as_array().ok_or("JSON Schema object required 必须是 array")).transpose()?;
            let required = required.map(Vec::as_slice).unwrap_or_default();
            let required = required.iter().map(|value| value.as_str().ok_or("JSON Schema object required 项必须是 string")).collect::<Result<Vec<_>, _>>()?;
            let mut compiled = Vec::with_capacity(properties.map_or(0, |properties| properties.len()));
            if let Some(properties) = properties {
                for (name, property) in properties {
                    let schema = compile_schema_inner(property, root, schemas, resolving)?;
                    compiled.push(Property { name: name.as_bytes().to_vec(), schema, required: required.contains(&name.as_str()) });
                }
            }
            if let Some(name) = required.iter().find(|name| !compiled.iter().any(|property| property.name == name.as_bytes())) {
                return Err(format!("JSON Schema object required 字段 {name:?} 未在 properties 中声明"));
            }
            let additional = match object.get("additionalProperties") {
                Some(Value::Object(_)) => Some(compile_schema_inner(&object["additionalProperties"], root, schemas, resolving)?),
                Some(Value::Bool(false)) | None | Some(Value::Bool(true)) => None,
                Some(_) => return Err("JSON Schema object additionalProperties 必须是 boolean 或 schema object".to_owned()),
            };
            // additionalProperties=true（以及标准缺省值）时，只生成 properties
            // 声明的字段，仍是原 schema 的合法子集；schema-valued map 则完整约束动态值。
            Schema::Object { properties: compiled, additional }
        }
        other => return Err(format!("JSON Schema type={other:?} 尚不支持结构化生成")),
    };
    push_schema(schemas, schema)
}

/// JSON Schema 允许 `$ref` 和组合关键字旁边继续声明约束。同层约束必须与
/// 引用或组合结果同时成立，不能把常见的 `type` 误报为未知关键字。
fn intersect_sibling_constraints(object: &serde_json::Map<String, Value>, keyword: &str, schema: usize, root: &Value, schemas: &mut Vec<Schema>, resolving: &mut HashSet<String>) -> Result<usize, String> {
    let mut siblings = object.clone();
    siblings.remove(keyword);
    if siblings.keys().all(|keyword| annotation_keyword(keyword)) {
        return Ok(schema);
    }
    let sibling = compile_schema_inner(&Value::Object(siblings), root, schemas, resolving)?;
    push_schema(schemas, Schema::Intersection { variants: vec![schema, sibling] })
}

fn push_schema(schemas: &mut Vec<Schema>, schema: Schema) -> Result<usize, String> {
    let index = schemas.len();
    schemas.push(schema);
    Ok(index)
}

fn annotation_keyword(keyword: &str) -> bool {
    matches!(keyword, "$comment" | "$defs" | "$id" | "$schema" | "default" | "definitions" | "deprecated" | "description" | "examples" | "readOnly" | "title" | "writeOnly")
}

fn validate_keywords(object: &serde_json::Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    if let Some(keyword) = object.keys().find(|keyword| !annotation_keyword(keyword) && !allowed.contains(&keyword.as_str())) {
        return Err(format!("JSON Schema 关键字 {keyword:?} 尚不支持严格生成"));
    }
    Ok(())
}

fn schema_usize(value: Option<&Value>, keyword: &str) -> Result<Option<usize>, String> {
    value.map(|value| value.as_u64().ok_or_else(|| format!("JSON Schema {keyword} 必须是非负整数")).and_then(|value| usize::try_from(value).map_err(|_| format!("JSON Schema {keyword} 超出 usize")))).transpose()
}

fn literal_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn number_bound(object: &serde_json::Map<String, Value>, inclusive: &str, exclusive: &str) -> Result<Option<(f64, bool)>, String> {
    if object.contains_key(inclusive) && object.contains_key(exclusive) {
        return Err(format!("JSON Schema number 不能同时声明 {inclusive} 与 {exclusive}"));
    }
    object
        .get(inclusive)
        .map(|value| value.as_f64().filter(|value| value.is_finite()).map(|value| (value, true)).ok_or_else(|| format!("JSON Schema number {inclusive} 必须是有限数字")))
        .or_else(|| object.get(exclusive).map(|value| value.as_f64().filter(|value| value.is_finite()).map(|value| (value, false)).ok_or_else(|| format!("JSON Schema number {exclusive} 必须是有限数字"))))
        .transpose()
}

fn integer_value(value: &Value, keyword: &str) -> Result<i128, String> {
    value.as_i64().map(i128::from).or_else(|| value.as_u64().map(i128::from)).ok_or_else(|| format!("JSON Schema integer {keyword} 必须是整数"))
}

fn integer_lower_bound(object: &serde_json::Map<String, Value>) -> Result<Option<i128>, String> {
    let inclusive = object.get("minimum").map(|value| integer_value(value, "minimum")).transpose()?;
    let exclusive = object.get("exclusiveMinimum").map(|value| integer_value(value, "exclusiveMinimum")?.checked_add(1).ok_or("JSON Schema integer exclusiveMinimum 溢出".to_owned())).transpose()?;
    Ok(inclusive.into_iter().chain(exclusive).max())
}

fn integer_upper_bound(object: &serde_json::Map<String, Value>) -> Result<Option<i128>, String> {
    let inclusive = object.get("maximum").map(|value| integer_value(value, "maximum")).transpose()?;
    let exclusive = object.get("exclusiveMaximum").map(|value| integer_value(value, "exclusiveMaximum")?.checked_sub(1).ok_or("JSON Schema integer exclusiveMaximum 溢出".to_owned())).transpose()?;
    Ok(inclusive.into_iter().chain(exclusive).min())
}

fn schemas_overlap(schemas: &[Schema], left: usize, right: usize) -> bool {
    match (&schemas[left], &schemas[right]) {
        (Schema::Union { variants }, _) => variants.iter().any(|&variant| schemas_overlap(schemas, variant, right)),
        (_, Schema::Union { variants }) => variants.iter().any(|&variant| schemas_overlap(schemas, left, variant)),
        (Schema::Intersection { .. }, _) | (_, Schema::Intersection { .. }) => true,
        (Schema::Literal { bytes: left }, Schema::Literal { bytes: right }) => left == right,
        (Schema::Literal { bytes }, schema) | (schema, Schema::Literal { bytes }) => literal_overlaps_schema(bytes, schema),
        (Schema::Integer { .. }, Schema::Number { .. }) | (Schema::Number { .. }, Schema::Integer { .. }) => true,
        (Schema::String { .. }, Schema::String { .. })
        | (Schema::Integer { .. }, Schema::Integer { .. })
        | (Schema::Number { .. }, Schema::Number { .. })
        | (Schema::Boolean, Schema::Boolean)
        | (Schema::Null, Schema::Null)
        | (Schema::Array { .. }, Schema::Array { .. })
        | (Schema::Object { .. }, Schema::Object { .. }) => true,
        _ => false,
    }
}

fn literal_overlaps_schema(bytes: &[u8], schema: &Schema) -> bool {
    let Ok(value) = serde_json::from_slice::<Value>(bytes) else { return true };
    match schema {
        Schema::String { .. } => value.is_string(),
        Schema::Integer { .. } => value.as_i64().is_some() || value.as_u64().is_some(),
        Schema::Number { .. } => value.is_number(),
        Schema::Boolean => value.is_boolean(),
        Schema::Null => value.is_null(),
        Schema::Array { .. } => value.is_array(),
        Schema::Object { .. } => value.is_object(),
        Schema::Literal { bytes: other } => bytes == other,
        Schema::Union { .. } | Schema::Intersection { .. } => true,
    }
}

fn advance_byte(schemas: &[Schema], state: &mut State, byte: u8) -> bool {
    for _ in 0..4 {
        match &mut state.mode {
            Mode::Value(schema) => {
                if byte.is_ascii_whitespace() {
                    return true;
                }
                let schema = *schema;
                match &schemas[schema] {
                    Schema::String { choices: None, .. } if byte == b'"' => {
                        state.mode = Mode::String { schema, escape: 0, utf8: Utf8State::default(), has_content: false, characters: 0, raw: string_requires_raw(schemas, schema).then(Vec::new) };
                    }
                    Schema::String { choices: Some(choices), .. } if byte == b'"' => {
                        state.mode = Mode::StringChoice { schema, candidates: (0..choices.len()).collect(), offset: 0 };
                    }
                    Schema::Integer { .. } | Schema::Number { .. } if byte == b'-' => {
                        state.mode = Mode::Number { schema, state: NumberState::Minus, negative: true, magnitude: 0, overflow: false, raw: vec![byte] };
                    }
                    Schema::Integer { .. } | Schema::Number { .. } if byte == b'0' => {
                        let mode = Mode::Number { schema, state: NumberState::Zero, negative: false, magnitude: 0, overflow: false, raw: vec![byte] };
                        if matches!(schemas[schema], Schema::Integer { .. }) && !number_mode_can_continue_or_complete(schemas, &mode) {
                            return false;
                        }
                        state.mode = mode;
                    }
                    Schema::Integer { .. } | Schema::Number { .. } if byte.is_ascii_digit() => {
                        let mode = Mode::Number { schema, state: NumberState::Integer, negative: false, magnitude: u128::from(byte - b'0'), overflow: false, raw: vec![byte] };
                        if matches!(schemas[schema], Schema::Integer { .. }) && !number_mode_can_continue_or_complete(schemas, &mode) {
                            return false;
                        }
                        state.mode = mode;
                    }
                    Schema::Boolean if byte == b't' => state.mode = Mode::Literal { bytes: b"true", offset: 1 },
                    Schema::Boolean if byte == b'f' => state.mode = Mode::Literal { bytes: b"false", offset: 1 },
                    Schema::Null if byte == b'n' => state.mode = Mode::Literal { bytes: b"null", offset: 1 },
                    Schema::Array { .. } if byte == b'[' => state.mode = Mode::ArrayStart { schema, count: 0, after_comma: false },
                    Schema::Object { properties, .. } if byte == b'{' => state.mode = Mode::ObjectStart { schema, seen: vec![false; properties.len()] },
                    Schema::Literal { bytes } if bytes.first().copied() == Some(byte) => {
                        if bytes.len() == 1 {
                            return finish_value(state);
                        }
                        state.mode = Mode::Fixed { schema, offset: 1 };
                    }
                    Schema::Union { variants } => {
                        let returns = std::mem::take(&mut state.returns);
                        let alternatives = variants
                            .iter()
                            .filter_map(|&variant| {
                                let mut alternative = State { mode: Mode::Value(variant), returns: returns.clone() };
                                advance_byte(schemas, &mut alternative, byte).then_some(alternative)
                            })
                            .collect::<Vec<_>>();
                        if alternatives.is_empty() {
                            return false;
                        }
                        state.mode = Mode::Union { alternatives };
                    }
                    Schema::Intersection { variants } => {
                        let returns = std::mem::take(&mut state.returns);
                        let alternatives = variants
                            .iter()
                            .map(|&variant| {
                                let mut alternative = State { mode: Mode::Value(variant), returns: returns.clone() };
                                advance_byte(schemas, &mut alternative, byte).then_some(alternative)
                            })
                            .collect::<Option<Vec<_>>>();
                        let Some(alternatives) = alternatives else { return false };
                        state.mode = Mode::Intersection { alternatives };
                    }
                    _ => return false,
                }
                return true;
            }
            Mode::ArrayStart { schema, count, after_comma } => {
                if byte.is_ascii_whitespace() {
                    return true;
                }
                let (item, min_items, max_items) = array_spec(schemas, *schema);
                if byte == b']' && !*after_comma && *count >= min_items {
                    return finish_value(state);
                }
                if max_items.is_some_and(|maximum| *count >= maximum) {
                    return false;
                }
                state.returns.push(Return::Array { schema: *schema, count: *count });
                state.mode = Mode::Value(item);
            }
            Mode::ArrayAfterValue { schema, count } => {
                if byte.is_ascii_whitespace() {
                    return true;
                }
                let (_, min_items, max_items) = array_spec(schemas, *schema);
                if byte == b']' && *count >= min_items {
                    return finish_value(state);
                }
                if byte == b',' && max_items.is_none_or(|maximum| *count < maximum) {
                    state.mode = Mode::ArrayStart { schema: *schema, count: *count, after_comma: true };
                    return true;
                }
                return false;
            }
            Mode::ObjectStart { schema, seen } => {
                if byte.is_ascii_whitespace() {
                    return true;
                }
                let schema_id = *schema;
                if byte == b'}' && required_seen(schemas, schema_id, seen) {
                    return finish_value(state);
                }
                if byte != b'"' {
                    return false;
                }
                if seen.iter().all(|seen| *seen) && object_additional(schemas, schema_id).is_none() {
                    return false;
                }
                let candidates = (0..seen.len()).collect();
                state.mode = Mode::ObjectKey { schema: schema_id, seen: seen.clone(), candidates, offset: 0, utf8: Utf8State::default() };
                return true;
            }
            Mode::ObjectKey { schema, seen, candidates, offset, utf8 } => {
                let properties = object_properties(schemas, *schema);
                if utf8.remaining != 0 {
                    if byte < utf8.min || byte > utf8.max {
                        return false;
                    }
                    utf8.remaining -= 1;
                    utf8.min = 0x80;
                    utf8.max = 0xbf;
                    candidates.retain(|&candidate| properties[candidate].name.get(*offset).copied() == Some(byte));
                    if candidates.is_empty() && object_additional(schemas, *schema).is_none() {
                        return false;
                    }
                    *offset += 1;
                    return true;
                }
                if byte == b'"' {
                    let property = candidates.iter().copied().find(|&candidate| properties[candidate].name.len() == *offset);
                    if property.is_some_and(|property| seen[property]) {
                        return false;
                    }
                    if property.is_none() && object_additional(schemas, *schema).is_none() {
                        return false;
                    }
                    state.mode = Mode::ObjectColon { schema: *schema, seen: seen.clone(), property };
                    return true;
                }
                match byte {
                    // 动态 key 选择不生成转义拼写，避免其解码后绕过声明字段的
                    // duplicate/type 约束；未转义 UTF-8 已覆盖协议常用 key。
                    b'\\' => return false,
                    0x00..=0x1f => return false,
                    0x20..=0x7f => {}
                    0xc2..=0xdf => *utf8 = Utf8State { remaining: 1, min: 0x80, max: 0xbf },
                    0xe0 => *utf8 = Utf8State { remaining: 2, min: 0xa0, max: 0xbf },
                    0xe1..=0xec | 0xee..=0xef => *utf8 = Utf8State { remaining: 2, min: 0x80, max: 0xbf },
                    0xed => *utf8 = Utf8State { remaining: 2, min: 0x80, max: 0x9f },
                    0xf0 => *utf8 = Utf8State { remaining: 3, min: 0x90, max: 0xbf },
                    0xf1..=0xf3 => *utf8 = Utf8State { remaining: 3, min: 0x80, max: 0xbf },
                    0xf4 => *utf8 = Utf8State { remaining: 3, min: 0x80, max: 0x8f },
                    _ => return false,
                }
                candidates.retain(|&candidate| properties[candidate].name.get(*offset).copied() == Some(byte));
                if candidates.is_empty() && object_additional(schemas, *schema).is_none() {
                    return false;
                }
                *offset += 1;
                return true;
            }
            Mode::ObjectColon { schema, seen, property } => {
                if byte.is_ascii_whitespace() {
                    return true;
                }
                if byte != b':' {
                    return false;
                }
                let child = property.map(|property| object_properties(schemas, *schema)[property].schema).or_else(|| object_additional(schemas, *schema)).expect("object key 已验证");
                state.returns.push(Return::Object { schema: *schema, seen: seen.clone(), property: *property });
                state.mode = Mode::Value(child);
                return true;
            }
            Mode::ObjectAfterValue { schema, seen } => {
                if byte.is_ascii_whitespace() {
                    return true;
                }
                if byte == b'}' && required_seen(schemas, *schema, seen) {
                    return finish_value(state);
                }
                if byte == b',' {
                    state.mode = Mode::ObjectStart { schema: *schema, seen: seen.clone() };
                    return true;
                }
                return false;
            }
            Mode::String { schema, escape, utf8, has_content, characters, raw } => {
                if *escape == 1 {
                    if let Some(raw) = raw.as_mut() {
                        raw.push(byte);
                    }
                    match byte {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            *escape = 0;
                            *has_content = true;
                            *characters += 1;
                        }
                        b'u' => *escape = 5,
                        _ => return false,
                    }
                    return string_prefix_within_max(schemas, *schema, *characters);
                }
                if *escape > 1 {
                    if !byte.is_ascii_hexdigit() {
                        return false;
                    }
                    if let Some(raw) = raw.as_mut() {
                        raw.push(byte);
                    }
                    *escape -= 1;
                    if *escape == 1 {
                        *escape = 0;
                        *has_content = true;
                        *characters += 1;
                    }
                    return string_prefix_within_max(schemas, *schema, *characters);
                }
                if utf8.remaining != 0 {
                    if byte < utf8.min || byte > utf8.max {
                        return false;
                    }
                    if let Some(raw) = raw.as_mut() {
                        raw.push(byte);
                    }
                    utf8.remaining -= 1;
                    utf8.min = 0x80;
                    utf8.max = 0xbf;
                    return true;
                }
                match byte {
                    b'"' => {
                        if !string_matches_schema(&schemas[*schema], *has_content, *characters, raw.as_deref()) {
                            return false;
                        }
                        return finish_value(state);
                    }
                    b'\\' => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *escape = 1;
                    }
                    0x00..=0x1f => return false,
                    0x20..=0x7f => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *has_content = true;
                        *characters += 1;
                    }
                    0xc2..=0xdf => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 1, min: 0x80, max: 0xbf };
                        *has_content = true;
                        *characters += 1;
                    }
                    0xe0 => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 2, min: 0xa0, max: 0xbf };
                        *has_content = true;
                        *characters += 1;
                    }
                    0xe1..=0xec | 0xee..=0xef => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 2, min: 0x80, max: 0xbf };
                        *has_content = true;
                        *characters += 1;
                    }
                    0xed => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 2, min: 0x80, max: 0x9f };
                        *has_content = true;
                        *characters += 1;
                    }
                    0xf0 => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 3, min: 0x90, max: 0xbf };
                        *has_content = true;
                        *characters += 1;
                    }
                    0xf1..=0xf3 => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 3, min: 0x80, max: 0xbf };
                        *has_content = true;
                        *characters += 1;
                    }
                    0xf4 => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 3, min: 0x80, max: 0x8f };
                        *has_content = true;
                        *characters += 1;
                    }
                    _ => return false,
                }
                return string_prefix_within_max(schemas, *schema, *characters);
            }
            Mode::RawString { schema, utf8, characters, raw } => {
                if utf8.remaining != 0 {
                    if byte < utf8.min || byte > utf8.max {
                        return false;
                    }
                    if let Some(raw) = raw.as_mut() {
                        raw.push(byte);
                    }
                    utf8.remaining -= 1;
                    utf8.min = 0x80;
                    utf8.max = 0xbf;
                    return true;
                }
                match byte {
                    b'\t' | b'\n' | b'\r' | 0x20..=0x7f => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *characters += 1;
                    }
                    0xc2..=0xdf => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 1, min: 0x80, max: 0xbf };
                        *characters += 1;
                    }
                    0xe0 => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 2, min: 0xa0, max: 0xbf };
                        *characters += 1;
                    }
                    0xe1..=0xec | 0xee..=0xef => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 2, min: 0x80, max: 0xbf };
                        *characters += 1;
                    }
                    0xed => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 2, min: 0x80, max: 0x9f };
                        *characters += 1;
                    }
                    0xf0 => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 3, min: 0x90, max: 0xbf };
                        *characters += 1;
                    }
                    0xf1..=0xf3 => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 3, min: 0x80, max: 0xbf };
                        *characters += 1;
                    }
                    0xf4 => {
                        if let Some(raw) = raw.as_mut() {
                            raw.push(byte);
                        }
                        *utf8 = Utf8State { remaining: 3, min: 0x80, max: 0x8f };
                        *characters += 1;
                    }
                    _ => return false,
                }
                return string_prefix_within_max(schemas, *schema, *characters);
            }
            Mode::StringChoice { schema, candidates, offset } => {
                let Schema::String { choices: Some(choices), .. } = &schemas[*schema] else { unreachable!() };
                if byte == b'"' {
                    if candidates.iter().any(|&candidate| choices[candidate].len() == *offset && string_matches_schema(&schemas[*schema], !choices[candidate].is_empty(), 0, Some(&choices[candidate]))) {
                        return finish_value(state);
                    }
                    return false;
                }
                candidates.retain(|&candidate| choices[candidate].get(*offset).copied() == Some(byte));
                if candidates.is_empty() {
                    return false;
                }
                *offset += 1;
                return true;
            }
            Mode::Number { schema, state: number, negative, magnitude, overflow, raw } => {
                let integer = matches!(schemas[*schema], Schema::Integer { .. });
                if advance_number(integer, number, byte) {
                    raw.push(byte);
                    if integer && byte.is_ascii_digit() {
                        if let Some(next) = magnitude.checked_mul(10).and_then(|value| value.checked_add(u128::from(byte - b'0'))) {
                            *magnitude = next;
                        } else {
                            *overflow = true;
                        }
                        if !integer_prefix_viable(schemas, *schema, *negative, *magnitude, *overflow) {
                            return false;
                        }
                    }
                    return true;
                }
                if matches!(number, NumberState::Zero | NumberState::Integer | NumberState::Fraction | NumberState::ExponentDigits) && number_matches_schema(schemas, *schema, *negative, *magnitude, *overflow, raw) {
                    if !finish_value(state) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            Mode::Literal { bytes, offset } => {
                if bytes.get(*offset).copied() != Some(byte) {
                    return false;
                }
                *offset += 1;
                if *offset == bytes.len() {
                    return finish_value(state);
                }
                return true;
            }
            Mode::Fixed { schema, offset } => {
                let Schema::Literal { bytes } = &schemas[*schema] else { unreachable!() };
                if bytes.get(*offset).copied() != Some(byte) {
                    return false;
                }
                *offset += 1;
                if *offset == bytes.len() {
                    return finish_value(state);
                }
                return true;
            }
            Mode::Union { alternatives } => {
                alternatives.retain_mut(|alternative| advance_byte(schemas, alternative, byte));
                return !alternatives.is_empty();
            }
            Mode::Intersection { alternatives } => {
                return alternatives.iter_mut().all(|alternative| advance_byte(schemas, alternative, byte));
            }
            Mode::Complete => return byte.is_ascii_whitespace(),
            Mode::Dead => return false,
        }
    }
    false
}

fn state_complete(schemas: &[Schema], state: &State) -> bool {
    match &state.mode {
        Mode::Complete => true,
        Mode::Number { schema, state: number, negative, magnitude, overflow, raw } if state.returns.is_empty() && matches!(number, NumberState::Zero | NumberState::Integer | NumberState::Fraction | NumberState::ExponentDigits) => {
            number_matches_schema(schemas, *schema, *negative, *magnitude, *overflow, raw)
        }
        Mode::RawString { schema, utf8, characters, raw } if utf8.remaining == 0 => raw_string_matches_schema(schemas, *schema, *characters, raw.as_deref()),
        Mode::Union { alternatives } => alternatives.iter().any(|alternative| state_complete(schemas, alternative)),
        Mode::Intersection { alternatives } => alternatives.iter().all(|alternative| state_complete(schemas, alternative)),
        _ => false,
    }
}

fn number_matches_schema(schemas: &[Schema], schema: usize, negative: bool, magnitude: u128, overflow: bool, raw: &[u8]) -> bool {
    let Schema::Integer { minimum, maximum } = &schemas[schema] else {
        let Schema::Number { minimum, maximum } = &schemas[schema] else { return true };
        let Ok(value) = serde_json::from_slice::<f64>(raw) else { return false };
        return minimum.is_none_or(|(minimum, inclusive)| if inclusive { value >= minimum } else { value > minimum }) && maximum.is_none_or(|(maximum, inclusive)| if inclusive { value <= maximum } else { value < maximum });
    };
    if overflow {
        return if negative { minimum.is_none() } else { maximum.is_none() };
    }
    let value = if negative { if magnitude == 1_u128 << 127 { Some(i128::MIN) } else { i128::try_from(magnitude).ok().and_then(i128::checked_neg) } } else { i128::try_from(magnitude).ok() };
    match value {
        Some(value) => minimum.is_none_or(|minimum| value >= minimum) && maximum.is_none_or(|maximum| value <= maximum),
        None => {
            if negative {
                minimum.is_none()
            } else {
                maximum.is_none()
            }
        }
    }
}

fn integer_prefix_viable(schemas: &[Schema], schema: usize, negative: bool, magnitude: u128, overflow: bool) -> bool {
    let Schema::Integer { minimum, maximum } = &schemas[schema] else { return true };
    if overflow {
        return if negative { minimum.is_none() } else { maximum.is_none() };
    }
    if negative {
        let value = if magnitude == 1_u128 << 127 { Some(i128::MIN) } else { i128::try_from(magnitude).ok().and_then(i128::checked_neg) };
        value.is_some_and(|value| minimum.is_none_or(|minimum| value >= minimum)) || minimum.is_none()
    } else {
        i128::try_from(magnitude).ok().is_some_and(|value| maximum.is_none_or(|maximum| value <= maximum)) || maximum.is_none()
    }
}

fn number_mode_can_continue_or_complete(schemas: &[Schema], mode: &Mode) -> bool {
    let Mode::Number { schema, state, negative, magnitude, overflow, raw } = mode else { return false };
    number_matches_schema(schemas, *schema, *negative, *magnitude, *overflow, raw) || *state == NumberState::Integer && integer_prefix_viable(schemas, *schema, *negative, *magnitude, *overflow)
}

fn schema_supports_raw_string(schemas: &[Schema], schema: usize) -> bool {
    match &schemas[schema] {
        Schema::String { .. } => true,
        Schema::Literal { bytes } => serde_json::from_slice::<Value>(bytes).is_ok_and(|value| value.is_string()),
        Schema::Union { variants } => variants.iter().any(|&variant| schema_supports_raw_string(schemas, variant)),
        Schema::Intersection { variants } => variants.iter().all(|&variant| schema_supports_raw_string(schemas, variant)),
        _ => false,
    }
}

fn string_requires_raw(schemas: &[Schema], schema: usize) -> bool {
    match &schemas[schema] {
        Schema::String { choices, pattern, format, .. } => choices.is_some() || pattern.is_some() || format.is_some(),
        Schema::Literal { .. } => true,
        Schema::Union { variants } | Schema::Intersection { variants } => variants.iter().any(|&variant| string_requires_raw(schemas, variant)),
        _ => false,
    }
}

fn string_prefix_within_max(schemas: &[Schema], schema: usize, characters: usize) -> bool {
    match &schemas[schema] {
        Schema::String { max_length, .. } => max_length.is_none_or(|maximum| characters <= maximum),
        Schema::Literal { bytes } => serde_json::from_slice::<String>(bytes).is_ok_and(|value| characters <= value.chars().count()),
        Schema::Union { variants } => variants.iter().any(|&variant| string_prefix_within_max(schemas, variant, characters)),
        Schema::Intersection { variants } => variants.iter().all(|&variant| string_prefix_within_max(schemas, variant, characters)),
        _ => false,
    }
}

fn string_matches_schema(schema: &Schema, has_content: bool, characters: usize, raw: Option<&[u8]>) -> bool {
    let Schema::String { min_length, max_length, pattern, format, .. } = schema else { return false };
    if raw.is_none() {
        return characters >= *min_length && max_length.is_none_or(|maximum| characters <= maximum) && (*min_length == 0 || has_content);
    }
    let mut quoted = Vec::with_capacity(raw.map_or(2, |raw| raw.len() + 2));
    quoted.push(b'"');
    quoted.extend_from_slice(raw.unwrap_or_default());
    quoted.push(b'"');
    let Ok(text) = serde_json::from_slice::<String>(&quoted) else { return false };
    let length = text.chars().count();
    length >= *min_length
        && max_length.is_none_or(|maximum| length <= maximum)
        && pattern.as_ref().is_none_or(|pattern| pattern.is_match(text.as_bytes()))
        && format.is_none_or(|format| match format {
            StringFormat::Uri => url::Url::parse(&text).is_ok(),
        })
}

fn raw_string_matches_schema(schemas: &[Schema], schema: usize, characters: usize, raw: Option<&[u8]>) -> bool {
    match &schemas[schema] {
        Schema::String { choices, min_length, max_length, pattern, format } => {
            if characters < *min_length || max_length.is_some_and(|maximum| characters > maximum) {
                return false;
            }
            let text = raw.unwrap_or_default();
            if choices.as_ref().is_some_and(|choices| {
                !choices.iter().any(|choice| {
                    let mut quoted = Vec::with_capacity(choice.len() + 2);
                    quoted.push(b'"');
                    quoted.extend_from_slice(choice);
                    quoted.push(b'"');
                    serde_json::from_slice::<String>(&quoted).is_ok_and(|choice| choice.as_bytes() == text)
                })
            }) {
                return false;
            }
            pattern.as_ref().is_none_or(|pattern| pattern.is_match(text))
                && format.is_none_or(|format| match format {
                    StringFormat::Uri => std::str::from_utf8(text).is_ok_and(|text| url::Url::parse(text).is_ok()),
                })
        }
        Schema::Literal { bytes } => serde_json::from_slice::<String>(bytes).is_ok_and(|value| value.as_bytes() == raw.unwrap_or_default()),
        Schema::Union { variants } => variants.iter().any(|&variant| raw_string_matches_schema(schemas, variant, characters, raw)),
        Schema::Intersection { variants } => variants.iter().all(|&variant| raw_string_matches_schema(schemas, variant, characters, raw)),
        _ => false,
    }
}

fn normalized_cache_state(schemas: &[Schema], state: &State, max_token_bytes: usize) -> Option<State> {
    let mut state = state.clone();
    normalize_cache_state(schemas, &mut state, max_token_bytes).then_some(state)
}

fn normalize_cache_state(schemas: &[Schema], state: &mut State, max_token_bytes: usize) -> bool {
    match &mut state.mode {
        Mode::String { schema, characters, raw, .. } => {
            if raw.is_some() {
                return false;
            }
            let Schema::String { min_length, max_length, .. } = &schemas[*schema] else { return true };
            if *characters >= *min_length {
                if let Some(maximum) = max_length {
                    if maximum.saturating_sub(*characters) > max_token_bytes {
                        *characters = maximum.saturating_sub(max_token_bytes + 1);
                    }
                } else {
                    *characters = *min_length;
                }
            }
            true
        }
        Mode::RawString { schema, characters, raw, .. } => {
            if raw.is_some() {
                return false;
            }
            let Schema::String { min_length, max_length, .. } = &schemas[*schema] else { return true };
            if *characters >= *min_length {
                if let Some(maximum) = max_length {
                    if maximum.saturating_sub(*characters) > max_token_bytes {
                        *characters = maximum.saturating_sub(max_token_bytes + 1);
                    }
                } else {
                    *characters = *min_length;
                }
            }
            true
        }
        Mode::Union { alternatives } | Mode::Intersection { alternatives } => alternatives.iter_mut().all(|alternative| normalize_cache_state(schemas, alternative, max_token_bytes)),
        _ => true,
    }
}

fn finish_value(state: &mut State) -> bool {
    state.mode = match state.returns.pop() {
        Some(Return::Array { schema, count }) => Mode::ArrayAfterValue { schema, count: count + 1 },
        Some(Return::Object { schema, mut seen, property }) => {
            if let Some(property) = property {
                seen[property] = true;
            }
            Mode::ObjectAfterValue { schema, seen }
        }
        None => Mode::Complete,
    };
    true
}

fn array_spec(schemas: &[Schema], schema: usize) -> (usize, usize, Option<usize>) {
    let Schema::Array { item, min_items, max_items } = &schemas[schema] else { unreachable!() };
    (*item, *min_items, *max_items)
}

fn object_properties(schemas: &[Schema], schema: usize) -> &[Property] {
    let Schema::Object { properties, .. } = &schemas[schema] else { unreachable!() };
    properties
}

fn object_additional(schemas: &[Schema], schema: usize) -> Option<usize> {
    let Schema::Object { additional, .. } = &schemas[schema] else { unreachable!() };
    *additional
}

fn required_seen(schemas: &[Schema], schema: usize, seen: &[bool]) -> bool {
    object_properties(schemas, schema).iter().zip(seen).all(|(property, seen)| !property.required || *seen)
}

fn advance_number(integer: bool, state: &mut NumberState, byte: u8) -> bool {
    match (*state, byte) {
        (NumberState::Minus, b'0') => *state = NumberState::Zero,
        (NumberState::Minus, b'1'..=b'9') => *state = NumberState::Integer,
        (NumberState::Zero | NumberState::Integer, b'0'..=b'9') if *state == NumberState::Integer => {}
        (NumberState::Zero | NumberState::Integer, b'.') if !integer => *state = NumberState::Dot,
        (NumberState::Zero | NumberState::Integer, b'e' | b'E') if !integer => *state = NumberState::Exponent,
        (NumberState::Dot, b'0'..=b'9') => *state = NumberState::Fraction,
        (NumberState::Fraction, b'0'..=b'9') => {}
        (NumberState::Fraction, b'e' | b'E') => *state = NumberState::Exponent,
        (NumberState::Exponent, b'+' | b'-') => *state = NumberState::ExponentSign,
        (NumberState::Exponent | NumberState::ExponentSign, b'0'..=b'9') => *state = NumberState::ExponentDigits,
        (NumberState::ExponentDigits, b'0'..=b'9') => {}
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn accepts(schema: &Value, text: &str) -> bool {
        let mut schemas = Vec::new();
        let root = compile_schema(schema, &mut schemas).unwrap();
        let mut state = State { mode: Mode::Value(root), returns: Vec::new() };
        text.bytes().all(|byte| advance_byte(&schemas, &mut state, byte)) && state_complete(&schemas, &state)
    }

    fn ascii_tokens() -> Arc<JsonTokenTable> {
        let mut bytes = (0u8..=127).map(|byte| Some(vec![byte].into_boxed_slice())).collect::<Vec<_>>();
        bytes.push(Some(b"</".to_vec().into_boxed_slice()));
        Arc::new(JsonTokenTable { bytes: bytes.into(), vocab_size: 129, max_bytes: 2 })
    }

    fn advance_text(fence: &mut JsonSchemaFence, text: &str) {
        for token in text.bytes().map(u32::from) {
            let current = fence.fence(128);
            assert_ne!(current.forced(), Some(128), "JSON 在输入结束前提前完成: {text}");
            assert!(current.excluded().binary_search(&token).is_err(), "JSON fence 拒绝合法 token={token} text={text:?}");
            fence.advance(token);
        }
    }

    #[test]
    fn todo_schema拒绝损坏结构与枚举截断() {
        let schema = json!({"type":"array","items":{"type":"object","properties":{"content":{"type":"string","minLength":1},"priority":{"type":"string","enum":["high","medium","low"]},"status":{"type":"string","enum":["pending","in_progress","completed"]}},"required":["content","priority","status"],"additionalProperties":false}});
        assert!(accepts(&schema, r#"[{"content":"read","priority":"high","status":"in_progress"}]"#));
        assert!(!accepts(&schema, r#"[{"content":"","priority":"high","status":"pending"}]"#));
        assert!(!accepts(&schema, r#"[{"content":"read","priority":"high","status":"in_pro"}]"#));
        assert!(!accepts(&schema, r#"[{"content":"read"},"priority":"high","status":"pending"}]"#));
        assert!(!accepts(&schema, r#"[{"content":"read","priority":"high","status":"pending","":"pending"}]"#));
    }

    #[test]
    fn object属性顺序不敏感且必填完整() {
        let schema = json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},"required":["a","b"],"additionalProperties":false});
        assert!(accepts(&schema, r#"{"b":true,"a":-12}"#));
        assert!(!accepts(&schema, r#"{"a":1}"#));
        assert!(!accepts(&schema, r#"{"a":01,"b":true}"#));
    }

    #[test]
    fn schema值additional_properties约束动态字段() {
        let schema = json!({
            "type":"object",
            "properties":{"questions":{"type":"array","items":{"type":"string"}}},
            "required":["questions"],
            "additionalProperties":{"type":"string"}
        });
        assert!(accepts(&schema, r#"{"questions":["继续？"],"answer":"是"}"#));
        assert!(accepts(&schema, r#"{"动态字段":"值","questions":[]}"#));
        assert!(!accepts(&schema, r#"{"questions":[],"answer":1}"#));
        assert!(!accepts(&schema, r#"{"questions":[],"questions":"损坏"}"#));
    }

    #[test]
    fn additional_properties缺省时生成声明字段合法子集() {
        let schema = json!({"type":"object","properties":{"值":{"type":"integer"}},"required":["值"]});
        assert!(accepts(&schema, r#"{"值":1}"#));
        assert!(!accepts(&schema, r#"{"值":1,"unknown":2}"#));
    }

    #[test]
    fn required必须引用已声明字段() {
        let mut schemas = Vec::new();
        let error = compile_schema(&json!({"type":"object","required":["missing"],"additionalProperties":{"type":"string"}}), &mut schemas).unwrap_err();
        assert!(error.contains("missing"));
    }

    #[test]
    fn one_of联合类型执行整数范围() {
        let schema = json!({"description":"relative delay","oneOf":[{"type":"integer","exclusiveMinimum":0,"maximum":525600},{"type":"null"}]});
        assert!(accepts(&schema, "1"));
        assert!(accepts(&schema, "525600"));
        assert!(accepts(&schema, "null"));
        assert!(!accepts(&schema, "0"));
        assert!(!accepts(&schema, "-1"));
        assert!(!accepts(&schema, "525601"));
        assert!(!accepts(&schema, r#""1""#));
    }

    #[test]
    fn one_of拒绝可能同时匹配的分支() {
        let mut schemas = Vec::new();
        let error = compile_schema(&json!({"oneOf":[{"type":"integer"},{"type":"number"}]}), &mut schemas).unwrap_err();
        assert!(error.contains("exactly-one"));
    }

    #[test]
    fn one_of围栏只在合法分支完整后闭合() {
        let schema = json!({"oneOf":[{"type":"integer","exclusiveMinimum":0,"maximum":3},{"type":"null"}]});
        let mut integer = JsonSchemaFence::new(&schema, ascii_tokens()).unwrap();
        advance_text(&mut integer, "3");
        assert!(integer.fence(128).excluded().binary_search(&128).is_err());
        assert!(integer.fence(128).excluded().binary_search(&u32::from(b'0')).is_ok());

        let mut null = JsonSchemaFence::new(&schema, ascii_tokens()).unwrap();
        advance_text(&mut null, "null");
        assert!(null.fence(128).excluded().binary_search(&128).is_err());

        let zero = JsonSchemaFence::new(&schema, ascii_tokens()).unwrap();
        assert!(zero.fence(128).excluded().binary_search(&u32::from(b'0')).is_ok());
    }

    #[test]
    fn array严格执行长度且拒绝尾逗号() {
        let schema = json!({"type":"array","items":{"type":"boolean"},"minItems":2,"maxItems":3});
        assert!(accepts(&schema, "[true,false]"));
        assert!(accepts(&schema, "[true,false,true]"));
        assert!(!accepts(&schema, "[]"));
        assert!(!accepts(&schema, "[true]"));
        assert!(!accepts(&schema, "[true,false,true,false]"));
        assert!(!accepts(&schema, "[true,false,]"));
    }

    #[test]
    fn string严格执行长度pattern与uri() {
        let pattern = json!({"type":"string","minLength":6,"maxLength":12,"pattern":"^sess_[A-Za-z0-9._-]+$"});
        assert!(accepts(&pattern, r#""sess_a-1""#));
        assert!(!accepts(&pattern, r#""sess_""#));
        assert!(!accepts(&pattern, r#""other_a""#));
        assert!(!accepts(&pattern, r#""sess_abcdefgh""#));

        let uri = json!({"type":"string","format":"uri"});
        assert!(accepts(&uri, r#""https://example.com/a?q=1""#));
        assert!(accepts(&uri, r#""mailto:user@example.com""#));
        assert!(!accepts(&uri, r#""not a uri""#));
    }

    #[test]
    fn dsml_raw_string在闭合标签候选前执行schema() {
        let schema = json!({"type":"string","minLength":6,"maxLength":12,"pattern":"^sess_[A-Za-z0-9._-]+$"});
        let mut valid = JsonSchemaFence::new_raw_string(&schema, &schema, ascii_tokens()).unwrap();
        advance_text(&mut valid, "sess_a-1");
        assert!(valid.fence(128).excluded().binary_search(&128).is_err());

        let mut invalid = JsonSchemaFence::new_raw_string(&schema, &schema, ascii_tokens()).unwrap();
        advance_text(&mut invalid, "other_a");
        assert!(invalid.fence(128).excluded().binary_search(&128).is_ok());
    }

    #[test]
    fn number范围在闭合候选前验证() {
        let schema = json!({"type":"number","minimum":0,"exclusiveMaximum":10});
        assert!(accepts(&schema, "0"));
        assert!(accepts(&schema, "9.5"));
        assert!(!accepts(&schema, "-0.1"));
        assert!(!accepts(&schema, "10"));
    }

    #[test]
    fn type_array_const_enum_ref与all_of使用同一组合状态机() {
        assert!(accepts(&json!({}), "null"));
        assert!(accepts(&json!({"description":"unconstrained"}), "null"));
        assert!(accepts(&json!({"type":"object","properties":{"value":{}},"required":["value"],"additionalProperties":false}), r#"{"value":null}"#));
        assert!(accepts(&json!({"type":["integer","null"]}), "null"));
        assert!(accepts(&json!({"type":["integer","null"]}), "7"));
        assert!(accepts(&json!({"const":{"ok":true}}), r#"{"ok":true}"#));
        assert!(accepts(&json!({"enum":[1,"x",null]}), r#""x""#));
        assert!(accepts(&json!({"allOf":[{"type":"integer","minimum":1},{"type":"integer","maximum":3}]}), "2"));
        assert!(!accepts(&json!({"allOf":[{"type":"integer","minimum":1},{"type":"integer","maximum":3}]}), "4"));

        let reference = json!({"type":"object","properties":{"value":{"$ref":"#/$defs/value"}},"required":["value"],"additionalProperties":false,"$defs":{"value":{"type":"string","minLength":2}}});
        assert!(accepts(&reference, r#"{"value":"ok"}"#));
        assert!(!accepts(&reference, r#"{"value":"x"}"#));
    }

    #[test]
    fn 组合与ref接受并执行同层type约束() {
        let any_of = json!({"type":"string","anyOf":[{"const":"read"},{"const":"write"}]});
        assert!(accepts(&any_of, r#""read""#));
        assert!(accepts(&any_of, r#""write""#));
        assert!(!accepts(&any_of, "1"));

        let one_of = json!({"type":"integer","oneOf":[{"const":1},{"const":2}]});
        assert!(accepts(&one_of, "1"));
        assert!(accepts(&one_of, "2"));
        assert!(!accepts(&one_of, r#""1""#));

        let all_of = json!({"type":"integer","allOf":[{"type":"integer","minimum":1},{"type":"integer","maximum":3}]});
        assert!(accepts(&all_of, "2"));
        assert!(!accepts(&all_of, "4"));

        let reference = json!({"$ref":"#/$defs/value","type":"string","maxLength":3,"$defs":{"value":{"type":"string","minLength":2}}});
        assert!(accepts(&reference, r#""ok""#));
        assert!(!accepts(&reference, r#""x""#));
        assert!(!accepts(&reference, r#""long""#));

        let mut raw = JsonSchemaFence::new_raw_string(&any_of, &any_of, ascii_tokens()).unwrap();
        advance_text(&mut raw, "read");
        assert!(raw.complete());

        let mut invalid_raw = JsonSchemaFence::new_raw_string(&any_of, &any_of, ascii_tokens()).unwrap();
        advance_text(&mut invalid_raw, "other");
        assert!(!invalid_raw.complete());
    }

    #[test]
    fn unsupported关键字不会静默丢失() {
        let mut schemas = Vec::new();
        assert!(compile_schema(&json!({"type":"array","items":{"type":"string"},"uniqueItems":true}), &mut schemas).unwrap_err().contains("uniqueItems"));
        schemas.clear();
        assert!(compile_schema(&json!({"type":"number","multipleOf":2}), &mut schemas).unwrap_err().contains("multipleOf"));
    }

    #[test]
    fn token围栏在采样前屏蔽损坏todo并只在完整后闭合() {
        let schema = json!({"type":"array","items":{"type":"object","properties":{"content":{"type":"string","minLength":1},"priority":{"type":"string","enum":["high","medium","low"]},"status":{"type":"string","enum":["pending","in_progress","completed"]}},"required":["content","priority","status"],"additionalProperties":false}});
        let mut fence = JsonSchemaFence::new(&schema, ascii_tokens()).unwrap();
        advance_text(&mut fence, r#"[{"content":"read","priority":"high","status":"in_pro"#);
        assert!(fence.fence(128).excluded().binary_search(&u32::from(b'"')).is_ok());
        advance_text(&mut fence, r#"gress"}]"#);
        assert_eq!(fence.fence(128).forced(), Some(128));

        let mut missing_object = JsonSchemaFence::new(&schema, ascii_tokens()).unwrap();
        advance_text(&mut missing_object, "[");
        assert!(missing_object.fence(128).excluded().binary_search(&u32::from(b'"')).is_ok());
    }

    #[test]
    fn root数字无需额外空格即可闭合参数() {
        let mut fence = JsonSchemaFence::new(&json!({"type":"integer"}), ascii_tokens()).unwrap();
        advance_text(&mut fence, "-12");
        let current = fence.fence(128);
        assert!(current.excluded().binary_search(&128).is_err());
        assert!(current.excluded().binary_search(&u32::from(b'3')).is_err());
    }
}
