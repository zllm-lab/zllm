//! 宿主无关的生成会话状态、运行指标与流式文本边界。

use std::{
    path::Path,
    sync::{Arc, Mutex, atomic::AtomicU64},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::kv_cache::terminal_cache::{ResidencyBudget, ResidencyReservation, TerminalInfo, TerminalResidencyStats, TerminalSessions, TerminalSnapshot};

pub type DynError = Box<dyn std::error::Error + Send + Sync>;

/// 生成持久化 cache 的模型身份。只读取文件元数据，不扫描大权重内容；调用方把
/// 会改变 cache 布局/执行状态的配置放入 `layout`。
pub fn model_cache_identity(path: &std::path::Path, layout: &str) -> Result<String, String> {
    fn visit(root: &std::path::Path, path: &std::path::Path, entries: &mut Vec<String>) -> Result<(), String> {
        let metadata = std::fs::metadata(path).map_err(|error| format!("读取模型 identity {}: {error}", path.display()))?;
        if metadata.is_dir() {
            let mut children = std::fs::read_dir(path).map_err(|error| format!("扫描模型 identity {}: {error}", path.display()))?.collect::<Result<Vec<_>, _>>().map_err(|error| format!("扫描模型 identity {}: {error}", path.display()))?;
            children.sort_by_key(|entry| entry.file_name());
            for child in children {
                visit(root, &child.path(), entries)?;
            }
        } else if metadata.is_file() {
            let relative = path.strip_prefix(root).unwrap_or(path);
            let modified = metadata.modified().ok().and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |duration| duration.as_nanos());
            entries.push(format!("{}:{}:{modified}", relative.display(), metadata.len()));
        }
        Ok(())
    }

    let canonical = path.canonicalize().map_err(|error| format!("解析模型 identity {}: {error}", path.display()))?;
    let mut entries = Vec::new();
    visit(&canonical, &canonical, &mut entries)?;
    let bytes = serde_json::to_vec(&(canonical, layout, entries)).map_err(|error| format!("序列化模型 identity: {error}"))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvCacheDeviceCapacity {
    pub device: String,
    pub available_bytes: u64,
    pub bytes_per_token: u64,
    pub token_capacity: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub backend: String,
    pub platform: String,
    pub architecture: String,
    pub accelerator: String,
    pub compute_units: Option<usize>,
    pub compute_unit_kind: String,
    pub memory_kind: String,
    pub unified_memory: bool,
    pub system_memory_bytes: Option<u64>,
    pub accelerator_memory_bytes: Option<u64>,
    pub recommended_working_set_bytes: Option<u64>,
    pub model_format: String,
    pub model_bytes: u64,
    pub max_seq_len: usize,
    pub kv_cache_format: String,
    #[serde(default)]
    pub kv_cache_devices: Vec<KvCacheDeviceCapacity>,
    #[serde(default)]
    pub kv_reservation_page_tokens: usize,
    #[serde(default)]
    pub task_kinds: Vec<String>,
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub output_modalities: Vec<String>,
    #[serde(default)]
    pub artifact_streaming: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RuntimeStatus {
    /// 模型权重、MTP/replay、视觉权重与长期 scratch 的当前常驻量。
    pub engine_resident_bytes: u64,
    /// 固定整块 session 的 KV/recurrent/draft cache 预算。
    pub session_resident_bytes: u64,
    pub accelerator_allocated_bytes: u64,
    pub accelerator_memory_used_bytes: Option<u64>,
    pub accelerator_utilization_percent: Option<f32>,
    pub temperature_celsius: Option<f32>,
    pub power_watts: Option<f32>,
    pub resident_expert_bytes: u64,
    pub kv_cache_entries: usize,
    pub kv_cache_tokens: usize,
    pub kv_cache_allocated_bytes: u64,
    pub kv_cache_used_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub current_batch_tokens: usize,
    /// 正在执行完整新会话 prefill 的 session 数。
    #[serde(default)]
    pub new_prefill: usize,
    /// 命中已有 cache 后正在追加 prefill 的 session 数。
    #[serde(default)]
    pub append_prefill: usize,
    /// 已进入生成阶段的 session 数（普通 decode 与 DSpark verify 都计一条）。
    #[serde(default)]
    pub decode: usize,
    /// 当前实际采用的 DSpark draft token 数；0 表示本轮未启用 DSpark。
    #[serde(default)]
    pub dspark_draft_tokens: usize,
    /// 当前可直接恢复的内存终点 cache。
    #[serde(default)]
    pub memory_cache_ids: Vec<String>,
    /// 当前可从本机 SSD 恢复的终点 cache。
    #[serde(default)]
    pub ssd_cache_ids: Vec<String>,
    pub compute_steps_total: u64,
}

impl RuntimeStatus {
    pub const MAX_SCHEDULING_PRESSURE: usize = 22;

    pub fn scheduling_pressure(&self) -> usize {
        self.new_prefill.saturating_mul(4).saturating_add(self.append_prefill).saturating_add(self.decode)
    }
}

pub fn effective_dspark_draft_tokens(configured: usize, decode: usize) -> usize {
    if decode == 0 { 0 } else { configured.min(32 / decode) }
}

#[derive(Debug, Default)]
pub struct AtomicCounterU64(AtomicU64);

impl AtomicCounterU64 {
    pub fn new(value: u64) -> Self {
        Self(AtomicU64::new(value))
    }

    pub fn fetch_add(&self, delta: u64) -> u64 {
        self.0.fetch_add(delta, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn load(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Clone for AtomicCounterU64 {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.load()))
    }
}

pub struct BatchTokenGuard {
    runtime: Arc<Mutex<RuntimeStatus>>,
    tokens: usize,
}

impl BatchTokenGuard {
    pub fn new(runtime: &Arc<Mutex<RuntimeStatus>>, tokens: usize) -> Self {
        if let Ok(mut runtime) = runtime.lock() {
            runtime.current_batch_tokens = runtime.current_batch_tokens.saturating_add(tokens);
        }
        Self { runtime: runtime.clone(), tokens }
    }
}

impl Drop for BatchTokenGuard {
    fn drop(&mut self) {
        if let Ok(mut runtime) = self.runtime.lock() {
            runtime.current_batch_tokens = runtime.current_batch_tokens.saturating_sub(self.tokens);
        }
    }
}

/// 库调用方和节点适配器共用的 KV 常驻快照；避免用位置元组传递五个同类整数。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvResidency {
    pub capacity_bytes: usize,
    pub active_bytes: usize,
    pub resident_bytes: usize,
    pub engine_resident_bytes: usize,
    pub session_resident_bytes: usize,
}

impl KvResidency {
    pub fn available_bytes(self) -> usize {
        self.capacity_bytes.saturating_sub(self.active_bytes.saturating_add(self.resident_bytes))
    }
}

/// 固定整块 cache 的 session admission 所有者。budget 与两类 footprint
/// 必须同步变化，因此收拢在一个实体里；模型只决定它们的真实字节数。
pub struct FixedSessionResidency {
    budget: ResidencyBudget,
    engine_resident_bytes: usize,
    session_resident_bytes: usize,
}

impl FixedSessionResidency {
    pub fn new(capacity_bytes: usize, engine_resident_bytes: usize, session_resident_bytes: usize) -> Result<Self, String> {
        if session_resident_bytes == 0 {
            return Err("固定 session resident footprint 不能为 0".to_owned());
        }
        // 权重等引擎常驻必须从 session 预算中扣除：只进报表会放行超卖配置，
        // 统一内存超卖下 GPU 命令缓冲区等页不返回，decode 表现为永久挂死
        // 而不是可恢复的分配错误。与首请求惰性增长的 consume_engine_growth
        // 同一口径（基线常驻在构造期一次性扣减）。
        let budget = ResidencyBudget::new(capacity_bytes.saturating_sub(engine_resident_bytes));
        Ok(Self { budget, engine_resident_bytes, session_resident_bytes })
    }

    pub fn configure(&self, capabilities: &mut NodeCapabilities, runtime: &Arc<Mutex<RuntimeStatus>>) {
        let max_seq_len = capabilities.max_seq_len.max(1);
        let sessions = self.budget.capacity() / self.session_resident_bytes.max(1);
        capabilities.kv_cache_devices = vec![KvCacheDeviceCapacity {
            device: capabilities.accelerator.clone(),
            available_bytes: self.budget.capacity() as u64,
            bytes_per_token: self.session_resident_bytes.div_ceil(max_seq_len) as u64,
            token_capacity: sessions.saturating_mul(max_seq_len),
        }];
        capabilities.kv_reservation_page_tokens = max_seq_len;
        if let Ok(mut runtime) = runtime.lock() {
            runtime.engine_resident_bytes = self.engine_resident_bytes as u64;
            runtime.session_resident_bytes = self.session_resident_bytes as u64;
        }
    }

    pub fn budget(&self) -> &ResidencyBudget {
        &self.budget
    }

    /// 没有 terminal resident 状态的模型直接准入一块 session。
    pub fn reserve(&self) -> Result<ResidencyReservation, String> {
        self.budget
            .try_reserve(self.session_resident_bytes)
            .ok_or_else(|| format!("KV_RESIDENCY_EXHAUSTED required={} available={} active={} resident=0 capacity={}", self.session_resident_bytes, self.budget.available(), self.budget.used(), self.budget.capacity(),))
    }

    /// 惰性创建视觉塔等长期资源前，先把所有 terminal session 换出并独占当前
    /// session budget。这样真实设备分配不会先于 admission；若仍有 active session，
    /// 直接失败而不是与其争抢显存。
    pub fn reserve_engine_growth<S: TerminalSnapshot>(&self, sessions: &mut TerminalSessions<S>) -> Result<ResidencyReservation, String> {
        let capacity = self.budget.capacity();
        sessions.reserve_with_eviction(&self.budget, capacity, |_, state| state.info().bytes as usize)
    }

    pub fn session_resident_bytes(&self) -> usize {
        self.session_resident_bytes
    }

    /// MTP/replay/视觉等惰性引擎资源一旦常驻，必须同时缩减后续
    /// session admission 容量，不能继续沿用模型刚加载时的快照。
    pub fn consume_engine_growth(&mut self, bytes: usize) {
        self.engine_resident_bytes = self.engine_resident_bytes.saturating_add(bytes);
        self.budget.consume_capacity(bytes);
    }

    pub fn report<S: TerminalSnapshot>(&self, sessions: &TerminalSessions<S>) -> KvResidency {
        KvResidency {
            capacity_bytes: self.budget.capacity(),
            active_bytes: self.budget.used(),
            resident_bytes: sessions.resident_cost(|_, state| state.info().bytes as usize),
            engine_resident_bytes: self.engine_resident_bytes,
            session_resident_bytes: self.session_resident_bytes,
        }
    }

    pub fn report_without_terminal(&self) -> KvResidency {
        KvResidency { capacity_bytes: self.budget.capacity(), active_bytes: self.budget.used(), resident_bytes: 0, engine_resident_bytes: self.engine_resident_bytes, session_resident_bytes: self.session_resident_bytes }
    }

    pub fn refresh<S: TerminalSnapshot>(&self, runtime: &Arc<Mutex<RuntimeStatus>>, sessions: &TerminalSessions<S>) {
        self.refresh_stats(runtime, sessions.residency_stats());
    }

    pub fn refresh_without_terminal(&self, runtime: &Arc<Mutex<RuntimeStatus>>) {
        self.refresh_stats(runtime, TerminalResidencyStats::default());
    }

    fn refresh_stats(&self, runtime: &Arc<Mutex<RuntimeStatus>>, stats: TerminalResidencyStats) {
        if let Ok(mut runtime) = runtime.lock() {
            runtime.engine_resident_bytes = self.engine_resident_bytes as u64;
            runtime.session_resident_bytes = self.session_resident_bytes as u64;
            runtime.kv_cache_entries = stats.resident_entries;
            runtime.kv_cache_tokens = stats.resident_tokens;
            runtime.kv_cache_allocated_bytes = stats.resident_bytes;
            runtime.kv_cache_used_bytes = stats.resident_bytes;
            // 固定整块 session 不能伪装成线性 token 成本；admission 粒度由
            // capabilities.kv_reservation_page_tokens 表达。
            runtime.kv_bytes_per_token = 0;
        }
    }
}

/// 汇总按页实际增长的 cache。`TerminalInfo.bytes` 已经是 backend 当前真实分配量，
/// 不能再除以 `max_seq_len` 把它伪装成整块预留；缺少页内逻辑用量时，allocated/used
/// 都按真实 resident bytes 上报，平均 token 成本只作为运行时观测值。
/// 固定整块 session 使用 [`FixedSessionResidency`]。
pub fn refresh_cache_runtime<'a>(runtime: &mut RuntimeStatus, infos: impl IntoIterator<Item = &'a TerminalInfo>, _max_seq_len: usize) {
    let mut entries = 0usize;
    let mut tokens = 0usize;
    let mut allocated = 0u64;
    for info in infos {
        entries += 1;
        tokens = tokens.saturating_add(info.prompt_tokens);
        allocated = allocated.saturating_add(info.bytes);
    }
    runtime.kv_cache_entries = entries;
    runtime.kv_cache_tokens = tokens;
    runtime.kv_cache_allocated_bytes = allocated;
    runtime.kv_cache_used_bytes = allocated;
    runtime.kv_bytes_per_token = allocated.div_ceil(tokens.max(1) as u64);
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

pub struct GenerationSummary {
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cache: Option<TerminalInfo>,
    pub tool_calls: Vec<ToolCall>,
}

pub enum ContentPiece {
    Text(String),
    Image { url: String },
}

pub fn text_content(content: Option<&Value>) -> Result<String, String> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    return Err("节点当前只支持 text content part".to_owned());
                }
                text.push_str(part.get("text").and_then(Value::as_str).ok_or("text content part 缺少 text")?);
            }
            Ok(text)
        }
        None | Some(Value::Null) => Ok(String::new()),
        _ => Err("message.content 必须是字符串或 text part 数组".to_owned()),
    }
}

fn local_image_matches(content: &str) -> Vec<(usize, usize, &str)> {
    fn is_local_image(reference: &str) -> bool {
        let path = reference.strip_prefix("mm_file://").or_else(|| reference.strip_prefix("file://")).unwrap_or(reference);
        let extension = Path::new(path).extension().and_then(|value| value.to_str()).unwrap_or_default();
        matches!(extension.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp") && Path::new(path).is_file()
    }

    let mut candidates = Vec::new();
    for (open_delimiter, close_delimiter) in [('`', '`'), ('"', '"'), ('\'', '\''), ('“', '”'), ('‘', '’')] {
        let mut offset = 0usize;
        while let Some(open) = content[offset..].find(open_delimiter) {
            let start = offset + open + open_delimiter.len_utf8();
            let Some(close) = content[start..].find(close_delimiter) else { break };
            let end = start + close;
            let reference = content[start..end].trim();
            if is_local_image(reference) {
                candidates.push((offset + open, end + close_delimiter.len_utf8(), reference));
            }
            offset = end + close_delimiter.len_utf8();
        }
    }
    let mut offset = 0usize;
    for word in content.split_whitespace() {
        let start = offset + content[offset..].find(word).expect("split_whitespace 片段必须来自原文");
        offset = start + word.len();
        if is_local_image(word) {
            candidates.push((start, start + word.len(), word));
            continue;
        }
        let trimmed =
            word.trim_matches(|character: char| matches!(character, '`' | '"' | '\'' | '“' | '”' | '‘' | '’' | '[' | ']' | '(' | ')' | '{' | '}' | '<' | '>' | ',' | '.' | ':' | ';' | '!' | '?' | '，' | '。' | '：' | '；' | '！' | '？'));
        if is_local_image(trimmed) {
            let path_start = start + word.find(trimmed).unwrap_or(0);
            candidates.push((path_start, path_start + trimmed.len(), trimmed));
            continue;
        }
        for (inner, _) in trimmed.match_indices('/') {
            let suffix = &trimmed[inner..];
            let end = suffix.find([',', ':', ';', '!', '?', '，', '。', '：', '；', '！', '？', '`', '"', '\'', '“', '”', '‘', '’']).unwrap_or(suffix.len());
            let path = &suffix[..end];
            if is_local_image(path) {
                let path_start = start + word.find(trimmed).unwrap_or(0) + inner;
                candidates.push((path_start, path_start + path.len(), path));
                break;
            }
        }
    }
    // 同一路径可能同时被“成对引号”和“词片段”命中；起点相同时优先保留
    // 覆盖整段引号的范围，split 时才能连同引号内误粘贴的空白一起移除。
    candidates.sort_by_key(|(start, end, _)| (*start, std::cmp::Reverse(end - start)));
    candidates
}

pub fn local_image_paths(content: &str) -> Vec<&str> {
    let mut images = Vec::new();
    for (_, _, path) in local_image_matches(content) {
        if !images.contains(&path) {
            images.push(path);
        }
    }
    images
}

pub fn split_local_images(content: &str) -> (String, Vec<&str>) {
    let matches = local_image_matches(content);
    let mut images = Vec::new();
    let mut text = String::with_capacity(content.len());
    let mut copied = 0usize;
    for (start, end, path) in matches {
        if !images.contains(&path) {
            images.push(path);
        }
        if start < copied {
            continue;
        }
        text.push_str(&content[copied..start]);
        copied = end;
    }
    text.push_str(&content[copied..]);
    (text, images)
}

pub fn content_pieces(content: Option<&Value>) -> Result<Vec<ContentPiece>, String> {
    match content {
        // 结构化请求不能因节点本地恰好存在某个路径而改变语义；图像必须使用
        // 显式 image_url content part，普通字符串始终是文本。
        Some(Value::String(text)) => Ok(vec![ContentPiece::Text(text.clone())]),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|part| match part.get("type").and_then(Value::as_str) {
                Some("text") => Ok(ContentPiece::Text(part.get("text").and_then(Value::as_str).ok_or("text content part 缺少 text")?.to_owned())),
                Some("image_url") => Ok(ContentPiece::Image { url: part.get("image_url").and_then(Value::as_object).and_then(|url| url.get("url")).and_then(Value::as_str).ok_or("image_url content part 缺少 url")?.to_owned() }),
                Some(other) => Err(format!("节点不支持 content type '{other}'")),
                None => Err("content part 缺少 type".to_owned()),
            })
            .collect(),
        None | Some(Value::Null) => Ok(Vec::new()),
        _ => Err("message.content 必须是字符串或 content part 数组".to_owned()),
    }
}

pub fn with_content_parts<T>(pieces: &[ContentPiece], run: impl FnOnce(&[crate::vision::ContentPart<'_>]) -> Result<T, String>) -> Result<T, String> {
    let images = pieces
        .iter()
        .filter_map(|piece| match piece {
            ContentPiece::Image { url } => Some(url),
            ContentPiece::Text(_) => None,
        })
        .enumerate()
        .map(|(index, url)| crate::vision::image_from_url(url).map_err(|error| format!("图像 {index}: {error}")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut next_image = 0usize;
    let parts = pieces
        .iter()
        .map(|piece| match piece {
            ContentPiece::Text(text) => crate::vision::ContentPart::Text(text),
            ContentPiece::Image { .. } => {
                let index = next_image;
                next_image += 1;
                crate::vision::ContentPart::Image(&images[index])
            }
        })
        .collect::<Vec<_>>();
    run(&parts)
}

pub fn parse_stops(value: Option<&Value>) -> Result<Vec<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(stop)) if !stop.is_empty() => Ok(vec![stop.clone()]),
        Some(Value::Array(stops)) => stops.iter().map(|stop| stop.as_str().filter(|stop| !stop.is_empty()).map(str::to_owned).ok_or("stop 数组只能包含非空字符串".to_owned())).collect(),
        _ => Err("stop 必须是非空字符串或字符串数组".to_owned()),
    }
}

/// OpenAI Chat/Responses 兼容入口共用同一输出预算优先级。
pub fn requested_completion_tokens(request: &Value) -> usize {
    let requested = match request.get("max_completion_tokens") {
        Some(value) if !value.is_null() => Some(value),
        _ => request.get("max_tokens").filter(|value| !value.is_null()),
    };
    requested.map_or(256, |value| value.as_u64().and_then(|value| usize::try_from(value).ok()).unwrap_or(0))
}

pub struct TextStream {
    utf8: crate::tokenizer::Utf8StreamDecoder,
    stops: Vec<String>,
    text: String,
    emitted: usize,
    hold_bytes: usize,
    searched_until: usize,
}

pub struct TextStreamUpdate {
    pub chunk: Option<String>,
    pub stopped: bool,
}

/// 模型无关的可见文本输出状态机。模型 runtime 只负责给出 token 字节；这里统一
/// completion 计数、stop 文本边界、客户端取消和最终 UTF-8 flush 语义。
pub struct GenerationOutput {
    stream: TextStream,
    completion_tokens: usize,
    finish_reason: &'static str,
}

impl GenerationOutput {
    pub fn new(stops: &[String]) -> Self {
        Self { stream: TextStream::new(stops), completion_tokens: 0, finish_reason: "length" }
    }

    /// 返回 false 表示请求已经结束，调用方不得再推进模型状态。
    pub fn push(&mut self, bytes: &[u8], emit: impl FnOnce(String) -> bool) -> bool {
        if self.finish_reason != "length" {
            return false;
        }
        self.completion_tokens += 1;
        let update = self.stream.push(bytes);
        if let Some(chunk) = update.chunk
            && !emit(chunk)
        {
            self.cancel();
            return false;
        }
        if update.stopped {
            self.stop();
            return false;
        }
        true
    }

    /// 只在正常长度终止时 flush；stop/cancelled 不能泄漏停止串或不完整输出。
    pub fn finish(&mut self, emit: impl FnOnce(String) -> bool) {
        if self.finish_reason == "length"
            && let Some(chunk) = self.stream.finish()
            && !emit(chunk)
        {
            self.cancel();
        }
    }

    pub fn stop(&mut self) {
        if self.finish_reason == "length" {
            self.finish_reason = "stop";
        }
    }

    pub fn cancel(&mut self) {
        self.finish_reason = "cancelled";
    }

    pub fn mark_tool_calls(&mut self) {
        if self.finish_reason != "cancelled" {
            self.finish_reason = "tool_calls";
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.finish_reason == "cancelled"
    }

    pub fn finish_reason(&self) -> &str {
        self.finish_reason
    }

    pub fn completion_tokens(&self) -> usize {
        self.completion_tokens
    }

    pub fn text(&self) -> &str {
        self.stream.text()
    }

    pub fn summary(&self, prompt_tokens: usize) -> GenerationSummary {
        GenerationSummary { finish_reason: self.finish_reason.to_owned(), prompt_tokens, completion_tokens: self.completion_tokens, cache: None, tool_calls: Vec::new() }
    }
}

impl TextStream {
    pub fn new(stops: &[String]) -> Self {
        Self { utf8: crate::tokenizer::Utf8StreamDecoder::default(), stops: stops.to_vec(), text: String::new(), emitted: 0, hold_bytes: stops.iter().map(String::len).max().unwrap_or(1).saturating_sub(1), searched_until: 0 }
    }

    pub fn push(&mut self, bytes: &[u8]) -> TextStreamUpdate {
        self.text.push_str(&self.utf8.push(bytes));
        let scan_start = char_boundary_at_or_before(&self.text, self.searched_until.saturating_sub(self.hold_bytes));
        let stop_at = self.stops.iter().filter_map(|stop| self.text[scan_start..].find(stop).map(|offset| scan_start + offset)).min();
        self.searched_until = self.text.len();
        if let Some(stop_at) = stop_at {
            let chunk = (stop_at > self.emitted).then(|| self.text[self.emitted..stop_at].to_owned());
            self.emitted = stop_at;
            return TextStreamUpdate { chunk, stopped: true };
        }
        let safe_end = char_boundary_at_or_before(&self.text, self.text.len().saturating_sub(self.hold_bytes));
        let chunk = (safe_end > self.emitted).then(|| {
            let chunk = self.text[self.emitted..safe_end].to_owned();
            self.emitted = safe_end;
            chunk
        });
        TextStreamUpdate { chunk, stopped: false }
    }

    pub fn finish(&mut self) -> Option<String> {
        self.text.push_str(&self.utf8.finish());
        (self.emitted < self.text.len()).then(|| {
            let chunk = self.text[self.emitted..].to_owned();
            self.emitted = self.text.len();
            chunk
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

fn char_boundary_at_or_before(text: &str, mut byte: usize) -> usize {
    byte = byte.min(text.len());
    while byte > 0 && !text.is_char_boundary(byte) {
        byte -= 1;
    }
    byte
}

pub fn scoped_cache_id(namespace: Option<&str>, cache_id: &str) -> String {
    let Some(namespace) = namespace else { return cache_id.to_owned() };
    let mut input = Vec::with_capacity(namespace.len() + cache_id.len() + 15);
    input.extend_from_slice(b"zllm-cache-v1");
    input.push(0);
    input.extend_from_slice(namespace.as_bytes());
    input.push(0);
    input.extend_from_slice(cache_id.as_bytes());
    blake3::hash(&input).to_hex().to_string()
}

fn strip_null_fields(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(map.iter().filter(|(_, value)| !value.is_null()).map(|(key, value)| (key.clone(), strip_null_fields(value))).collect()),
        Value::Array(items) => Value::Array(items.iter().map(strip_null_fields).collect()),
        other => other.clone(),
    }
}

pub fn conversation_hash(model: &str, messages: &[Value]) -> Result<String, serde_json::Error> {
    let normalized: Vec<Value> = messages.iter().map(strip_null_fields).collect();
    serde_json::to_vec(&(model, normalized)).map(|bytes| blake3::hash(&bytes).to_hex().to_string())
}

fn request_conversation_hash(request: &Value, messages: &[Value]) -> Result<String, String> {
    let model = request.get("model").and_then(Value::as_str).ok_or("model 必须是字符串")?;
    let context = ["tools", "tool_choice", "parallel_tool_calls", "reasoning_effort", "thinking_token_budget", "thinking", "enable_thinking"]
        .into_iter()
        .filter_map(|key| request.get(key).filter(|value| !value.is_null()).map(|value| (key.to_owned(), value.clone())))
        .collect::<serde_json::Map<_, _>>();
    let normalized: Vec<Value> = messages.iter().map(strip_null_fields).collect();
    serde_json::to_vec(&(model, normalized, context)).map(|bytes| blake3::hash(&bytes).to_hex().to_string()).map_err(|error| format!("计算 conversation hash: {error}"))
}

pub fn terminal_cache_id(request: &Value, response: &str, tool_calls: &[ToolCall]) -> Result<String, String> {
    let source = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    let mut messages = source.clone();
    let response = client_replayable_response(request, response);
    let content = if response.is_empty() && !tool_calls.is_empty() { Value::Null } else { Value::String(response.to_owned()) };
    messages.push(serde_json::json!({ "role": "assistant", "content": content, "reasoning_content": null, "name": null, "tool_call_id": null, "tool_calls": if tool_calls.is_empty() { Value::Null } else { serde_json::json!(tool_calls) } }));
    let cache_id = request_conversation_hash(request, &messages)?;
    Ok(scoped_cache_id(request.get("_zllm_cache_namespace").and_then(Value::as_str), &cache_id))
}

pub fn client_replayable_response<'a>(request: &Value, response: &'a str) -> &'a str {
    let split_reasoning = request.get("reasoning_effort").is_some_and(|value| !value.is_null()) || request.get("thinking_token_budget").is_some_and(|value| !value.is_null());
    if split_reasoning { response.split_once("</think>").map_or("", |(_, visible)| visible) } else { response }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerminalResume {
    None,
    Match { cache_id: String, assistant: usize },
    Mismatch { requested: String, expected: String },
}

pub fn request_resume_boundary(request: &Value) -> Result<Option<(String, usize)>, String> {
    let source = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    let Some(assistant) = source.iter().rposition(|message| message.get("role").and_then(Value::as_str) == Some("assistant")) else { return Ok(None) };
    let hash = request_conversation_hash(request, &source[..=assistant])?;
    Ok(Some((scoped_cache_id(request.get("_zllm_cache_namespace").and_then(Value::as_str), &hash), assistant)))
}

pub fn request_terminal_resume(request: &Value) -> Result<TerminalResume, String> {
    let Some(requested) = request.get("cache_id").and_then(Value::as_str) else { return Ok(TerminalResume::None) };
    let Some((expected, assistant)) = request_resume_boundary(request)? else { return Ok(TerminalResume::None) };
    Ok(if requested == expected { TerminalResume::Match { cache_id: requested.to_owned(), assistant } } else { TerminalResume::Mismatch { requested: requested.to_owned(), expected } })
}

pub fn resume_terminal_session<S: crate::kv_cache::terminal_cache::TerminalSnapshot>(
    sessions: &mut crate::kv_cache::terminal_cache::TerminalSessions<S>,
    request: &Value,
    resources: &S::Resources,
) -> Result<Option<(usize, (Vec<u32>, S))>, String> {
    Ok(match request_terminal_resume(request)? {
        TerminalResume::Match { cache_id, assistant } => match sessions.resume(&cache_id, resources) {
            Some(Ok(entry)) => Some((assistant, entry)),
            Some(Err(error)) => {
                eprintln!("[zllm-runtime] {error}");
                None
            }
            None => None,
        },
        TerminalResume::Mismatch { requested, expected } => {
            eprintln!("[zllm-runtime] terminal cache hash 不匹配 id={requested} expected={expected}");
            None
        }
        TerminalResume::None => None,
    })
}

pub fn resume_terminal_append<S: crate::kv_cache::terminal_cache::TerminalSnapshot>(
    sessions: &mut crate::kv_cache::terminal_cache::TerminalSessions<S>,
    request: &Value,
    render_suffix: impl FnOnce(usize) -> Result<Vec<u32>, String>,
    resources: &S::Resources,
) -> Result<Option<(S, Vec<u32>)>, String> {
    Ok(match resume_terminal_session(sessions, request, resources)? {
        Some((assistant, (cached, state))) => {
            let suffix = render_suffix(assistant)?;
            eprintln!("[zllm-runtime] terminal cache 拼接命中 cached={} new={}", cached.len(), suffix.len());
            Some((state, suffix))
        }
        None => None,
    })
}

pub fn retain_terminal_session<S: crate::kv_cache::terminal_cache::TerminalSnapshot>(sessions: &mut crate::kv_cache::terminal_cache::TerminalSessions<S>, state: S, info: TerminalInfo) -> Option<TerminalInfo> {
    let cache_id = info.cache_id.clone();
    let terminal_tokens = state.terminal_tokens().to_vec();
    sessions.retain(cache_id, terminal_tokens, state).then_some(info)
}

pub fn activate_terminal_append<S: crate::kv_cache::terminal_cache::TerminalSnapshot>(
    sessions: &mut crate::kv_cache::terminal_cache::TerminalSessions<S>,
    budget: &crate::kv_cache::terminal_cache::ResidencyBudget,
    required: usize,
    request: &Value,
    render_suffix: impl FnOnce(usize) -> Result<Vec<u32>, String>,
    resources: &S::Resources,
) -> Result<(Option<(S, Vec<u32>)>, crate::kv_cache::terminal_cache::ResidencyReservation), String> {
    let (cache_id, assistant) = match request_terminal_resume(request)? {
        TerminalResume::Match { cache_id, assistant } => (Some(cache_id), Some(assistant)),
        TerminalResume::Mismatch { requested, expected } => {
            eprintln!("[zllm-runtime] terminal cache hash 不匹配 id={requested} expected={expected}");
            (None, None)
        }
        TerminalResume::None => (None, None),
    };
    let suffix = assistant.map(render_suffix).transpose()?;
    let (state, reservation) = sessions.activate(cache_id.as_deref(), budget, required, |_, state| state.info().bytes as usize, resources)?;
    let resumed = match (state, suffix) {
        (Some((cached, state)), Some(suffix)) => {
            eprintln!("[zllm-runtime] terminal cache 拼接命中 cached={} new={}", cached.len(), suffix.len());
            Some((state, suffix))
        }
        _ => None,
    };
    Ok((resumed, reservation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 本地图片路径支持中英文成对引号() {
        let path = std::env::temp_dir().join(format!("zllm-session-image-{}.png", std::process::id()));
        std::fs::write(&path, b"image placeholder").unwrap();
        let path = path.to_string_lossy();
        for input in [format!("`{path}`"), format!("\"{path}\""), format!("“{path}”"), format!("‘{path}’"), format!("\" {path} \""), format!("“ {path} ”")] {
            let (text, images) = split_local_images(&input);
            assert_eq!(images, [path.as_ref()]);
            assert!(text.is_empty());
        }
        std::fs::remove_file(path.as_ref()).unwrap();
    }

    #[test]
    fn 固定整块session容量不允许半块拼接() {
        let runtime = Arc::new(Mutex::new(RuntimeStatus::default()));
        let mut capabilities = NodeCapabilities { accelerator: "device".to_owned(), max_seq_len: 10, ..NodeCapabilities::default() };
        let residency = FixedSessionResidency::new(1000, 30, 101).unwrap();
        residency.configure(&mut capabilities, &runtime);

        assert_eq!(capabilities.kv_cache_devices[0].bytes_per_token, 11);
        assert_eq!(capabilities.kv_cache_devices[0].token_capacity, 90);
        assert_eq!(capabilities.kv_reservation_page_tokens, 10);
        let status = runtime.lock().unwrap();
        assert_eq!(status.engine_resident_bytes, 30);
        assert_eq!(status.session_resident_bytes, 101);
    }

    #[test]
    fn 引擎常驻从session预算扣除() {
        let runtime = Arc::new(Mutex::new(RuntimeStatus::default()));
        let mut capabilities = NodeCapabilities { accelerator: "device".to_owned(), max_seq_len: 10, ..NodeCapabilities::default() };
        // 容量 1000、引擎常驻 950：剩余 50 放不下一块 101 的 session，
        // 超卖配置必须在准入期拒绝而不是装载后 decode 挂死
        let residency = FixedSessionResidency::new(1000, 950, 101).unwrap();
        residency.configure(&mut capabilities, &runtime);
        assert_eq!(capabilities.kv_cache_devices[0].token_capacity, 0);
        assert!(residency.reserve().is_err());
        assert_eq!(runtime.lock().unwrap().engine_resident_bytes, 950);
    }

    #[test]
    fn completion预算遵循新字段优先级与默认值() {
        assert_eq!(requested_completion_tokens(&serde_json::json!({})), 256);
        assert_eq!(requested_completion_tokens(&serde_json::json!({ "max_tokens": 17 })), 17);
        assert_eq!(requested_completion_tokens(&serde_json::json!({ "max_completion_tokens": 23, "max_tokens": 17 })), 23);
        assert_eq!(requested_completion_tokens(&serde_json::json!({ "max_completion_tokens": null, "max_tokens": 17 })), 17);
        assert_eq!(requested_completion_tokens(&serde_json::json!({ "max_tokens": null })), 256);
        assert_eq!(requested_completion_tokens(&serde_json::json!({ "max_tokens": -1 })), 0);
        assert_eq!(requested_completion_tokens(&serde_json::json!({ "max_tokens": "17" })), 0);
    }

    #[test]
    fn 分页cache指标报告真实常驻量() {
        let infos = [
            TerminalInfo { cache_id: "a".to_owned(), model_key: "model".to_owned(), cache_format: "test".to_owned(), last_layer: 1, prompt_tokens: 25, bytes: 1000, modified_unix: 1 },
            TerminalInfo { cache_id: "b".to_owned(), model_key: "model".to_owned(), cache_format: "test".to_owned(), last_layer: 1, prompt_tokens: 50, bytes: 2000, modified_unix: 1 },
        ];
        let mut runtime = RuntimeStatus::default();
        refresh_cache_runtime(&mut runtime, &infos, 100);

        assert_eq!(runtime.kv_cache_entries, 2);
        assert_eq!(runtime.kv_cache_tokens, 75);
        assert_eq!(runtime.kv_cache_allocated_bytes, 3000);
        assert_eq!(runtime.kv_cache_used_bytes, 3000);
        assert_eq!(runtime.kv_bytes_per_token, 40);
    }

    #[test]
    fn dspark深度随decode并发收缩到总量三十二() {
        assert_eq!(effective_dspark_draft_tokens(8, 0), 0);
        assert_eq!(effective_dspark_draft_tokens(8, 4), 8);
        assert_eq!(effective_dspark_draft_tokens(8, 5), 6);
        assert_eq!(effective_dspark_draft_tokens(8, 16), 2);
        assert_eq!(effective_dspark_draft_tokens(8, 22), 1);
    }

    #[test]
    fn generation_output跨token识别stop且不发出停止串() {
        let mut output = GenerationOutput::new(&["STOP".to_owned()]);
        let mut chunks = Vec::new();
        assert!(output.push(b"abST", |chunk| {
            chunks.push(chunk);
            true
        }));
        assert!(!output.push(b"OPtail", |chunk| {
            chunks.push(chunk);
            true
        }));
        output.finish(|chunk| {
            chunks.push(chunk);
            true
        });

        assert_eq!(chunks.concat(), "ab");
        assert_eq!(output.finish_reason(), "stop");
        assert_eq!(output.completion_tokens(), 2);
    }

    #[test]
    fn generation_output统一回调取消与正常flush() {
        let mut cancelled = GenerationOutput::new(&[]);
        assert!(!cancelled.push(b"cancel", |_| false));
        cancelled.finish(|_| panic!("取消后不应 flush"));
        assert_eq!(cancelled.finish_reason(), "cancelled");

        let mut finished = GenerationOutput::new(&["STOP".to_owned()]);
        let mut chunks = Vec::new();
        assert!(finished.push(b"ok", |chunk| {
            chunks.push(chunk);
            true
        }));
        finished.finish(|chunk| {
            chunks.push(chunk);
            true
        });
        assert_eq!(chunks.concat(), "ok");
        assert_eq!(finished.finish_reason(), "length");
    }
}
