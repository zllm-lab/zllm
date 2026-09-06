//! 连续对话终点缓存：只管理 resident 状态所有权与 FIFO 淘汰，不感知模型和 backend。

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

/// Terminal snapshot 的小端安全编解码器。模型仍拥有字段顺序与版本，公共层只处理
/// 长度、整数和越界检查，不定义跨模型持久化格式。
#[derive(Default)]
pub struct SnapshotWriter {
    bytes: Vec<u8>,
}

impl SnapshotWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn flag(&mut self, value: bool) {
        self.bytes.push(u8::from(value));
    }

    pub fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn usize(&mut self, value: usize) -> Result<(), String> {
        self.u64(u64::try_from(value).map_err(|_| format!("snapshot usize={value} 超过 u64"))?);
        Ok(())
    }

    pub fn string(&mut self, value: &str) -> Result<(), String> {
        self.usize(value.len())?;
        self.bytes(value.as_bytes());
        Ok(())
    }

    pub fn optional_string(&mut self, value: Option<&str>) -> Result<(), String> {
        self.flag(value.is_some());
        if let Some(value) = value {
            self.string(value)?;
        }
        Ok(())
    }

    pub fn u32s(&mut self, values: &[u32]) -> Result<(), String> {
        self.u32(u32::try_from(values.len()).map_err(|_| format!("snapshot u32 数组过长: {}", values.len()))?);
        self.bytes.reserve(values.len().saturating_mul(4));
        for &value in values {
            self.u32(value);
        }
        Ok(())
    }

    pub fn u16s(&mut self, values: &[u16]) -> Result<(), String> {
        self.usize(values.len())?;
        self.bytes.reserve(values.len().saturating_mul(2));
        for &value in values {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

pub struct SnapshotReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SnapshotReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self.offset.checked_add(len).ok_or_else(|| format!("snapshot {what} offset 溢出"))?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| format!("snapshot {what} 字节不足: offset={} need={len} total={}", self.offset, self.bytes.len()))?;
        self.offset = end;
        Ok(value)
    }

    pub fn u32(&mut self, what: &str) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().expect("已检查 u32 字节")))
    }

    pub fn flag(&mut self, what: &str) -> Result<bool, String> {
        match self.take(1, what)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(format!("snapshot {what} bool={value} 非法")),
        }
    }

    pub fn u64(&mut self, what: &str) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().expect("已检查 u64 字节")))
    }

    pub fn i64(&mut self, what: &str) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8, what)?.try_into().expect("已检查 i64 字节")))
    }

    pub fn usize(&mut self, what: &str) -> Result<usize, String> {
        usize::try_from(self.u64(what)?).map_err(|_| format!("snapshot {what} 超过 usize"))
    }

    /// 从不可信快照读取集合数量，并用每项最小编码字节数阻止畸形长度触发巨量分配。
    pub fn count(&mut self, what: &str, minimum_item_bytes: usize) -> Result<usize, String> {
        let count = self.usize(what)?;
        if minimum_item_bytes > 0 && count > self.remaining().len() / minimum_item_bytes {
            return Err(format!("snapshot {what} count={count} 超过剩余 {} bytes 可容纳数量", self.remaining().len()));
        }
        Ok(count)
    }

    pub fn string(&mut self, what: &str) -> Result<String, String> {
        let len = self.usize(what)?;
        let bytes = self.take(len, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|error| format!("snapshot {what} 不是 UTF-8: {error}"))
    }

    pub fn optional_string(&mut self, what: &str) -> Result<Option<String>, String> {
        if self.flag(what)? { self.string(what).map(Some) } else { Ok(None) }
    }

    pub fn u32s(&mut self, what: &str) -> Result<Vec<u32>, String> {
        let count = self.u32(what)? as usize;
        let bytes = self.take(count.checked_mul(4).ok_or_else(|| format!("snapshot {what} 数组大小溢出"))?, what)?;
        Ok(bytes.chunks_exact(4).map(|value| u32::from_le_bytes(value.try_into().expect("u32 数组分块"))).collect())
    }

    pub fn u16s(&mut self, what: &str) -> Result<Vec<u16>, String> {
        let count = self.usize(what)?;
        let bytes = self.take(count.checked_mul(2).ok_or_else(|| format!("snapshot {what} 数组大小溢出"))?, what)?;
        Ok(bytes.chunks_exact(2).map(|value| u16::from_le_bytes(value.try_into().expect("u16 数组分块"))).collect())
    }

    pub fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.offset..]
    }

    pub fn finish(self) -> Result<(), String> {
        if self.offset == self.bytes.len() { Ok(()) } else { Err(format!("snapshot 尾部仍有 {} 字节", self.bytes.len() - self.offset)) }
    }
}

/// active cache 的通用容量门票；单位由模型/backend 适配器定义，可以是 token 或字节。
#[derive(Clone)]
pub struct ResidencyBudget {
    inner: Arc<ResidencyCounter>,
}

struct ResidencyCounter {
    state: Mutex<ResidencyState>,
}

struct ResidencyState {
    capacity: usize,
    used: usize,
}

/// 与一个 active session 同寿命，退出时自动归还容量。
pub struct ResidencyReservation {
    inner: Arc<ResidencyCounter>,
    cost: usize,
}

impl ResidencyBudget {
    pub fn new(capacity: usize) -> Self {
        Self { inner: Arc::new(ResidencyCounter { state: Mutex::new(ResidencyState { capacity, used: 0 }) }) }
    }

    pub fn capacity(&self) -> usize {
        self.inner.state.lock().expect("residency budget 锁中毒").capacity
    }

    /// 引擎惰性创建长期资源后，从 session admission 总预算中永久扣除。
    pub fn consume_capacity(&self, bytes: usize) {
        let mut state = self.inner.state.lock().expect("residency budget 锁中毒");
        state.capacity = state.capacity.saturating_sub(bytes);
    }

    pub fn used(&self) -> usize {
        self.inner.state.lock().expect("residency budget 锁中毒").used
    }

    pub fn available(&self) -> usize {
        let state = self.inner.state.lock().expect("residency budget 锁中毒");
        state.capacity.saturating_sub(state.used)
    }

    pub fn try_reserve(&self, cost: usize) -> Option<ResidencyReservation> {
        let mut state = self.inner.state.lock().expect("residency budget 锁中毒");
        let next = state.used.checked_add(cost)?;
        if next > state.capacity {
            return None;
        }
        state.used = next;
        drop(state);
        Some(ResidencyReservation { inner: self.inner.clone(), cost })
    }
}

impl ResidencyReservation {
    pub fn cost(&self) -> usize {
        self.cost
    }

    /// active cache 只在真实跨过 backend 分配粒度时增长，不根据声明的最大生成长度预留。
    pub fn try_grow(&mut self, additional: usize) -> bool {
        let mut state = self.inner.state.lock().expect("residency budget 锁中毒");
        let Some(next) = state.used.checked_add(additional) else { return false };
        if next > state.capacity {
            return false;
        }
        state.used = next;
        self.cost += additional;
        true
    }
}

impl Drop for ResidencyReservation {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock().expect("residency budget 锁中毒");
        state.used = state.used.checked_sub(self.cost).expect("residency reservation 归还超过已用容量");
    }
}

pub struct TerminalState<C, H, R = ()> {
    pub cache: C,
    pub hidden: H,
    pub recurrent: R,
}

/// terminal session 的宿主无关元数据。scheduler 只负责上报和路由，不拥有
/// 这份生命周期数据，因此服务层通过类型别名复用该定义。
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TerminalInfo {
    pub cache_id: String,
    pub model_key: String,
    pub cache_format: String,
    pub last_layer: usize,
    pub prompt_tokens: usize,
    pub bytes: u64,
    pub modified_unix: u64,
}

pub struct TerminalCache<S> {
    entries: HashMap<String, (Vec<u32>, S)>,
    order: VecDeque<String>,
    limit: usize,
}

/// 前缀复用的最小公共前缀:低于此长度时截断/续写的调度成本不如全量 prefill。
pub const MIN_PREFIX_RESUME_TOKENS: usize = 32;

/// 前缀复用恢复策略(llama slot 截断复用的通用形,模型无关):
/// cached 恰好是新 prompt 的真前缀时直接续写;否则截断到 lcp-1,由 `extend`
/// 重算边界 hidden。`truncate`/`extend` 由模型/backend 适配器提供;返回续写起点(计费用)。
pub fn resume_by_prefix<S>(state: &mut S, tokens: &[u32], lcp: usize, cached_len: usize, truncate: impl FnOnce(&mut S, usize), mut extend: impl FnMut(&mut S, &[u32]) -> Result<(), String>) -> Result<usize, String> {
    debug_assert!(lcp >= 1 && lcp <= tokens.len() && lcp <= cached_len, "lcp={lcp} 必须是两侧公共前缀(tokens={}, cached={cached_len})", tokens.len());
    if lcp == cached_len && cached_len < tokens.len() {
        extend(state, &tokens[cached_len..])?;
        Ok(cached_len)
    } else {
        let resume_at = lcp - 1;
        truncate(state, resume_at);
        extend(state, &tokens[resume_at..])?;
        Ok(resume_at)
    }
}

/// `cache_id` 只索引不可变的 backend block graph；命中时由 backend 从 graph
/// fork 出可写 session。会话主表只保留最新终点，旧终点只有位于前
/// `prefix_rounds` 轮时才进入全局前缀池；全部 graph 共同受 LRU 硬上限约束。
pub trait SharedBlockGraph {
    type Session;
    type Error;

    fn materialize(&self) -> Result<Self::Session, Self::Error>;
}

pub struct SharedBlockCache<G> {
    entries: HashMap<String, SharedBlockEntry<G>>,
    order: VecDeque<String>,
    limit: usize,
    prefix_rounds: usize,
}

pub struct RemovedSharedBlock<G> {
    pub cache_id: String,
    pub tokens: Vec<u32>,
    pub graph: Arc<G>,
    pub round: usize,
    pub head: bool,
    pub persist: bool,
}

struct SharedBlockEntry<G> {
    tokens: Vec<u32>,
    graph: Arc<G>,
    round: usize,
    head: bool,
}

impl<G: SharedBlockGraph> SharedBlockCache<G> {
    pub fn new(limit: usize, prefix_rounds: usize) -> Self {
        Self { entries: HashMap::new(), order: VecDeque::new(), limit, prefix_rounds }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn cached_tokens(&self, cache_id: &str) -> Option<&[u32]> {
        self.entries.get(cache_id).map(|entry| entry.tokens.as_slice())
    }

    pub fn is_head(&self, cache_id: &str) -> Option<bool> {
        self.entries.get(cache_id).map(|entry| entry.head)
    }

    pub fn entries(&self) -> impl Iterator<Item = (&str, &[u32], &G)> {
        self.order.iter().filter_map(|cache_id| self.entries.get(cache_id).map(|entry| (cache_id.as_str(), entry.tokens.as_slice(), entry.graph.as_ref())))
    }

    pub fn drain(&mut self) -> impl Iterator<Item = Arc<G>> + '_ {
        self.order.clear();
        self.entries.drain().map(|(_, entry)| entry.graph)
    }

    pub fn drain_entries(&mut self) -> Vec<RemovedSharedBlock<G>> {
        self.order.clear();
        self.entries.drain().map(|(cache_id, entry)| RemovedSharedBlock { cache_id, tokens: entry.tokens, graph: entry.graph, round: entry.round, head: entry.head, persist: true }).collect()
    }

    pub fn restore(&mut self, cache_id: String, tokens: Vec<u32>, round: usize, head: bool, graph: G) -> Vec<RemovedSharedBlock<G>> {
        let mut removed = self.remove(&cache_id).map(|entry| RemovedSharedBlock { cache_id: cache_id.clone(), tokens: entry.tokens, graph: entry.graph, round: entry.round, head: entry.head, persist: false }).into_iter().collect::<Vec<_>>();
        self.order.push_back(cache_id.clone());
        self.entries.insert(cache_id, SharedBlockEntry { tokens, graph: Arc::new(graph), round, head });
        self.evict_to_limit(&mut removed);
        removed
    }

    /// 命中只共享读取 graph，不消费当前 entry；backend capability 负责构造可写 session。
    pub fn materialize(&mut self, cache_id: &str, tokens: &[u32]) -> Option<Result<(Vec<u32>, usize, G::Session), G::Error>> {
        let entry = self.entries.get(cache_id)?;
        if !tokens.starts_with(&entry.tokens) {
            return None;
        }
        let cached = entry.tokens.clone();
        let round = entry.round;
        let graph = entry.graph.clone();
        self.touch(cache_id);
        Some(graph.materialize().map(|session| (cached, round, session)))
    }

    pub fn materialize_longest_prefix(&mut self, tokens: &[u32], mut eligible: impl FnMut(&G) -> bool) -> Option<Result<(String, Vec<u32>, usize, G::Session), G::Error>> {
        let mut best = None::<(String, Vec<u32>, usize, _)>;
        let mut near_miss = None::<(usize, usize, String)>;
        for (cache_id, entry) in self.entries.iter() {
            if entry.head || entry.tokens.is_empty() || !eligible(&entry.graph) {
                continue;
            }
            if tokens.starts_with(&entry.tokens) {
                if best.as_ref().is_none_or(|(_, candidate, _, _)| entry.tokens.len() > candidate.len()) {
                    best = Some((cache_id.clone(), entry.tokens.clone(), entry.round, entry.graph.clone()));
                }
            } else {
                // 前缀链诊断:快照与请求几乎重合却在尾部分叉,几乎总是 assistant
                // 回显序列化与生成 token 不一致导致 terminal cache 无法接续。
                let common = entry.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count();
                if common >= 1024 && common * 10 >= entry.tokens.len() * 9 && near_miss.as_ref().is_none_or(|(length, _, _)| common > *length) {
                    near_miss = Some((common, entry.tokens.len(), cache_id.clone()));
                }
            }
        }
        if let Some((common, snapshot_len, cache_id)) = &near_miss
            && best.as_ref().is_none_or(|(best_id, _, _, _)| best_id != cache_id)
        {
            eprintln!("[terminal-cache] 前缀近失配 cache_id={cache_id} 公共前缀={common} 快照长度={snapshot_len} prompt 长度={}", tokens.len());
        }
        let (cache_id, cached, round, graph) = best?;
        self.touch(&cache_id);
        Some(graph.materialize().map(|session| (cache_id, cached, round, session)))
    }

    /// `parent` 携带命中时的轮次，即使请求执行期间父节点被 LRU 换出，
    /// 新 head 也不会错误回到第一轮。返回退出的 graph；唯一引用可由 backend 回收。
    pub fn insert(&mut self, cache_id: String, tokens: Vec<u32>, parent: Option<(String, usize)>, graph: G) -> Vec<RemovedSharedBlock<G>> {
        if self.limit == 0 {
            return vec![RemovedSharedBlock { cache_id, tokens, graph: Arc::new(graph), round: parent.as_ref().map_or(1, |(_, round)| round.saturating_add(1)), head: true, persist: true }];
        }
        let mut removed = Vec::new();
        let round = parent.as_ref().map_or(1, |(_, round)| round.saturating_add(1));
        if let Some((parent_id, _)) = &parent
            && parent_id != &cache_id
            && let Some(parent) = self.entries.get_mut(parent_id)
            && parent.head
        {
            if parent.round <= self.prefix_rounds {
                parent.head = false;
            } else if let Some(parent) = self.remove(parent_id) {
                removed.push(RemovedSharedBlock { cache_id: parent_id.clone(), tokens: parent.tokens, graph: parent.graph, round: parent.round, head: false, persist: false });
            }
        }
        if let Some(previous) = self.remove(&cache_id) {
            removed.push(RemovedSharedBlock { cache_id: cache_id.clone(), tokens: previous.tokens, graph: previous.graph, round: previous.round, head: previous.head, persist: false });
        }
        self.order.push_back(cache_id.clone());
        self.entries.insert(cache_id, SharedBlockEntry { tokens, graph: Arc::new(graph), round, head: true });
        self.evict_to_limit(&mut removed);
        removed
    }

    fn evict_to_limit(&mut self, removed: &mut Vec<RemovedSharedBlock<G>>) {
        while self.entries.len() > self.limit {
            let Some(evicted) = self.order.pop_front() else { break };
            if let Some(entry) = self.entries.remove(&evicted) {
                removed.push(RemovedSharedBlock { cache_id: evicted, tokens: entry.tokens, graph: entry.graph, round: entry.round, head: entry.head, persist: true });
            }
        }
    }

    fn remove(&mut self, cache_id: &str) -> Option<SharedBlockEntry<G>> {
        self.order.retain(|existing| existing != cache_id);
        self.entries.remove(cache_id)
    }

    fn touch(&mut self, cache_id: &str) {
        self.order.retain(|existing| existing != cache_id);
        self.order.push_back(cache_id.to_owned());
    }
}

impl<S> TerminalCache<S> {
    pub fn new(limit: usize) -> Self {
        Self { entries: HashMap::new(), order: VecDeque::new(), limit }
    }

    pub fn capacity(&self) -> usize {
        self.limit
    }

    pub fn take_matching(&mut self, cache_id: &str, tokens: &[u32]) -> Option<(Vec<u32>, S)> {
        if self.entries.get(cache_id).is_some_and(|(cached, _)| tokens.starts_with(cached)) {
            self.order.retain(|existing| existing != cache_id);
            return self.entries.remove(cache_id);
        }
        None
    }

    /// 在满足调用方隔离条件的 entry 中取最长 token 前缀。cache_id 只负责所有权，
    /// 不能把内容相同但会话 hash 不同的 prompt 排除在 prefix cache 之外。
    pub fn take_longest_prefix(&mut self, tokens: &[u32], mut eligible: impl FnMut(&S) -> bool) -> Option<(String, Vec<u32>, S)> {
        let cache_id = self.entries.iter().filter(|(_, (cached, state))| !cached.is_empty() && tokens.starts_with(cached) && eligible(state)).max_by_key(|(_, (cached, _))| cached.len()).map(|(cache_id, _)| cache_id.clone())?;
        self.order.retain(|existing| existing != &cache_id);
        self.entries.remove(&cache_id).map(|(cached, state)| (cache_id, cached, state))
    }

    pub fn take(&mut self, cache_id: &str) -> Option<(Vec<u32>, S)> {
        if let Some(entry) = self.entries.remove(cache_id) {
            self.order.retain(|existing| existing != cache_id);
            return Some(entry);
        }
        None
    }

    /// 取出最早的内存 entry。模型可在 cache miss 时复用其中的 resident 资源，
    /// 避免为新会话重复准备同一份层权重。
    pub fn take_oldest(&mut self) -> Option<(String, Vec<u32>, S)> {
        while let Some(cache_id) = self.order.pop_front() {
            if let Some((tokens, state)) = self.entries.remove(&cache_id) {
                return Some((cache_id, tokens, state));
            }
        }
        None
    }

    pub fn oldest(&self) -> Option<(&str, &[u32], &S)> {
        self.order.iter().find_map(|cache_id| self.entries.get(cache_id).map(|(tokens, state)| (cache_id.as_str(), tokens.as_slice(), state)))
    }

    /// 跳过被 pin 的条目取最早内存 entry,供 append 排队期间的换出保护;
    /// 全部候选被 pin 时返回 None,由调用方决定报错或把新状态直接落盘。
    pub fn oldest_unpinned(&self, pinned: &HashSet<String>) -> Option<(&str, &[u32], &S)> {
        self.order.iter().filter(|cache_id| !pinned.contains(*cache_id)).find_map(|cache_id| self.entries.get(cache_id).map(|(tokens, state)| (cache_id.as_str(), tokens.as_slice(), state)))
    }

    /// 内容最长公共前缀匹配(与 cache_id 无关):返回 lcp 最长的 entry。
    /// 与 `take_longest_prefix` 的差别是允许 cached 比新 tokens 更长——调用方
    /// 按 lcp 与 cached/new 长度的关系自行决定续写或截断复用(llama slot 截断复用)。
    /// lcp 低于 [`MIN_PREFIX_RESUME_TOKENS`] 视为无复用价值,不消费 entry。
    pub fn take_longest_common_prefix(&mut self, tokens: &[u32], mut eligible: impl FnMut(&S) -> bool) -> Option<(String, Vec<u32>, usize, S)> {
        let (cache_id, lcp) = self
            .entries
            .iter()
            .filter(|(_, (cached, state))| !cached.is_empty() && eligible(state))
            .map(|(cache_id, (cached, _))| (cache_id.clone(), cached.iter().zip(tokens.iter()).take_while(|(left, right)| left == right).count()))
            .filter(|(_, lcp)| *lcp >= MIN_PREFIX_RESUME_TOKENS)
            .max_by_key(|(_, lcp)| *lcp)?;
        let (cached, state) = self.entries.remove(&cache_id)?;
        self.order.retain(|existing| existing != &cache_id);
        Some((cache_id, cached, lcp, state))
    }

    pub fn cached_tokens(&self, cache_id: &str) -> Option<&[u32]> {
        self.entries.get(cache_id).map(|(tokens, _)| tokens.as_slice())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.limit
    }

    pub fn insert(&mut self, cache_id: String, tokens: Vec<u32>, state: S) -> bool {
        if self.limit == 0 {
            return false;
        }
        self.entries.remove(&cache_id);
        self.order.retain(|existing| existing != &cache_id);
        while self.entries.len() >= self.limit {
            let Some(evicted) = self.order.pop_front() else { break };
            self.entries.remove(&evicted);
        }
        self.order.push_back(cache_id.clone());
        self.entries.insert(cache_id, (tokens, state));
        true
    }

    pub fn states(&self) -> impl Iterator<Item = &S> {
        self.order.iter().filter_map(|cache_id| self.entries.get(cache_id).map(|(_, state)| state))
    }

    pub fn entries(&self) -> impl Iterator<Item = (&str, &[u32], &S)> {
        self.order.iter().filter_map(|cache_id| self.entries.get(cache_id).map(|(tokens, state)| (cache_id.as_str(), tokens.as_slice(), state)))
    }

    /// 由模型/backend 适配器把 resident state 折算为与 [`ResidencyBudget`] 相同的单位。
    pub fn resident_cost(&self, mut cost: impl FnMut(&[u32], &S) -> usize) -> usize {
        self.order.iter().filter_map(|cache_id| self.entries.get(cache_id)).fold(0usize, |total, (tokens, state)| total.saturating_add(cost(tokens, state)))
    }
}

/// 换出编码边界:引擎把 resident 终点状态编码/恢复为字节,Resources 携带引擎侧
/// 重建资源(如 backend context)。本层只管字节与生命周期,保持模型/后端无关。
pub trait TerminalSnapshot: Sized {
    type Resources;
    fn encode(&self) -> Result<Vec<u8>, String>;
    fn decode(bytes: &[u8], resources: &Self::Resources) -> Result<Self, String>;
    /// 恢复后的已处理 token 数(resume 计费与边界判断用)。
    fn terminal_tokens(&self) -> &[u32];
    /// 该 state 对应的 cache 元数据(供 `swap_infos` 暴露给 scheduler 端做命中查找)。
    fn info(&self) -> &TerminalInfo;
    /// SSD envelope 是 metadata 的唯一持久化来源；恢复后由通用层注回模型状态。
    fn set_info(&mut self, _info: TerminalInfo) {}
    /// 会话所属租户(协议层 cache namespace),用于内容前缀匹配的跨租户隔离;
    /// 与 `terminal_tokens` 同属 entry 元数据。默认 None:无租户隔离。
    fn cache_namespace(&self) -> Option<&str> {
        None
    }
}

const TERMINAL_MAGIC: [u8; 8] = *b"ZLLMTS01";
const TERMINAL_VERSION: u32 = 2;
const TERMINAL_INFO_MAGIC: [u8; 8] = *b"ZLLMTI01";
const TERMINAL_INFO_VERSION: u32 = 1;
const TERMINAL_INFO_PREFIX: &str = "@info:";

fn terminal_info_key(cache_id: &str) -> String {
    format!("{TERMINAL_INFO_PREFIX}{cache_id}")
}

/// pin 数量上限:排队风暴不允许锁死整个 resident,超出的 pin 退化为尽力而为。
/// 由节点命令层在写入 pin 句柄时执行。
pub const TERMINAL_PIN_LIMIT: usize = 16;

/// 内存 LRU + fjall 换出的统一终点会话管理:resident 满时最旧条目快照落盘,
/// resume 先查内存、miss 再从盘上恢复。swap 持久化时同时保存 CacheInfo 供上报。
pub struct TerminalSessions<S: TerminalSnapshot> {
    resident: TerminalCache<S>,
    swap: Option<crate::kv_cache::fjall::FjallValueStore>,
    swap_infos: HashMap<String, TerminalInfo>,
    /// scheduler 在 append 请求排队期间 pin 的 cache_id:换出循环跳过它们,
    /// 排到队时命中内存而不是 swap 慢路径。共享句柄让 pin/unpin 不进 engine
    /// 命令队列(队列在满载 batch 期间会推迟命令,pin 恰恰在这个窗口必须生效)。
    /// 上限见 [`TERMINAL_PIN_LIMIT`],由写入方(节点命令层)执行。
    pinned: Arc<Mutex<HashSet<String>>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerminalResidencyStats {
    pub resident_entries: usize,
    pub resident_tokens: usize,
    pub resident_bytes: u64,
    pub swapped_entries: usize,
    pub swapped_tokens: usize,
    pub swapped_bytes: u64,
}

impl<S: TerminalSnapshot> TerminalSessions<S> {
    pub fn new(limit: usize, swap: Option<crate::kv_cache::fjall::FjallValueStore>) -> Self {
        let mut swap_infos = HashMap::new();
        if let Some(store) = &swap
            && let Ok(keys) = store.keys()
        {
            let present = keys.iter().map(String::as_str).collect::<HashSet<_>>();
            for cache_id in &keys {
                let Some(snapshot_id) = cache_id.strip_prefix(TERMINAL_INFO_PREFIX) else { continue };
                if !present.contains(snapshot_id) {
                    eprintln!("[terminal-cache] 隔离缺少 snapshot 的 swap metadata cache_id={snapshot_id}");
                    let _ = store.remove(cache_id);
                    continue;
                }
                match store.get(cache_id).and_then(|bytes| bytes.map(|bytes| decode_terminal_info(&bytes)).transpose()) {
                    Ok(Some(info)) => {
                        swap_infos.insert(snapshot_id.to_owned(), info);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        eprintln!("[terminal-cache] 隔离损坏的 swap metadata cache_id={snapshot_id}: {error}");
                        let _ = store.remove_pair(snapshot_id, cache_id);
                    }
                }
            }
        }
        Self { resident: TerminalCache::new(limit), swap, swap_infos, pinned: Arc::new(Mutex::new(HashSet::new())) }
    }

    /// pin 集合的共享写句柄:节点命令层据此直接响应 scheduler 的 PinCache/UnpinCache,
    /// 不经 engine 命令队列。上限限流由写入方执行。
    pub fn pin_handle(&self) -> Arc<Mutex<HashSet<String>>> {
        self.pinned.clone()
    }

    pub fn resident_cost(&self, cost: impl FnMut(&[u32], &S) -> usize) -> usize {
        self.resident.resident_cost(cost)
    }

    /// cache_id 在 resident 或 swap 任一层可恢复。调用方只据此选择恢复路径，
    /// 真正取得状态仍必须走 [`activate`](Self::activate) 完成容量准入。
    pub fn contains(&self, cache_id: &str) -> bool {
        self.resident.cached_tokens(cache_id).is_some() || self.swap_infos.contains_key(cache_id)
    }

    pub fn residency_stats(&self) -> TerminalResidencyStats {
        let mut stats = TerminalResidencyStats::default();
        for state in self.resident.states() {
            stats.resident_entries += 1;
            stats.resident_tokens = stats.resident_tokens.saturating_add(state.info().prompt_tokens);
            stats.resident_bytes = stats.resident_bytes.saturating_add(state.info().bytes);
        }
        for info in self.swap_infos.values() {
            stats.swapped_entries += 1;
            stats.swapped_tokens = stats.swapped_tokens.saturating_add(info.prompt_tokens);
            stats.swapped_bytes = stats.swapped_bytes.saturating_add(info.bytes);
        }
        stats
    }

    /// 列出 swap 上所有持久化 cache 的元数据;供节点心跳把"换出但仍在本地"的 cache_id
    /// 报给 scheduler,避免 server 误以为 cache 不在本地而把请求分到其他节点。空 swap 返回空。
    /// 启动时也走这条路径:打开已有 swap store 后立刻 `swap_infos()` 拿到视图。
    pub fn swap_infos(&self) -> Vec<TerminalInfo> {
        let mut infos: Vec<_> = self.swap_infos.values().cloned().collect();
        infos.sort_by(|left, right| left.cache_id.cmp(&right.cache_id));
        infos
    }

    /// 节点调度看到的是本节点可恢复的全部终点会话，而不是物理驻留位置。
    /// resident 优先覆盖同 id 的旧 swap metadata；这也容忍提交 resident 后
    /// 删除旧持久化副本失败的可恢复状态。
    pub fn infos(&self) -> Vec<TerminalInfo> {
        let mut infos = self.swap_infos.clone();
        for state in self.resident.states() {
            infos.insert(state.info().cache_id.clone(), state.info().clone());
        }
        let mut infos: Vec<_> = infos.into_values().collect();
        infos.sort_by(|left, right| left.cache_id.cmp(&right.cache_id));
        infos
    }

    /// 为 active session 预留 resident 容量；不足时按 LRU 换出终点会话。
    /// 被排队 append pin 的 cache 跳过不换,全部候选被 pin 时直接报容量不足。
    /// `cost` 与 budget 使用同一单位（token page 或字节），因此可复用在多节点与库模式。
    pub fn reserve_with_eviction(&mut self, budget: &ResidencyBudget, required: usize, mut cost: impl FnMut(&[u32], &S) -> usize) -> Result<ResidencyReservation, String> {
        if budget.used().saturating_add(required) > budget.capacity() {
            return Err(format!("KV_RESIDENCY_EXHAUSTED required={required} available={} active={} resident={} capacity={}", budget.available(), budget.used(), self.resident.resident_cost(&mut cost), budget.capacity(),));
        }
        while budget.used().saturating_add(self.resident.resident_cost(&mut cost)).saturating_add(required) > budget.capacity() {
            let pinned_count = {
                let pinned = self.pinned.lock().expect("pin 集合锁中毒");
                match self.resident.oldest_unpinned(&pinned) {
                    Some((cache_id, _, state)) => {
                        let cache_id = cache_id.to_owned();
                        if let Some(swap) = &self.swap {
                            persist_terminal(swap, &cache_id, state).map_err(|error| format!("KV 换出失败 cache_id={cache_id}: {error}"))?;
                            self.swap_infos.insert(cache_id.clone(), state.info().clone());
                        }
                        self.resident.take(&cache_id);
                        continue;
                    }
                    None => pinned.len(),
                }
            };
            return Err(format!(
                "KV_RESIDENCY_EXHAUSTED required={required} available={} active={} resident={} capacity={} pinned={pinned_count}",
                budget.capacity().saturating_sub(budget.used().saturating_add(self.resident.resident_cost(&mut cost))),
                budget.used(),
                self.resident.resident_cost(&mut cost),
                budget.capacity(),
            ));
        }
        budget.try_reserve(required).ok_or_else(|| format!("KV_RESIDENCY_RACE required={required} available={}", budget.available()))
    }

    /// 把命中的 terminal session 从 resident 所有权转成 active reservation。
    /// resident 命中先移出再 admission，避免同一份 cache 被重复计费；swap 命中
    /// 则先取得容量门票再解码，防止恢复分配发生在容量检查之前。
    pub fn activate(&mut self, cache_id: Option<&str>, budget: &ResidencyBudget, required: usize, cost: impl FnMut(&[u32], &S) -> usize, resources: &S::Resources) -> Result<(Option<(Vec<u32>, S)>, ResidencyReservation), String> {
        let resident = cache_id.and_then(|cache_id| self.resident.take(cache_id));
        let reservation = match self.reserve_with_eviction(budget, required, cost) {
            Ok(reservation) => reservation,
            Err(error) => {
                if let Some((tokens, state)) = resident {
                    self.resident.insert(cache_id.expect("resident 命中必有 cache_id").to_owned(), tokens, state);
                }
                return Err(error);
            }
        };
        if resident.is_some() {
            return Ok((resident, reservation));
        }
        let restored = match cache_id.filter(|cache_id| self.swap_infos.contains_key(*cache_id)) {
            Some(cache_id) => match self.resume(cache_id, resources) {
                Some(Ok(state)) => Some(state),
                Some(Err(error)) => return Err(error),
                None => None,
            },
            None => None,
        };
        Ok((restored, reservation))
    }

    /// 把最长公共前缀命中的 resident session 转成 active reservation。前缀匹配
    /// 不主动解码 swap；准入失败时把已取出的状态放回，不能为了失败请求丢 cache。
    pub fn activate_longest_common_prefix(
        &mut self,
        tokens: &[u32],
        namespace: Option<&str>,
        budget: &ResidencyBudget,
        required: usize,
        cost: impl FnMut(&[u32], &S) -> usize,
    ) -> Result<(Option<(Vec<u32>, usize, S)>, ResidencyReservation), String> {
        let resident = self.resident.take_longest_common_prefix(tokens, |state| state.cache_namespace() == namespace);
        let reservation = match self.reserve_with_eviction(budget, required, cost) {
            Ok(reservation) => reservation,
            Err(error) => {
                if let Some((cache_id, cached, _, state)) = resident {
                    self.resident.insert(cache_id, cached, state);
                }
                return Err(error);
            }
        };
        Ok((resident.map(|(_, cached, lcp, state)| (cached, lcp, state)), reservation))
    }

    pub fn retain(&mut self, cache_id: String, tokens: Vec<u32>, state: S) -> bool {
        let replaces_swap = self.swap_infos.contains_key(&cache_id);
        // resident 容量为 0(全 SSD 模式)：状态编码后直接落盘，不在内存保留任何会话。
        if self.resident.capacity() == 0 {
            let Some(swap) = &self.swap else { return false };
            return match persist_terminal(swap, &cache_id, &state) {
                Ok(()) => {
                    self.swap_infos.insert(cache_id, state.info().clone());
                    let _ = tokens;
                    true
                }
                Err(error) => {
                    eprintln!("[terminal-cache] 落盘写入失败 cache_id={cache_id}: {error}");
                    false
                }
            };
        }
        while self.resident.is_full() {
            let evicted = {
                let pinned = self.pinned.lock().expect("pin 集合锁中毒");
                self.resident.oldest_unpinned(&pinned).map(|(cache_id, _, _)| cache_id.to_owned())
            };
            let Some(evicted_id) = evicted else {
                // resident 满且可换出的全部被 pin:新终态直接落盘,排队者的换出
                // 保护不被破坏;无 swap 时只能丢弃(pin 上限之外的最后一道闸)。
                let _ = tokens;
                let Some(swap) = &self.swap else { return false };
                return match persist_terminal(swap, &cache_id, &state) {
                    Ok(()) => {
                        self.swap_infos.insert(cache_id, state.info().clone());
                        true
                    }
                    Err(error) => {
                        eprintln!("[terminal-cache] 落盘写入失败 cache_id={cache_id}: {error}");
                        false
                    }
                };
            };
            let Some((evicted_tokens, evicted)) = self.resident.take(&evicted_id) else { continue };
            if let Some(swap) = &self.swap {
                if let Err(error) = persist_terminal(swap, &evicted_id, &evicted) {
                    eprintln!("[terminal-cache] 换出写入失败 cache_id={evicted_id}: {error}");
                    // 放回保住状态,本次 retain 失败
                    let _ = self.resident.insert(evicted_id, evicted_tokens, evicted);
                    return false;
                }
                self.swap_infos.insert(evicted_id.clone(), evicted.info().clone());
            }
        }
        let inserted = self.resident.insert(cache_id.clone(), tokens, state);
        if inserted && replaces_swap {
            // 新 resident 已经取得所有权后再删除旧持久化副本；删除失败只留下可清理的
            // 重复副本，不能反过来丢弃已经可用的新状态。
            if let Some(swap) = &self.swap {
                match swap.remove_pair(&cache_id, &terminal_info_key(&cache_id)) {
                    Ok(()) => {
                        self.swap_infos.remove(&cache_id);
                    }
                    Err(error) => eprintln!("[terminal-cache] 新 resident 已提交，但删除旧 swap 失败 cache_id={cache_id}: {error}"),
                }
            }
        }
        inserted
    }

    /// 优雅关闭时把全部 resident session 写盘。正常请求完成只保留内存状态，
    /// 不在 decode 尾部同步编码或写盘；进程异常退出允许丢失尚未换出的 session。
    pub fn persist_resident(&mut self) -> Result<usize, String> {
        let Some(swap) = &self.swap else { return Ok(0) };
        let mut persisted = 0usize;
        let mut first_error = None;
        while let Some((cache_id, _, state)) = self.resident.oldest() {
            let cache_id = cache_id.to_owned();
            match persist_terminal(swap, &cache_id, state) {
                Ok(()) => {
                    self.swap_infos.insert(cache_id.clone(), state.info().clone());
                    self.resident.take(&cache_id);
                    persisted += 1;
                }
                Err(error) => {
                    first_error.get_or_insert_with(|| format!("cache_id={cache_id}: {error}"));
                    break;
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(persisted),
        }
    }

    /// 命中返回内存条目;内存 miss 且盘上有快照时恢复。两者都没有时返回 None。
    ///
    /// SSD 快照在恢复成功后仍作为本次 active session 的 checkpoint 保留，直到
    /// [`retain`](Self::retain) 成功提交新终点才删除。这样 backend 执行失败不会先
    /// 消费掉唯一可恢复副本；损坏快照则立即隔离，避免后续请求反复恢复失败。
    pub fn resume(&mut self, cache_id: &str, resources: &S::Resources) -> Option<Result<(Vec<u32>, S), String>> {
        if let Some(entry) = self.resident.take(cache_id) {
            return Some(Ok(entry));
        }
        let swap = self.swap.as_ref()?;
        let bytes = match swap.get(cache_id) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                let _ = swap.remove(&terminal_info_key(cache_id));
                self.swap_infos.remove(cache_id);
                return None;
            }
            Err(error) => return Some(Err(format!("swap 读取失败 cache_id={cache_id}: {error}"))),
        };
        match decode_persisted_terminal(&bytes) {
            Ok((info, state_bytes)) => match S::decode(state_bytes, resources) {
                Ok(mut state) => {
                    state.set_info(info);
                    let tokens = state.terminal_tokens().to_vec();
                    Some(Ok((tokens, state)))
                }
                Err(error) => {
                    let _ = swap.remove_pair(cache_id, &terminal_info_key(cache_id));
                    self.swap_infos.remove(cache_id);
                    Some(Err(format!("终点快照恢复失败且已隔离 cache_id={cache_id}: {error}")))
                }
            },
            Err(error) => {
                let _ = swap.remove_pair(cache_id, &terminal_info_key(cache_id));
                self.swap_infos.remove(cache_id);
                Some(Err(format!("swap 持久化格式解析失败且已隔离 cache_id={cache_id}: {error}")))
            }
        }
    }

    /// 内容最长公共前缀匹配(只查 resident;swap 条目不解码不参与)。
    /// `namespace` 是协议层 cache namespace(请求原样转发),只匹配同租户 entry。
    /// 返回 (cached_tokens, lcp, state);恢复策略见 [`resume_by_prefix`]。
    pub fn resume_longest_common_prefix(&mut self, tokens: &[u32], namespace: Option<&str>) -> Option<(Vec<u32>, usize, S)> {
        self.resident.take_longest_common_prefix(tokens, |state| state.cache_namespace() == namespace).map(|(_, cached, lcp, state)| (cached, lcp, state))
    }

    /// 丢弃内存与盘上的条目(会话失效/重建)。
    pub fn discard(&mut self, cache_id: &str) {
        self.resident.take(cache_id);
        if let Some(swap) = &self.swap {
            let _ = swap.remove_pair(cache_id, &terminal_info_key(cache_id));
        }
        self.swap_infos.remove(cache_id);
    }

    pub fn states(&self) -> impl Iterator<Item = &S> {
        self.resident.states()
    }
}

fn persist_terminal<S: TerminalSnapshot>(swap: &crate::kv_cache::fjall::FjallValueStore, cache_id: &str, state: &S) -> Result<(), String> {
    let state_bytes = state.encode().map_err(|error| format!("编码失败: {error}"))?;
    let info = encode_terminal_info(state.info())?;
    let info_len = u32::try_from(info.len()).map_err(|_| format!("持久化 metadata {} bytes 超过 u32", info.len()))?;
    let mut bytes = Vec::with_capacity(16usize.saturating_add(info.len()).saturating_add(state_bytes.len()));
    bytes.extend_from_slice(&TERMINAL_MAGIC);
    bytes.extend_from_slice(&TERMINAL_VERSION.to_le_bytes());
    bytes.extend_from_slice(&info_len.to_le_bytes());
    bytes.extend_from_slice(&info);
    bytes.extend_from_slice(&state_bytes);
    swap.put_pair(cache_id, &bytes, &terminal_info_key(cache_id), &info).map_err(|error| format!("写入失败: {error}"))
}

fn decode_persisted_terminal(bytes: &[u8]) -> Result<(TerminalInfo, &[u8]), String> {
    if bytes.len() < 16 || bytes[..8] != TERMINAL_MAGIC {
        return Err("terminal snapshot magic 不匹配".to_owned());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("固定四字节"));
    if version != TERMINAL_VERSION {
        return Err(format!("terminal snapshot version={version}，当前支持 {TERMINAL_VERSION}"));
    }
    let info_len = u32::from_le_bytes(bytes[12..16].try_into().expect("固定四字节")) as usize;
    let state_offset = 16usize.checked_add(info_len).ok_or("terminal metadata 长度溢出")?;
    let info_bytes = bytes.get(16..state_offset).ok_or_else(|| format!("terminal metadata 声明 {info_len} bytes，实际总长 {}", bytes.len()))?;
    let info = decode_terminal_info(info_bytes)?;
    Ok((info, &bytes[state_offset..]))
}

fn encode_terminal_info(info: &TerminalInfo) -> Result<Vec<u8>, String> {
    let mut writer = SnapshotWriter::new();
    writer.bytes(&TERMINAL_INFO_MAGIC);
    writer.u32(TERMINAL_INFO_VERSION);
    writer.string(&info.cache_id)?;
    writer.string(&info.model_key)?;
    writer.string(&info.cache_format)?;
    writer.usize(info.last_layer)?;
    writer.usize(info.prompt_tokens)?;
    writer.u64(info.bytes);
    writer.u64(info.modified_unix);
    Ok(writer.into_inner())
}

fn decode_terminal_info(bytes: &[u8]) -> Result<TerminalInfo, String> {
    let mut reader = SnapshotReader::new(bytes);
    if reader.take(8, "terminal info magic")? != TERMINAL_INFO_MAGIC {
        return Err("terminal info magic 不匹配".to_owned());
    }
    let version = reader.u32("terminal info version")?;
    if version != TERMINAL_INFO_VERSION {
        return Err(format!("terminal info version={version}，当前支持 {TERMINAL_INFO_VERSION}"));
    }
    let info = TerminalInfo {
        cache_id: reader.string("terminal cache_id")?,
        model_key: reader.string("terminal model_key")?,
        cache_format: reader.string("terminal cache_format")?,
        last_layer: reader.usize("terminal last_layer")?,
        prompt_tokens: reader.usize("terminal prompt_tokens")?,
        bytes: reader.u64("terminal bytes")?,
        modified_unix: reader.u64("terminal modified_unix")?,
    };
    reader.finish()?;
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestGraph(u64);

    impl SharedBlockGraph for TestGraph {
        type Session = u64;
        type Error = std::convert::Infallible;

        fn materialize(&self) -> Result<Self::Session, Self::Error> {
            Ok(self.0)
        }
    }

    #[test]
    fn shared_graph会话只留最新head且全局池只留前四轮() {
        let mut cache = SharedBlockCache::new(16, 4);
        cache.insert("a".into(), vec![1], None, TestGraph(1));
        cache.insert("b".into(), vec![1, 2], Some(("a".into(), 1)), TestGraph(2));
        cache.insert("c".into(), vec![1, 2, 3], Some(("b".into(), 2)), TestGraph(3));
        cache.insert("d".into(), vec![1, 2, 3, 4], Some(("c".into(), 3)), TestGraph(4));
        cache.insert("e".into(), vec![1, 2, 3, 4, 5], Some(("d".into(), 4)), TestGraph(5));
        let removed = cache.insert("f".into(), vec![1, 2, 3, 4, 5, 6], Some(("e".into(), 5)), TestGraph(6));

        assert_eq!(removed.len(), 1, "第五轮旧 head 应立即退出");
        for cache_id in ["a", "b", "c", "d", "f"] {
            assert!(cache.cached_tokens(cache_id).is_some(), "{cache_id} 应在前缀池或会话 head");
        }
        assert!(cache.cached_tokens("e").is_none(), "第五轮旧 head 不进入全局前缀池");
        assert_eq!(cache.is_head("b"), Some(false));
        assert_eq!(cache.is_head("f"), Some(true));
        assert_eq!(cache.materialize("b", &[1, 2, 9]).unwrap().unwrap(), (vec![1, 2], 2, 2));
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn shared_graph从全局前缀分支不覆盖原会话head() {
        let mut cache = SharedBlockCache::new(16, 4);
        cache.insert("a".into(), vec![1], None, TestGraph(1));
        cache.insert("b".into(), vec![1, 2], Some(("a".into(), 1)), TestGraph(2));
        cache.insert("c".into(), vec![1, 2, 3], Some(("b".into(), 2)), TestGraph(3));
        cache.insert("x".into(), vec![1, 2, 9], Some(("b".into(), 2)), TestGraph(9));

        assert!(cache.cached_tokens("b").is_some());
        assert!(cache.cached_tokens("c").is_some());
        assert!(cache.cached_tokens("x").is_some());
        let (cache_id, _, _, _) = cache.materialize_longest_prefix(&[1, 2, 3, 4], |_| true).unwrap().unwrap();
        assert_eq!(cache_id, "b", "会话 head 不参与全局最长前缀匹配");
    }

    #[test]
    fn shared_graph全局lru有固定上限() {
        let mut cache = SharedBlockCache::new(2, 8);
        cache.insert("a".into(), vec![1], None, TestGraph(1));
        cache.insert("b".into(), vec![2], None, TestGraph(2));
        cache.materialize("a", &[1]).unwrap().unwrap();
        cache.insert("c".into(), vec![3], None, TestGraph(3));

        assert!(cache.cached_tokens("a").is_some());
        assert!(cache.cached_tokens("b").is_none());
        assert!(cache.cached_tokens("c").is_some());
    }

    #[test]
    fn no_swap_evicts_oldest_and_drops() {
        let mut cache = TerminalCache::<u64>::new(2);
        cache.insert("a".into(), vec![1, 2], 10);
        cache.insert("b".into(), vec![3, 4], 20);
        cache.insert("c".into(), vec![5, 6], 30); // 容量 2，淘汰 a 并丢弃。
        assert!(cache.cached_tokens("a").is_none());
        assert!(cache.take_matching("a", &[1, 2]).is_none());
        assert_eq!(cache.take_matching("b", &[3, 4]), Some((vec![3, 4], 20)));
    }

    #[test]
    fn longest_prefix跨cache_id选择最长且服从隔离条件() {
        let mut cache = TerminalCache::new(4);
        cache.insert("short".into(), vec![1, 2], ("tenant-a", 10));
        cache.insert("long".into(), vec![1, 2, 3], ("tenant-a", 20));
        cache.insert("other".into(), vec![1, 2, 3, 4], ("tenant-b", 30));

        let (cache_id, tokens, state) = cache.take_longest_prefix(&[1, 2, 3, 4, 5], |state| state.0 == "tenant-a").unwrap();
        assert_eq!(cache_id, "long");
        assert_eq!(tokens, vec![1, 2, 3]);
        assert_eq!(state, ("tenant-a", 20));
        assert!(cache.cached_tokens("long").is_none());
        assert!(cache.cached_tokens("other").is_some());
    }

    #[test]
    fn take_oldest_removes_fifo_entry() {
        let mut cache = TerminalCache::new(2);
        cache.insert("a".into(), vec![1], 10);
        cache.insert("b".into(), vec![2], 20);
        assert_eq!(cache.len(), 2);
        assert!(cache.is_full());
        assert_eq!(cache.take_oldest(), Some(("a".into(), vec![1], 10)));
        assert!(!cache.is_full());
        assert_eq!(cache.take("b"), Some((vec![2], 20)));
        assert!(cache.is_empty());
    }

    #[test]
    fn longest_common_prefix允许cached比新tokens更长() {
        let long: Vec<u32> = (0..64).collect();
        let mut cache = TerminalCache::new(4);
        cache.insert("full".into(), long.clone(), ("tenant-a", 10));
        cache.insert("short".into(), long[..40].to_vec(), ("tenant-a", 20));
        cache.insert("other".into(), long[..40].to_vec(), ("tenant-b", 30));

        // 新 prompt 是 cached 的真前缀(bench 重复 prompt 场景):lcp = 新长度。
        let query = long[..48].to_vec();
        let (cache_id, cached, lcp, _) = cache.take_longest_common_prefix(&query, |_| true).unwrap();
        assert_eq!(cache_id, "full");
        assert_eq!(cached, long);
        assert_eq!(lcp, 48);
        // 隔离条件:tenant-b 的 entry 不参与 tenant-a 匹配;短于阈值不消费 entry。
        let (cache_id, _, lcp, _) = cache.take_longest_common_prefix(&query, |state| state.0 == "tenant-a").unwrap();
        assert_eq!(cache_id, "short");
        assert_eq!(lcp, 40);
        // tenant-b 仍能命中自己的 entry;不存在的租户则匹配为空。
        assert!(cache.take_longest_common_prefix(&query, |state| state.0 == "tenant-b").is_some());
        assert!(cache.take_longest_common_prefix(&query, |state| state.0 == "tenant-c").is_none());
    }

    #[test]
    fn residency_reservation_releases_capacity_on_drop() {
        let budget = ResidencyBudget::new(10);
        let mut first = budget.try_reserve(7).unwrap();
        assert_eq!(first.cost(), 7);
        assert_eq!(budget.available(), 3);
        assert!(budget.try_reserve(4).is_none());
        assert!(first.try_grow(3));
        assert_eq!(first.cost(), 10);
        assert!(!first.try_grow(1));
        drop(first);
        assert_eq!(budget.available(), 10);
    }

    #[test]
    fn resident_cost_is_defined_by_adapter() {
        let mut cache = TerminalCache::new(2);
        cache.insert("a".into(), vec![1, 2], 10usize);
        cache.insert("b".into(), vec![3], 20usize);
        assert_eq!(cache.resident_cost(|tokens, state| tokens.len() * state), 40);
    }

    #[test]
    fn snapshot_codec_round_trip_and_bounds() {
        let mut writer = SnapshotWriter::new();
        writer.u32(7);
        writer.u64(9);
        writer.i64(-3);
        writer.u32s(&[11, 12]).unwrap();
        writer.bytes(&[1, 2]);
        let bytes = writer.into_inner();

        let mut reader = SnapshotReader::new(&bytes);
        assert_eq!(reader.u32("version").unwrap(), 7);
        assert_eq!(reader.u64("size").unwrap(), 9);
        assert_eq!(reader.i64("delta").unwrap(), -3);
        assert_eq!(reader.u32s("tokens").unwrap(), [11, 12]);
        assert_eq!(reader.take(2, "tail").unwrap(), [1, 2]);
        reader.finish().unwrap();

        let mut truncated = SnapshotReader::new(&bytes[..3]);
        assert!(truncated.u32("version").unwrap_err().contains("字节不足"));
    }

    #[test]
    fn terminal_info使用版本化二进制而不是json() {
        let info = TerminalInfo { cache_id: "cache".into(), model_key: "model".into(), cache_format: "q8g64".into(), last_layer: 7, prompt_tokens: 32, bytes: 4096, modified_unix: 9 };
        let bytes = encode_terminal_info(&info).unwrap();
        assert_eq!(&bytes[..8], &TERMINAL_INFO_MAGIC);
        assert_eq!(decode_terminal_info(&bytes).unwrap(), info);
        assert!(decode_terminal_info(br#"{"cache_id":"cache"}"#).is_err());
    }
}

#[cfg(test)]
mod swap_tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct CountingState {
        tokens: Vec<u32>,
        marker: u32,
    }

    struct MustNotEncode;

    struct TrackingState {
        tokens: Vec<u32>,
        encodes: Arc<AtomicUsize>,
    }

    /// 带真实 CacheInfo 的 CountingState,用于验证 swap_infos 序列化往返。
    struct CountingStateWithInfo {
        tokens: Vec<u32>,
        marker: u32,
        info: TerminalInfo,
    }

    impl TerminalSnapshot for CountingStateWithInfo {
        type Resources = ();
        fn encode(&self) -> Result<Vec<u8>, String> {
            let mut bytes = self.marker.to_le_bytes().to_vec();
            bytes.extend((self.tokens.len() as u32).to_le_bytes());
            bytes.extend(self.tokens.iter().flat_map(|token| token.to_le_bytes()));
            Ok(bytes)
        }
        fn decode(bytes: &[u8], _resources: &Self::Resources) -> Result<Self, String> {
            let marker = u32::from_le_bytes(bytes[..4].try_into().map_err(|_| "字节长度非法".to_owned())?);
            let count = u32::from_le_bytes(bytes[4..8].try_into().map_err(|_| "字节长度非法".to_owned())?) as usize;
            let tokens = (0..count).map(|index| u32::from_le_bytes(bytes[8 + index * 4..12 + index * 4].try_into().expect("token 字节"))).collect();
            Ok(CountingStateWithInfo { tokens, marker, info: TerminalInfo::default() })
        }
        fn terminal_tokens(&self) -> &[u32] {
            &self.tokens
        }
        fn info(&self) -> &TerminalInfo {
            &self.info
        }
        fn set_info(&mut self, info: TerminalInfo) {
            self.info = info;
        }
    }

    impl TerminalSnapshot for MustNotEncode {
        type Resources = ();
        fn encode(&self) -> Result<Vec<u8>, String> {
            panic!("关闭持久化后不应编码 KV cache")
        }
        fn decode(_bytes: &[u8], _resources: &()) -> Result<Self, String> {
            unreachable!()
        }
        fn terminal_tokens(&self) -> &[u32] {
            &[]
        }
        fn info(&self) -> &TerminalInfo {
            static EMPTY: std::sync::OnceLock<TerminalInfo> = std::sync::OnceLock::new();
            EMPTY.get_or_init(TerminalInfo::default)
        }
    }

    impl TerminalSnapshot for CountingState {
        type Resources = ();
        fn encode(&self) -> Result<Vec<u8>, String> {
            let mut bytes = self.marker.to_le_bytes().to_vec();
            bytes.extend((self.tokens.len() as u32).to_le_bytes());
            bytes.extend(self.tokens.iter().flat_map(|token| token.to_le_bytes()));
            Ok(bytes)
        }
        fn decode(bytes: &[u8], _resources: &()) -> Result<Self, String> {
            let marker = u32::from_le_bytes(bytes[..4].try_into().map_err(|_| "字节长度非法".to_owned())?);
            let count = u32::from_le_bytes(bytes[4..8].try_into().map_err(|_| "字节长度非法".to_owned())?) as usize;
            let tokens = (0..count).map(|index| u32::from_le_bytes(bytes[8 + index * 4..12 + index * 4].try_into().expect("token 字节"))).collect();
            Ok(CountingState { tokens, marker })
        }
        fn terminal_tokens(&self) -> &[u32] {
            &self.tokens
        }
        fn info(&self) -> &TerminalInfo {
            static EMPTY: std::sync::OnceLock<TerminalInfo> = std::sync::OnceLock::new();
            EMPTY.get_or_init(TerminalInfo::default)
        }
    }

    impl TerminalSnapshot for TrackingState {
        type Resources = ();
        fn encode(&self) -> Result<Vec<u8>, String> {
            self.encodes.fetch_add(1, Ordering::Relaxed);
            Ok(Vec::new())
        }
        fn decode(_bytes: &[u8], _resources: &()) -> Result<Self, String> {
            unreachable!()
        }
        fn terminal_tokens(&self) -> &[u32] {
            &self.tokens
        }
        fn info(&self) -> &TerminalInfo {
            static EMPTY: std::sync::OnceLock<TerminalInfo> = std::sync::OnceLock::new();
            EMPTY.get_or_init(TerminalInfo::default)
        }
    }

    #[test]
    fn pin保护resident换出且全pin时新终态落盘() {
        let directory = std::env::temp_dir().join(format!("zllm-terminal-pin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let swap = crate::kv_cache::fjall::FjallValueStore::open(&directory, "terminal-test").expect("打开 fjall");
        let mut sessions = TerminalSessions::<CountingState>::new(1, Some(swap));
        assert!(sessions.retain("a".into(), vec![1], CountingState { tokens: vec![1], marker: 11 }));
        // pin 住唯一 resident 条目:新终态不能靠换出 a 腾位,只能自己落盘,a 留在内存
        let pins = sessions.pin_handle();
        pins.lock().unwrap().insert("a".to_owned());
        assert!(sessions.retain("b".into(), vec![2], CountingState { tokens: vec![2], marker: 22 }));
        assert!(sessions.resident.cached_tokens("a").is_some(), "被 pin 的 a 必须留在内存");
        assert_eq!(sessions.resume("b", &()).expect("b 应已落盘").expect("b 应解码成功").1.marker, 22);
        // 解除 pin 后常规换出恢复
        pins.lock().unwrap().remove("a");
        assert!(sessions.retain("c".into(), vec![3], CountingState { tokens: vec![3], marker: 33 }));
        assert!(sessions.resident.cached_tokens("a").is_none(), "unpin 后 a 恢复为可换出");
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn pin保护准入换出并报容量不足() {
        let mut sessions = TerminalSessions::<CountingState>::new(2, None);
        sessions.retain("a".into(), vec![1], CountingState { tokens: vec![1], marker: 1 });
        sessions.retain("b".into(), vec![2], CountingState { tokens: vec![2], marker: 2 });
        let budget = ResidencyBudget::new(20);
        let cost = |tokens: &[u32], _: &CountingState| tokens.len() * 10;
        // 全部候选被 pin:无法靠换出腾容量,报 KV_RESIDENCY_EXHAUSTED 并带上 pinned 计数
        let pins = sessions.pin_handle();
        pins.lock().unwrap().insert("a".to_owned());
        pins.lock().unwrap().insert("b".to_owned());
        let error = match sessions.reserve_with_eviction(&budget, 15, cost) {
            Err(error) => error,
            Ok(_) => panic!("全部候选被 pin 时准入应失败"),
        };
        assert!(error.contains("KV_RESIDENCY_EXHAUSTED") && error.contains("pinned=2"), "{error}");
        // 解 pin 后换出 a/b 腾出容量,准入成功
        pins.lock().unwrap().clear();
        assert!(sessions.reserve_with_eviction(&budget, 15, cost).is_ok());
        assert!(sessions.resident.is_empty(), "无 pin 时 a、b 都应被换出(无 swap 即丢弃)");
    }

    #[test]
    fn retain_evicts_to_swap_and_resume_restores() {
        let directory = std::env::temp_dir().join(format!("zllm-terminal-swap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let swap = crate::kv_cache::fjall::FjallValueStore::open(&directory, "terminal-test").expect("打开 fjall");
        let mut sessions = TerminalSessions::<CountingState>::new(1, Some(swap));
        assert!(sessions.retain("a".into(), vec![1], CountingState { tokens: vec![1], marker: 11 }));
        // 容量 1:第二条把 a 换出到盘
        assert!(sessions.retain("b".into(), vec![2], CountingState { tokens: vec![2], marker: 22 }));
        // b 在内存;a 从盘上恢复
        let resumed = sessions.resume("a", &()).expect("盘上快照可恢复").expect("解码成功");
        assert_eq!(resumed.0, vec![1]);
        assert_eq!(resumed.1.marker, 11);
        // active 执行完成前旧 SSD checkpoint 不消费，失败后仍可重新恢复。
        assert!(sessions.resume("a", &()).is_some());
        // 新终点成功进入 resident 后才提交并删除旧 checkpoint。
        assert!(sessions.retain("a".into(), vec![1, 3], CountingState { tokens: vec![1, 3], marker: 33 }));
        assert!(sessions.swap.as_ref().unwrap().get("a").unwrap().is_none());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn disabled_persistence_does_not_encode_zero_resident_cache() {
        let mut sessions = TerminalSessions::<MustNotEncode>::new(0, None);
        assert!(!sessions.retain("benchmark".into(), vec![1], MustNotEncode));
    }

    #[test]
    fn admission不足时换出resident且无可换出时失败() {
        let budget = ResidencyBudget::new(10);
        let mut sessions = TerminalSessions::<CountingState>::new(2, None);
        assert!(sessions.retain("old".into(), vec![1], CountingState { tokens: vec![1], marker: 6 }));
        let reservation = sessions.reserve_with_eviction(&budget, 6, |_, state| state.marker as usize).expect("换出后应可预留");
        assert_eq!(reservation.cost(), 6);
        assert!(sessions.states().next().is_none());
        let error = match sessions.reserve_with_eviction(&budget, 5, |_, state| state.marker as usize) {
            Ok(_) => panic!("active reservation 已占用容量时不应继续接纳"),
            Err(error) => error,
        };
        assert!(error.contains("KV_RESIDENCY_EXHAUSTED"));
    }

    #[test]
    fn admission确定失败时不驱逐resident() {
        let budget = ResidencyBudget::new(10);
        let _active = budget.try_reserve(7).expect("active");
        let mut sessions = TerminalSessions::<CountingState>::new(1, None);
        assert!(sessions.retain("keep".into(), vec![1], CountingState { tokens: vec![1], marker: 2 }));
        assert!(sessions.reserve_with_eviction(&budget, 4, |_, state| state.marker as usize).is_err());
        assert_eq!(sessions.states().count(), 1, "确定无法 admission 时不得破坏 resident cache");
    }

    #[test]
    fn 惰性引擎资源永久缩减session预算() {
        let budget = ResidencyBudget::new(100);
        let reservation = budget.try_reserve(40).expect("初始 session 应可准入");
        budget.consume_capacity(30);
        assert_eq!(budget.capacity(), 70);
        assert_eq!(budget.available(), 30);
        assert!(budget.try_reserve(31).is_none());
        drop(reservation);
        assert_eq!(budget.available(), 70);
    }

    #[test]
    fn activate命中resident时不重复计费() {
        let budget = ResidencyBudget::new(10);
        let mut sessions = TerminalSessions::<CountingState>::new(1, None);
        assert!(sessions.retain("hit".into(), vec![1], CountingState { tokens: vec![1], marker: 6 }));
        let (resumed, reservation) = sessions.activate(Some("hit"), &budget, 6, |_, state| state.marker as usize, &()).expect("命中的 resident 应直接转成 active");
        assert_eq!(reservation.cost(), 6);
        assert_eq!(resumed.expect("应命中").1.marker, 6);
        assert!(sessions.states().next().is_none());
    }

    #[test]
    fn 前缀activate转移resident所有权且失败时放回() {
        let tokens: Vec<u32> = (0..64).collect();
        let budget = ResidencyBudget::new(10);
        let mut sessions = TerminalSessions::<CountingState>::new(1, None);
        assert!(sessions.retain("prefix".into(), tokens.clone(), CountingState { tokens: tokens.clone(), marker: 6 }));

        let (resumed, reservation) = sessions.activate_longest_common_prefix(&tokens, None, &budget, 6, |_, state| state.marker as usize).expect("前缀命中应转成 active");
        assert_eq!(reservation.cost(), 6);
        assert_eq!(resumed.expect("应命中").1, 64);
        assert!(sessions.states().next().is_none());
        drop(reservation);

        assert!(sessions.retain("prefix".into(), tokens.clone(), CountingState { tokens: tokens.clone(), marker: 6 }));
        let _active = budget.try_reserve(7).expect("active");
        assert!(sessions.activate_longest_common_prefix(&tokens, None, &budget, 6, |_, state| state.marker as usize).is_err());
        assert!(sessions.contains("prefix"), "准入失败后必须恢复 resident cache");
    }

    #[test]
    fn persisted_terminal保留raw_state并独立保存metadata() {
        let directory = std::env::temp_dir().join(format!("zllm-terminal-envelope-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let swap = crate::kv_cache::fjall::FjallValueStore::open(&directory, "terminal-envelope").expect("打开 fjall");
        let info = TerminalInfo { cache_id: "raw".into(), model_key: "m".into(), cache_format: "test".into(), last_layer: 1, prompt_tokens: 2, bytes: 12, modified_unix: 3 };
        let mut sessions = TerminalSessions::<CountingStateWithInfo>::new(0, Some(swap));
        assert!(sessions.retain("raw".into(), vec![7, 8], CountingStateWithInfo { tokens: vec![7, 8], marker: 0x01020304, info: info.clone() }));
        drop(sessions);

        let swap = crate::kv_cache::fjall::FjallValueStore::open(&directory, "terminal-envelope").expect("重新打开 fjall");
        let snapshot = swap.get("raw").unwrap().expect("snapshot");
        assert_eq!(&snapshot[..8], &TERMINAL_MAGIC);
        let (_, raw) = decode_persisted_terminal(&snapshot).expect("decode envelope");
        assert_eq!(raw, CountingStateWithInfo { tokens: vec![7, 8], marker: 0x01020304, info: info.clone() }.encode().unwrap());
        let metadata = swap.get(&terminal_info_key("raw")).unwrap().expect("metadata");
        assert_eq!(decode_terminal_info(&metadata).unwrap(), info);
        let mut restored_sessions = TerminalSessions::<CountingStateWithInfo>::new(0, Some(swap));
        let (_, restored) = restored_sessions.resume("raw", &()).expect("存在快照").expect("恢复成功");
        assert_eq!(restored.info, info, "envelope metadata 必须注回恢复状态");
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn swap_infos_lists_persisted_caches() {
        // swap 持久化的 cache_id 也要让 server 看到(用户原话:"内存中 cache id 也要报告"):
        // 节点通过 `swap_infos` 把换出/全 SSD 模式下留在盘上的 cache_id 报给 scheduler。
        let directory = std::env::temp_dir().join(format!("zllm-terminal-swap-infos-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let swap = crate::kv_cache::fjall::FjallValueStore::open(&directory, "swap-infos-test").expect("打开 fjall");
        // 构造带 info 的 TerminalState:CountingState 默认 info 是空 CacheInfo,
        // 这里手写一个带 cache_id 的真实 info 测一遍。
        let mut sessions = TerminalSessions::<CountingStateWithInfo>::new(0, Some(swap));
        // 全 SSD 模式 (capacity=0) → retain 直接落盘。
        assert!(sessions.retain(
            "swap-a".into(),
            vec![1, 2, 3],
            CountingStateWithInfo { tokens: vec![1, 2, 3], marker: 1, info: TerminalInfo { cache_id: "swap-a".into(), model_key: "m".into(), cache_format: "mla".into(), last_layer: 4, prompt_tokens: 3, bytes: 128, modified_unix: 7 } }
        ));
        assert!(sessions.retain(
            "swap-b".into(),
            vec![4],
            CountingStateWithInfo { tokens: vec![4], marker: 2, info: TerminalInfo { cache_id: "swap-b".into(), model_key: "m".into(), cache_format: "mla".into(), last_layer: 4, prompt_tokens: 1, bytes: 64, modified_unix: 8 } }
        ));
        let mut infos = sessions.swap_infos();
        infos.sort_by(|left, right| left.cache_id.cmp(&right.cache_id));
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].cache_id, "swap-a");
        assert_eq!(infos[0].prompt_tokens, 3);
        assert_eq!(infos[0].bytes, 128);
        assert_eq!(infos[1].cache_id, "swap-b");
        assert_eq!(infos[1].prompt_tokens, 1);
        // 打开新 store 模拟节点重启,swap_infos 仍能列出(持久化生效)。
        drop(sessions);
        let swap_reopen = crate::kv_cache::fjall::FjallValueStore::open(&directory, "swap-infos-test").expect("重启打开 fjall");
        let sessions = TerminalSessions::<CountingStateWithInfo>::new(0, Some(swap_reopen));
        let infos = sessions.swap_infos();
        assert_eq!(infos.len(), 2, "重启后 swap_infos 仍能从盘上恢复出全部 cache_id");
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn 启动时不报告缺少snapshot的孤立metadata() {
        let directory = std::env::temp_dir().join(format!("zllm-terminal-stale-info-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let swap = crate::kv_cache::fjall::FjallValueStore::open(&directory, "stale-info-test").expect("打开 fjall");
        let info = TerminalInfo { cache_id: "stale".into(), model_key: "m".into(), cache_format: "test".into(), last_layer: 1, prompt_tokens: 1, bytes: 4, modified_unix: 1 };
        swap.put(&terminal_info_key("stale"), &encode_terminal_info(&info).unwrap()).unwrap();

        let sessions = TerminalSessions::<CountingStateWithInfo>::new(0, Some(swap));
        assert!(!sessions.contains("stale"));
        assert!(sessions.swap_infos().is_empty());
        assert!(sessions.swap.as_ref().unwrap().get(&terminal_info_key("stale")).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn infos统一列出resident与swap会话() {
        let directory = std::env::temp_dir().join(format!("zllm-terminal-infos-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let swap = crate::kv_cache::fjall::FjallValueStore::open(&directory, "infos-test").expect("打开 fjall");
        let mut sessions = TerminalSessions::<CountingStateWithInfo>::new(1, Some(swap));
        let state = |cache_id: &str, token| CountingStateWithInfo {
            tokens: vec![token],
            marker: token,
            info: TerminalInfo { cache_id: cache_id.into(), model_key: "m".into(), cache_format: "test".into(), last_layer: 1, prompt_tokens: 1, bytes: 4, modified_unix: token as u64 },
        };
        assert!(sessions.retain("swap-a".into(), vec![1], state("swap-a", 1)));
        assert!(sessions.retain("resident-b".into(), vec![2], state("resident-b", 2)));

        let infos = sessions.infos();
        assert_eq!(infos.iter().map(|info| info.cache_id.as_str()).collect::<Vec<_>>(), ["resident-b", "swap-a"]);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn resident只在换出与优雅关闭时编码() {
        let directory = std::env::temp_dir().join(format!("zllm-terminal-lifecycle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let swap = crate::kv_cache::fjall::FjallValueStore::open(&directory, "terminal-lifecycle").expect("打开 fjall");
        let encodes = Arc::new(AtomicUsize::new(0));
        let mut sessions = TerminalSessions::<TrackingState>::new(1, Some(swap));
        assert!(sessions.retain("a".into(), vec![1], TrackingState { tokens: vec![1], encodes: encodes.clone() }));
        assert_eq!(encodes.load(Ordering::Relaxed), 0, "请求完成不能同步编码 resident session");
        assert!(sessions.retain("b".into(), vec![2], TrackingState { tokens: vec![2], encodes: encodes.clone() }));
        assert_eq!(encodes.load(Ordering::Relaxed), 1, "容量换出时应编码旧 session");
        assert_eq!(sessions.persist_resident().unwrap(), 1);
        assert_eq!(encodes.load(Ordering::Relaxed), 2, "优雅关闭时应编码剩余 session");
        let _ = std::fs::remove_dir_all(&directory);
    }
}
