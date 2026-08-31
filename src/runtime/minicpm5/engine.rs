//! MiniCPM5 × Metal 嵌入式推理引擎：统一请求准备、session 与生成生命周期。
//!
//! MiniCPM5-1B-Instruct Q4_K_M（~657 MB）在 M5 26GB 上零压力：
//! 权重 0.7 GB + KV (Q8G64) ~negligible + workspace ~0.1 GB ≈ 1 GB。
//! `max_sequence_length` 默认 131k（MiniCPM5 原始 context），但 standalone yaml 可按需收紧。

#[cfg(target_os = "macos")]
use super::protocol::{chat_prompt as minicpm5_chat_prompt, chat_prompt_suffix as minicpm5_chat_prompt_suffix};
#[cfg(target_os = "macos")]
use crate::runtime::session::{FixedSessionResidency, GenerationSummary, KvResidency};
#[cfg(target_os = "macos")]
use crate::{
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::session::{AtomicCounterU64, DynError, NodeCapabilities, RuntimeStatus as NodeRuntime},
};
#[cfg(target_os = "macos")]
use serde_json::Value;
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(target_os = "macos")]
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

#[cfg(target_os = "macos")]
use crate::{
    attention::{gqa::CausalWindow, rope::RopeTable},
    backend::{
        Backend, BackendResources, LinearWeight,
        metal::{MetalContext, MetalKvCache, MetalTensor, MetalWeight},
    },
    kv_cache::{KvCacheLayerMap, KvCacheSpec},
    runtime::minicpm5::{self, MiniCpm5Config, MiniCpm5OutputHead, MiniCpm5Weights},
    tokenizer::{Detokenizer, Tokenizer},
};

#[cfg(target_os = "macos")]
#[path = "metal_session.rs"]
mod metal_session;
#[cfg(target_os = "macos")]
use metal_session::{MiniCpm5MetalSequence, MiniCpm5MetalSession};

/// MiniCPM5 Metal 引擎主体。
#[cfg(target_os = "macos")]
pub struct MiniCpm5Engine {
    session: MiniCpm5MetalSession,
    residency: FixedSessionResidency,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
    /// 连续对话终点缓存(通用 kv_cache::terminal_cache 机制,内存 LRU)。
    terminal_states: crate::kv_cache::terminal_cache::TerminalSessions<MiniCpm5TerminalState>,
}

/// 会话终点状态:KV cache + 边界 hidden + 全量 token + length 截断时未前向的 pending token。
/// 落盘编码未实现(persist_kv_cache 对 MiniCPM5 不生效,swap 不创建,trait 方法不会被调用)。
#[cfg(target_os = "macos")]
struct MiniCpm5TerminalState {
    sequence: MiniCpm5MetalSequence,
    pending: Vec<u32>,
    info: CacheInfo,
    /// 内容前缀匹配的租户隔离:不同 namespace 的会话即使 token 相同也互不复用。
    cache_namespace: Option<String>,
}

#[cfg(target_os = "macos")]
impl crate::kv_cache::terminal_cache::TerminalSnapshot for MiniCpm5TerminalState {
    type Resources = ();

    fn encode(&self) -> Result<Vec<u8>, String> {
        Err("MiniCPM5 终点快照落盘未实现".to_owned())
    }

    fn decode(_bytes: &[u8], _resources: &Self::Resources) -> Result<Self, String> {
        Err("MiniCPM5 终点快照落盘未实现".to_owned())
    }

    fn terminal_tokens(&self) -> &[u32] {
        &self.sequence.tokens
    }

    fn info(&self) -> &crate::kv_cache::terminal_cache::TerminalInfo {
        &self.info
    }

    fn set_info(&mut self, info: crate::kv_cache::terminal_cache::TerminalInfo) {
        self.info = info;
    }

    fn cache_namespace(&self) -> Option<&str> {
        self.cache_namespace.as_deref()
    }
}

#[cfg(target_os = "macos")]
impl MiniCpm5Engine {
    pub fn load(
        model_path: &Path,
        max_seq_len: usize,
        kv_f16: bool,
        replay_enabled: bool,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        runtime: Arc<Mutex<NodeRuntime>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        let session = MiniCpm5MetalSession::load_with_replay(model_path, max_seq_len, kv_f16, replay_enabled, lm_head_quantization).map_err(|error| -> DynError { error.into() })?;
        let session_resident_bytes = session.session_capacity_bytes();
        let available = crate::backend::metal::available_residency_bytes(session.context()) as usize;
        let engine_resident_bytes = session.context().device.current_allocated_size() as usize;
        let residency = FixedSessionResidency::new(available, engine_resident_bytes, session_resident_bytes)?;
        let mut capabilities = crate::runtime::metal_node::text_capabilities(
            session.context(),
            crate::runtime::node::SessionDescriptor { model_format: "gguf-mixed", model_bytes: session.model_bytes(), max_seq_len, kv_cache_format: if kv_f16 { "f16" } else { "q8g64" }, input_modalities: &["text"] },
        );
        residency.configure(&mut capabilities, &runtime);
        eprintln!("[minicpm5-kv-admission] available={:.1} MiB session={:.1} MiB", available as f64 / 1048576.0, session_resident_bytes as f64 / 1048576.0);
        Ok(Self { session, residency, capabilities, runtime, compute_steps, terminal_states: crate::kv_cache::terminal_cache::TerminalSessions::new(1, None) })
    }

    pub(crate) fn kv_residency(&self) -> KvResidency {
        self.residency.report(&self.terminal_states)
    }

    pub(crate) fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        crate::runtime::metal_node::startup_info(self.session.context(), &self.capabilities)
    }

    pub(crate) fn refresh_runtime(&self) {
        self.residency.refresh(&self.runtime, &self.terminal_states);
    }

    pub(crate) fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        self.terminal_states.infos()
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        // 终点缓存是内存 LRU,无落盘;接口形状与 gemma4/qwen36 保持一致。
        Ok(())
    }

    /// 嵌入式入口:与 Node 路径共享 generate_one(模板、采样、terminal cache、
    /// 异步流水线全部一致),只是不走调度器。
    pub fn generate(&mut self, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(Option<u32>, String) -> bool) -> Result<GenerationSummary, String> {
        self.generate_one("", request, cancellation, on_token)
    }
}

#[cfg(target_os = "macos")]
impl MiniCpm5Engine {
    fn generate_one(&mut self, _request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(Option<u32>, String) -> bool) -> Result<GenerationSummary, String> {
        use crate::runtime::session::{BatchTokenGuard, GenerationOutput, parse_stops, requested_completion_tokens};
        // 思考开关：默认开（官方默认）；thinking.type=disabled / enable_thinking=false 关闭。
        let thinking_disabled = request.get("thinking").and_then(|t| t.get("type")).and_then(Value::as_str) == Some("disabled") || request.get("enable_thinking").and_then(Value::as_bool) == Some(false);
        let prompt = minicpm5_chat_prompt(request, !thinking_disabled)?;
        // 请求带 temperature>0/top_p 时按其采样（MiniCPM5-1B 纯贪心会复读循环）；缺省保持贪心。
        // temperature=0 视为贪心(不改变语义),让异步流水线继续可用。
        let sampling_state = match (request.get("temperature").and_then(Value::as_f64), request.get("top_p").and_then(Value::as_f64)) {
            (Some(temperature), top_p) if temperature > 0.0 => {
                let config = crate::runtime::output::SamplingConfig { temperature: temperature as f32, top_p: top_p.unwrap_or(1.0) as f32, seed: 0 };
                crate::runtime::output::SamplingState::new(config).ok()
            }
            _ => None,
        };
        let mut sampling_state = sampling_state;
        let tokens = self.session.tokenize(&prompt);
        let max_completion = requested_completion_tokens(request);
        if max_completion == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        if tokens.is_empty() {
            return Err("MiniCPM5 prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.session.max_seq_len() {
            return Err(format!("MiniCPM5 prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.session.max_seq_len()));
        }
        let stops = parse_stops(request.get("stop"))?;
        // cache_id 边界命中优先；否则尝试同 namespace 的内容前缀。两条路径
        // 都在取得 resident 状态时转成 active reservation，miss 则先换出再分配。
        let namespace = request.get("_zllm_cache_namespace").and_then(Value::as_str);
        let terminal_resume = crate::runtime::session::request_terminal_resume(request)?;
        let exact = match &terminal_resume {
            crate::runtime::session::TerminalResume::Match { cache_id, assistant } if self.terminal_states.contains(cache_id) => Some((cache_id.as_str(), *assistant)),
            crate::runtime::session::TerminalResume::Mismatch { requested, expected } => {
                eprintln!("[zllm-runtime] terminal cache hash 不匹配 id={requested} expected={expected}");
                None
            }
            _ => None,
        };
        let (resumed, prefix, _kv_reservation) = if let Some((cache_id, assistant)) = exact {
            let suffix = self.session.tokenize(&minicpm5_chat_prompt_suffix(request, assistant, !thinking_disabled)?);
            let (state, reservation) =
                self.terminal_states.activate(Some(cache_id), self.residency.budget(), self.residency.session_resident_bytes(), |_, state| state.info.bytes as usize, &()).map_err(|error| format!("MiniCPM5 {error}; 请重新开始会话"))?;
            let resumed = state.map(|(cached, state)| {
                eprintln!("[zllm-runtime] terminal cache 拼接命中 cached={} new={}", cached.len(), suffix.len());
                (state, suffix)
            });
            (resumed, None, reservation)
        } else {
            let (prefix, reservation) = self
                .terminal_states
                .activate_longest_common_prefix(&tokens, namespace, self.residency.budget(), self.residency.session_resident_bytes(), |_, state| state.info.bytes as usize)
                .map_err(|error| format!("MiniCPM5 {error}; 请重新开始会话"))?;
            (None, prefix, reservation)
        };
        // resume 命中时只对 suffix 计费,避免 dispatch 把整段历史算到当前 batch。
        // 三条路径:cache_id 边界 resume(多轮)、内容前缀复用(重复/重叠 prompt)、全量 prefill。
        let mut billed_tokens = tokens.len();
        let mut sequence = if let Some((state, suffix)) = resumed {
            let mut sequence = state.sequence;
            let mut append = state.pending;
            append.extend(suffix);
            billed_tokens = append.len();
            self.session.extend(&mut sequence, &append)?;
            sequence
        } else {
            // 内容前缀复用(llama slot 截断复用的等价物);匹配与恢复策略都在
            // kv_cache::terminal_cache,这里只提供 truncate/extend 两个模型操作。
            match prefix {
                Some((_cached, lcp, state)) => {
                    let mut sequence = state.sequence;
                    let cached_len = sequence.tokens.len();
                    let resume_at = crate::kv_cache::terminal_cache::resume_by_prefix(
                        &mut sequence,
                        &tokens,
                        lcp,
                        cached_len,
                        |sequence, len| {
                            sequence.tokens.truncate(len);
                            sequence.cache.truncate(len);
                        },
                        |sequence, suffix| self.session.extend(sequence, suffix),
                    )?;
                    eprintln!("[zllm-node] minicpm5 前缀复用 lcp={lcp} resume_at={resume_at} suffix={}", tokens.len() - resume_at);
                    billed_tokens = tokens.len() - resume_at;
                    sequence
                }
                None => self.session.prefill(tokens.clone())?,
            }
        };
        let _batch_guard = BatchTokenGuard::new(&self.runtime, billed_tokens);
        let max_tokens = max_completion.min(self.session.max_seq_len() - sequence.token_count());
        if max_tokens == 0 {
            return Err("会话已没有可用的生成位置".to_owned());
        }
        let mut output = GenerationOutput::new(&stops);
        macro_rules! emit_token {
            ($token:expr) => {{
                let token = $token;
                (|| -> Result<bool, String> {
                    if self.session.is_eos(token) {
                        output.stop();
                        return Ok(true);
                    }
                    let bytes = self.session.decode_bytes(token)?;
                    Ok(!output.push(&bytes, |chunk| on_token(Some(token), chunk)))
                })()
            }};
        }
        // 异步流水线:argmax → embedding gather → decode round 在 GPU 上闭环,
        // CPU 读回 token N 与 GPU 跑第 N+1 轮重叠,消掉逐 token 同步泡(实测 ~1.1ms/tok)。
        // 采样请求(temperature/top_p)走同步循环:流水线的输出步是 argmax 贪心语义,
        // 采样进流水线是后续工作。
        let async_decode = self.session.async_decode_available() && sampling_state.is_none();
        if async_decode {
            self.session.begin_async_decode();
        }
        // 最后一次采样的 token:length 截断时它尚未前向,retain 时作为 pending 留给续写。
        let mut last_token: Option<u32> = None;
        let loop_result = if async_decode {
            let mut step_result: Result<(), String> = Ok(());
            'outer: {
                let mut pending = match self.session.submit_output(&sequence.hidden, sequence.tokens.len()) {
                    Ok(pending) => pending,
                    Err(error) => {
                        step_result = Err(error);
                        break 'outer;
                    }
                };
                for step in 0..max_tokens {
                    if cancellation.load(Ordering::Acquire) {
                        output.cancel();
                        break;
                    }
                    // 推测提交下一轮:EOS 时这轮白算,代价是一轮 GPU 时间;末轮不再推测。
                    let checkpoint_len = sequence.token_count();
                    let checkpoint_hidden = (step + 1 < max_tokens).then(|| sequence.hidden.clone());
                    let next = if step + 1 < max_tokens {
                        match self.session.submit_step(&mut sequence, &mut pending) {
                            Ok(next) => Some(next),
                            Err(error) => {
                                step_result = Err(error);
                                break 'outer;
                            }
                        }
                    } else {
                        None
                    };
                    let token = self.session.wait_token(&mut sequence, &pending);
                    last_token = Some(token);
                    match emit_token!(token) {
                        Ok(true) => {
                            // submit_step 已投机把当前 token 前向。EOS 在同步路径不会进入
                            // session；这里必须同时回滚 token、KV 游标和边界 hidden，避免
                            // terminal resume 再追加模板终止符时重复一个 EOS。
                            if self.session.is_eos(token)
                                && let Some(hidden) = checkpoint_hidden
                            {
                                sequence.tokens.truncate(checkpoint_len);
                                sequence.cache.truncate(checkpoint_len);
                                sequence.hidden = hidden;
                            }
                            break;
                        }
                        Ok(false) => {}
                        Err(error) => {
                            step_result = Err(error);
                            break 'outer;
                        }
                    }
                    self.compute_steps.fetch_add(1);
                    let Some(next) = next else { break };
                    pending = next;
                }
            }
            step_result
        } else {
            let mut step_result: Result<(), String> = Ok(());
            'outer: {
                for step in 0..max_tokens {
                    if cancellation.load(Ordering::Acquire) {
                        output.cancel();
                        break;
                    }
                    let sample = sampling_state.as_mut().map(|state| state.next());
                    let token = match self.session.next_token(&sequence, sample.as_ref()) {
                        Ok(token) => token,
                        Err(error) => {
                            step_result = Err(error);
                            break 'outer;
                        }
                    };
                    last_token = Some(token);
                    match emit_token!(token) {
                        Ok(true) => break,
                        Ok(false) => {}
                        Err(error) => {
                            step_result = Err(error);
                            break 'outer;
                        }
                    }
                    if step + 1 == max_tokens {
                        break;
                    }
                    if let Err(error) = self.session.decode_token(&mut sequence, token) {
                        step_result = Err(error);
                        break 'outer;
                    }
                    self.compute_steps.fetch_add(1);
                }
            }
            step_result
        };
        if async_decode {
            self.session.end_async_decode();
        }
        loop_result?;
        output.finish(|chunk| on_token(None, chunk));
        let output_text = output.text().to_owned();
        let mut summary = output.summary(tokens.len());
        // 终点缓存 retain(与 gemma4/ornith 同一协议):cache_id 由客户端可重放的消息历史哈希,
        // 下一轮同会话请求经 request_terminal_resume 命中,跳过全部历史 prefill。
        // length 截断时最后采样的 token 尚未前向,作为 pending 留给续写。
        // stop 已由下一轮模板后缀表达,重复补入 eos 会破坏多轮缓存对齐。
        if summary.finish_reason != "cancelled" && stops.is_empty() {
            let pending = if summary.finish_reason == "length" { last_token.into_iter().collect() } else { Vec::new() };
            let cache_id = crate::runtime::session::terminal_cache_id(request, &output_text, &[])?;
            let info = CacheInfo {
                cache_id,
                model_key: "minicpm5".to_owned(),
                cache_format: "minicpm5-terminal-v1".to_owned(),
                last_layer: self.session.layer_count().saturating_sub(1),
                prompt_tokens: sequence.token_count(),
                bytes: self.residency.session_resident_bytes() as u64,
                modified_unix: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
            };
            let cache_namespace = request.get("_zllm_cache_namespace").and_then(Value::as_str).map(str::to_owned);
            let state = MiniCpm5TerminalState { sequence, pending, info: info.clone(), cache_namespace };
            summary.cache = crate::runtime::session::retain_terminal_session(&mut self.terminal_states, state, info);
        }
        Ok(summary)
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::metal_session::MiniCpm5MetalSession;
    use super::{CacheInfo, MiniCpm5Engine, MiniCpm5TerminalState, NodeRuntime};
    use std::path::Path;

    const WEIGHTS: &str = "/path/to/MiniCPM5-1B-Q4_K_M.gguf";

    /// 端到端回归:真实权重 prefill + 8 轮 greedy decode,逐 token 对拍
    /// llama.cpp 2.29.1 参考序列(2026-08-27 重录:思考模板打开 <think> 块)。
    /// 锁定三个历史 bug:BOS 必须前置、EOS 必须含 <|im_end|>、RoPE 必须
    /// Interleaved(SplitHalf 在第 3 个 token 处分叉并退化为复读)。
    #[test]
    fn minicpm5_decode_matches_llama_reference() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        let session = MiniCpm5MetalSession::load(Path::new(WEIGHTS), 8192, false, crate::weight::LmHeadQuantization::Native).unwrap();
        let prompt = crate::runtime::minicpm5::minicpm5_instruct_prompt("用一句话解释什么是注意力机制", None, true);
        let tokens = session.tokenize(&prompt);
        assert_eq!(tokens, [0, 130072, 8448, 220, 1066, 49667, 12778, 42013, 43211, 15963, 130073, 220, 130072, 130071, 220, 8, 220], "prompt 编码与 llama.cpp /tokenize 不一致");
        let mut sequence = session.prefill(tokens).unwrap();
        let mut generated = Vec::new();
        for _ in 0..8 {
            let token = session.next_token(&sequence, None).unwrap();
            generated.push(token);
            session.decode_token(&mut sequence, token).unwrap();
        }
        // llama.cpp greedy 参考:["嗯", "，", "用户", "问", "的", "是", "“", "用"]
        assert_eq!(generated, [16431, 82579, 2210, 5443, 574, 1066, 49667, 12778], "decode 序列与 llama.cpp 参考不一致");

        // 异步流水线(argmax → 设备端 embedding gather → 推测下一轮)必须与同步路径逐 token 一致
        assert!(session.async_decode_available(), "设备端 embedding 来源缺失,异步流水线不可用");
        let mut sequence = session.prefill(session.tokenize(&prompt)).unwrap();
        session.begin_async_decode();
        let mut pending = session.submit_output(&sequence.hidden, sequence.tokens.len()).unwrap();
        let mut generated_async = Vec::new();
        for step in 0..8 {
            let next = if step + 1 < 8 { Some(session.submit_step(&mut sequence, &mut pending).unwrap()) } else { None };
            generated_async.push(session.wait_token(&mut sequence, &pending));
            match next {
                Some(next) => pending = next,
                None => break,
            }
        }
        session.end_async_decode();
        assert_eq!(generated_async, generated, "异步流水线 decode 序列与同步路径不一致");

        // 长 prompt(316 tokens)decode:覆盖 kv 256..512 的 append 直通区间
        // (kv>256 曾回落 split+merge 慢路径,长生成实测掉速 ~10%)。
        // llama.cpp greedy 参考:["好的","，","用户","让我","详细","介绍","注意力",...]
        // 注意:q8g64 KV 在长上下文累计舍入与 llama(f16 KV)不可比,第 8 个 token 起
        // 可能出现刀锋 argmax 分叉(实测 split 路径 15158 / append 路径 2595,两者
        // 数值差 4.7e-4,见 direct_q8_append_matches_split_kv);故只对拍前 7 个。
        let long_user = "请介绍一下注意力机制。".to_owned() + &"详细说明。".repeat(100);
        let long_prompt = crate::runtime::minicpm5::minicpm5_instruct_prompt(&long_user, None, true);
        let mut sequence = session.prefill(session.tokenize(&long_prompt)).unwrap();
        let mut long_generated = Vec::new();
        for _ in 0..8 {
            let token = session.next_token(&sequence, None).unwrap();
            long_generated.push(token);
            session.decode_token(&mut sequence, token).unwrap();
        }
        assert_eq!(long_generated[..7], [6805, 82579, 19979, 26458, 13498, 43211, 15963], "长 prompt decode 前 7 token 与 llama.cpp 参考不一致");
    }

    /// 同步路径对照:两轮 fresh prefill + 48 步同步 decode 必须逐 token 一致。
    #[test]
    fn minicpm5_sync_decode_repeat_deterministic() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        let session = MiniCpm5MetalSession::load(Path::new(WEIGHTS), 8192, false, crate::weight::LmHeadQuantization::Native).unwrap();
        let prompt = crate::runtime::minicpm5::minicpm5_instruct_prompt("用一句话解释什么是注意力机制", None, true);
        let tokens = session.tokenize(&prompt);
        let mut rounds = Vec::new();
        for _ in 0..2 {
            let mut sequence = session.prefill(tokens.clone()).unwrap();
            let mut generated = Vec::new();
            for _ in 0..48 {
                let token = session.next_token(&sequence, None).unwrap();
                generated.push(token);
                session.decode_token(&mut sequence, token).unwrap();
            }
            rounds.push(generated);
        }
        assert_eq!(rounds[0], rounds[1], "同步路径两轮 decode 不一致");
    }

    /// terminal-cache resume 的核心等价性:全量 prefill 与"前缀 prefill + 尾段
    /// extend"必须产生逐 token 相同的 decode 序列。extend 的 KV 写入/读取任何
    /// 错位都会在后续 decode 的 attention 上暴露。suffix 长度覆盖 kv≤256 直通
    /// 与 >256 回落两条 attention 路径;split=1 的极端分段近似"全部走 extend"。
    #[test]
    fn minicpm5_extend_matches_full_prefill() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        let session = MiniCpm5MetalSession::load(Path::new(WEIGHTS), 8192, false, crate::weight::LmHeadQuantization::Native).unwrap();
        let user = "我住的 city 是北京,请记住。现在回答:我住哪个 city?只回答城市名".to_owned();
        let prompt = crate::runtime::minicpm5::minicpm5_instruct_prompt(&user, None, false);
        let tokens = session.tokenize(&prompt);
        let mut sequence = session.prefill(tokens.clone()).unwrap();
        let mut reference = Vec::new();
        for _ in 0..8 {
            let token = session.next_token(&sequence, None).unwrap();
            reference.push(token);
            session.decode_token(&mut sequence, token).unwrap();
        }
        for suffix_len in [1usize, 8, 64, 300] {
            let split = tokens.len().saturating_sub(suffix_len).max(1);
            let mut resumed = session.prefill(tokens[..split].to_vec()).unwrap();
            session.extend(&mut resumed, &tokens[split..]).unwrap();
            assert_eq!(resumed.tokens, tokens, "suffix_len={suffix_len} extend 后 token 序列与全量不一致");
            let mut generated = Vec::new();
            for _ in 0..8 {
                let token = session.next_token(&resumed, None).unwrap();
                generated.push(token);
                session.decode_token(&mut resumed, token).unwrap();
            }
            assert_eq!(generated, reference, "suffix_len={suffix_len} 分段 extend 与全量 prefill 的 decode 序列分叉");
        }
    }

    /// 引擎级多轮等价性:与 zllm-metal 相同形状的请求,显式 cache_id 链(token 对齐
    /// resume)与无 cache_id 链(LCP/全量)三轮输出必须逐字节一致;另设思考开的
    /// 参考臂打印模型多轮召回(MiniCPM5-1B 在空围栏 no-think 下召回弱是模型特性)。
    #[test]
    fn minicpm5_engine_multiturn_alignment_semantics() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        let runtime = std::sync::Arc::new(std::sync::Mutex::new(NodeRuntime::default()));
        let compute_steps = std::sync::Arc::new(crate::runtime::session::AtomicCounterU64::new(0));
        let mut engine = MiniCpm5Engine::load(Path::new(WEIGHTS), 8192, false, true, crate::weight::LmHeadQuantization::Native, runtime, compute_steps).expect("engine");
        let cancellation = std::sync::atomic::AtomicBool::new(false);
        let mut history: Vec<(String, String)> = Vec::new();
        let mut cache_id: Option<String> = None;
        let mut answers: Vec<String> = Vec::new();
        for turn in ["我住的 city 是北京,请记住", "好的", "我住哪个 city?只回答城市名"] {
            history.push(("user".to_owned(), turn.to_owned()));
            let messages: Vec<serde_json::Value> = history.iter().map(|(role, content)| serde_json::json!({ "role": role, "content": content })).collect();
            let request = serde_json::json!({ "model": "minicpm5", "messages": messages, "max_completion_tokens": 4096, "cache_id": cache_id, "temperature": 0.7, "top_p": 0.8, "enable_thinking": false });
            let mut content = String::new();
            let summary = engine
                .generate(&request, &cancellation, &mut |_token, text| {
                    content.push_str(&text);
                    true
                })
                .expect("generate");
            assert_eq!(summary.finish_reason, "stop", "turn={turn} content={content}");
            cache_id = summary.cache.as_ref().map(|cache| cache.cache_id.clone());
            answers.push(content.clone());
            history.push(("assistant".to_owned(), content));
        }
        eprintln!("[multiturn] cache_id 链 第三轮回答: {}", answers[2]);
        // 对照:同一形状但不带 cache_id(引擎走 LCP 前缀复用/全量),新引擎隔离终态。
        let runtime = std::sync::Arc::new(std::sync::Mutex::new(NodeRuntime::default()));
        let compute_steps = std::sync::Arc::new(crate::runtime::session::AtomicCounterU64::new(0));
        let mut engine = MiniCpm5Engine::load(Path::new(WEIGHTS), 8192, false, true, crate::weight::LmHeadQuantization::Native, runtime, compute_steps).expect("engine");
        let mut history: Vec<(String, String)> = Vec::new();
        let mut no_cache_answers: Vec<String> = Vec::new();
        for turn in ["我住的 city 是北京,请记住", "好的", "我住哪个 city?只回答城市名"] {
            history.push(("user".to_owned(), turn.to_owned()));
            let messages: Vec<serde_json::Value> = history.iter().map(|(role, content)| serde_json::json!({ "role": role, "content": content })).collect();
            let request = serde_json::json!({ "model": "minicpm5", "messages": messages, "max_completion_tokens": 4096, "temperature": 0.7, "top_p": 0.8, "enable_thinking": false });
            let mut content = String::new();
            let summary = engine
                .generate(&request, &cancellation, &mut |_token, text| {
                    content.push_str(&text);
                    true
                })
                .expect("generate");
            assert_eq!(summary.finish_reason, "stop");
            no_cache_answers.push(content.clone());
            history.push(("assistant".to_owned(), content));
        }
        eprintln!("[multiturn] 无 cache_id 对照 第三轮回答: {}", no_cache_answers[2]);
        // 第二对照:开思考(官方默认形态)验证模型多轮记忆本身是否正常。
        let runtime = std::sync::Arc::new(std::sync::Mutex::new(NodeRuntime::default()));
        let compute_steps = std::sync::Arc::new(crate::runtime::session::AtomicCounterU64::new(0));
        let mut engine = MiniCpm5Engine::load(Path::new(WEIGHTS), 8192, false, true, crate::weight::LmHeadQuantization::Native, runtime, compute_steps).expect("engine");
        let mut history: Vec<(String, String)> = Vec::new();
        let mut thinking_answers: Vec<String> = Vec::new();
        for turn in ["我住的 city 是北京,请记住", "好的", "我住哪个 city?只回答城市名"] {
            history.push(("user".to_owned(), turn.to_owned()));
            let messages: Vec<serde_json::Value> = history.iter().map(|(role, content)| serde_json::json!({ "role": role, "content": content })).collect();
            let request = serde_json::json!({ "model": "minicpm5", "messages": messages, "max_completion_tokens": 4096, "temperature": 0.7, "top_p": 0.8 });
            let mut content = String::new();
            let summary = engine
                .generate(&request, &cancellation, &mut |_token, text| {
                    content.push_str(&text);
                    true
                })
                .expect("generate");
            assert_eq!(summary.finish_reason, "stop");
            thinking_answers.push(content.clone());
            history.push(("assistant".to_owned(), content));
        }
        eprintln!("[multiturn] 思考开 第三轮回答: {}", thinking_answers[2]);
        // 核心回归:显式 cache_id 链(token 对齐 resume)与无 cache_id 链(LCP/全量)
        // 的三轮输出必须逐字节一致——对齐机制不得改变生成语义。采样 seed 固定为 0,
        // 两条链 hidden 一致时输出确定。
        assert_eq!(answers, no_cache_answers, "cache_id 链与无 cache_id 链输出分叉(resume 改变了语义)");
    }

    /// 生产形态的多轮链式对拍:每轮 decode 到自然 stop(最后一个 eos/im_end
    /// 不前向),resume 后以"全量 token 数组截取的 suffix"续写——与
    /// resume_terminal_tokens 的对齐路径完全同构。assistant 段直接回填生成的
    /// token(不经文本重分词,隔离 BPE 漂移),两侧 token 序列天然一致,
    /// 任何分叉都指向 KV 状态本身。
    #[test]
    fn minicpm5_chained_stop_form_matches_full_prefill() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        let session = MiniCpm5MetalSession::load(Path::new(WEIGHTS), 8192, false, crate::weight::LmHeadQuantization::Native).unwrap();
        let turns = ["我住的 city 是北京,请记住", "好的", "我住哪个 city?只回答城市名"];
        let fence = "<think>\n\n</think>\n\n";
        let user_block = |turn: &str| format!("<|im_start|>user\n{turn}<|im_end|>\n<|im_start|>assistant\n{fence}");
        let mut sessions = crate::kv_cache::terminal_cache::TerminalSessions::<MiniCpm5TerminalState>::new(1, None);
        let mut chained_state: Option<MiniCpm5TerminalState> = None;
        // 全量侧的 token 数组:与链式侧逐 token 相同(assistant 段直接用生成 token)。
        let mut prompt_tokens: Vec<u32> = session.tokenize("<s>");
        for (round, turn) in turns.iter().enumerate() {
            prompt_tokens.extend(session.tokenize(&user_block(turn)));
            let mut sequence = match chained_state.take() {
                Some(state) => {
                    let mut sequence = state.sequence;
                    let cached = sequence.tokens.clone();
                    assert!(prompt_tokens.starts_with(&cached), "第 {round} 轮前缀必须对齐");
                    session.extend(&mut sequence, &prompt_tokens[cached.len()..]).unwrap();
                    sequence
                }
                None => session.prefill(prompt_tokens.clone()).unwrap(),
            };
            // decode 直到产出 eos/im_end(不前向),与生成循环的 stop 形态一致。
            let mut assistant_tokens = Vec::new();
            let mut sampling = crate::runtime::output::SamplingState::new(crate::runtime::output::SamplingConfig { temperature: 0.7, top_p: 0.8, seed: 0 }).ok();
            for _ in 0..64 {
                let sample = sampling.as_mut().map(|state| state.next());
                let token = session.next_token(&sequence, sample.as_ref()).unwrap();
                if session.is_eos(token) {
                    break;
                }
                assistant_tokens.push(token);
                session.decode_token(&mut sequence, token).unwrap();
            }
            prompt_tokens.extend(&assistant_tokens);
            prompt_tokens.extend(session.tokenize("<|im_end|>\n"));
            let info = CacheInfo { cache_id: format!("round{round}"), model_key: "minicpm5".to_owned(), cache_format: "t".to_owned(), last_layer: 0, prompt_tokens: sequence.token_count(), bytes: 0, modified_unix: 0 };
            assert!(sessions.retain(format!("round{round}"), sequence.tokens.clone(), MiniCpm5TerminalState { sequence, pending: Vec::new(), info, cache_namespace: None }), "retain 失败");
            let (_, state) = sessions.resume(&format!("round{round}"), &()).expect("resume").expect("ok");
            chained_state = Some(state);
        }
        // 终态对照:链式 vs 同 token 数组全量重算,后续 decode 必须逐 token 一致。
        let mut chained = chained_state.expect("chained").sequence;
        // 末轮生成的 stop 段(im_end+换行)按 pending 语义不在链式侧,截断对齐。
        prompt_tokens.truncate(chained.tokens.len());
        assert_eq!(chained.tokens, prompt_tokens, "链式与全量 token 数组不一致");
        let mut via_full = session.prefill(prompt_tokens).unwrap();
        let mut chained_out = Vec::new();
        let mut full_out = Vec::new();
        for _ in 0..8 {
            let a = session.next_token(&chained, None).unwrap();
            chained_out.push(a);
            session.decode_token(&mut chained, a).unwrap();
            let b = session.next_token(&via_full, None).unwrap();
            full_out.push(b);
            session.decode_token(&mut via_full, b).unwrap();
        }
        assert_eq!(chained_out, full_out, "多轮 stop 形态链式 resume 与全量重算分叉(KV 损坏)");
    }

    /// terminal cache 往返等价性:retain 保存的 sequence 经 resume 取回后 extend
    /// 续写,与"同一 token 数组重新全量 prefill"必须产生逐 token 相同的 decode。
    /// 两侧 token 数组完全一致,唯一差异是 KV 来自 retain 往返还是重新计算,
    /// 以此隔离 terminal cache 的状态管理问题。
    #[test]
    fn minicpm5_terminal_resume_matches_full_prefill() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        let session = MiniCpm5MetalSession::load(Path::new(WEIGHTS), 8192, false, crate::weight::LmHeadQuantization::Native).unwrap();
        let user = "我住的 city 是北京,请记住".to_owned();
        let tokens = session.tokenize(&crate::runtime::minicpm5::minicpm5_instruct_prompt(&user, None, false));
        let mut sequence = session.prefill(tokens.clone()).unwrap();
        let mut assistant_tokens = Vec::new();
        for _ in 0..6 {
            let token = session.next_token(&sequence, None).unwrap();
            assistant_tokens.push(token);
            session.decode_token(&mut sequence, token).unwrap();
        }
        let suffix: Vec<u32> = session.tokenize("\n<|im_start|>user\n我住哪个 city?只回答城市名<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");

        // 侧 A:retain → resume → extend(生产 resume 路径)
        let mut sessions = crate::kv_cache::terminal_cache::TerminalSessions::<MiniCpm5TerminalState>::new(2, None);
        let info = CacheInfo {
            cache_id: "test".to_owned(),
            model_key: "minicpm5".to_owned(),
            cache_format: "minicpm5-terminal-v1".to_owned(),
            last_layer: session.layer_count() - 1,
            prompt_tokens: sequence.token_count(),
            bytes: session.session_capacity_bytes() as u64,
            modified_unix: 0,
        };
        let state = MiniCpm5TerminalState { sequence, pending: Vec::new(), info, cache_namespace: None };
        assert!(sessions.retain("test".to_owned(), state.sequence.tokens.clone(), state), "retain 失败");
        let (_, resumed) = sessions.resume("test", &()).expect("resume miss").expect("resume err");
        let mut via_cache = resumed.sequence;
        session.extend(&mut via_cache, &suffix).unwrap();

        // 侧 B:同一 token 数组全量重算
        let mut full_tokens = tokens.clone();
        full_tokens.extend(&assistant_tokens);
        full_tokens.extend(&suffix);
        let mut via_full = session.prefill(full_tokens).unwrap();

        assert_eq!(via_cache.tokens, via_full.tokens, "resume 侧与全量侧 token 数组不一致");
        let mut cache_generated = Vec::new();
        let mut full_generated = Vec::new();
        for _ in 0..8 {
            let a = session.next_token(&via_cache, None).unwrap();
            cache_generated.push(a);
            session.decode_token(&mut via_cache, a).unwrap();
            let b = session.next_token(&via_full, None).unwrap();
            full_generated.push(b);
            session.decode_token(&mut via_full, b).unwrap();
        }
        assert_eq!(cache_generated, full_generated, "terminal cache 往返后的 KV 与全量重算分叉");

        // 第二轮链式:把 extend+decode 出来的 sequence 再次 retain → resume → extend,
        // 复现多轮会话的完整生产路径;任一轮 KV 与全量重算分叉都会在此暴露。
        let mut sessions = crate::kv_cache::terminal_cache::TerminalSessions::<MiniCpm5TerminalState>::new(2, None);
        let mut cache_sequence = session.prefill(tokens.clone()).unwrap();
        let mut replayed: Vec<u32> = tokens.clone();
        for round in 0..2 {
            let mut assistant = Vec::new();
            for _ in 0..6 {
                let token = session.next_token(&cache_sequence, None).unwrap();
                assistant.push(token);
                session.decode_token(&mut cache_sequence, token).unwrap();
            }
            replayed.extend(&assistant);
            let info = CacheInfo {
                cache_id: format!("round{round}"),
                model_key: "minicpm5".to_owned(),
                cache_format: "minicpm5-terminal-v1".to_owned(),
                last_layer: session.layer_count() - 1,
                prompt_tokens: cache_sequence.token_count(),
                bytes: session.session_capacity_bytes() as u64,
                modified_unix: 0,
            };
            let state = MiniCpm5TerminalState { sequence: std::mem::replace(&mut cache_sequence, session.prefill(tokens.clone()).unwrap()), pending: Vec::new(), info, cache_namespace: None };
            assert!(sessions.retain(format!("round{round}"), state.sequence.tokens.clone(), state), "retain 失败");
            let (_, resumed) = sessions.resume(&format!("round{round}"), &()).expect("resume miss").expect("resume err");
            cache_sequence = resumed.sequence;
            replayed.extend(&suffix);
            session.extend(&mut cache_sequence, &suffix).unwrap();
        }
        let mut full_sequence = session.prefill(replayed).unwrap();
        let mut chained_cache = Vec::new();
        let mut chained_full = Vec::new();
        for _ in 0..8 {
            let a = session.next_token(&cache_sequence, None).unwrap();
            chained_cache.push(a);
            session.decode_token(&mut cache_sequence, a).unwrap();
            let b = session.next_token(&full_sequence, None).unwrap();
            chained_full.push(b);
            session.decode_token(&mut full_sequence, b).unwrap();
        }
        assert_eq!(cache_sequence.tokens, full_sequence.tokens, "链式 resume 与全量 token 数组不一致");
        assert_eq!(chained_cache, chained_full, "两轮链式 terminal resume 与全量重算的 decode 序列分叉");
    }

    /// 串行-CPU-embedding 对照实验:每轮等 token 结算后再提交下一轮(CPU 上传 embedding,
    /// 无设备 gather、无推测重叠),但保持 deferred batching 与输出步并入提交窗口。
    /// 两轮必须一致——若此形态分叉,竞态在 deferred 轮内容本身;若干净,则在 gather/重叠。
    #[test]
    fn minicpm5_serial_cpu_embed_repeat_deterministic() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        const STEPS: usize = 48;
        let session = MiniCpm5MetalSession::load(Path::new(WEIGHTS), 8192, false, crate::weight::LmHeadQuantization::Native).unwrap();
        let prompt = crate::runtime::minicpm5::minicpm5_instruct_prompt("用一句话解释什么是注意力机制", None, true);
        let tokens = session.tokenize(&prompt);
        let mut rounds = Vec::new();
        for _ in 0..2 {
            let mut sequence = session.prefill(tokens.clone()).unwrap();
            session.begin_async_decode();
            let mut pending = session.submit_output(&sequence.hidden, sequence.tokens.len()).unwrap();
            let mut generated = Vec::new();
            for step in 0..STEPS {
                let token = session.wait_token(&mut sequence, &pending);
                generated.push(token);
                if step + 1 < STEPS {
                    pending = session.submit_step_cpu_embed(&mut sequence, token).unwrap();
                }
            }
            session.end_async_decode();
            rounds.push(generated);
        }
        assert_eq!(rounds[0], rounds[1], "串行 CPU embedding 两轮 decode 不一致(deferred 轮内容竞态)");
    }

    /// 异步流水线读回竞态回归:两轮 fresh prefill + 48 步异步 decode 必须逐 token
    /// 一致。2-slot id buffer 时代,argmax(N+2) 与 CPU 读回 slot(N) 竞态曾致
    /// token ~17 起分叉;per-position 读回区按位置寻址后必须稳定。
    /// 循环内刻意不做全同步:竞态的前提是 GPU 跑在 CPU 读回前面,逐步 synchronize
    /// 会把流水线串行化,测试退化成同步路径,永远抓不住这类回归。
    #[test]
    fn minicpm5_async_decode_repeat_deterministic() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        const STEPS: usize = 48;
        let session = MiniCpm5MetalSession::load(Path::new(WEIGHTS), 8192, false, crate::weight::LmHeadQuantization::Native).unwrap();
        assert!(session.async_decode_available(), "设备端 embedding 来源缺失,异步流水线不可用");
        let prompt = crate::runtime::minicpm5::minicpm5_instruct_prompt("用一句话解释什么是注意力机制", None, true);
        let tokens = session.tokenize(&prompt);
        let mut rounds = Vec::new();
        for _ in 0..2 {
            let mut sequence = session.prefill(tokens.clone()).unwrap();
            session.begin_async_decode();
            let mut pending = session.submit_output(&sequence.hidden, sequence.tokens.len()).unwrap();
            let mut generated = Vec::new();
            let mut snapshots: Vec<Vec<u16>> = Vec::new();
            // 从 prefill 输出 H_0 开始打点:iteration i 的 wait(current) 保证其之前的
            // 提交全部完成,stash 可无同步读;与在飞推测轮并发正是竞态窗口。
            let mut stashed: Option<crate::backend::metal::MetalTensor> = Some(sequence.hidden.clone());
            for step in 0..STEPS {
                let mut current = pending;
                // 与 generate_one 同构:先提交下一轮推测再等当前轮,末轮不再推测。
                let next = if step + 1 < STEPS { Some(session.submit_step(&mut sequence, &mut current).unwrap()) } else { None };
                generated.push(session.wait_token(&mut sequence, &current));
                // 诊断:上一轮 stash 的 hidden 已被 current.command(FIFO 更晚)保证完成,
                // 无需全同步即可读快照;与在飞推测轮并发正是竞态窗口。
                if let Some(hidden) = stashed.take() {
                    let bits = unsafe { std::slice::from_raw_parts(hidden.buffer.contents() as *const u16, hidden.len()) };
                    snapshots.push(bits.to_vec());
                }
                stashed = Some(sequence.hidden.clone());
                let Some(next) = next else { break };
                pending = next;
            }
            session.end_async_decode();
            // 排空后复读读回区:wait_token 在竞态窗口内读到的值必须与结算值一致,
            // 不一致即"读回早于 argmax 写入"——单轮内的竞态,两轮对比抓不住。
            let prompt_len = tokens.len();
            let settled: Vec<u32> = (0..STEPS).map(|index| session.readback_at(prompt_len + index)).collect();
            assert_eq!(generated, settled, "wait_token 读回值与 GPU 结算值不一致(读回早于写入)");
            rounds.push((generated, snapshots));
        }
        let hidden_diff = rounds[0].1.iter().zip(rounds[1].1.iter()).position(|(a, b)| a != b);
        eprintln!("首个分叉 hidden(0=prefill 输出, k=第 k 轮 decode 输出): {hidden_diff:?}");
        if let Some(step) = hidden_diff {
            for prior in step..=(step + 1).min(rounds[0].1.len() - 1) {
                let (a, b) = (&rounds[0].1[prior], &rounds[1].1[prior]);
                let diffs: Vec<usize> = a.iter().zip(b.iter()).enumerate().filter_map(|(i, (x, y))| (x != y).then_some(i)).collect();
                let max_abs = diffs.iter().fold(0.0f32, |m, &i| {
                    let da = (half::f16::from_bits(a[i]).to_f32() - half::f16::from_bits(b[i]).to_f32()).abs();
                    m.max(da)
                });
                eprintln!("  快照 {prior}(0=prefill): {} 个元素不同, max|Δ|={max_abs:.6}, 前 8 个下标 {:?}", diffs.len(), &diffs[..diffs.len().min(8)]);
                if let Some(&first) = diffs.first() {
                    eprintln!("    首个不同元素[{first}]: round0={} round1={}", half::f16::from_bits(a[first]).to_f32(), half::f16::from_bits(b[first]).to_f32());
                }
            }
        }
        assert_eq!(rounds[0].0, rounds[1].0, "异步流水线两轮 decode 不一致(读回竞态回归)");
    }

    /// decode 逐算子 GPU profile(诊断用,不随 CI 跑):
    /// `cargo test --release --lib minicpm5_decode_profile -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn minicpm5_decode_profile() {
        if !Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        let session = MiniCpm5MetalSession::load(Path::new(WEIGHTS), 8192, false, crate::weight::LmHeadQuantization::Native).unwrap();
        let prompt = crate::runtime::minicpm5::minicpm5_instruct_prompt("用一句话解释什么是注意力机制", None, true);
        let tokens = session.tokenize(&prompt);
        // prefill 耗时单独计时(server 每请求重建 KV cache + prefill,曾见 ~100ms 开销)
        let prefill_started = std::time::Instant::now();
        let mut sequence = session.prefill(tokens).unwrap();
        session.context().synchronize();
        eprintln!("prefill(cold cache alloc): {:.1}ms", prefill_started.elapsed().as_secs_f64() * 1e3);
        let prefill_started = std::time::Instant::now();
        let _sequence2 = session.prefill(session.tokenize(&prompt)).unwrap();
        session.context().synchronize();
        eprintln!("prefill(warm): {:.1}ms", prefill_started.elapsed().as_secs_f64() * 1e3);
        let ctx = session.context();
        // 异步流水线(argmax→gather→推测下一轮):热身 8 token 后开始计时
        session.begin_async_decode();
        let mut pending = session.submit_output(&sequence.hidden, sequence.tokens.len()).unwrap();
        for _ in 0..8 {
            let mut current = pending;
            let next = session.submit_step(&mut sequence, &mut current).unwrap();
            session.wait_token(&mut sequence, &current);
            pending = next;
        }
        ctx.reset_gpu_stats();
        let started = std::time::Instant::now();
        const STEPS: usize = 32;
        for _ in 0..STEPS {
            let mut current = pending;
            let next = session.submit_step(&mut sequence, &mut current).unwrap();
            session.wait_token(&mut sequence, &current);
            pending = next;
        }
        session.end_async_decode();
        ctx.synchronize();
        let wall = started.elapsed();
        let gpu = ctx.gpu_stats();
        let (alloc_ns, commit_ns) = ctx.decode_cpu_breakdown();
        eprintln!(
            "minicpm5 decode: {} tok/s, wall={:.2}ms/tok gpu={:.2}ms/tok commands={} cpu_alloc={:.2}ms cpu_commit={:.2}ms submit_wait={:.2}ms gap={:.2}ms tail={:.2}ms",
            STEPS as f64 / wall.as_secs_f64(),
            wall.as_secs_f64() * 1e3 / STEPS as f64,
            gpu.seconds * 1e3 / STEPS as f64,
            gpu.command_buffers / STEPS as u64,
            alloc_ns as f64 * 1e-6 / STEPS as f64,
            commit_ns as f64 * 1e-6 / STEPS as f64,
            gpu.submit_wait_seconds * 1e3 / STEPS as f64,
            gpu.inter_command_gap_seconds * 1e3 / STEPS as f64,
            gpu.completion_tail_seconds * 1e3 / STEPS as f64
        );
        for operator in ctx.gpu_profile().into_iter().take(12) {
            eprintln!("  {:>9.3} ms x{} | {} {}", operator.gpu_seconds * 1e3 / STEPS as f64, operator.calls / STEPS as u64, operator.operator, operator.shape);
        }
        // 第二段:1 op/command buffer,拿逐算子精确归因( dispatch 间隔变大,仅用于归因)
        ctx.set_decode_batch_max_operations(1);
        ctx.reset_gpu_stats();
        let mut token = 0u32;
        for _ in 0..STEPS {
            token = session.next_token(&sequence, None).unwrap();
            session.decode_token(&mut sequence, token).unwrap();
        }
        let _ = token;
        ctx.synchronize();
        eprintln!("--- per-op (1 op/CB) ---");
        for operator in ctx.gpu_profile().into_iter().take(24) {
            eprintln!("  {:>9.3} ms x{} | {} {}", operator.gpu_seconds * 1e3 / STEPS as f64, operator.calls / STEPS as u64, operator.operator, operator.shape);
        }
        ctx.set_decode_batch_max_operations(16);
    }
}
