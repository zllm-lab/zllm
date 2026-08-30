//! 模型无关的 prefill、stage、chunk 与连续事件调度。
//!
//! 模型只配置工作优先级、工作量、batch 上限并提供单批执行；backend 只提供
//! completion event。完成后先取 decode，再取 prefill；低压 decode 不等待凑批，
//! 只合并计算期间已就绪的同批工作。

use crate::backend::{Backend, BackendError};

/// 模型无关的 prefill chunk 策略：首次、追加和长上下文只描述调度工作量，
/// 模型负责从 token 构造输入，backend 负责执行。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveChunkPolicy {
    pub initial_chunk_size: usize,
    pub append_chunk_size: usize,
    pub long_context_threshold_tokens: usize,
    pub long_context_chunk_size: usize,
}

impl AdaptiveChunkPolicy {
    pub fn chunk_size(self, suffix_start: usize, position: usize) -> usize {
        let configured = if suffix_start == 0 { self.initial_chunk_size } else { self.append_chunk_size }.max(1);
        if position >= self.long_context_threshold_tokens { configured.min(self.long_context_chunk_size.max(1)) } else { configured }
    }
}

/// 通用分块 prefill 驱动：把 `[offset, offset+len)` 顺序切成 chunk，逐块交给
/// `step(相对区间, 区间首 token 的绝对位置)`——相对区间直接索引调用方的 suffix
/// 切片，绝对位置用于模型的 cache/rope 位置。模型无关，只管边界与顺序；错误
/// 类型由 step 决定，不绑定 backend。各模型 node/一次性入口的分块 prefill 统一
/// 走这里，避免每个 runtime 手写一遍 while 循环。
pub fn run_chunked_prefill<E>(offset: usize, len: usize, chunk_size: usize, mut step: impl FnMut(std::ops::Range<usize>, usize) -> Result<(), E>) -> Result<(), E> {
    let mut position = offset;
    while position < offset + len {
        let end = (position + chunk_size.max(1)).min(offset + len);
        step(position - offset..end - offset, position)?;
        position = end;
    }
    Ok(())
}

pub fn run_token_chunk_stage_pipeline<T, S, F>(chunks: Vec<(usize, T)>, states: Vec<S>, run_stage: F) -> Result<Vec<(usize, T)>, BackendError>
where
    T: Send,
    S: Send,
    F: Fn(&mut S, usize, usize, T) -> Result<T, BackendError> + Sync,
{
    if chunks.is_empty() || states.is_empty() {
        return Err(BackendError::Compute { msg: format!("prefill pipeline 参数非法: chunks={} stages={}", chunks.len(), states.len()) });
    }
    std::thread::scope(|scope| {
        let (input, mut receiver) = std::sync::mpsc::channel::<Result<(usize, T), BackendError>>();
        for (stage, mut state) in states.into_iter().enumerate() {
            let (sender, next_receiver) = std::sync::mpsc::channel();
            let stage_receiver = receiver;
            let run_stage = &run_stage;
            scope.spawn(move || {
                for item in stage_receiver {
                    let output = item.and_then(|(position, hidden)| run_stage(&mut state, stage, position, hidden).map(|hidden| (position, hidden)));
                    let failed = output.is_err();
                    if sender.send(output).is_err() || failed {
                        break;
                    }
                }
            });
            receiver = next_receiver;
        }
        for chunk in chunks {
            input.send(Ok(chunk)).map_err(|_| BackendError::Compute { msg: "prefill pipeline 输入 stage 已退出".to_owned() })?;
        }
        drop(input);
        receiver.into_iter().collect()
    })
}

/// 非阻塞输入让 runtime 区分“暂时没有数据”和“流已结束”，避免为了凑 batch 阻塞整个流水线。
pub enum TokenStreamBatchPoll<T> {
    Ready(Result<T, BackendError>),
    Pending,
    Closed,
}

pub fn run_full_token_layers<B, F>(backend: &B, token_count: usize, layer_count: usize, hidden: B::Tensor, mut run_layer: F) -> Result<B::Tensor, BackendError>
where
    B: Backend,
    F: FnMut(usize, B::Tensor) -> Result<B::Tensor, BackendError>,
{
    run_full_token_layers_prefetched(backend, token_count, layer_count, 0, hidden, &mut (), |_, _| Ok(()), |_, layer, hidden| run_layer(layer, hidden))
}

#[allow(clippy::too_many_arguments)]
pub fn run_full_token_layers_prefetched<B, S, P, F>(backend: &B, token_count: usize, layer_count: usize, start_layer: usize, mut hidden: B::Tensor, state: &mut S, mut prefetch_layer: P, mut run_layer: F) -> Result<B::Tensor, BackendError>
where
    B: Backend,
    P: FnMut(&mut S, usize) -> Result<(), BackendError>,
    F: FnMut(&mut S, usize, B::Tensor) -> Result<B::Tensor, BackendError>,
{
    if token_count == 0 || layer_count == 0 || start_layer > layer_count {
        return Err(BackendError::Compute { msg: format!("prefill 参数非法: tokens={token_count} layers={layer_count} start_layer={start_layer}") });
    }
    if backend.token_rows(&hidden) != token_count {
        return Err(BackendError::Compute { msg: format!("prefill 初始 hidden 行数 {}，期望 {token_count}", backend.token_rows(&hidden)) });
    }
    if start_layer == layer_count {
        return Ok(hidden);
    }
    prefetch_layer(state, start_layer).map_err(|error| BackendError::Compute { msg: format!("prefill L{start_layer} 预取失败: {error:?}") })?;
    for layer in start_layer..layer_count {
        let _layer_scope = backend.layer_scope();
        if layer + 1 < layer_count {
            prefetch_layer(state, layer + 1).map_err(|error| BackendError::Compute { msg: format!("prefill L{} 预取失败: {error:?}", layer + 1) })?;
        }
        hidden = run_layer(state, layer, hidden).map_err(|error| BackendError::Compute { msg: format!("prefill L{layer} 失败: {error:?}") })?;
        backend.submit_batch();
        let rows = backend.token_rows(&hidden);
        if rows != token_count {
            return Err(BackendError::Compute { msg: format!("prefill L{layer} 输出行数 {rows}，期望 {token_count}") });
        }
    }
    Ok(hidden)
}

pub use super::prefill_scheduler::*;
