//! 八卡 VAE 时间微帧流水；每段保留完整空间帧及自己的 causal history。

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::mpsc::{Receiver, SyncSender, sync_channel},
    thread,
    time::{Duration, Instant},
};

use crate::{
    backend::{
        Backend, BackendError, SegmentedTensorBackend,
        rocm::{RocmContext, RocmTensor, RocmWeight},
    },
    weight::model::seedvr2::SeedVr2VaeSource,
};

use super::{
    rocm::SeedVr2RocmGroup,
    vae::{self, PreparedVae, VaeStageCache, VaeVideo},
};

fn error(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}

fn activate(contexts: &[RocmContext], stage: usize) -> Result<(), BackendError> {
    // ordered P2P 在接收线程查源流 TLS；不能只登记自己的目的流。
    for context in contexts {
        context.activate().map_err(error)?;
    }
    contexts[stage].activate().map_err(error)
}

/// 每卡独立驻留完整 VAE 权重；算法分段只决定本卡执行哪些层。
pub fn prepare(group: &SeedVr2RocmGroup, source: &SeedVr2VaeSource) -> Result<Vec<PreparedVae<RocmWeight>>, BackendError> {
    if group.contexts().len() != 8 {
        return Err(error("SeedVR2 VAE 流水要求八张不同设备"));
    }
    let mut weights = Vec::with_capacity(8);
    let result = (|| {
        for context in group.contexts() {
            context.activate().map_err(error)?;
            weights.push(vae::prepare_vae(context, source)?);
        }
        Ok(())
    })();
    if let Err(drain) = group.synchronize_all() {
        std::mem::forget(weights);
        return Err(error(format!("SeedVR2 VAE prepare={result:?}; drain={drain}")));
    }
    result?;
    Ok(weights)
}

fn required_frames(encode: bool, stage: usize, first: bool) -> usize {
    if encode && matches!(stage, 2 | 3) && !first { 2 } else { 1 }
}

fn expected_shape(shape: [usize; 3], encode: bool) -> Result<[usize; 3], BackendError> {
    let [t, h, w] = shape;
    if shape.contains(&0) || t.checked_mul(h).and_then(|n| n.checked_mul(w)).is_none() {
        return Err(error(format!("SeedVR2 VAE 流水 shape={shape:?} 为空或溢出")));
    }
    if encode {
        if !(t - 1).is_multiple_of(4) || !h.is_multiple_of(8) || !w.is_multiple_of(8) {
            return Err(error(format!("SeedVR2 VAE encode shape={shape:?} 要求 T=4n+1、H/W 为 8 的倍数")));
        }
        Ok([(t - 1) / 4 + 1, h / 8, w / 8])
    } else {
        let t = (t - 1).checked_mul(4).and_then(|n| n.checked_add(1));
        match (t, h.checked_mul(8), w.checked_mul(8)) {
            (Some(t), Some(h), Some(w)) if t.checked_mul(h).and_then(|n| n.checked_mul(w)).is_some() => Ok([t, h, w]),
            _ => Err(error("SeedVR2 VAE decode 输出 shape 溢出")),
        }
    }
}

fn frame(context: &RocmContext, video: &VaeVideo<RocmTensor>, index: usize, tokens: bool) -> Result<VaeVideo<RocmTensor>, BackendError> {
    let [t, h, w] = video.shape;
    if index >= t {
        return Err(error(format!("SeedVR2 VAE frame={index} 超过 T={t}")));
    }
    let plane = h * w;
    let tensor = if t == 1 {
        video.tensor.clone()
    } else if tokens {
        context.slice_token_rows(&video.tensor, index * plane, plane)?
    } else {
        // C×THW 的帧不连续；只复制所需列，禁止整段转置再截取。
        context.slice_columns_range(&video.tensor, index * plane..(index + 1) * plane)?
    };
    Ok(VaeVideo { tensor, shape: [1, h, w] })
}

fn send_frames(context: &RocmContext, output: VaeVideo<RocmTensor>, tokens: bool, sender: &SyncSender<VaeVideo<RocmTensor>>) -> Result<(), BackendError> {
    for index in 0..output.shape[0] {
        let part = frame(context, &output, index, tokens)?;
        let part = VaeVideo { tensor: context.tensor_to_stable_deferred(part.tensor)?, shape: part.shape };
        // 背压必须约束已完成的微帧；否则 host 可以把整段 GPU 工作提前排完。
        if let Err(drain) = context.synchronize_compute_stream() {
            std::mem::forget(part);
            std::mem::forget(output);
            return Err(drain);
        }
        sender.send(part).map_err(|_| error("SeedVR2 VAE 下游已退出"))?;
    }
    Ok(())
}

fn worker(
    contexts: &[RocmContext],
    stage: usize,
    encode: bool,
    initial: Option<VaeVideo<RocmTensor>>,
    receiver: Option<Receiver<VaeVideo<RocmTensor>>>,
    sender: SyncSender<VaeVideo<RocmTensor>>,
    weights: &PreparedVae<RocmWeight>,
) -> Result<(), BackendError> {
    let context = &contexts[stage];
    let mut cache = VaeStageCache::new();
    let started = Instant::now();
    let mut input_wall = Duration::ZERO;
    let mut compute_wall = Duration::ZERO;
    let mut output_wall = Duration::ZERO;
    let mut chunks = 0usize;
    let mut output_frames = 0usize;
    // 目标流若无法排空，额外源引用保留跨线程 P2P 读取，不能随 TLS 退出归还池。
    let mut sources = Vec::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        activate(contexts, stage)?;
        let mut count = 0usize;
        loop {
            let input_started = Instant::now();
            let needed = required_frames(encode, stage, count == 0);
            let mut input: Option<VaeVideo<RocmTensor>> = None;
            for part in 0..needed {
                let next = if let Some(video) = initial.as_ref() { if count == video.shape[0] { None } else { Some(frame(context, video, count, !encode)?) } } else { receiver.as_ref().expect("非首段有输入队列").recv().ok() };
                let Some(next) = next else {
                    if part != 0 {
                        return Err(error(format!("SeedVR2 VAE stage={stage} 第 {count} 片缺少配对帧 {part}/{needed}")));
                    }
                    return Ok(());
                };
                if next.shape[0] != 1 {
                    return Err(error(format!("SeedVR2 VAE stage={stage} 输入不是单帧: {:?}", next.shape)));
                }
                sources.push(next.tensor.clone());
                let tensor = context.tensor_on_device_ordered(next.tensor)?;
                input = Some(match input {
                    None => VaeVideo { tensor, shape: next.shape },
                    Some(previous) => {
                        if previous.shape[1..] != next.shape[1..] || previous.tensor.rows != tensor.rows {
                            return Err(error(format!("SeedVR2 VAE stage={stage} 相邻帧 shape 不一致")));
                        }
                        VaeVideo { tensor: context.concat_columns(&previous.tensor, &tensor)?, shape: [previous.shape[0] + 1, next.shape[1], next.shape[2]] }
                    }
                });
            }
            let input = input.expect("微帧数量非零");
            input_wall += input_started.elapsed();
            let compute_started = Instant::now();
            let output = if encode { vae::encode_stage(context, stage, input, weights, &mut cache) } else { vae::decode_stage(context, stage, input, weights, &mut cache) };
            if let Err(drain) = context.synchronize_compute_stream() {
                std::mem::forget(output);
                return Err(drain);
            }
            compute_wall += compute_started.elapsed();
            // pending refs 属于本线程；主线程的 group.synchronize_all 不会退休它们。
            context.retire_ordered_p2p_sources();
            sources.clear();
            let output_started = Instant::now();
            output_frames += output.as_ref().map_or(0, |video| video.shape[0]);
            send_frames(context, output?, encode && stage == 7, &sender)?;
            output_wall += output_started.elapsed();
            count += 1;
            chunks += 1;
        }
    }));
    // 包含算子失败、下游拒收及 panic；每段先结束本流再销毁自己的 cache。
    if let Err(drain) = context.synchronize_compute_stream() {
        std::mem::forget(cache);
        std::mem::forget(sources);
        std::mem::forget(initial);
        return Err(error(format!("SeedVR2 VAE stage={stage} device={} drain={drain}", context.device_id())));
    }
    context.retire_ordered_p2p_sources();
    eprintln!(
        "[seedvr2-native-vae] mode={} stage={stage} device={} chunks={chunks} output_frames={output_frames} input={:.3}s compute={:.3}s output={:.3}s wall={:.3}s",
        if encode { "encode" } else { "decode" },
        context.device_id(),
        input_wall.as_secs_f64(),
        compute_wall.as_secs_f64(),
        output_wall.as_secs_f64(),
        started.elapsed().as_secs_f64(),
    );
    match result {
        Ok(result) => result.map_err(|reason| error(format!("SeedVR2 VAE stage={stage} device={}: {reason}", context.device_id()))),
        Err(_) => Err(error(format!("SeedVR2 VAE stage={stage} device={} worker panic", context.device_id()))),
    }
}

fn run_pipeline(group: &SeedVr2RocmGroup, input: VaeVideo<RocmTensor>, weights: &[PreparedVae<RocmWeight>], encode: bool, mut emit: impl FnMut(usize, VaeVideo<RocmTensor>) -> Result<(), BackendError>) -> Result<[usize; 3], BackendError> {
    if group.contexts().len() != 8 || weights.len() != 8 {
        return Err(error(format!("SeedVR2 VAE 流水要求八卡八份权重，实际 devices={} weights={}", group.contexts().len(), weights.len())));
    }
    let input_shape = input.shape;
    let expected = expected_shape(input.shape, encode)?;
    let elements = input.shape.into_iter().product::<usize>();
    let channels = if encode { weights[0].config.in_channels } else { weights[0].config.latent_channels };
    let matrix = if encode { [channels, elements] } else { [elements, channels] };
    if [input.tensor.rows, input.tensor.cols] != matrix || input.tensor.device.as_ref().is_none_or(|b| b.device_id() != group.primary().device_id()) {
        return Err(error(format!("SeedVR2 VAE 流水输入要求主卡 resident {matrix:?}，实际 [{},{}]", input.tensor.rows, input.tensor.cols)));
    }
    activate(group.contexts(), 0)?;
    let input = VaeVideo { tensor: group.primary().tensor_to_stable_deferred(input.tensor)?, shape: input.shape };
    group.synchronize_all()?;
    let mut initial = Some(input);
    thread::scope(|scope| {
        let mut previous = None;
        let mut workers = Vec::with_capacity(8);
        for (stage, weight) in weights.iter().enumerate() {
            // 额外排队一帧吸收阶段耗时波动；用最大通道数估计显存上界，
            // 高分辨率仍保持单槽背压，避免累积大尺寸激活。
            let full_shape = if encode { input_shape } else { expected };
            let frame_bound = full_shape[1].saturating_mul(full_shape[2]).saturating_mul(*weights[0].config.block_channels.iter().max().unwrap()).saturating_mul(std::mem::size_of::<f32>());
            let capacity = if frame_bound <= 2 * 1024 * 1024 * 1024 { 2 } else { 1 };
            let (sender, receiver) = sync_channel(capacity);
            let source = previous.take();
            let first = if stage == 0 { initial.take() } else { None };
            workers.push(scope.spawn(move || worker(group.contexts(), stage, encode, first, source, sender, weight)));
            previous = Some(receiver);
        }
        let output = previous.expect("八段至少有一个输出队列");
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut frames = 0usize;
            group.contexts()[7].activate().map_err(error)?;
            while let Ok(video) = output.recv() {
                if video.shape != [1, expected[1], expected[2]] || frames >= expected[0] {
                    return Err(error(format!("SeedVR2 VAE 输出 frame={frames} shape={:?}，期望 {expected:?}", video.shape)));
                }
                emit(frames, video)?;
                frames += 1;
            }
            if frames != expected[0] {
                return Err(error(format!("SeedVR2 VAE 输出缺帧: {frames}/{}", expected[0])));
            }
            Ok(expected)
        }));
        // 消费回调失败也必须先断开末端，再 join；断链会逐段唤醒阻塞 send/recv。
        drop(output);
        let mut failures = Vec::new();
        for (stage, handle) in workers.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(())) => (),
                Ok(Err(reason)) => failures.push(reason.to_string()),
                Err(_) => failures.push(format!("SeedVR2 VAE stage={stage} worker cleanup panic")),
            }
        }
        if let Err(reason) = group.synchronize_all() {
            failures.push(reason.to_string());
        }
        let result = result.unwrap_or_else(|_| Err(error("SeedVR2 VAE emit callback panic")));
        if !failures.is_empty() {
            if let Err(reason) = result {
                failures.insert(0, reason.to_string());
            }
            return Err(error(failures.join("; ")));
        }
        result
    })
}

/// RGB C×THW 输入；末段的短 latent 全收齐后才回主卡，避免流水首尾流依赖成环。
pub fn encode(group: &SeedVr2RocmGroup, input: VaeVideo<RocmTensor>, weights: &[PreparedVae<RocmWeight>]) -> Result<VaeVideo<RocmTensor>, BackendError> {
    let mut frames = Vec::new();
    let shape = run_pipeline(group, input, weights, true, |_, video| {
        frames.push(video.tensor);
        Ok(())
    })?;
    activate(group.contexts(), 0)?;
    let mut local = Vec::with_capacity(frames.len());
    let result = (|| {
        for frame in &frames {
            local.push(group.primary().tensor_on_device_ordered(frame.clone())?);
        }
        group.primary().concat_token_rows(&local.iter().collect::<Vec<_>>())
    })();
    if let Err(drain) = group.synchronize_all() {
        std::mem::forget(frames);
        std::mem::forget(local);
        std::mem::forget(result);
        return Err(drain);
    }
    Ok(VaeVideo { tensor: result?, shape })
}

/// 回调收到末卡上已完成的单 RGB 帧；只在视频输出边界允许下载，不能积攒全片激活。
/// 调用期间不得在同一 group 上并发提交另一条 DiT/VAE 执行链。
pub fn decode_stream(group: &SeedVr2RocmGroup, latent: VaeVideo<RocmTensor>, weights: &[PreparedVae<RocmWeight>], emit: impl FnMut(usize, VaeVideo<RocmTensor>) -> Result<(), BackendError>) -> Result<[usize; 3], BackendError> {
    run_pipeline(group, latent, weights, false, emit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn microframe_compression_and_expansion_preserve_order() {
        for count in [1usize, 5, 9, 17, 33, 65] {
            let mut frames = count;
            for stage in 0..8 {
                let mut pending = frames;
                let mut calls = 0;
                while pending != 0 {
                    let needed = required_frames(true, stage, calls == 0);
                    assert!(pending >= needed);
                    pending -= needed;
                    calls += 1;
                }
                frames = calls;
            }
            assert_eq!(frames, (count - 1) / 4 + 1);
            for stage in 0..8 {
                assert_eq!(required_frames(false, stage, false), 1);
                if matches!(stage, 0 | 1) {
                    frames = frames * 2 - 1;
                }
            }
            assert_eq!(frames, count);
        }
    }

    #[test]
    fn reject_unpaired_tail_and_dimension_overflow_before_spawning() {
        assert_eq!(expected_shape([33, 1536, 2688], true).unwrap(), [9, 192, 336]);
        assert_eq!(expected_shape([9, 192, 336], false).unwrap(), [33, 1536, 2688]);
        for shape in [[0, 8, 8], [2, 8, 8], [3, 8, 8], [33, 15, 16], [usize::MAX, 8, 8]] {
            assert!(expected_shape(shape, true).is_err());
        }
        assert!(expected_shape([1, usize::MAX, 1], false).is_err());
    }
}
