//! SeedVR2 的 ROCm sequence/head 交换；每个 rank 处理同一视频的部分 token。

use std::{collections::HashSet, ops::Range};

use crate::backend::{
    Backend, BackendError, SegmentedTensorBackend, VaeBackend,
    rocm::{RocmContext, RocmTensor, RocmWeight},
};

use super::SeedVr2WindowPlan;

fn error(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}

/// 权重由模型持有；组只拥有每卡独立 compute stream 和通信顺序。
pub struct SeedVr2RocmGroup {
    contexts: Vec<RocmContext>,
}

fn ranges(rows: usize, ranks: usize) -> Result<Vec<Range<usize>>, BackendError> {
    if !matches!(ranks, 1 | 2 | 4 | 8) || rows < ranks {
        return Err(error(format!("SeedVR2 sequence rows={rows} ranks={ranks} 非法，要求非空的 1/2/4/8 个 shard")));
    }
    let mut start = 0;
    Ok((0..ranks)
        .map(|rank| {
            let end = start + rows / ranks + usize::from(rank < rows % ranks);
            let range = start..end;
            start = end;
            range
        })
        .collect())
}

impl SeedVr2RocmGroup {
    pub fn new(devices: &[i32]) -> Result<Self, String> {
        if !matches!(devices.len(), 1 | 2 | 4 | 8) || devices.iter().copied().collect::<HashSet<_>>().len() != devices.len() {
            return Err(format!("SeedVR2 devices={devices:?} 必须为不重复的 1/2/4/8 张卡"));
        }
        // 2K VAE 的大块激活与常驻条件统一使用显式池，避免混用异步池时条件被改写。
        crate::kernel::rocm::hip::enable_device_buffer_reuse();
        let contexts =
            devices.iter().map(|&device| RocmContext::configured(device, false).and_then(|context| context.with_independent_stream()).map_err(|reason| format!("SeedVR2 初始化 device={device}: {reason}"))).collect::<Result<Vec<_>, _>>()?;
        for destination in &contexts {
            for source in &contexts {
                if destination.device_id() != source.device_id() {
                    destination.enable_peer_access_from(source.device_id()).map_err(|reason| format!("SeedVR2 P2P {} <- {}: {reason}", destination.device_id(), source.device_id()))?;
                }
            }
        }
        Ok(Self { contexts })
    }

    pub fn contexts(&self) -> &[RocmContext] {
        &self.contexts
    }
    pub fn primary(&self) -> &RocmContext {
        &self.contexts[0]
    }

    pub fn sequence_ranges(&self, rows: usize) -> Result<Vec<Range<usize>>, BackendError> {
        ranges(rows, self.contexts.len())
    }

    fn activate_all(&self) -> Result<(), BackendError> {
        for context in &self.contexts {
            context.activate().map_err(error)?;
        }
        Ok(())
    }

    pub fn synchronize_all(&self) -> Result<(), BackendError> {
        // 即使一张卡报错，也尝试等待其它卡。只有全部完成才允许退休跨卡源引用。
        let mut failures = Vec::new();
        for context in &self.contexts {
            if let Err(reason) = context.synchronize_compute_stream() {
                failures.push(format!("device={}: {reason}", context.device_id()));
            }
        }
        if !failures.is_empty() {
            return Err(error(format!("SeedVR2 collective 等待失败: {}", failures.join("; "))));
        }
        for context in &self.contexts {
            context.retire_ordered_p2p_sources();
        }
        Ok(())
    }

    fn exchange(&self, sources: Vec<Vec<RocmTensor>>) -> Result<Vec<Vec<RocmTensor>>, BackendError> {
        self.activate_all()?;
        let ranks = self.contexts.len();
        if sources.len() != ranks || sources.iter().any(|row| row.len() != ranks) {
            return Err(error(format!("SeedVR2 all-to-all sources={} ranks={ranks} 不匹配", sources.len())));
        }
        let mut destinations = vec![vec![None; ranks]; ranks];
        let submitted = (|| {
            for rank in 0..ranks {
                destinations[rank][rank] = Some(sources[rank][rank].clone());
            }
            // XOR 每轮每卡只参与一对；双向 helper 先记录两端 ready，再排入两向复制。
            for round in 1..ranks {
                for left in 0..ranks {
                    let right = left ^ round;
                    if left < right {
                        let (left_on_right, right_on_left) = RocmContext::exchange_stable_tensors_ordered(&sources[left][right], &sources[right][left], self.primary().device_id())?;
                        destinations[right][left] = Some(left_on_right);
                        destinations[left][right] = Some(right_on_left);
                    }
                }
            }
            Ok::<_, BackendError>(())
        })();
        if let Err(drain) = self.synchronize_all() {
            // 上一对成功、下一对失败也可能留下目标写入；不能把这些地址提早交还复用池。
            std::mem::forget(sources);
            std::mem::forget(destinations);
            return Err(error(format!("SeedVR2 all-to-all 提交={submitted:?}; {drain}")));
        }
        submitted?;
        destinations.into_iter().enumerate().map(|(destination, row)| row.into_iter().enumerate().map(|(source, tensor)| tensor.ok_or_else(|| error(format!("SeedVR2 exchange 缺少 {source}->{destination}")))).collect()).collect()
    }

    pub fn scatter_sequence(&self, tensor: &RocmTensor) -> Result<Vec<RocmTensor>, BackendError> {
        self.activate_all()?;
        let shards = self.sequence_ranges(tensor.rows)?;
        let mut outputs = Vec::with_capacity(shards.len());
        for (context, shard) in self.contexts.iter().zip(shards) {
            let local = self.primary().slice_token_rows(tensor, shard.start, shard.len())?;
            let stable = self.primary().tensor_to_stable_deferred(local)?;
            outputs.push(context.tensor_on_device_ordered(stable)?);
        }
        self.synchronize_all()?;
        Ok(outputs)
    }

    pub fn gather_sequence(&self, tensors: &[RocmTensor]) -> Result<RocmTensor, BackendError> {
        self.activate_all()?;
        let rows = tensors.iter().try_fold(0usize, |rows, tensor| rows.checked_add(tensor.rows)).ok_or_else(|| error("SeedVR2 gather rows 溢出"))?;
        self.validate_shards(tensors, rows, tensors.first().map_or(0, |tensor| tensor.cols))?;
        let mut outputs = Vec::with_capacity(tensors.len());
        for (context, tensor) in self.contexts.iter().zip(tensors) {
            outputs.push(self.primary().tensor_on_device_ordered(context.tensor_to_stable_deferred(tensor.clone())?)?);
        }
        self.synchronize_all()?;
        self.primary().concat_token_rows(&outputs.iter().collect::<Vec<_>>())
    }

    fn validate_shards(&self, tensors: &[RocmTensor], rows: usize, columns: usize) -> Result<(), BackendError> {
        if tensors.len() != self.contexts.len() {
            return Err(error(format!("SeedVR2 tensors={} ranks={} 不匹配", tensors.len(), self.contexts.len())));
        }
        for ((context, tensor), shard) in self.contexts.iter().zip(tensors).zip(self.sequence_ranges(rows)?) {
            if tensor.rows != shard.len() || tensor.cols != columns || tensor.device.as_deref().is_none_or(|device| device.device_id() != context.device_id()) {
                return Err(error(format!("SeedVR2 device={} shard=[{},{}]，期望 [{},{}] 常驻本卡", context.device_id(), tensor.rows, tensor.cols, shard.len(), columns)));
            }
        }
        Ok(())
    }

    fn sequence_to_heads(&self, qkv: &[RocmTensor], heads: usize, head_dim: usize) -> Result<Vec<RocmTensor>, BackendError> {
        let local_heads = heads / self.contexts.len();
        let compact = self
            .contexts
            .iter()
            .zip(qkv)
            .map(|(context, tensor)| {
                let tensor = context.tensor_as_f32(tensor.clone())?;
                (0..self.contexts.len())
                    .map(|rank| {
                        let part = context.compact_qkv_heads(&tensor, heads, rank * local_heads..(rank + 1) * local_heads, head_dim)?;
                        context.tensor_to_stable_deferred(part)
                    })
                    .collect::<Result<Vec<_>, BackendError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.contexts.iter().zip(self.exchange(compact)?).map(|(context, parts)| context.concat_token_rows(&parts.iter().collect::<Vec<_>>())).collect()
    }

    fn heads_to_sequence(&self, heads: Vec<RocmTensor>, rows: usize) -> Result<Vec<RocmTensor>, BackendError> {
        let shards = self.sequence_ranges(rows)?;
        let compact = self
            .contexts
            .iter()
            .zip(heads)
            .map(|(context, tensor)| {
                shards
                    .iter()
                    .map(|shard| {
                        let part = context.slice_token_rows(&tensor, shard.start, shard.len())?;
                        context.tensor_to_stable_deferred(part)
                    })
                    .collect::<Result<Vec<_>, BackendError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.contexts
            .iter()
            .zip(self.exchange(compact)?)
            .map(|(context, columns)| {
                let mut columns = columns.into_iter();
                let mut output = columns.next().expect("rank 非空");
                for column in columns {
                    output = context.concat_columns(&output, &column)?;
                }
                Ok(output)
            })
            .collect()
    }

    /// norms 顺序为 video-Q、video-K、text-Q、text-K；输出仍为两条原序列的 shard。
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn attention_qkv(
        &self,
        video_qkv: &[RocmTensor],
        text_qkv: &[RocmTensor],
        norms: &[(&RocmWeight, &RocmWeight, &RocmWeight, &RocmWeight)],
        plan: &SeedVr2WindowPlan,
        head_count: usize,
        head_dim: usize,
        rotary_dim: usize,
        eps: f32,
    ) -> Result<(Vec<RocmTensor>, Vec<RocmTensor>), BackendError> {
        self.activate_all()?;
        if head_count == 0 || !head_count.is_multiple_of(self.contexts.len()) || head_dim == 0 || norms.len() != self.contexts.len() {
            return Err(error(format!("SeedVR2 attention heads={head_count} dim={head_dim} norms={} ranks={} 不兼容", norms.len(), self.contexts.len())));
        }
        let columns = head_count.checked_mul(head_dim).and_then(|v| v.checked_mul(3)).ok_or_else(|| error("SeedVR2 QKV columns 溢出"))?;
        self.validate_shards(video_qkv, plan.video_rows, columns)?;
        self.validate_shards(text_qkv, plan.text_rows, columns)?;
        let video = self.sequence_to_heads(video_qkv, head_count, head_dim)?;
        let text = self.sequence_to_heads(text_qkv, head_count, head_dim)?;
        let local_heads = head_count / self.contexts.len();
        // 现有 attention pack 仍有本卡 fence；独立提交线程让一张卡的窗口等待
        // 不阻止其它卡启动。线程返回前稳定化输出，后续 P2P 仍由同一组 stream 排序。
        let outputs = std::thread::scope(|scope| {
            let handles = self
                .contexts
                .iter()
                .zip(video)
                .zip(text)
                .zip(norms)
                .map(|(((context, video), text), &norms)| {
                    scope.spawn(move || {
                        context.activate().map_err(error)?;
                        let (video, text) = window_attention(context, &video, &text, norms, plan, local_heads, head_dim, rotary_dim, eps)?;
                        let outputs = (context.tensor_to_stable_deferred(video)?, context.tensor_to_stable_deferred(text)?);
                        context.synchronize_compute_stream()?;
                        Ok::<_, BackendError>(outputs)
                    })
                })
                .collect::<Vec<_>>();
            // 所有 worker 都回收后再传播错误，不能因前一个失败漏掉后续 panic/join。
            handles.into_iter().map(|handle| handle.join().unwrap_or_else(|_| Err(error("SeedVR2 window attention worker panic")))).collect::<Vec<_>>()
        });
        let (video_outputs, text_outputs): (Vec<_>, Vec<_>) = outputs.into_iter().collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
        Ok((self.heads_to_sequence(video_outputs, plan.video_rows)?, self.heads_to_sequence(text_outputs, plan.text_rows)?))
    }
}

/// 单份 head 的模型算法也供 CPU oracle 使用，通信只改变 head/sequence 所有权。
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn window_attention<B: VaeBackend>(
    backend: &B,
    video: &B::Tensor,
    text: &B::Tensor,
    norms: (&B::Weight, &B::Weight, &B::Weight, &B::Weight),
    plan: &SeedVr2WindowPlan,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
) -> Result<(B::Tensor, B::Tensor), BackendError> {
    if plan.windows.is_empty()
        || plan.text_indices.len() != plan.windows.len()
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || plan.cosine.len() != plan.video_rows.checked_mul(rotary_dim / 2).ok_or_else(|| error("SeedVR2 RoPE elements 溢出"))?
        || plan.sine.len() != plan.cosine.len()
    {
        return Err(error("SeedVR2 window attention 窗口/RoPE 元数据不匹配"));
    }
    let columns = heads.checked_mul(head_dim).ok_or_else(|| error("SeedVR2 attention columns 溢出"))?;
    let video = backend.select_rows(video, &plan.partition)?;
    let (vq, vkv) = backend.split_columns(&video, columns)?;
    let (vk, vv) = backend.split_columns(&vkv, columns)?;
    let (tq, tkv) = backend.split_columns(text, columns)?;
    let (tk, tv) = backend.split_columns(&tkv, columns)?;
    let vq = backend.rmsnorm_heads(&vq, norms.0, heads, head_dim, eps)?;
    let vk = backend.rmsnorm_heads(&vk, norms.1, heads, head_dim, eps)?;
    let tq = backend.rmsnorm_heads(&tq, norms.2, heads, head_dim, eps)?;
    let tk = backend.rmsnorm_heads(&tk, norms.3, heads, head_dim, eps)?;
    let (vq, vk) = backend.rope_pair_prefix(vq, vk, heads, rotary_dim, crate::attention::rope::RotaryLayout::Interleaved, 0, &plan.cosine, &plan.sine)?;
    let joint = |video: &B::Tensor, text: &B::Tensor| backend.select_rows(&backend.concat_rows(video, text)?, &plan.joint_indices);
    let mut offsets = Vec::with_capacity(plan.windows.len() + 1);
    offsets.push(0);
    for window in &plan.windows {
        if window.joint.start != *offsets.last().expect("首边界已添加") {
            return Err(error("SeedVR2 window joint 边界不连续"));
        }
        offsets.push(window.joint.end);
    }
    let attended = backend.varlen_attention(joint(&vq, &tq)?, joint(&vk, &tk)?, joint(&vv, &tv)?, &offsets, &offsets, heads, head_dim, (head_dim as f32).sqrt().recip())?;
    let video = backend.select_rows(&backend.select_rows(&attended, &plan.video_indices)?, &plan.reverse)?;
    let mut texts = plan.text_indices.iter();
    let mut text = backend.select_rows(&attended, texts.next().expect("窗口非空"))?;
    for indices in texts {
        text = backend.add(&text, &backend.select_rows(&attended, indices)?)?;
    }
    let text = backend.scale_tensor(&text, (plan.windows.len() as f32).recip())?;
    Ok((video, text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendResources;

    #[test]
    fn sequence_ranges_cover_uneven_rows() {
        for ranks in [1, 2, 4, 8] {
            let shards = ranges(19, ranks).unwrap();
            assert_eq!(shards.iter().flat_map(Clone::clone).collect::<Vec<_>>(), (0..19).collect::<Vec<_>>());
            assert!(shards.iter().map(Range::len).max().unwrap() - shards.iter().map(Range::len).min().unwrap() <= 1);
        }
        assert!(ranges(7, 8).is_err());
        assert!(ranges(10, 3).is_err());
    }

    #[test]
    #[ignore = "占用 ROCm 0..7，仅显式运行"]
    fn eight_rank_window_attention_matches_cpu() {
        use crate::backend::{LinearWeight, cpu::CpuContext};
        let group = SeedVr2RocmGroup::new(&(0..8).collect::<Vec<_>>()).unwrap();
        let heads = 24;
        let dim = 128;
        let columns = heads * dim * 3;
        let cpu = CpuContext;
        let values = |rows: usize, seed: usize| (0..rows * columns).map(|i| ((i * 17 + seed) % 101) as f32 / 83.0 - 0.6).collect::<Vec<_>>();
        let video = values(19, 7);
        let text = values(11, 13);
        let norm_data = (0..dim).map(|i| 0.8 + i as f32 / 1024.0).collect::<Vec<_>>();
        let cpu_norm = cpu.prepare_weight(LinearWeight::F32(&norm_data), 1, dim).unwrap();
        let norms = group.contexts().iter().map(|context| context.prepare_weight(LinearWeight::F32(&norm_data), 1, dim).unwrap()).collect::<Vec<_>>();
        let norms = norms.iter().map(|weight| (weight, weight, weight, weight)).collect::<Vec<_>>();
        let video_device = group.primary().tensor_from_f32(video.clone(), 19, columns).unwrap();
        let text_device = group.primary().tensor_from_f32(text.clone(), 11, columns).unwrap();
        let video_shards = group.scatter_sequence(&video_device).unwrap();
        let text_shards = group.scatter_sequence(&text_device).unwrap();
        assert_eq!(group.primary().tensor_to_f32(&group.gather_sequence(&video_shards).unwrap()).unwrap(), video);
        assert_eq!(group.primary().tensor_to_f32(&group.gather_sequence(&text_shards).unwrap()).unwrap(), text);
        for shifted in [false, true] {
            let plan = SeedVr2WindowPlan::new([19, 1, 1], 11, [4, 3, 3], shifted, &[0.25; 10]).unwrap();
            let video_cpu = cpu.vae_tensor_from_f32(video.clone(), 19, columns).unwrap();
            let text_cpu = cpu.vae_tensor_from_f32(text.clone(), 11, columns).unwrap();
            let expected = window_attention(&cpu, &video_cpu, &text_cpu, (&cpu_norm, &cpu_norm, &cpu_norm, &cpu_norm), &plan, heads, dim, 60, 1e-6).unwrap();
            let actual = group.attention_qkv(&video_shards, &text_shards, &norms, &plan, heads, dim, 60, 1e-6).unwrap();
            for (name, expected, shards) in [("video", expected.0, actual.0), ("text", expected.1, actual.1)] {
                let actual = group.primary().tensor_to_f32(&group.gather_sequence(&shards).unwrap()).unwrap();
                let expected = cpu.vae_tensor_to_f32(&expected).unwrap();
                assert_eq!(actual.len(), expected.len());
                let mut max_error = 0.0f32;
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    max_error = max_error.max((actual - expected).abs());
                    assert!(actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1e-2 + 1e-2 * expected.abs(), "{name} shifted={shifted} index={index} actual={actual} expected={expected}");
                }
                eprintln!("[seedvr2-native-oracle] ranks=8 local_heads=3 {name} shifted={shifted} elements={} max_error={max_error}", actual.len());
            }
        }
        group.synchronize_all().unwrap();
    }
}
