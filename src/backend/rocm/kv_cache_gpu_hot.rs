//! GPU 选集对应的 RAM 历史注册与热槽元数据；模型与持久化格式保持独立。

use super::kv_cache::MlaLayerSerde;
use super::{BackendError, compute_error, ops};
use std::sync::Arc;

type Buffer = ops::hip::DeviceBuffer;

pub(super) struct GpuHotCache {
    // 注册借用外层 mirror 的 allocation，外层必须先销毁本结构再销毁 mirror。
    registered: Option<[ops::hip::RegisteredHostBuffer; 3]>,
    registered_rows: usize,
    metadata: Option<Buffer>,
    map_rows: usize,
    epoch: u32,
    cache_rows: usize,
    output: Option<[Arc<Buffer>; 3]>,
    table: Option<Arc<Buffer>>,
    indices: Option<Arc<Buffer>>,
    output_rows: usize,
    stream: usize,
    trace_counts: Option<Buffer>,
}

impl GpuHotCache {
    pub(super) fn new(_device_id: i32, cache_rows: usize) -> Result<Self, BackendError> {
        if cache_rows == 0 || cache_rows > u32::MAX as usize {
            return Err(compute_error(format!("GPU hot cache_rows={cache_rows} 非法")));
        }
        Ok(Self { registered: None, registered_rows: 0, metadata: None, map_rows: 0, epoch: 0, cache_rows, output: None, table: None, indices: None, output_rows: 0, stream: 0, trace_counts: None })
    }

    pub(super) fn register_history(&mut self, device_id: i32, mirror: &mut MlaLayerSerde, capacity: usize, required_rows: usize) -> Result<(), BackendError> {
        if self.registered_rows >= required_rows {
            return Ok(());
        }
        // 每 64K 行才扩大一次注册范围；Vec 一开始预留逻辑容量，后续追加不 realloc。
        // 已注册前缀只供 GPU 读取，CPU 仅追加尚不可见的尾部字节。
        self.registered = None;
        self.registered_rows = 0;
        let rows = required_rows.div_ceil(65536).saturating_mul(65536).min(capacity);
        let row_bytes = [mirror.latent_cols, mirror.latent_cols / mirror.latent_group_size * 2, mirror.rope_cols * 2];
        let buffers = [&mut mirror.latent, mirror.latent_scales.as_mut().expect("GPU hot 必有 scales"), &mut mirror.rope];
        let mut registered = Vec::with_capacity(3);
        for (buffer, row_bytes) in buffers.into_iter().zip(row_bytes) {
            let reserved = capacity.checked_mul(row_bytes).ok_or_else(|| compute_error("GPU hot host capacity 溢出"))?;
            buffer.reserve_exact(reserved.saturating_sub(buffer.len()));
            let bytes = rows.checked_mul(row_bytes).ok_or_else(|| compute_error("GPU hot registered bytes 溢出"))?;
            // 先实写将要注册的尾页，避免后续 append 首次写入已经映射的零页。
            // 只初始化 spare capacity，逻辑长度仍由完成的 mirror 回传推进。
            let spare_bytes = bytes.saturating_sub(buffer.len());
            buffer.spare_capacity_mut()[..spare_bytes].fill(std::mem::MaybeUninit::new(0));
            // allocation 由外层 mirror 拥有，容量固定；guard 的析构先等待设备。
            registered.push(unsafe { ops::hip::RegisteredHostBuffer::register(device_id, buffer.as_mut_ptr(), bytes) }.map_err(compute_error)?);
        }
        self.registered = Some(registered.try_into().map_err(|_| compute_error("GPU hot history 注册数量异常"))?);
        self.registered_rows = rows;
        Ok(())
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn prepare(
        &mut self,
        device_id: i32,
        mirror: &mut MlaLayerSerde,
        capacity: usize,
        cache: [&Buffer; 3],
        selection: &Buffer,
        query_rows: usize,
        width: usize,
        context_rows: usize,
        recent_rows: usize,
    ) -> Result<([Arc<Buffer>; 3], Arc<Buffer>, Option<Arc<Buffer>>, usize), BackendError> {
        self.register_history(device_id, mirror, capacity, mirror.rows)?;
        let stream = ops::hip::active_compute_stream() as usize;
        if self.metadata.is_some() && self.stream != stream {
            ops::hip::order_stream_after(device_id, self.stream, stream).map_err(compute_error)?;
        }
        self.stream = stream;
        // pin 的高位区分本轮新填槽；它们到下一轮才可作为命中来源。
        self.epoch = self.epoch.wrapping_add(1) & 0x7fff_ffff;
        if context_rows > self.map_rows || self.epoch == 0 {
            let map_rows = context_rows.checked_next_power_of_two().ok_or_else(|| compute_error("GPU hot token map 溢出"))?.max(self.map_rows);
            let mut words = vec![0_u32; map_rows + self.cache_rows * 2 + 1];
            words[..map_rows + self.cache_rows].fill(u32::MAX);
            let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 4) };
            let metadata = Buffer::allocate_cache(device_id, bytes.len()).map_err(compute_error)?;
            metadata.copy_from_host(bytes).map_err(compute_error)?;
            if let Some(previous) = &self.metadata {
                metadata.copy_from_device(0, previous, 0, self.map_rows * 4).map_err(compute_error)?;
                // 扩容只移动 metadata 尾部；epoch 回绕时保留 tag，重新开始 pin。
                let tail_words = if self.epoch == 0 { self.cache_rows } else { self.cache_rows * 2 + 1 };
                metadata.copy_from_device(map_rows * 4, previous, self.map_rows * 4, tail_words * 4).map_err(compute_error)?;
            }
            self.metadata = Some(metadata);
            self.map_rows = map_rows;
            self.epoch = self.epoch.max(1);
        }
        let count = query_rows.checked_mul(width).ok_or_else(|| compute_error("GPU hot selection 大小溢出"))?;
        let columns = [mirror.latent_cols, mirror.latent_cols / mirror.latent_group_size, mirror.rope_cols];
        if count > self.output_rows {
            let output = [columns[0], columns[1] * 2, columns[2] * 2].map(|row_bytes| {
                let bytes = count.checked_mul(row_bytes).ok_or_else(|| compute_error("GPU hot gather scratch 溢出"))?;
                Buffer::allocate_cache(device_id, bytes).map(Arc::new).map_err(compute_error)
            });
            let [latent, scales, rope] = output;
            self.output = Some([latent?, scales?, rope?]);
            let ids = (0..count).map(|i| u32::try_from(i).map_err(|_| compute_error("GPU hot identity 超过 u32"))).collect::<Result<Vec<_>, _>>()?;
            let bytes = unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids.len() * 4) };
            let identity = super::upload_cache_buffer(device_id, bytes, bytes.len())?;
            self.table = Some(identity.clone());
            self.indices = Some(identity);
            self.output_rows = count;
        }
        let output = self.output.as_ref().ok_or_else(|| compute_error("GPU hot selection 为空"))?;
        let registered = self.registered.as_ref().ok_or_else(|| compute_error("GPU hot history 为空"))?;
        if ops::hip::options().mla_hot_trace && self.trace_counts.is_none() {
            self.trace_counts = Some(Buffer::upload(device_id, &[0; 12]).map_err(compute_error)?);
        }
        // 只在诊断模式每 128 轮排空采样；该运行不能用于吞吐验收。
        let sample = self.trace_counts.is_some() && self.epoch.is_multiple_of(128);
        let mut before_counts = [0; 12];
        if sample {
            ops::hip::synchronize_device(device_id, "GPU hot gather sample begin").map_err(compute_error)?;
            self.trace_counts.as_ref().expect("trace 已创建").copy_to_host(&mut before_counts).map_err(compute_error)?;
        }
        let started = sample.then(std::time::Instant::now);
        let gather = || {
            ops::hip::try_mla_gpu_hot_gather_q8(
                device_id,
                registered.each_ref(),
                cache,
                self.metadata.as_ref().expect("metadata 已创建"),
                self.map_rows,
                self.epoch,
                selection,
                output.each_ref().map(Arc::as_ref),
                count,
                columns,
                self.cache_rows,
                recent_rows,
                mirror.rows,
                context_rows,
                self.trace_counts.as_ref(),
            )
            .map_err(compute_error)
        };
        gather()?;
        if let Some(started) = started {
            ops::hip::synchronize_device(device_id, "GPU hot gather sample end").map_err(compute_error)?;
            let gather_ms = started.elapsed().as_secs_f64() * 1e3;
            let mut bytes = [0; 12];
            self.trace_counts.as_ref().expect("trace 已创建").copy_to_host(&mut bytes).map_err(compute_error)?;
            let counts = bytes.chunks_exact(4).map(|v| u32::from_ne_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            let sample_miss = counts[1].wrapping_sub(u32::from_ne_bytes(before_counts[4..8].try_into().unwrap()));
            // 选集不变，第二遍已填入热槽；同时记录 warm miss，不能假设其为零。
            // 时差包含 RAM miss、置换以及 L2 冷热差异，不能单独等同 PCIe 带宽。
            let warm_started = std::time::Instant::now();
            gather()?;
            ops::hip::synchronize_device(device_id, "GPU hot gather warm sample").map_err(compute_error)?;
            let warm_ms = warm_started.elapsed().as_secs_f64() * 1e3;
            self.trace_counts.as_ref().expect("trace 已创建").copy_to_host(&mut bytes).map_err(compute_error)?;
            let warm_miss = u32::from_ne_bytes(bytes[4..8].try_into().unwrap()).wrapping_sub(counts[1]);
            eprintln!(
                "[mla-gpu-hot-trace] device={device_id} metadata={} epoch={} hit={} miss={} recent={} sample_miss={sample_miss} gather_sample_ms={gather_ms:.3} warm_sample_ms={warm_ms:.3} warm_miss={warm_miss}",
                self.metadata.as_ref().unwrap().device_pointer(),
                self.epoch,
                counts[0],
                counts[1],
                counts[2]
            );
        }
        Ok((output.clone(), self.table.as_ref().expect("GPU hot table 已创建").clone(), (query_rows > 1).then(|| self.indices.as_ref().expect("GPU hot indices 已创建").clone()), count))
    }

    pub(super) fn truncate(&self, device_id: i32, keep: usize, end: usize) -> Result<(), BackendError> {
        if let Some(metadata) = &self.metadata {
            let previous = ops::hip::compute_stream_for(device_id) as usize;
            ops::hip::activate_compute_stream(device_id, self.stream).map_err(compute_error)?;
            let invalidated = ops::hip::try_mla_gpu_hot_invalidate(device_id, metadata, self.map_rows, self.cache_rows, keep, end).map_err(compute_error);
            ops::hip::activate_compute_stream(device_id, previous).map_err(compute_error)?;
            invalidated?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn gpu_hot_collisions_reuse_and_rollback_preserve_bytes() {
        let columns = [512, 16, 128];
        let capacity = 4096;
        let host_rows = 128;
        let mut context_rows = 131;
        let cache_rows = 32;
        let recent_rows = 4;
        let mut expected = columns.map(|cols| (0..300 * cols).map(|i| ((i * 37 + i / cols * 19) % 251) as u8).collect::<Vec<_>>());
        let mut host = columns.map(|cols| Vec::with_capacity(capacity * cols));
        for i in 0..3 {
            host[i].extend_from_slice(&expected[i][..host_rows * columns[i]]);
        }
        let [latent, scales, rope] = host;
        let mut mirror = MlaLayerSerde { rows: host_rows, ownership: super::super::kv_cache::RocmKvOwnership::Full, latent_cols: columns[0], rope_cols: columns[2] / 2, latent_group_size: 64, latent, latent_scales: Some(scales), rope };
        let mut gpu = GpuHotCache::new(0, cache_rows).unwrap();
        let cache = columns.map(|cols| {
            let mut bytes = vec![0; (cache_rows + recent_rows) * cols];
            let source = &expected[if cols == 512 {
                0
            } else if cols == 16 {
                1
            } else {
                2
            }];
            for token in host_rows..context_rows {
                let slot = cache_rows + token % recent_rows;
                bytes[slot * cols..(slot + 1) * cols].copy_from_slice(&source[token * cols..(token + 1) * cols]);
            }
            Buffer::upload(0, &bytes).unwrap()
        });
        for (generation, count) in [16, 24, 96, 12, 12, 24, 24].into_iter().enumerate() {
            if generation == 3 {
                for (buffer, source, cols) in [(&mut mirror.latent, &expected[0], columns[0]), (mirror.latent_scales.as_mut().unwrap(), &expected[1], columns[1]), (&mut mirror.rope, &expected[2], columns[2])] {
                    buffer.extend_from_slice(&source[host_rows * cols..context_rows * cols]);
                }
                mirror.rows = context_rows;
            }
            if generation == 4 {
                gpu.truncate(0, 129, 131).unwrap();
                for (i, buffer) in [&mut mirror.latent, mirror.latent_scales.as_mut().unwrap(), &mut mirror.rope].into_iter().enumerate() {
                    for byte in &mut expected[i][129 * columns[i]..131 * columns[i]] {
                        *byte ^= 0x5a;
                    }
                    buffer[129 * columns[i]..].copy_from_slice(&expected[i][129 * columns[i]..131 * columns[i]]);
                }
            }
            if generation == 5 {
                context_rows = 300;
                for (i, buffer) in [&mut mirror.latent, mirror.latent_scales.as_mut().unwrap(), &mut mirror.rope].into_iter().enumerate() {
                    buffer.extend_from_slice(&expected[i][131 * columns[i]..]);
                }
                mirror.rows = context_rows;
            }
            if generation == 6 {
                gpu.truncate(0, 131, context_rows).unwrap();
                context_rows = 131;
                mirror.rows = context_rows;
                // epoch 回绕和历史缩短同时发生时，token map 仍必须保留已有容量。
                gpu.epoch = u32::MAX;
            }
            let mut ids = (0..count).map(|i| ((i * 17 + generation * 7) % context_rows) as u32).collect::<Vec<_>>();
            // 两个被回滚的 token 最后提交，确保命中其旧槽位才能暴露失效缺陷。
            ids[count - 2] = 129;
            ids[count - 1] = 130;
            ids[count - 3] = ids[0];
            let selection = Buffer::upload(0, unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids.len() * 4) }).unwrap();
            let query_rows = if count == 96 { 3 } else { 1 };
            let (output, _, _, _) = gpu.prepare(0, &mut mirror, capacity, cache.each_ref(), &selection, query_rows, count / query_rows, context_rows, recent_rows).unwrap();
            for i in 0..3 {
                let mut actual = vec![0; output[i].bytes()];
                output[i].copy_to_host(&mut actual).unwrap();
                for (row, &token) in ids.iter().enumerate() {
                    let cols = columns[i];
                    assert_eq!(&actual[row * cols..(row + 1) * cols], &expected[i][token as usize * cols..(token as usize + 1) * cols], "generation={generation} plane={i} row={row} token={token}");
                }
            }
        }
    }
    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn gpu_hot_single_and_multi_query_attention_match_paged() {
        const CONTEXT: usize = 50027;
        const LATENT: usize = 512;
        const ROPE: usize = 64;
        const HEADS: usize = 32;
        const Q: usize = 256;
        const KV: usize = 448;
        let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
        let bytes = |values: Vec<u16>| values.into_iter().flat_map(u16::to_ne_bytes).collect::<Vec<_>>();
        let mut mirror = MlaLayerSerde {
            rows: CONTEXT - 4,
            ownership: super::super::kv_cache::RocmKvOwnership::Full,
            latent_cols: LATENT,
            rope_cols: ROPE,
            latent_group_size: 64,
            latent: (0..CONTEXT * LATENT).map(|i| ((i * 29 + i / LATENT * 7) % 63 + 1) as u8).collect(),
            latent_scales: Some(bytes(vec![bf16(1.0 / 256.0); CONTEXT * (LATENT / 64)])),
            rope: bytes((0..CONTEXT * ROPE).map(|i| bf16(((i * 13 % 127) as f32 - 63.0) / 128.0)).collect()),
        };
        let full = [&mirror.latent, mirror.latent_scales.as_ref().unwrap(), &mirror.rope].map(|v| Buffer::upload(0, v).unwrap());
        let cache_rows = 32704;
        let recent_rows = 64;
        let mut plane = 0;
        let cache = [LATENT, LATENT / 64 * 2, ROPE * 2].map(|cols| {
            let source = [&mirror.latent, mirror.latent_scales.as_ref().unwrap(), &mirror.rope][plane];
            plane += 1;
            let mut data = vec![0xff; (cache_rows + recent_rows) * cols];
            for token in mirror.rows..CONTEXT {
                let slot = cache_rows + token % recent_rows;
                data[slot * cols..(slot + 1) * cols].copy_from_slice(&source[token * cols..(token + 1) * cols]);
            }
            Buffer::upload(0, &data).unwrap()
        });
        let mut gpu = GpuHotCache::new(0, cache_rows).unwrap();
        let weight = Buffer::upload(0, &bytes((0..HEADS * KV * LATENT).map(|i| bf16(((i * 17 % 127) as f32 - 63.0) / 4096.0)).collect())).unwrap();
        let weight_scales = Buffer::upload(0, &bytes(vec![bf16(1.0)])).unwrap();
        let table = Buffer::upload(0, &(0..CONTEXT as u32).flat_map(u32::to_ne_bytes).collect::<Vec<_>>()).unwrap();
        // 首次直接进入多行 verify，选集高度重叠，覆盖同一 kernel 内的重复 miss。
        for query_rows in [4, 1, 2, 1] {
            let query = Buffer::upload_f32(0, &(0..query_rows * HEADS * Q).map(|i| ((i % 31) as f32 - 15.0) / 32.0).collect::<Vec<_>>()).unwrap();
            let width = 2048;
            let ids = (0..query_rows).flat_map(|row| (0..width).map(move |i| if i + 4 >= width { (CONTEXT - query_rows + row + 1 - (width - i)) as u32 } else { ((i * 251) % (CONTEXT - query_rows)) as u32 })).collect::<Vec<_>>();
            let selection = Buffer::upload(0, &ids.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>()).unwrap();
            let (compact, compact_table, compact_indices, compact_rows) = gpu.prepare(0, &mut mirror, 65536, cache.each_ref(), &selection, query_rows, width, CONTEXT, recent_rows).unwrap();
            // 未填槽位用 NaN 字节污染，重复 miss 不能从同一 kernel 正在填的
            // scale cache line 读到旧内容。先逐字节检查，避免小幅输入被 atol 掩盖。
            for (plane, cols) in [LATENT, LATENT / 64 * 2, ROPE * 2].into_iter().enumerate() {
                let mut actual = vec![0; compact[plane].bytes()];
                compact[plane].copy_to_host(&mut actual).unwrap();
                let source = [&mirror.latent, mirror.latent_scales.as_ref().unwrap(), &mirror.rope][plane];
                for (row, &token) in ids.iter().enumerate() {
                    assert_eq!(&actual[row * cols..(row + 1) * cols], &source[token as usize * cols..(token as usize + 1) * cols], "rows={query_rows} plane={plane} row={row} token={token}");
                }
            }
            let element_bytes = if query_rows == 1 { 4 } else { 2 };
            let output_bytes = query_rows * HEADS * Q * element_bytes;
            // 多行契约为 BF16；尾部 guard 直接捕获误写 F32。
            let reference = Buffer::upload(0, &vec![0xa5; output_bytes * 2]).unwrap();
            let actual = Buffer::upload(0, &vec![0xa5; output_bytes * 2]).unwrap();
            for (data, table, indices, rows, output) in [(full.each_ref(), &table, Some(&selection), CONTEXT, &reference), (compact.each_ref().map(Arc::as_ref), compact_table.as_ref(), compact_indices.as_deref(), compact_rows, &actual)] {
                ops::hip::try_paged_mla_attention_ct_into(
                    0,
                    &query,
                    data[0],
                    Some(data[1]),
                    64,
                    data[2],
                    table,
                    indices,
                    ops::hip::CtMlaWeightRef { packed: &weight, scales: &weight_scales, rows: HEADS * KV, cols: LATENT, group_size: LATENT, scale_dtype: 0, bits: 16 },
                    query_rows,
                    rows,
                    rows - query_rows,
                    HEADS * Q,
                    HEADS,
                    ROPE,
                    width,
                    1,
                    output,
                    None,
                )
                .unwrap();
            }
            let decode = |output: &Buffer| {
                let mut data = vec![0; output_bytes * 2];
                output.copy_to_host(&mut data).unwrap();
                assert!(data[output_bytes..].iter().all(|&byte| byte == 0xa5), "MLA 输出越界: rows={query_rows} element_bytes={element_bytes}");
                data[..output_bytes].chunks_exact(element_bytes).map(|v| if element_bytes == 4 { f32::from_ne_bytes(v.try_into().unwrap()) } else { f32::from_bits(u32::from(u16::from_ne_bytes(v.try_into().unwrap())) << 16) }).collect::<Vec<_>>()
            };
            for (index, (actual, expected)) in decode(&actual).into_iter().zip(decode(&reference)).enumerate() {
                assert!(actual.is_finite() && (actual - expected).abs() <= 1e-2 + 1e-2 * expected.abs(), "rows={query_rows} element={index} actual={actual} expected={expected}");
            }
        }
    }
}
