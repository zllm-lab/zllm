//! GPU 选集对应的 RAM 历史注册与热槽元数据；模型与持久化格式保持独立。

use super::kv_cache::MlaLayerSerde;
use super::{BackendError, compute_error, ops};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

type Buffer = ops::hip::DeviceBuffer;
type RegisteredPlanes = [ops::hip::RegisteredHostBuffer; 3];

// 只限制提前注册；繁忙时下一轮再尝试，不为每一层创建排队线程。
static REGISTRATION_PREFETCH_WORKERS: AtomicUsize = AtomicUsize::new(0);

pub(super) struct GpuHotCache {
    // 注册借用外层 mirror 的 allocation，外层必须先销毁本结构再销毁 mirror。
    registered: Option<RegisteredPlanes>,
    registered_rows: usize,
    registration_prefetch: Option<(usize, std::thread::JoinHandle<Result<RegisteredPlanes, String>>)>,
    registration_prefetch_disabled: bool,
    metadata: Option<Buffer>,
    map_rows: usize,
    epoch: u32,
    cache_rows: usize,
    stream: usize,
    trace_counts: Option<Buffer>,
}

impl Drop for GpuHotCache {
    fn drop(&mut self) {
        // 后台注册借用 mirror，必须在外层释放 allocation 之前收回并注销。
        if let Some((_, worker)) = self.registration_prefetch.take() {
            let _ = worker.join();
        }
    }
}

impl GpuHotCache {
    pub(super) fn metadata_reservation_bytes(&self, rows: usize) -> Result<usize, BackendError> {
        if rows <= self.map_rows {
            return Ok(0);
        }
        let map_rows = rows.checked_next_power_of_two().ok_or_else(|| compute_error("GPU hot token map 溢出"))?;
        Ok((map_rows + self.cache_rows * 2 + 1) * 4)
    }

    pub(super) fn reserve_metadata(&mut self, device_id: i32, rows: usize) -> Result<(), BackendError> {
        if rows <= self.map_rows && self.epoch != 0 {
            return Ok(());
        }
        let map_rows = rows.checked_next_power_of_two().ok_or_else(|| compute_error("GPU hot token map 溢出"))?.max(self.map_rows);
        let mut words = vec![0_u32; map_rows + self.cache_rows * 2 + 1];
        words[..map_rows + self.cache_rows].fill(u32::MAX);
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 4) };
        let metadata = Buffer::allocate_cache(device_id, bytes.len()).map_err(compute_error)?;
        metadata.copy_from_host(bytes).map_err(compute_error)?;
        if let Some(previous) = &self.metadata {
            metadata.copy_from_device(0, previous, 0, self.map_rows * 4).map_err(compute_error)?;
            let tail_words = if self.epoch == 0 { self.cache_rows } else { self.cache_rows * 2 + 1 };
            metadata.copy_from_device(map_rows * 4, previous, self.map_rows * 4, tail_words * 4).map_err(compute_error)?;
        }
        self.metadata = Some(metadata);
        self.map_rows = map_rows;
        // 准入只分配元数据，不注册历史，也不改变 Full MLA 的执行路径。
        self.epoch = self.epoch.max(1);
        Ok(())
    }

    pub(super) fn new(_device_id: i32, cache_rows: usize) -> Result<Self, BackendError> {
        if cache_rows == 0 || cache_rows > u32::MAX as usize {
            return Err(compute_error(format!("GPU hot cache_rows={cache_rows} 非法")));
        }
        Ok(Self { registered: None, registered_rows: 0, registration_prefetch: None, registration_prefetch_disabled: false, metadata: None, map_rows: 0, epoch: 0, cache_rows, stream: 0, trace_counts: None })
    }

    pub(super) fn register_history(&mut self, device_id: i32, mirror: &mut MlaLayerSerde, capacity: usize, required_rows: usize) -> Result<(), BackendError> {
        self.finish_registration_prefetch(required_rows > self.registered_rows);
        if self.registered_rows >= required_rows {
            self.prefetch_registration(device_id, mirror, capacity, required_rows)?;
            return Ok(());
        }
        let started = std::time::Instant::now();
        let old_rows = self.registered_rows;
        // 只注册新增的 4096 行页；保留旧前缀，避免越过边界时重映射全部历史。
        // Vec 一开始预留逻辑容量，后续追加不 realloc。
        let previous = self.registered.take().map(|planes| planes.map(Some)).unwrap_or([None, None, None]);
        self.registered_rows = 0;
        let mut unregister_elapsed = std::time::Duration::ZERO;
        let rows = required_rows.div_ceil(4096).saturating_mul(4096).min(capacity);
        let row_bytes = [mirror.latent_cols, mirror.latent_cols / mirror.latent_group_size * 2, mirror.rope_cols * 2];
        let buffers = [&mut mirror.latent, mirror.latent_scales.as_mut().expect("GPU hot 必有 scales"), &mut mirror.rope];
        let mut registered = Vec::with_capacity(3);
        let mut prepare_elapsed = std::time::Duration::ZERO;
        let mut register_elapsed = std::time::Duration::ZERO;
        let mut register_bytes = 0;
        let mut plane_register_ms = [0.0; 3];
        for (plane, ((buffer, row_bytes), previous)) in buffers.into_iter().zip(row_bytes).zip(previous).enumerate() {
            let prepare_started = std::time::Instant::now();
            let reserved = capacity.checked_mul(row_bytes).ok_or_else(|| compute_error("GPU hot host capacity 溢出"))?;
            if previous.is_some() && buffer.capacity() < reserved {
                return Err(compute_error(format!("GPU hot 已注册 allocation 不能扩容: device={device_id} plane={plane} capacity_bytes={} required_bytes={reserved}", buffer.capacity())));
            }
            buffer.reserve_exact(reserved.saturating_sub(buffer.len()));
            let bytes = rows.checked_mul(row_bytes).ok_or_else(|| compute_error("GPU hot registered bytes 溢出"))?;
            // 先实写将要注册的尾页，避免后续 append 首次写入已经映射的零页。
            // 只初始化 spare capacity，逻辑长度仍由完成的 mirror 回传推进。
            let spare_bytes = bytes.saturating_sub(buffer.len());
            buffer.spare_capacity_mut()[..spare_bytes].fill(std::mem::MaybeUninit::new(0));
            prepare_elapsed += prepare_started.elapsed();
            // allocation 由外层 mirror 拥有，容量固定；guard 的析构先等待设备。
            let register_started = std::time::Instant::now();
            let registration = (|| {
                if let Some(mut previous) = previous {
                    let added = bytes.saturating_sub(previous.bytes());
                    if unsafe { previous.try_extend(bytes) }? {
                        register_bytes += added;
                        return Ok(previous);
                    }
                    // 某些设备的相邻 host 段映射不连续，保持原有整段注册路径。
                    let unregister_started = std::time::Instant::now();
                    drop(previous);
                    unregister_elapsed += unregister_started.elapsed();
                }
                register_bytes += bytes;
                unsafe { ops::hip::RegisteredHostBuffer::register(device_id, buffer.as_mut_ptr(), bytes) }
            })()
            .map_err(|error| {
                let error = format!(
                    "GPU hot history device={device_id} plane={plane} previous_rows={old_rows} required_rows={required_rows} registered_rows={rows} capacity_rows={capacity} len_bytes={} capacity_bytes={}: {error}",
                    buffer.len(),
                    buffer.capacity()
                );
                // 失败后清理大量注册可能很慢，先保留原始错误再析构已注册平面。
                eprintln!("[mla-host-register-error] {error}");
                compute_error(error)
            })?;
            registered.push(registration);
            let elapsed = register_started.elapsed();
            register_elapsed += elapsed;
            plane_register_ms[plane] = elapsed.as_secs_f64() * 1e3;
        }
        self.registered = Some(registered.try_into().map_err(|_| compute_error("GPU hot history 注册数量异常"))?);
        self.registered_rows = rows;
        let elapsed = started.elapsed();
        if elapsed.as_millis() >= 20 {
            // 冷恢复同时可能复制 Vec、实写尾页及锁定页，不能把总耗时都归因于 HIP。
            eprintln!(
                "[mla-host-register-slow] device={device_id} previous_rows={old_rows} rows={rows} bytes={} register_bytes={register_bytes} wall_ms={:.3} prepare_ms={:.3} register_ms={:.3} unregister_ms={:.3} plane_register_ms={plane_register_ms:.3?}",
                rows * row_bytes.iter().sum::<usize>(),
                elapsed.as_secs_f64() * 1e3,
                prepare_elapsed.as_secs_f64() * 1e3,
                register_elapsed.as_secs_f64() * 1e3,
                unregister_elapsed.as_secs_f64() * 1e3
            );
        }
        self.prefetch_registration(device_id, mirror, capacity, required_rows)?;
        Ok(())
    }

    fn finish_registration_prefetch(&mut self, wait: bool) {
        if self.registration_prefetch.as_ref().is_none_or(|(_, worker)| !wait && !worker.is_finished()) {
            return;
        }
        let (rows, worker) = self.registration_prefetch.take().expect("prefetch 已检查");
        match worker.join().map_err(|_| "主存尾页注册线程 panic".to_owned()).and_then(|result| result) {
            Ok(tail) => {
                let previous = self.registered.as_mut().expect("prefetch 期间旧映射保持所有权");
                if previous.iter().zip(&tail).all(|(old, new)| old.is_contiguous_tail(new)) {
                    for (old, new) in previous.iter_mut().zip(tail) {
                        assert!(old.join_tail(new));
                    }
                    self.registered_rows = rows;
                } else {
                    // 连续地址不是 HIP 契约；不拼接时保留旧映射，越界才走同步恢复。
                    self.registration_prefetch_disabled = true;
                    eprintln!("[mla-host-register-prefetch-fallback] rows={rows} 新尾段设备地址不连续");
                }
            }
            Err(error) => {
                self.registration_prefetch_disabled = true;
                eprintln!("[mla-host-register-prefetch-fallback] rows={rows}: {error}");
            }
        }
    }

    fn prefetch_registration(&mut self, device_id: i32, mirror: &mut MlaLayerSerde, capacity: usize, required_rows: usize) -> Result<(), BackendError> {
        if self.registration_prefetch_disabled || self.registration_prefetch.is_some() || self.registered_rows == 0 || self.registered_rows >= capacity || required_rows.saturating_add(1024) < self.registered_rows {
            return Ok(());
        }
        let previous_rows = self.registered_rows;
        let rows = previous_rows.saturating_add(4096).min(capacity);
        let columns = [mirror.latent_cols, mirror.latent_cols / mirror.latent_group_size * 2, mirror.rope_cols * 2];
        let buffers = [&mut mirror.latent, mirror.latent_scales.as_mut().expect("GPU hot 必有 scales"), &mut mirror.rope];
        let mut regions = [(0usize, 0usize); 3];
        let mut offsets = [(0usize, 0usize); 3];
        for (plane, (buffer, width)) in buffers.iter().zip(columns).enumerate() {
            let begin = previous_rows.checked_mul(width).ok_or_else(|| compute_error("主存预注册起点溢出"))?;
            let end = rows.checked_mul(width).ok_or_else(|| compute_error("主存预注册终点溢出"))?;
            if end > buffer.capacity() {
                return Err(compute_error(format!("主存预注册不能 realloc: device={device_id} plane={plane} rows={rows} bytes={end} capacity={}", buffer.capacity())));
            }
            regions[plane] = ((buffer.as_ptr() as usize).checked_add(begin).ok_or_else(|| compute_error("主存预注册地址溢出"))?, end - begin);
            offsets[plane] = (begin.max(buffer.len()) - buffer.len(), end.saturating_sub(buffer.len()));
        }
        if REGISTRATION_PREFETCH_WORKERS.fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| (count < 4).then_some(count + 1)).is_err() {
            return Ok(());
        }
        // 只实写尚未进入逻辑历史的下一页；已有前缀和已完成的 mirror 行不再改写。
        for (buffer, (begin, end)) in buffers.into_iter().zip(offsets) {
            if begin < end {
                buffer.spare_capacity_mut()[begin..end].fill(std::mem::MaybeUninit::new(0));
            }
        }
        let started = std::time::Instant::now();
        let worker = std::thread::Builder::new().name(format!("mla-host-page-{device_id}")).spawn(move || {
            let result: Result<RegisteredPlanes, String> = std::panic::catch_unwind(|| {
                let registered = regions
                    .into_iter()
                    .enumerate()
                    .map(|(plane, (pointer, bytes))| {
                        // mirror 的稳定容量由 GpuHotCache 拥有的 join 生命周期保护。
                        unsafe { ops::hip::RegisteredHostBuffer::register(device_id, pointer as *mut u8, bytes) }.map_err(|error| format!("device={device_id} plane={plane} rows={previous_rows}..{rows}: {error}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                registered.try_into().map_err(|_| "主存尾页注册平面数量错误".to_owned())
            })
            .unwrap_or_else(|_| Err(format!("主存尾页注册线程 panic device={device_id} rows={rows}")));
            REGISTRATION_PREFETCH_WORKERS.fetch_sub(1, Ordering::AcqRel);
            eprintln!(
                "[mla-host-register-prefetch] device={device_id} previous_rows={previous_rows} rows={rows} bytes={} wall_ms={:.3} success={}",
                regions.iter().map(|(_, bytes)| bytes).sum::<usize>(),
                started.elapsed().as_secs_f64() * 1e3,
                result.is_ok()
            );
            result
        });
        match worker {
            Ok(worker) => self.registration_prefetch = Some((rows, worker)),
            Err(error) => {
                REGISTRATION_PREFETCH_WORKERS.fetch_sub(1, Ordering::AcqRel);
                self.registration_prefetch_disabled = true;
                eprintln!("[mla-host-register-prefetch-fallback] device={device_id} rows={rows} 启动失败: {error}");
            }
        }
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
        let count = query_rows.checked_mul(width).filter(|&count| count != 0).ok_or_else(|| compute_error("GPU hot selection 大小非法"))?;
        // 单次 gather 的候选多于槽位时旧 kernel 不填热缓存。拆搬运范围，
        // 让后续范围复用已填好的行，attention 的 query 批次与舍入路径不变。
        let gather_limit = self.cache_rows.saturating_sub(1).max(1);
        let gathers = u32::try_from(count.div_ceil(gather_limit)).ok().filter(|&n| n <= 0x7fff_ffff).ok_or_else(|| compute_error("GPU hot gather epoch 数溢出"))?;
        let previous_epoch = self.epoch;
        // pin 高位区分同 kernel 新填槽；跨 kernel 后才允许命中这些槽。
        self.epoch = self.epoch.checked_add(gathers).filter(|&epoch| epoch <= 0x7fff_ffff).unwrap_or(0);
        if context_rows > self.map_rows || self.epoch == 0 {
            self.reserve_metadata(device_id, context_rows)?;
            self.epoch = self.epoch.max(gathers);
        }
        let columns = [mirror.latent_cols, mirror.latent_cols / mirror.latent_group_size, mirror.rope_cols];
        let (output, identity) = ops::hip::prepare_paged_mla_hot_gather(device_id, count, [columns[0], columns[1] * 2, columns[2] * 2]).map_err(compute_error)?;
        let registered = self.registered.as_ref().ok_or_else(|| compute_error("GPU hot history 为空"))?;
        if ops::hip::options().mla_hot_trace && self.trace_counts.is_none() {
            self.trace_counts = Some(Buffer::upload(device_id, &[0; 12]).map_err(compute_error)?);
        }
        // 只在诊断模式每 128 轮排空采样；该运行不能用于吞吐验收。
        let sample = self.trace_counts.is_some() && self.epoch / 128 != previous_epoch / 128;
        let mut before_counts = [0; 12];
        if sample {
            ops::hip::synchronize_device(device_id, "GPU hot gather sample begin").map_err(compute_error)?;
            self.trace_counts.as_ref().expect("trace 已创建").copy_to_host(&mut before_counts).map_err(compute_error)?;
        }
        let started = sample.then(std::time::Instant::now);
        let gather = || {
            for (part, offset) in (0..count).step_by(gather_limit).enumerate() {
                ops::hip::try_mla_gpu_hot_gather_q8(
                    device_id,
                    registered.each_ref(),
                    cache,
                    self.metadata.as_ref().expect("metadata 已创建"),
                    self.map_rows,
                    self.epoch - gathers + 1 + part as u32,
                    selection,
                    output.each_ref().map(Arc::as_ref),
                    offset,
                    (count - offset).min(gather_limit),
                    columns,
                    self.cache_rows,
                    recent_rows,
                    mirror.rows,
                    context_rows,
                    self.trace_counts.as_ref(),
                )
                .map_err(compute_error)?;
            }
            Ok::<_, BackendError>(())
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
        Ok((output, identity.clone(), (query_rows > 1).then_some(identity), count))
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
    fn gpu_hot_pending_registration_joins_on_drop_and_recovers_error() {
        let (release, pending) = std::sync::mpsc::channel();
        let (finished, done) = std::sync::mpsc::channel();
        let (entered, dropping) = std::sync::mpsc::channel();
        let mut gpu = GpuHotCache::new(0, 32).unwrap();
        gpu.registration_prefetch = Some((
            4096,
            std::thread::spawn(move || {
                pending.recv().unwrap();
                Err("测试注册失败".to_owned())
            }),
        ));
        let dropper = std::thread::spawn(move || {
            entered.send(()).unwrap();
            drop(gpu);
            finished.send(()).unwrap();
        });
        dropping.recv().unwrap();
        assert!(matches!(done.recv_timeout(std::time::Duration::from_millis(30)), Err(std::sync::mpsc::RecvTimeoutError::Timeout)), "后台仍借用 allocation 时不能结束析构");
        release.send(()).unwrap();
        dropper.join().unwrap();
        done.recv().unwrap();

        let mut gpu = GpuHotCache::new(0, 32).unwrap();
        gpu.registered_rows = 4096;
        gpu.registration_prefetch = Some((8192, std::thread::spawn(|| Err("测试提前注册失败".to_owned()))));
        gpu.finish_registration_prefetch(true);
        assert!(gpu.registration_prefetch.is_none());
        assert!(gpu.registration_prefetch_disabled);
        assert_eq!(gpu.registered_rows, 4096, "失败不能宣布新页已可用");
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn gpu_hot_registration_growth_preserves_prefix_and_appended_pages() {
        let columns = [512, 16, 128];
        let capacity = 73728;
        let cache_rows = 32;
        let recent_rows = 4;
        let expected = columns.map(|cols| (0..capacity * cols).map(|i| ((i * 37 + i / cols * 19) % 251) as u8).collect::<Vec<_>>());
        let [latent, scales, rope] = columns.map(|cols| Vec::with_capacity(capacity * cols));
        let mut mirror = MlaLayerSerde { rows: 0, ownership: super::super::kv_cache::RocmKvOwnership::Full, latent_cols: columns[0], rope_cols: columns[2] / 2, latent_group_size: 64, latent, latent_scales: Some(scales), rope };
        let cache = columns.map(|cols| Buffer::upload(0, &vec![0xff; (cache_rows + recent_rows) * cols]).unwrap());
        let mut gpu = GpuHotCache::new(0, cache_rows).unwrap();
        let mut first_pointers = None;
        // 跨过旧 64K 边界，再跨一页；缩短后重用已注册范围并再次追加。
        for rows in [65535, 65537, 69634, 65533, 65539] {
            for (i, buffer) in [&mut mirror.latent, mirror.latent_scales.as_mut().unwrap(), &mut mirror.rope].into_iter().enumerate() {
                if rows < mirror.rows {
                    buffer.truncate(rows * columns[i]);
                } else {
                    buffer.extend_from_slice(&expected[i][mirror.rows * columns[i]..rows * columns[i]]);
                }
            }
            mirror.rows = rows;
            // 每轮强制 miss，防止旧热槽掩盖主存注册损坏。
            gpu.truncate(0, 0, gpu.map_rows).unwrap();
            let ids = [0_u32, 129, 32767, rows as u32 - 2, rows as u32 - 1];
            let selection = Buffer::upload(0, &ids.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>()).unwrap();
            let (output, _, _, _) = gpu.prepare(0, &mut mirror, capacity, cache.each_ref(), &selection, 1, ids.len(), rows, recent_rows).unwrap();
            let pointers = gpu.registered.as_ref().unwrap().each_ref().map(|buffer| buffer.device_pointer());
            if let Some(first) = first_pointers {
                assert_eq!(pointers, first, "连续扩展不应改变旧前缀设备地址");
            } else {
                first_pointers = Some(pointers);
            }
            assert!(gpu.registered_rows >= rows && gpu.registered_rows <= capacity);
            if rows == 65537 {
                assert_eq!(gpu.registered_rows, 69632, "只增加一页，不能注册完整 128K");
            }
            for i in 0..3 {
                let mut actual = vec![0; output[i].bytes()];
                output[i].copy_to_host(&mut actual).unwrap();
                for (row, &token) in ids.iter().enumerate() {
                    let cols = columns[i];
                    assert_eq!(&actual[row * cols..(row + 1) * cols], &expected[i][token as usize * cols..(token as usize + 1) * cols], "rows={rows} plane={i} token={token}");
                }
            }
            if rows == 65535 {
                assert!(gpu.registration_prefetch.is_some(), "页尾前应已启动下一页的后台注册");
                gpu.finish_registration_prefetch(true);
                assert_eq!(gpu.registered_rows, 69632, "边界前只提前一页，不能预留最大输出");
                assert_eq!(gpu.registered.as_ref().unwrap().each_ref().map(|buffer| buffer.device_pointer()), pointers);
            }
        }
    }

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
        for (generation, count) in [96, 24, 96, 12, 12, 24, 24].into_iter().enumerate() {
            if generation == 2 {
                // 同一次多段 gather 跨过 epoch 上界，也必须先清理旧 pin。
                gpu.epoch = 0x7fff_fffe;
            }
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
            if generation == 0 {
                let mut bytes = vec![0; gpu.map_rows * 4];
                gpu.metadata.as_ref().unwrap().copy_to_host(&mut bytes).unwrap();
                assert!(bytes.chunks_exact(4).any(|v| u32::from_le_bytes(v.try_into().unwrap()) < cache_rows as u32), "大于热窗的选集也必须填入后续可复用的行");
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
                data[..output_bytes]
                    .chunks_exact(element_bytes)
                    .map(|v| if element_bytes == 4 { f32::from_ne_bytes(v.try_into().unwrap()) } else { f32::from_bits(u32::from(u16::from_ne_bytes(v.try_into().unwrap())) << 16) })
                    .collect::<Vec<_>>()
            };
            for (index, (actual, expected)) in decode(&actual).into_iter().zip(decode(&reference)).enumerate() {
                assert!(actual.is_finite() && (actual - expected).abs() <= 1e-2 + 1e-2 * expected.abs(), "rows={query_rows} element={index} actual={actual} expected={expected}");
            }
        }
    }
}
