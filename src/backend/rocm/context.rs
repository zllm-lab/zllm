use super::*;

impl crate::backend::MemoryPool for RocmContext {
    type Memory = Arc<ops::hip::DeviceBuffer>;

    fn allocate_memory(&self, request: crate::backend::MemoryRequest) -> Result<Self::Memory, BackendError> {
        let request = request.validate()?;
        if request.bytes == 0 {
            return Err(compute_error("ROCm memory pool 申请字节数不能为 0"));
        }
        if request.alignment != 1 {
            return Err(compute_error(format!("ROCm memory pool 尚不支持 alignment={}", request.alignment)));
        }
        let buffer = match request.lifetime {
            crate::backend::MemoryLifetime::Operation | crate::backend::MemoryLifetime::Stage => ops::hip::DeviceBuffer::allocate_reusable(self.device_id, request.bytes),
            crate::backend::MemoryLifetime::Session | crate::backend::MemoryLifetime::Model => ops::hip::DeviceBuffer::allocate_cache(self.device_id, request.bytes),
        }
        .map_err(compute_error)?;
        Ok(Arc::new(buffer))
    }

    fn memory_bytes(&self, memory: &Self::Memory) -> u64 {
        memory.allocation_bytes() as u64
    }
}

impl RocmContext {
    pub(crate) fn profile_stage_begin(&self, detailed_eligible: bool) -> Result<(), BackendError> {
        ops::hip::device_profile_stage_begin(self.device_id, detailed_eligible).map_err(compute_error)
    }

    pub(crate) fn profile_stage_end(&self) -> Result<(), BackendError> {
        ops::hip::device_profile_stage_end(self.device_id).map_err(compute_error)
    }

    pub(crate) fn profile_scope_begin(&self, label: &'static str) -> Result<(), BackendError> {
        ops::hip::device_profile_scope_begin(self.device_id, label).map_err(compute_error)
    }

    pub(crate) fn profile_scope_end(&self) -> Result<(), BackendError> {
        ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)
    }

    /// 提交线程绑定到 `ZLLM_ROCM_SUBMIT_CPUS` 指定的 CPU 列表（一次设置）。
    /// gfx1100 实测跨 NUMA 节点的 hipLaunchKernel 约 2.4µs、本节点约 1.0µs；
    /// 目标 8-GPU 机器上全部以 NUMA0(0-31,64-95) 最快。未设置时完全不动。
    pub(crate) fn pin_submission_thread_to_configured_cpus(&self) {
        static CPUS: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let Some(cpus) = CPUS.get_or_init(|| std::env::var("ZLLM_ROCM_SUBMIT_CPUS").ok().filter(|value| !value.trim().is_empty())) else { return };
        if let Err(error) = crate::kernel::cpu::set_current_thread_affinity(cpus) {
            eprintln!("[rocm-submit-affinity] device={} 绑定 {cpus} 失败: {error}", self.device_id);
        }
    }
}

impl crate::backend::StageExecutionBackend for RocmContext {
    type Completion = RocmStageCompletion;

    fn trace_stage_work_begin(&self, label: &str) -> Result<(), BackendError> {
        ops::hip::roctx_stage_work_begin(&format!("zllm device={} {label}", self.device_id)).map_err(compute_error)
    }

    fn trace_stage_work_end(&self) -> Result<(), BackendError> {
        ops::hip::roctx_stage_work_end().map_err(compute_error)
    }

    fn profile_stage_begin(&self, detailed_eligible: bool) -> Result<(), BackendError> {
        if ops::hip::options().kernel_profile || ops::hip::device_profile_enabled() {
            RocmContext::profile_stage_begin(self, detailed_eligible)?;
        }
        Ok(())
    }

    fn profile_stage_end(&self) -> Result<(), BackendError> {
        if ops::hip::options().kernel_profile || ops::hip::device_profile_enabled() {
            RocmContext::profile_stage_end(self)?;
        }
        Ok(())
    }

    fn stage_available_bytes(&self) -> Result<usize, BackendError> {
        ops::hip::device_memory_info(self.device_id).map(|(free, _)| free).map_err(compute_error)
    }

    fn stage_total_bytes(&self) -> Result<usize, BackendError> {
        ops::hip::device_memory_info(self.device_id).map(|(_, total)| total).map_err(compute_error)
    }

    fn activate_stage_submission(&self, kind: crate::backend::StageSubmissionKind) -> Result<(), BackendError> {
        if kind == crate::backend::StageSubmissionKind::Latency {
            ops::hip::device_profile_decode_boundary();
        }
        let stream = match kind {
            crate::backend::StageSubmissionKind::Latency => self.compute_stream,
            crate::backend::StageSubmissionKind::Background if self.compute_stream == 0 => {
                let stream = ops::hip::background_stage_stream(self.device_id).map_err(compute_error)?;
                // stage0 的 embedding/hidden 以及前一轮 cache 可能刚由 latency
                // default stream 产生；background consumer 必须显式接在其后。
                ops::hip::order_stream_after(self.device_id, self.compute_stream, stream).map_err(compute_error)?;
                stream
            }
            crate::backend::StageSubmissionKind::Background => self.compute_stream,
        };
        ops::hip::activate_compute_stream(self.device_id, stream).map_err(compute_error)
    }

    fn supports_concurrent_stage_submissions(&self) -> bool {
        self.compute_stream == 0
    }

    fn pin_submission_thread(&self) {
        self.pin_submission_thread_to_configured_cpus();
    }

    fn max_queued_latency_submissions(&self) -> usize {
        // latency work 共用一条有序 compute stream。默认仍只允许一份在途；C6+
        // 诊断显示 stage 墙钟占用已约 90% 但每批 submit→complete 里约一半是
        // 主机提交，第二份在途可以把下一批的主机提交叠到本批 GPU 尾部之下。
        // 回收批次/完成事件均按提交顺序在每批 record 时切分，两份在途不共享
        // 可写 scratch；>2 没有证据支持，先封顶 4 只做实验对照。
        static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *LIMIT.get_or_init(|| {
            std::env::var("ZLLM_ROCM_DECODE_IN_FLIGHT").ok().and_then(|value| value.parse::<usize>().ok()).filter(|&value| (1..=4).contains(&value)).unwrap_or(1)
        })
    }

    fn begin_stage_submission(&self) -> Result<(), BackendError> {
        ops::hip::begin_stage_buffer_recycle(self.device_id).map_err(compute_error)
    }

    fn abort_stage_submission(&self) -> Result<(), BackendError> {
        ops::hip::abort_stage_buffer_recycle(self.device_id).map_err(compute_error)
    }

    fn record_stage_completion(&self) -> Result<Self::Completion, BackendError> {
        let owner = ops::hip::DeviceCompletion::record(self.device_id).map_err(compute_error)?;
        let peers = super::pair_worker::take_pair_stage_completions(self.device_id);
        Ok(RocmStageCompletion { owner, peers })
    }

    fn stage_completion_ready(&self, completion: &Self::Completion) -> Result<bool, BackendError> {
        if !completion.owner.is_complete().map_err(compute_error)? {
            return Ok(false);
        }
        completion.peers.iter().try_fold(true, |ready, peer| peer.is_complete().map(|peer_ready| ready && peer_ready))
    }

    fn wait_stage_completion(&self, completion: &Self::Completion) -> Result<(), BackendError> {
        completion.owner.wait().map_err(compute_error)?;
        for peer in &completion.peers {
            peer.wait()?;
        }
        Ok(())
    }

    fn retire_ordered_stage_completion(&self, completion: &Self::Completion) -> Result<(), BackendError> {
        completion.owner.retire_ordered();
        for peer in &completion.peers {
            peer.retire_ordered()?;
        }
        Ok(())
    }

    fn finish_stage_session(&self) -> Result<(), BackendError> {
        ops::hip::release_tensor_workspace(self.device_id).map_err(compute_error)
    }
}

impl crate::backend::StageTensorBackend for RocmContext {
    fn stage_label(&self) -> String {
        format!("ROCm device {}", self.device_id)
    }

    fn activate_stage(&self) -> Result<(), BackendError> {
        // submission stream 已由 StageExecutionBackend 在 batch 外层选择；这里只
        // 切换 HIP current device。若重置为 context 默认流，会让 background P2P
        // 与随后在 default stream 执行的首层计算失去依赖。
        ops::hip::set_device(self.device_id).map_err(compute_error)
    }

    fn stage_tensor_from_f32(&self, values: Vec<f32>, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        self.tensor_from_f32_independent(values, rows, cols).map_err(compute_error)
    }

    fn move_tensor_to_stage(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError> {
        self.tensor_on_device(tensor)
    }

    fn move_tensor_to_stage_ordered(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError> {
        self.tensor_on_device_ordered(tensor)
    }

    fn stabilize_stage_tensor(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError> {
        self.tensor_to_stable_deferred(tensor)
    }

    fn compact_stage_tensor(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError> {
        self.tensor_as_bf16(tensor)
    }

    fn warmup_stage(&self) -> Result<(), BackendError> {
        self.warmup_quantized().map_err(compute_error)?;
        ops::hip::warmup_paged_mla(self.device_id).map_err(compute_error)
    }
}

impl crate::backend::DsaStageBackend for RocmContext {
    type DsaSelection = RocmDsaSelection;

    fn new_stage_cache(&self, layer_count: usize, max_seq_len: usize) -> Result<Self::Cache, BackendError> {
        Ok(RocmKvCache::with_capacity(layer_count, max_seq_len))
    }

    fn new_stage_dsa(&self, layer_count: usize, max_seq_len: usize, head_dim: usize, top_k: usize) -> Result<Self::DsaState, BackendError> {
        RocmDsaState::new(layer_count, max_seq_len, head_dim, top_k).map_err(compute_error)
    }

    fn set_stage_decode_parallelism(&self, dsa: &mut Self::DsaState, sessions: usize) {
        dsa.decode_parallelism = sessions.max(1);
    }

    fn set_stage_cache_decode(&self, cache: &mut Self::Cache, decode: bool) -> Result<(), BackendError> {
        cache.set_stage_decode(decode)
    }

    fn truncate_stage_state(&self, cache: &mut Self::Cache, dsa: &mut Self::DsaState, rows: usize) -> Result<(), BackendError> {
        cache.truncate_rows(rows)?;
        dsa.truncate_rows(rows)
    }

    fn stage_cache_allocated_bytes(&self, cache: &Self::Cache, dsa: &Self::DsaState) -> u64 {
        cache.allocated_bytes().saturating_add(dsa.allocated_bytes())
    }

    fn import_stage_selection(&self, state: &mut Self::DsaState, selection: Option<Self::DsaSelection>) -> Result<(), BackendError> {
        state.import_selection(self, selection)
    }

    fn export_stage_selection(&self, state: &Self::DsaState) -> Option<Self::DsaSelection> {
        state.export_selection()
    }

    fn move_selection_to_stage_ordered(&self, selection: Self::DsaSelection) -> Result<Self::DsaSelection, BackendError> {
        selection.move_to_device_ordered(self.device_id)
    }

    fn stabilize_stage_selection(&self, selection: Self::DsaSelection) -> Result<Self::DsaSelection, BackendError> {
        selection.move_to_stable_deferred()
    }
}

impl crate::backend::SegmentedTensorBackend for RocmContext {
    fn argmax_rows_excluding(&self, input: &Self::Tensor, excluded: &[u32]) -> Result<Vec<u32>, BackendError> {
        let input = self.tensor_as_f32(input.clone())?;
        let device = input.device.as_deref().ok_or_else(|| compute_error("ROCm row argmax 缺少 device buffer"))?;
        ops::hip::try_argmax_rows_excluding_resident_f32(self.device_id, device, input.rows, input.cols, excluded).map_err(compute_error)
    }

    fn sample_rows_excluding(&self, input: &Self::Tensor, sampling: &[crate::backend::TokenSampling], excluded: &[u32]) -> Result<Vec<u32>, BackendError> {
        let input = self.tensor_as_f32(input.clone())?;
        let device = input.device.as_deref().ok_or_else(|| compute_error("ROCm row sampling 缺少 device buffer"))?;
        ops::hip::try_sample_top_p_rows_excluding_resident_f32(self.device_id, device, input.rows, input.cols, sampling, excluded).map_err(compute_error)
    }

    fn concat_token_rows(&self, tensors: &[&Self::Tensor]) -> Result<Self::Tensor, BackendError> {
        let profile_started = ops::hip::options().kernel_profile.then(std::time::Instant::now);
        let first = tensors.first().ok_or_else(|| compute_error("ROCm token concat 输入为空"))?;
        if first.rows == 0 || first.cols == 0 {
            return Err(compute_error("ROCm token concat 不接受空 tensor"));
        }
        let _first_device = first.device.as_deref().ok_or_else(|| compute_error("ROCm token concat 缺少 device buffer"))?;
        let row_bytes = first.cols.checked_mul(first.dtype.element_bytes()).ok_or_else(|| compute_error("ROCm token concat 行跨度溢出"))?;
        let mut total_rows = 0usize;
        let mut total_bytes = 0usize;
        for tensor in tensors {
            let device = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm token concat 缺少 device buffer"))?;
            if tensor.rows == 0
                || tensor.cols != first.cols
                || tensor.dtype != first.dtype
                || tensor.layout != first.layout
                || device.device_id() != self.device_id
                || device.bytes() != tensor.rows.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm token concat 大小溢出"))?
            {
                return Err(compute_error(format!("ROCm token concat shape/device 不一致: rows={} cols={} bytes={} device={}", tensor.rows, tensor.cols, device.bytes(), device.device_id())));
            }
            total_rows = total_rows.checked_add(tensor.rows).ok_or_else(|| compute_error("ROCm token concat rows 溢出"))?;
            total_bytes = total_bytes.checked_add(device.bytes()).ok_or_else(|| compute_error("ROCm token concat bytes 溢出"))?;
        }
        // 单个输入已经是目标行序；只共享只读 allocation，不物化等价副本。
        if tensors.len() == 1 {
            return Ok(device_tensor_with_arc(first.device.as_ref().expect("上方已检查 device").clone(), total_rows, first.cols, first.dtype));
        }
        let output = <Self as crate::backend::MemoryPool>::allocate_memory(self, crate::backend::MemoryRequest::new(total_bytes, crate::backend::MemoryKind::Activation, crate::backend::MemoryLifetime::Operation))?;
        let mut offset = 0usize;
        for tensor in tensors {
            let device = tensor.device.as_deref().expect("上方已检查 device");
            output.copy_from_device(offset, device, 0, device.bytes()).map_err(compute_error)?;
            offset += device.bytes();
        }
        if let Some(started) = profile_started {
            ops::hip::synchronize_device(self.device_id, "token row concat profile").map_err(compute_error)?;
            eprintln!("[rocm-kernel] concat-token-rows device={} tensors={} rows={} cols={} bytes={} wall={:.6}s", self.device_id, tensors.len(), total_rows, first.cols, total_bytes, started.elapsed().as_secs_f64());
        }
        Ok(device_tensor_with_arc(output, total_rows, first.cols, first.dtype))
    }

    fn concat_token_rows_reserved(&self, tensors: &[&Self::Tensor], capacity_rows: usize) -> Result<Self::Tensor, BackendError> {
        let first = tensors.first().ok_or_else(|| compute_error("ROCm reserved token concat 输入为空"))?;
        if first.rows == 0 || first.cols == 0 {
            return Err(compute_error("ROCm reserved token concat 不接受空 tensor"));
        }
        let _first_device = first.device.as_deref().ok_or_else(|| compute_error("ROCm reserved token concat 缺少 device buffer"))?;
        let row_bytes = first.cols.checked_mul(first.dtype.element_bytes()).ok_or_else(|| compute_error("ROCm reserved token concat 行跨度溢出"))?;
        let mut total_rows = 0usize;
        for tensor in tensors {
            let device = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm reserved token concat 缺少 device buffer"))?;
            if tensor.rows == 0
                || tensor.cols != first.cols
                || tensor.dtype != first.dtype
                || tensor.layout != first.layout
                || device.device_id() != self.device_id
                || device.bytes() != tensor.rows.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm reserved token concat 大小溢出"))?
            {
                return Err(compute_error(format!("ROCm reserved token concat shape/device 不一致: rows={} cols={} bytes={} device={}", tensor.rows, tensor.cols, device.bytes(), device.device_id())));
            }
            total_rows = total_rows.checked_add(tensor.rows).ok_or_else(|| compute_error("ROCm reserved token concat rows 溢出"))?;
        }
        if total_rows > capacity_rows {
            return Err(compute_error(format!("ROCm reserved token concat rows={total_rows} 超过 capacity={capacity_rows}")));
        }
        let used_bytes = total_rows.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm reserved token concat used bytes 溢出"))?;
        let capacity_bytes = capacity_rows.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm reserved token concat capacity bytes 溢出"))?;
        let owner = if tensors.len() == 2 {
            let prefix = tensors[0].device.as_deref().expect("上方已检查 device");
            prefix.prefix_capacity_owner().filter(|owner| owner.bytes() >= capacity_bytes)
        } else {
            None
        };
        let owner = match owner {
            Some(owner) => {
                let suffix = tensors[1].device.as_deref().expect("上方已检查 suffix device");
                owner.copy_from_device(tensors[0].rows * row_bytes, suffix, 0, suffix.bytes()).map_err(compute_error)?;
                owner
            }
            None => {
                let owner = <Self as crate::backend::MemoryPool>::allocate_memory(self, crate::backend::MemoryRequest::new(capacity_bytes, crate::backend::MemoryKind::Activation, crate::backend::MemoryLifetime::Stage))?;
                let mut offset = 0usize;
                for tensor in tensors {
                    let device = tensor.device.as_deref().expect("上方已检查 device");
                    owner.copy_from_device(offset, device, 0, device.bytes()).map_err(compute_error)?;
                    offset += device.bytes();
                }
                owner
            }
        };
        let view = ops::hip::DeviceBuffer::view(owner, 0, used_bytes).map_err(compute_error)?;
        Ok(device_tensor_with_dtype(view, total_rows, first.cols, first.dtype))
    }

    fn slice_token_rows(&self, tensor: &Self::Tensor, row_start: usize, rows: usize) -> Result<Self::Tensor, BackendError> {
        if rows == 0 || row_start.checked_add(rows).is_none_or(|end| end > tensor.rows) {
            return Err(compute_error(format!("ROCm token slice 越界: start={row_start} rows={rows} tensor_rows={}", tensor.rows)));
        }
        let owner = tensor.device.as_ref().cloned().ok_or_else(|| compute_error("ROCm token slice 缺少 device buffer"))?;
        if owner.device_id() != self.device_id {
            return Err(compute_error(format!("ROCm token slice device={}，当前 device={}", owner.device_id(), self.device_id)));
        }
        let row_bytes = tensor.cols.checked_mul(tensor.dtype.element_bytes()).ok_or_else(|| compute_error("ROCm token slice 行跨度溢出"))?;
        if owner.bytes() != tensor.rows.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm token slice tensor 大小溢出"))? {
            return Err(compute_error(format!("ROCm token slice dtype={:?} layout={:?} buffer={} 与 shape=[{},{}] 不匹配", tensor.dtype, tensor.layout, owner.bytes(), tensor.rows, tensor.cols)));
        }
        let offset = row_start.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm token slice offset 溢出"))?;
        let bytes = rows.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm token slice bytes 溢出"))?;
        let owner = if offset == 0 { owner.prefix_capacity_owner().unwrap_or(owner) } else { owner };
        let view = ops::hip::DeviceBuffer::view(owner, offset, bytes).map_err(compute_error)?;
        let mut output = device_tensor_with_dtype(view, rows, tensor.cols, tensor.dtype);
        if let Some(replica) = tensor.replica.as_ref() {
            let replica_row_bytes = tensor.cols.checked_mul(replica.dtype.element_bytes()).ok_or_else(|| compute_error("ROCm token replica slice 行跨度溢出"))?;
            let replica_expected = tensor.rows.checked_mul(replica_row_bytes).ok_or_else(|| compute_error("ROCm token replica slice tensor 大小溢出"))?;
            if replica.device.bytes() != replica_expected {
                return Err(compute_error(format!("ROCm token replica slice dtype={:?} buffer={} 与 shape=[{},{}] 不匹配", replica.dtype, replica.device.bytes(), tensor.rows, tensor.cols)));
            }
            let replica_offset = row_start.checked_mul(replica_row_bytes).ok_or_else(|| compute_error("ROCm token replica slice offset 溢出"))?;
            let replica_bytes = rows.checked_mul(replica_row_bytes).ok_or_else(|| compute_error("ROCm token replica slice bytes 溢出"))?;
            let replica_owner = if replica_offset == 0 { replica.device.prefix_capacity_owner().unwrap_or_else(|| replica.device.clone()) } else { replica.device.clone() };
            let replica_view = ops::hip::DeviceBuffer::view(replica_owner, replica_offset, replica_bytes).map_err(compute_error)?;
            output.replica = Some(RocmTensorReplica { device_id: replica.device_id, dtype: replica.dtype, device: Arc::new(replica_view) });
        }
        Ok(output)
    }
}

impl BackendResources for RocmContext {
    type Tensor = RocmTensor;
    type Weight = RocmWeight;
    type Cache = RocmKvCache;
    type LayerScope<'a>
        = ()
    where
        Self: 'a;

    fn layer_scope(&self) -> Self::LayerScope<'_> {
        ()
    }

    fn token_rows(&self, tensor: &Self::Tensor) -> usize {
        tensor.rows
    }

    fn token_cols(&self, tensor: &Self::Tensor) -> usize {
        tensor.cols
    }

    fn tensor_allocated_bytes(&self, tensor: &Self::Tensor) -> u64 {
        tensor.device.as_deref().map_or_else(|| tensor.data.capacity().saturating_mul(std::mem::size_of::<f32>()) as u64, |buffer| buffer.allocation_bytes() as u64)
    }

    fn begin_batch(&self) {}

    fn finish_batch(&self) {
        if ops::hip::options().release_batch_workspace {
            let _ = ops::hip::release_tensor_workspace(self.device_id);
        }
    }

    fn finish_stream_chunk(&self) {
        let _ = ops::hip::release_tensor_workspace(self.device_id);
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        ops::hip::synchronize_device(self.device_id, "ROCm profile synchronize").map_err(compute_error)
    }

    fn profile_device_operator(&self, label: &'static str) -> Result<(), BackendError> {
        ops::hip::device_profile_operator(self.device_id, label).map_err(compute_error)
    }

    fn prepare_grouped_block_fp8(&self, matrix: &crate::weight::format::block_fp8::BlockFp8Matrix, groups: usize, rows_per_group: usize) -> Result<Vec<Self::Weight>, BackendError> {
        if groups == 0 || matrix.rows != groups.checked_mul(rows_per_group).ok_or_else(|| compute_error("ROCm grouped BlockFP8 rows 溢出"))? {
            return Err(compute_error(format!("ROCm grouped BlockFP8 shape=[{},{}] groups={groups} rows={rows_per_group}", matrix.rows, matrix.cols)));
        }
        let (block_rows, block_cols) = matrix.block_shape();
        if !rows_per_group.is_multiple_of(block_rows) {
            return Err(compute_error(format!("ROCm grouped BlockFP8 rows={rows_per_group} 不是 block_rows={block_rows} 的倍数")));
        }
        let code_bytes = rows_per_group.checked_mul(matrix.cols).ok_or_else(|| compute_error("ROCm grouped BlockFP8 code bytes 溢出"))?;
        let scale_bytes = (rows_per_group / block_rows).checked_mul(matrix.cols.div_ceil(block_cols)).ok_or_else(|| compute_error("ROCm grouped BlockFP8 scale bytes 溢出"))?;
        let code_owner = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, matrix.codes()).map_err(compute_error)?);
        let scale_owner = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, matrix.scales()).map_err(compute_error)?);
        (0..groups)
            .map(|group| {
                let codes = Arc::new(ops::hip::DeviceBuffer::view(code_owner.clone(), group * code_bytes, code_bytes).map_err(compute_error)?);
                let scales = Arc::new(ops::hip::DeviceBuffer::view(scale_owner.clone(), group * scale_bytes, scale_bytes).map_err(compute_error)?);
                Ok(RocmWeight {
                    rows: rows_per_group,
                    cols: matrix.cols,
                    inner: RocmWeightInner::Quantized(RocmQuantizedWeight::BlockFp8 { codes, scales, block_rows, block_cols, bf16_cache: Arc::default() }),
                    expert_gguf: None,
                    cpu_mla_data: None,
                })
            })
            .collect()
    }

    fn prepare_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        let expected = checked_elements(rows, cols, "ROCm weight")?;
        if let LinearWeight::Quantized(QuantizedMatrixRef::W8A16(matrix)) = weight
            && let Some(group_size) = matrix.convrot_group_size()
        {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute_error(format!("ROCm INT8 ConvRot weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            let packed = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, matrix.packed()).map_err(compute_error)?);
            let scales = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, matrix.scales()).map_err(compute_error)?);
            return Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Quantized(RocmQuantizedWeight::ConvRotInt8 { packed, scales, group_size }), expert_gguf: None, cpu_mla_data: None });
        }
        let quantized = match weight {
            LinearWeight::Quantized(QuantizedMatrixRef::W4A16(matrix)) => Some((4, matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), matrix.rows, matrix.cols)),
            LinearWeight::Quantized(QuantizedMatrixRef::W8A16(matrix)) => Some((8, matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), matrix.rows, matrix.cols)),
            _ => None,
        };
        if let Some((bits, packed, scales, scale_dtype, group_size, actual_rows, actual_cols)) = quantized {
            if actual_rows != rows || actual_cols != cols {
                return Err(compute_error(format!("ROCm W{bits}A16 weight shape [{actual_rows},{actual_cols}]，期望 [{rows},{cols}]")));
            }
            let packed = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, packed).map_err(compute_error)?);
            let scales = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, scales).map_err(compute_error)?);
            let quantized = if bits == 4 { RocmQuantizedWeight::W4A16 { packed, scales, scale_dtype, group_size } } else { RocmQuantizedWeight::W8A16 { packed, scales, scale_dtype, group_size } };
            return Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Quantized(quantized), expert_gguf: None, cpu_mla_data: None });
        }
        if let LinearWeight::Quantized(QuantizedMatrixRef::Fp8(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute_error(format!("ROCm FP8 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            // GLM 官方 FP8 使用 F32 scale_inv，与 E8M0 Block-FP8 不是同一格式。
            // RDNA3 无原生 FP8 tensor core，驻留时转为 BF16 后复用 WMMA/GEMV 正确路径。
            let values = matrix.decode_bf16();
            let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values.as_slice())) };
            let resident = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, bytes).map_err(compute_error)?);
            return Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Dense { data: Vec::new(), resident: Some(resident), resident_bf16: true, router_bf16: Arc::default() }, expert_gguf: None, cpu_mla_data: None });
        }
        if let LinearWeight::Quantized(QuantizedMatrixRef::BlockFp8(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute_error(format!("ROCm BlockFp8 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            let (block_rows, block_cols) = matrix.block_shape();
            let codes = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, matrix.codes()).map_err(compute_error)?);
            let scales = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, matrix.scales()).map_err(compute_error)?);
            return Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Quantized(RocmQuantizedWeight::BlockFp8 { codes, scales, block_rows, block_cols, bf16_cache: Arc::default() }), expert_gguf: None, cpu_mla_data: None });
        }
        if let LinearWeight::Quantized(QuantizedMatrixRef::Mxfp4(matrix)) = weight {
            if matrix.rows() != rows || matrix.cols() != cols {
                return Err(compute_error(format!("ROCm MXFP4 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows(), matrix.cols())));
            }
            let packed = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, matrix.packed()).map_err(compute_error)?);
            let scales = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, matrix.scales()).map_err(compute_error)?);
            return Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Quantized(RocmQuantizedWeight::Mxfp4 { packed, scales }), expert_gguf: None, cpu_mla_data: None });
        }
        if let LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(matrix)) = weight {
            if matrix.rows != rows || matrix.columns != cols {
                return Err(compute_error(format!("ROCm GGUF weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
            }
            let bytes = matrix.read_bytes().map_err(compute_error)?;
            match matrix.tensor_type.0 {
                // Q8_0(32 元素块 = f16 scale + int8)重排为 CT W8A16 布局(group 32, F16 scale)：
                // 仅做字节拆分与 +128 偏移包装，不做数值反量化；执行侧复用 WMMA 快路径。
                8 => {
                    return self.prepare_gguf_q8_bytes(&bytes, rows, cols);
                }
                // K-quant 保持 GGUF 打包形态常驻。Qwen3.8 的 CPU/ROCm 分层路径
                // 依赖这里避免把 4/5/6 bit 权重展开为 F32（显存会膨胀约 5-8 倍）。
                // linear 与 expert kernel 都直接消费原始 block，不创建 host/device F32 shadow。
                11 | 12 | 13 | 14 | 21 | 23 => return self.prepare_gguf_packed(matrix),
                other => return Err(compute_error(format!("ROCm GGUF type {other} 无设备路径(支持 Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ3_S/IQ4_XS)"))),
            }
        }
        let bf16_bytes = match &weight {
            LinearWeight::Bf16Bytes(bytes) if rows > 1 && cols > 1 => Some(*bytes),
            _ => None,
        };
        let data = match weight {
            LinearWeight::F32(values) => values.to_vec(),
            LinearWeight::F16(values) => values.iter().map(|value| value.to_f32()).collect(),
            LinearWeight::Bf16Bytes(_) if bf16_bytes.is_some() => Vec::new(),
            LinearWeight::Bf16Bytes(bytes) => {
                let expected_bytes = expected.checked_mul(2).ok_or_else(|| compute_error("ROCm BF16 weight 大小溢出"))?;
                if bytes.len() != expected_bytes {
                    return Err(compute_error(format!("ROCm BF16 weight 字节数 {}，期望 {expected_bytes}", bytes.len())));
                }
                bytes.chunks_exact(2).map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32()).collect()
            }
            LinearWeight::Quantized(matrix) => {
                if matrix.rows() != rows || matrix.cols() != cols {
                    return Err(compute_error(format!("{} weight shape [{},{}]，期望 [{rows},{cols}]", matrix.name(), matrix.rows(), matrix.cols())));
                }
                matrix.decode().map_err(compute_error)?
            }
        };
        if bf16_bytes.is_none() && data.len() != expected {
            return Err(compute_error(format!("ROCm weight 元素数 {}，期望 {expected}", data.len())));
        }
        let resident = if let Some(bytes) = bf16_bytes {
            Some(Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, bytes).map_err(compute_error)?))
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * std::mem::size_of::<f32>()) };
            Some(Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, bytes).map_err(compute_error)?))
        };
        Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Dense { data, resident, resident_bf16: bf16_bytes.is_some(), router_bf16: Arc::default() }, expert_gguf: None, cpu_mla_data: None })
    }

    fn prepare_expert_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        if let LinearWeight::Quantized(QuantizedMatrixRef::Gguf(matrix)) = weight
            && matrix.tensor_type.0 == 8
        {
            if matrix.rows != rows || matrix.columns != cols {
                return Err(compute_error(format!("ROCm GGUF expert weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
            }
            let packed = self.prepare_gguf_packed(matrix)?;
            let (codes, tensor_type) = packed.expert_gguf().expect("GGUF packed expert sidecar 已准备");
            let mut weight = self.prepare_weight(weight, rows, cols)?;
            weight.expert_gguf = Some((codes.clone(), tensor_type));
            return Ok(weight);
        }
        self.prepare_weight(weight, rows, cols)
    }

    fn prepare_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        let expected = checked_elements(rows, cols, "ROCm F32 weight")?;
        if values.len() != expected {
            return Err(compute_error(format!("ROCm F32 weight 元素数 {}，期望 {expected}", values.len())));
        }
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * std::mem::size_of::<f32>()) };
        let resident = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, bytes).map_err(compute_error)?);
        Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Dense { data: values.to_vec(), resident: Some(resident), resident_bf16: false, router_bf16: Arc::default() }, expert_gguf: None, cpu_mla_data: None })
    }
}

impl RocmContext {
    pub(crate) fn prepare_gguf_q8_bytes(&self, bytes: &[u8], rows: usize, cols: usize) -> Result<RocmWeight, BackendError> {
        if cols % 32 != 0 || cols % 4 != 0 {
            return Err(compute_error(format!("ROCm GGUF Q8_0 columns={cols} 不是 32 的倍数")));
        }
        let blocks = cols / 32;
        if bytes.len() != rows * blocks * 34 {
            return Err(compute_error(format!("ROCm GGUF Q8_0 字节数 {}，期望 {}", bytes.len(), rows * blocks * 34)));
        }
        let mut packed = vec![0_u8; rows * cols];
        let mut scales = Vec::with_capacity(rows * blocks * 2);
        for row in 0..rows {
            let source = &bytes[row * blocks * 34..][..blocks * 34];
            for block in 0..blocks {
                let source = &source[block * 34..][..34];
                scales.extend_from_slice(&source[..2]);
                let destination = &mut packed[row * cols + block * 32..][..32];
                for (destination, quant) in destination.iter_mut().zip(&source[2..]) {
                    *destination = quant.wrapping_add(128);
                }
            }
        }
        let packed = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, &packed).map_err(compute_error)?);
        let scales = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, &scales).map_err(compute_error)?);
        Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Quantized(RocmQuantizedWeight::W8A16 { packed, scales, scale_dtype: ScaleDType::F16, group_size: 32 }), expert_gguf: None, cpu_mla_data: None })
    }

    /// Q8_0 的 32 列 block 与 MLA head 分界对齐；逐行抽取目标 block 后直接
    /// 转成 W8A16 resident，避免为 o_proj 在两卡各保留一份完整权重。
    pub(crate) fn prepare_gguf_q8_column_shard(&self, matrix: &crate::weight::container::gguf::GgufMatrix, range: std::ops::Range<usize>) -> Result<RocmWeight, BackendError> {
        if matrix.tensor_type.0 != 8 || range.start >= range.end || range.end > matrix.columns || !range.start.is_multiple_of(32) || !range.end.is_multiple_of(32) {
            return Err(compute_error(format!("ROCm GGUF Q8_0 column shard={range:?}/{} 非法", matrix.columns)));
        }
        let source = matrix.read_bytes().map_err(compute_error)?;
        let source_row_bytes = matrix.columns / 32 * 34;
        let start = range.start / 32 * 34;
        let row_bytes = range.len() / 32 * 34;
        let mut shard = Vec::with_capacity(matrix.rows * row_bytes);
        for row in 0..matrix.rows {
            let offset = row * source_row_bytes + start;
            shard.extend_from_slice(&source[offset..offset + row_bytes]);
        }
        self.prepare_gguf_q8_bytes(&shard, matrix.rows, range.len())
    }

    /// Q8_0 每个输出行独立连续，直接只上传目标行；operator peer 不应为了
    /// 一个 head 半片先常驻完整 q_b 再建立 view。
    pub(crate) fn prepare_gguf_q8_row_shard(&self, matrix: &crate::weight::container::gguf::GgufMatrix, range: std::ops::Range<usize>) -> Result<RocmWeight, BackendError> {
        if matrix.tensor_type.0 != 8 || range.start >= range.end || range.end > matrix.rows || !matrix.columns.is_multiple_of(32) {
            return Err(compute_error(format!("ROCm GGUF Q8_0 row shard={range:?}/{} 非法", matrix.rows)));
        }
        let source = matrix.read_bytes().map_err(compute_error)?;
        let row_bytes = matrix.columns / 32 * 34;
        let start = range.start * row_bytes;
        let end = range.end * row_bytes;
        self.prepare_gguf_q8_bytes(&source[start..end], range.len(), matrix.columns)
    }

    pub(crate) fn prepare_gguf_packed_bytes(&self, bytes: &[u8], tensor_type: u32, rows: usize, cols: usize) -> Result<RocmWeight, BackendError> {
        let block_bytes = match tensor_type {
            8 => 272,
            11 | 21 => 110,
            12 => 144,
            13 => 176,
            14 => 210,
            23 => 136,
            other => return Err(compute_error(format!("ROCm GGUF packed type {other} 不支持"))),
        };
        if cols % 256 != 0 || bytes.len() != rows * (cols / 256) * block_bytes {
            return Err(compute_error(format!("ROCm GGUF packed shape/type 不匹配: rows={rows}, cols={cols}, bytes={}", bytes.len())));
        }
        let codes = Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, bytes).map_err(compute_error)?);
        let q8_gemv = if tensor_type == 8 {
            let repacked = ops::hip::repack_q8_0_rows(bytes, rows, cols).map_err(compute_error)?;
            Some(Arc::new(ops::hip::DeviceBuffer::upload(self.device_id, &repacked).map_err(compute_error)?))
        } else {
            None
        };
        Ok(RocmWeight { rows, cols, inner: RocmWeightInner::Quantized(RocmQuantizedWeight::GgufPacked { codes, tensor_type, q8_gemv }), expert_gguf: None, cpu_mla_data: None })
    }

    pub(crate) fn prepare_gguf_packed(&self, matrix: &crate::weight::container::gguf::GgufMatrix) -> Result<RocmWeight, BackendError> {
        let rows = matrix.rows;
        let cols = matrix.columns;
        let tensor_type = matrix.tensor_type.0;
        let bytes = matrix.read_bytes().map_err(compute_error)?;
        self.prepare_gguf_packed_bytes(&bytes, tensor_type, rows, cols)
    }
}
