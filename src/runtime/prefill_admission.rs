//! Decode 活跃时的 opportunistic prefill 准入策略。

use super::prefill_scheduler::StageFlowSnapshot;
#[derive(Default)]
pub struct PrefillAdmissionCursor {
    pub(super) session: usize,
    pub(super) submitted: usize,
}

/// decode 未占满 stage 时直接准入一块 prefill；stage 内的执行槽和优先级
/// 决定何时提交。只有 decode 占满流水线时才使用空闲与公平份额预算。
#[derive(Default)]
pub struct OpportunisticPrefillAdmission {
    cursor: PrefillAdmissionCursor,
    observed: StageFlowSnapshot,
    credit_micros: u64,
    prefill_micros_per_unit: u64,
    prefill_micros_per_batch: u64,
    sample_started: Option<StageFlowSnapshot>,
    sample_work_units: usize,
    sample_work_items: usize,
    decode_limited: bool,
    bootstrap_submitted: bool,
}

impl OpportunisticPrefillAdmission {
    const DECODE_FAIR_SHARE_DENOMINATOR: u64 = 32;

    pub fn cursor_mut(&mut self) -> &mut PrefillAdmissionCursor {
        &mut self.cursor
    }

    /// 请求末块可以小于配置分块，不能为凑满分块读越 token 边界。
    pub fn chunk_size(&self, requested: usize, limit: usize, remaining: usize) -> usize {
        requested.min(limit).min(remaining)
    }

    /// 短请求不背负上一块的长期 GPU 时间债，但必须等待上一块在所有 stage
    /// 完整退休；调用方仍只准入一个 work，stage 内继续 decode-first。
    pub fn short_request_ready(&self) -> bool {
        self.decode_limited && self.sample_started.is_none()
    }

    fn observe(&mut self, snapshot: StageFlowSnapshot, ceiling: usize, stage_count: usize) {
        let decode = snapshot.decode_micros.saturating_sub(self.observed.decode_micros);
        let idle = snapshot.idle_micros.saturating_sub(self.observed.idle_micros).saturating_add(snapshot.prefill_idle_micros.saturating_sub(self.observed.prefill_idle_micros));
        self.observed = snapshot;
        if let Some(started) = self.sample_started
            && snapshot.prefill_batches > started.prefill_batches
            && snapshot.prefill_active == 0
        {
            let batches = snapshot.prefill_batches - started.prefill_batches;
            // stage 已经空闲时不再消耗计算资源；等待下一轮 decode 的时间
            // 不能算成 prefill 成本，否则会同时增加债务并剥夺空闲预算。
            let micros = snapshot.prefill_micros.saturating_sub(started.prefill_micros);
            let batch_sample = micros.div_ceil(batches).max(1);
            self.prefill_micros_per_batch = if self.prefill_micros_per_batch == 0 { batch_sample } else { self.prefill_micros_per_batch.saturating_mul(3).saturating_add(batch_sample).div_ceil(4) };
            if batches == u64::try_from(self.sample_work_items).unwrap_or(u64::MAX) && self.sample_work_units != 0 {
                let unit_sample = micros.div_ceil(u64::try_from(self.sample_work_units).unwrap_or(u64::MAX)).max(1);
                self.prefill_micros_per_unit = if self.prefill_micros_per_unit == 0 { unit_sample } else { self.prefill_micros_per_unit.saturating_mul(3).saturating_add(unit_sample).div_ceil(4) };
            }
            self.sample_started = None;
            self.sample_work_units = 0;
            self.sample_work_items = 0;
        }
        // 各 stage idle 求和用于观测真实空泡，但 admission 只消费平均单卡
        // 空泡，避免把错峰历史一次性兑换成多个不可抢占的 prefill work。
        let refill = idle / u64::try_from(stage_count.max(1)).unwrap_or(u64::MAX);
        let refill = refill.saturating_add(decode / Self::DECODE_FAIR_SHARE_DENOMINATOR);
        self.credit_micros = self.credit_micros.saturating_add(refill);
        if self.prefill_micros_per_batch != 0 {
            let variable = self.prefill_micros_per_unit.saturating_mul(u64::try_from(ceiling.max(1)).unwrap_or(u64::MAX));
            let cap = self.prefill_micros_per_batch.max(variable).saturating_mul(2);
            self.credit_micros = self.credit_micros.min(cap);
        }
    }

    /// 返回可在当前时刻立即发射的 prefill 数量和单块上限。0 表示没有已赚取的
    /// GPU 时间预算；调用方继续驱动 decode，不等待也不轮询凑批。
    pub fn limits(&mut self, snapshot: StageFlowSnapshot, pipeline_work_window: usize, stage_count: usize, decode_active: bool, base: usize, ceiling: usize) -> (usize, usize, Option<usize>) {
        let base = base.max(1);
        let ceiling = ceiling.max(base);
        self.observe(snapshot, ceiling, stage_count);
        self.decode_limited = decode_active;
        if !decode_active {
            return (pipeline_work_window, pipeline_work_window, None);
        }
        let mixed_floor = base;
        let work_items = 1;
        if self.sample_started.is_some() {
            return (0, work_items, Some(mixed_floor));
        }
        // 不能把“存在 decode”误当成“全部 stage 已满”。仍只放一块，
        // 防止后台排队淹没 latency 队列；实际设备提交服从 stage 执行槽。
        if snapshot.decode_busy_stages < u64::try_from(stage_count.max(1)).unwrap_or(u64::MAX) {
            return (1, work_items, Some(mixed_floor));
        }
        if self.prefill_micros_per_batch == 0 {
            return (usize::from(!self.bootstrap_submitted), work_items, Some(mixed_floor));
        }
        let fixed = self.prefill_micros_per_batch.saturating_mul(u64::try_from(work_items).unwrap_or(u64::MAX));
        let cost = |chunk: usize| fixed.max(self.prefill_micros_per_unit.saturating_mul(u64::try_from(chunk).unwrap_or(u64::MAX)));
        if self.credit_micros < cost(mixed_floor) {
            return (0, work_items, Some(mixed_floor));
        }
        let mut chunk = mixed_floor;
        while chunk <= ceiling / 2 && cost(chunk.saturating_mul(2)) <= self.credit_micros {
            chunk *= 2;
        }
        (1, work_items, Some(chunk.min(ceiling)))
    }

    pub fn commit_work(&mut self, work_units: usize, work_items: usize) {
        if !self.decode_limited {
            return;
        }
        self.bootstrap_submitted = true;
        self.sample_started = Some(self.observed);
        self.sample_work_units = work_units;
        self.sample_work_items = work_items.max(1);
        eprintln!(
            "[opportunistic-prefill-admit] work_units={work_units} work_items={} credit_ms={:.3} fixed_ms={:.3} unit_us={}",
            self.sample_work_items,
            self.credit_micros as f64 / 1000.0,
            self.prefill_micros_per_batch as f64 / 1000.0,
            self.prefill_micros_per_unit,
        );
        if self.prefill_micros_per_batch == 0 {
            self.credit_micros = 0;
        } else {
            let fixed = self.prefill_micros_per_batch.saturating_mul(u64::try_from(work_items.max(1)).unwrap_or(u64::MAX));
            let variable = self.prefill_micros_per_unit.saturating_mul(u64::try_from(work_units).unwrap_or(u64::MAX));
            self.credit_micros = self.credit_micros.saturating_sub(fixed.max(variable));
        }
    }
}
