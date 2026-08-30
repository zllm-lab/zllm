//! 多请求连续批调度：只决定本轮推进哪些 prefill/decode 请求，不感知模型和 backend。

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestPhase {
    PrefillReady { position: usize, end: usize },
    DecodeReady { position: usize },
    DecodeInFlight { position: usize },
    Pending,
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillSlice {
    pub request: usize,
    pub position: usize,
    pub len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchPlan {
    pub prefill: Vec<PrefillSlice>,
    pub decode: Vec<usize>,
}

/// 在各模型 request state 之外保存公平轮转位置；模型 runtime 执行计划并更新 phase。
pub struct BatchScheduler {
    cursor: usize,
    prefill_token_budget: usize,
    decode_batch_limit: usize,
    initial_batch: usize,
    initial_batch_released: bool,
}

impl BatchScheduler {
    pub fn new(prefill_token_budget: usize, decode_batch_limit: usize) -> Result<Self, String> {
        if prefill_token_budget == 0 || decode_batch_limit == 0 {
            return Err("multiplex batch budget 必须大于 0".to_owned());
        }
        Ok(Self { cursor: 0, prefill_token_budget, decode_batch_limit, initial_batch: 0, initial_batch_released: true })
    }

    /// 初始批在所有成员到达 decode 边界后一起释放。这里只建立一次性边界；释放后
    /// 完全按 completion 驱动，后续请求不会反向阻塞已经运行的会话，也不等待凑批。
    pub fn align_initial_batch(&mut self, requests: usize) {
        self.initial_batch = requests;
        self.initial_batch_released = requests == 0;
    }

    pub fn initial_batch_released(&self) -> bool {
        self.initial_batch_released
    }

    /// 只安排已经收到输入的 ready 请求；Pending 不占槽，也不阻塞其他 ready 工作。
    /// 每轮同时推进 decode 和一份公平轮转的 prefill 预算，避免两类任务互相饿死。
    pub fn next(&mut self, phases: &[RequestPhase]) -> Option<BatchPlan> {
        self.next_with_completion_alignment(phases, false)
    }

    /// 同一轮尚有 decode 在途时暂存已完成成员；达到上限或本轮全部完成后立即
    /// 释放。这里只按状态对齐，不按时间等待，也不让后来 prefill 建立 barrier。
    pub fn next_completion_wave(&mut self, phases: &[RequestPhase]) -> Option<BatchPlan> {
        self.next_with_completion_alignment(phases, true)
    }

    fn next_with_completion_alignment(&mut self, phases: &[RequestPhase], align_completion: bool) -> Option<BatchPlan> {
        if phases.is_empty() {
            self.cursor = 0;
            return None;
        }
        self.cursor %= phases.len();

        if !self.initial_batch_released {
            let initial = self.initial_batch.min(phases.len());
            self.initial_batch_released = initial == self.initial_batch && phases[..initial].iter().all(|phase| matches!(phase, RequestPhase::DecodeReady { .. } | RequestPhase::Finished));
        }
        let mut decode = Vec::new();
        let ready = phases.iter().filter(|phase| matches!(phase, RequestPhase::DecodeReady { .. })).count();
        let in_flight = phases.iter().filter(|phase| matches!(phase, RequestPhase::DecodeInFlight { .. })).count();
        let wave_ready = !align_completion || ready >= self.decode_batch_limit.min(ready.saturating_add(in_flight));
        if self.initial_batch_released && wave_ready {
            for offset in 0..phases.len() {
                let request = (self.cursor + offset) % phases.len();
                if matches!(phases[request], RequestPhase::DecodeReady { .. }) {
                    decode.push(request);
                    if decode.len() == self.decode_batch_limit {
                        break;
                    }
                }
            }
        }
        let mut budget = self.prefill_token_budget;
        let mut prefill = Vec::new();
        for offset in 0..phases.len() {
            let request = (self.cursor + offset) % phases.len();
            let RequestPhase::PrefillReady { position, end } = phases[request] else {
                continue;
            };
            if position >= end {
                continue;
            }
            let len = (end - position).min(budget);
            prefill.push(PrefillSlice { request, position, len });
            budget -= len;
            if budget == 0 {
                break;
            }
        }
        if let Some(&last) = decode.last() {
            self.cursor = (last + 1) % phases.len();
        } else if !prefill.is_empty() {
            self.cursor = (self.cursor + 1) % phases.len();
        }
        (!decode.is_empty() || !prefill.is_empty()).then_some(BatchPlan { prefill, decode })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_ready_requests_form_one_batch() {
        let mut scheduler = BatchScheduler::new(1024, 2).unwrap();
        let phases = [RequestPhase::PrefillReady { position: 0, end: 4096 }, RequestPhase::DecodeReady { position: 9 }, RequestPhase::DecodeReady { position: 17 }];
        assert_eq!(scheduler.next(&phases), Some(BatchPlan { prefill: vec![PrefillSlice { request: 0, position: 0, len: 1024 }], decode: vec![1, 2] }));
    }

    #[test]
    fn prefill_budget_is_shared_in_round_robin_order() {
        let mut scheduler = BatchScheduler::new(1024, 8).unwrap();
        let phases = [RequestPhase::PrefillReady { position: 0, end: 700 }, RequestPhase::PrefillReady { position: 10, end: 710 }];
        assert_eq!(scheduler.next(&phases), Some(BatchPlan { prefill: vec![PrefillSlice { request: 0, position: 0, len: 700 }, PrefillSlice { request: 1, position: 10, len: 324 }], decode: vec![] }));
        assert_eq!(scheduler.next(&phases), Some(BatchPlan { prefill: vec![PrefillSlice { request: 1, position: 10, len: 700 }, PrefillSlice { request: 0, position: 0, len: 324 }], decode: vec![] }));
    }

    #[test]
    fn pending_decode_does_not_block_prefill() {
        let mut scheduler = BatchScheduler::new(1024, 8).unwrap();
        let phases = [RequestPhase::Pending, RequestPhase::PrefillReady { position: 0, end: 700 }];
        assert_eq!(scheduler.next(&phases), Some(BatchPlan { prefill: vec![PrefillSlice { request: 1, position: 0, len: 700 }], decode: vec![] }));
        assert_eq!(scheduler.next(&[RequestPhase::Pending]), None);
    }

    #[test]
    fn finished_requests_produce_no_work() {
        let mut scheduler = BatchScheduler::new(1024, 8).unwrap();
        assert_eq!(scheduler.next(&[RequestPhase::Finished, RequestPhase::Finished]), None);
    }

    #[test]
    fn initial_batch_waits_for_every_member_then_releases_once() {
        let mut scheduler = BatchScheduler::new(1024, 8).unwrap();
        scheduler.align_initial_batch(3);
        assert_eq!(scheduler.next(&[RequestPhase::DecodeReady { position: 1 }, RequestPhase::Pending, RequestPhase::DecodeReady { position: 3 }]), None,);
        assert_eq!(scheduler.next(&[RequestPhase::DecodeReady { position: 1 }, RequestPhase::Finished, RequestPhase::DecodeReady { position: 3 }]), Some(BatchPlan { prefill: vec![], decode: vec![0, 2] }),);
        assert_eq!(
            scheduler.next(&[RequestPhase::DecodeReady { position: 1 }, RequestPhase::Pending, RequestPhase::DecodeReady { position: 3 }, RequestPhase::DecodeReady { position: 4 }]),
            Some(BatchPlan { prefill: vec![], decode: vec![0, 2, 3] }),
            "初始边界释放后，新请求不能重新建立 barrier",
        );
    }

    #[test]
    fn initial_decode_alignment_does_not_block_prefill_progress() {
        let mut scheduler = BatchScheduler::new(4, 8).unwrap();
        scheduler.align_initial_batch(2);
        assert_eq!(scheduler.next(&[RequestPhase::DecodeReady { position: 1 }, RequestPhase::PrefillReady { position: 4, end: 12 }]), Some(BatchPlan { prefill: vec![PrefillSlice { request: 1, position: 4, len: 4 }], decode: vec![] }),);
    }

    #[test]
    fn completion_wave只等待当前在途成员() {
        let mut scheduler = BatchScheduler::new(4, 4).unwrap();
        assert_eq!(
            scheduler.next_completion_wave(&[RequestPhase::DecodeReady { position: 1 }, RequestPhase::DecodeInFlight { position: 2 }, RequestPhase::DecodeInFlight { position: 3 }, RequestPhase::DecodeInFlight { position: 4 },]),
            None,
        );
        assert_eq!(
            scheduler.next_completion_wave(&[RequestPhase::DecodeReady { position: 1 }, RequestPhase::DecodeReady { position: 2 }, RequestPhase::DecodeReady { position: 3 }, RequestPhase::DecodeReady { position: 4 },]),
            Some(BatchPlan { prefill: vec![], decode: vec![0, 1, 2, 3] }),
        );
    }
}
