//! Metal 平铺录制与 decode 重放执行器(模型无关)。
//!
//! `ReplayPlan` 把任意静态 dispatch 序列录制成可重编码的命令表，
//! `DualReplay` 再把 A/B 两份计划组成深度-2 流水。模型侧只负责用
//! position/state buffer 组装可录制的一步，以及在提交前更新自己的
//! KV 游标和输入；录制、buffer 重映射、编码与提交生命周期属于 backend。

use crate::backend::{
    BackendError,
    metal::{
        MetalContext,
        api::{Buffer, CommandBuffer, CommandList, RecordedComputeOp, Transcriber},
    },
};

/// 单份模型无关平铺重放计划。
pub struct ReplayPlan {
    commands: CommandList,
}

pub struct DualReplay {
    plans: [ReplayPlan; 2],
}

/// 一步已提交未等待的重放:CB 句柄即精确等待点(queue FIFO)。
pub struct ReplayStep {
    pub command: CommandBuffer,
    pub parity: usize,
}

impl ReplayPlan {
    /// 录制闭包中的 Metal dispatch，同时返回闭包产生的模型状态。
    /// 录制期只固化命令，不提交 GPU 工作。
    pub fn record<T>(record: impl FnOnce() -> Result<T, BackendError>) -> Result<(Self, T), BackendError> {
        Transcriber::begin_flat().map_err(|msg| BackendError::Compute { msg })?;
        let result = record();
        let transcriber = Transcriber::end().ok_or_else(|| BackendError::Compute { msg: "重放转录器未在进行".to_owned() })?;
        let commands = transcriber.into_command_list().map_err(|msg| BackendError::Compute { msg })?;
        Ok((Self { commands }, result?))
    }

    pub fn command_count(&self) -> usize {
        self.commands.ops.len()
    }

    /// 静态命令表只对性能消融和诊断开放。
    pub fn ops(&self) -> &[RecordedComputeOp] {
        &self.commands.ops
    }

    /// 请求级重映射：把录制时绑定的 cache/input buffer 替换为当前实例。
    pub fn remap_buffers(&mut self, replacements: &[(Buffer, Buffer)]) {
        self.commands.remap_buffers(replacements);
    }

    /// 将录制计划重编码为一个 command buffer 并提交，不等待。
    pub fn submit(&self, ctx: &MetalContext) -> CommandBuffer {
        self.submit_filtered(ctx, &|_| true)
    }

    /// 诊断用子集提交；返回值不具有模型数值语义。
    pub fn submit_filtered(&self, ctx: &MetalContext, keep: &dyn Fn(&RecordedComputeOp) -> bool) -> CommandBuffer {
        let command = ctx.command_buffer();
        {
            let encoder = command.new_compute_command_encoder();
            for op in &self.commands.ops {
                if keep(op) {
                    encoder.encode_recorded(op);
                }
            }
            encoder.end_encoding();
        }
        command.commit();
        command
    }
}

impl DualReplay {
    /// 录制 A/B 两份命令表。`record(parity)` 用模型自备的 per-parity 资源
    /// (input/state/readback 等)组装同构命令序列;两份必须 dispatch 序列相同、
    /// 仅 buffer 绑定不同,否则交替重放的数值语义不成立。
    pub fn record(mut record: impl FnMut(usize) -> Result<(), BackendError>) -> Result<Self, BackendError> {
        let mut plans = Vec::with_capacity(2);
        for parity in 0..2 {
            let (plan, ()) = ReplayPlan::record(|| record(parity))?;
            plans.push(plan);
        }
        let plan_a = plans.pop().expect("两份命令表");
        let plan_b = plans.pop().expect("两份命令表");
        Ok(Self { plans: [plan_b, plan_a] })
    }

    pub fn command_count(&self) -> usize {
        self.plans[0].command_count()
    }

    /// 请求级重映射:录制期钉住的 KV cache 等 buffer 换成新请求实例。
    pub fn remap_buffers(&mut self, replacements: &[(Buffer, Buffer)]) {
        self.plans[0].remap_buffers(replacements);
        self.plans[1].remap_buffers(replacements);
    }

    /// 提交一个 decode 步:按 parity 重编码命令表为一个 command buffer 并 commit,
    /// 不等待;后续步的 GPU 执行与本 CB 按提交顺序串行。
    pub fn submit(&self, ctx: &MetalContext, parity: usize) -> Result<ReplayStep, BackendError> {
        let command = self.plans[parity].submit(ctx);
        Ok(ReplayStep { command, parity })
    }
}
