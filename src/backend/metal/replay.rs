//! 双缓冲 decode 重放执行器(模型无关)。
//!
//! 平铺转录把一个 decode 步的 dispatch 序列固化成 CommandList;本执行器把它
//! 生产化:A/B 两份命令表交替重编码提交,深度-2 流水下 CPU 写与 GPU 读落在
//! 不同 parity 的 buffer 上,消除 shared memory 竞态。模型侧只提供录制闭包
//! (用 position_tensor 类间接寻址原语组装一步序列,保证命令表 position 无关)
//! 与每步的 CPU 状态写入(KV 游标推进、state 槽填充)。

use crate::backend::{
    BackendError,
    metal::{
        MetalContext,
        api::{Buffer, CommandBuffer, CommandList, Transcriber},
    },
};

pub struct DualReplay {
    plans: [CommandList; 2],
}

/// 一步已提交未等待的重放:CB 句柄即精确等待点(queue FIFO)。
pub struct ReplayStep {
    pub command: CommandBuffer,
    pub parity: usize,
}

impl DualReplay {
    /// 录制 A/B 两份命令表。`record(parity)` 用模型自备的 per-parity 资源
    /// (input/state/readback 等)组装同构命令序列;两份必须 dispatch 序列相同、
    /// 仅 buffer 绑定不同,否则交替重放的数值语义不成立。
    pub fn record(mut record: impl FnMut(usize) -> Result<(), BackendError>) -> Result<Self, BackendError> {
        let mut plans = Vec::with_capacity(2);
        for parity in 0..2 {
            Transcriber::begin_flat().map_err(|msg| BackendError::Compute { msg })?;
            let result = record(parity);
            let transcriber = Transcriber::end().ok_or_else(|| BackendError::Compute { msg: "重放转录器未在进行".to_owned() })?;
            let plan = transcriber.into_command_list().map_err(|msg| BackendError::Compute { msg })?;
            result?;
            plans.push(plan);
        }
        let plan_a = plans.pop().expect("两份命令表");
        let plan_b = plans.pop().expect("两份命令表");
        Ok(Self { plans: [plan_b, plan_a] })
    }

    pub fn command_count(&self) -> usize {
        self.plans[0].ops.len()
    }

    /// 请求级重映射:录制期钉住的 KV cache 等 buffer 换成新请求实例。
    pub fn remap_buffers(&mut self, replacements: &[(Buffer, Buffer)]) {
        self.plans[0].remap_buffers(replacements);
        self.plans[1].remap_buffers(replacements);
    }

    /// 提交一个 decode 步:按 parity 重编码命令表为一个 command buffer 并 commit,
    /// 不等待;后续步的 GPU 执行与本 CB 按提交顺序串行。
    pub fn submit(&self, ctx: &MetalContext, parity: usize) -> Result<ReplayStep, BackendError> {
        let command = ctx.command_buffer();
        {
            let encoder = command.new_compute_command_encoder();
            for op in &self.plans[parity].ops {
                encoder.encode_recorded(op);
            }
            encoder.end_encoding();
        }
        command.commit();
        Ok(ReplayStep { command, parity })
    }
}
