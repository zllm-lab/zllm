//! 前馈层(FFN)领域层。包括规格、路由、通用数据流与 reference 实现。

pub mod dense_mlp;
pub mod expert_predictor;
pub mod latent_moe;
pub mod prefill;
pub mod routing;
pub mod topk_moe;

/// 不驻留 expert 权重的 backend 共享状态；只记录真实路由量。
#[derive(Debug, Default, Clone, Copy)]
pub struct UncachedMoeState {
    routed_experts: usize,
}

impl UncachedMoeState {
    pub fn record_routed_experts(&mut self, count: usize) {
        self.routed_experts = self.routed_experts.saturating_add(count);
    }

    pub fn routed_experts(&self) -> usize {
        self.routed_experts
    }
}

pub use dense_mlp::Activation;

/// 已准备到 backend 的 gate/up/down 权重；模型 runtime 只负责组合其执行顺序。
pub struct DenseFfn<W> {
    pub gate: W,
    pub up: W,
    pub down: W,
}

/// 前馈层规格。
#[derive(Debug)]
pub enum FeedforwardSpec {
    /// 单专家 dense FFN(前几层、或非 MoE 模型)。
    Dense(dense_mlp::DenseMlpSpec),
    /// Top-k 路由 MoE。
    TopkMoe(topk_moe::TopkMoeSpec),
    /// routed experts 在低维 latent 空间计算，共享 MLP 仍读取原 hidden。
    LatentTopkMoe(latent_moe::LatentTopkMoeSpec),
}
