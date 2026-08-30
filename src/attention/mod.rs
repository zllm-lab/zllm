//! 注意力机制领域层。保存架构规格、平台无关算法与 reference 实现。
//!
//! backend 根据 [`AttentionSpec`] 选择具体存储与 kernel。

pub mod attn_res;
pub mod block;
pub mod compressed_sparse;
pub mod dsa;
pub mod gated_delta_net;
pub mod gqa;
pub mod hybrid;
pub mod hyper_connection;
pub mod kda;
pub mod mla;
pub mod msa;
pub mod rope;

mod recurrent_state;

/// 注意力机制规格。
///
/// 字段全是架构常量,与硬件无关。`Backend` 据此选择并配置 kernel。
#[derive(Debug)]
pub enum AttentionSpec {
    /// query 与 KV 可不同长度、每个 query 有显式可见区间的块注意力。
    Block(block::BlockAttentionSpec),
    /// 分组查询注意力(MiniMax-M3、LLaMA 等)。
    Gqa(gqa::GqaSpec),
    /// 带短卷积和固定 recurrent state 的线性注意力。
    GatedDeltaNet(gated_delta_net::GatedDeltaNetSpec),
    /// Kimi Delta Attention，使用短卷积与矩阵 recurrent state。
    Kda(kda::KdaSpec),
    /// 多头潜在注意力(GLM-5.2、DeepSeek-V3)。
    Mla(mla::MlaSpec),
    /// 带输出门的多头潜在注意力(Kimi-K3)。
    GatedMla(mla::GatedMlaSpec),
    /// MiniMax 稀疏注意力(MiniMax-M3 长上下文)。
    Msa(msa::MsaSpec),
}

impl AttentionSpec {
    /// 该注意力对应的 RoPE 旋转维度(0 表示不使用 RoPE)。
    pub fn rope_dim(&self) -> usize {
        match self {
            Self::Block(_) => 0,
            Self::Gqa(s) => s.rope_dim,
            Self::GatedDeltaNet(_) => 0,
            Self::Kda(_) => 0,
            Self::Mla(s) => s.qk_rope_head_dim,
            Self::GatedMla(s) => {
                if s.use_rope {
                    s.mla.qk_rope_head_dim
                } else {
                    0
                }
            }
            Self::Msa(s) => s.gqa.rope_dim,
        }
    }

    /// RoPE 的基数(theta)。
    pub fn rope_theta(&self) -> f32 {
        match self {
            Self::Block(_) => 0.0,
            Self::Gqa(s) => s.rope_theta,
            Self::GatedDeltaNet(_) => 0.0,
            Self::Kda(_) => 0.0,
            Self::Mla(s) => s.rope_theta,
            Self::GatedMla(s) => {
                if s.use_rope {
                    s.mla.rope_theta
                } else {
                    0.0
                }
            }
            Self::Msa(s) => s.gqa.rope_theta,
        }
    }
}
