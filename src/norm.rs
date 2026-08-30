//! 归一化规格。

/// RMSNorm 变体。区别在于权重 gamma 的解释方式。
#[derive(Debug)]
pub enum NormSpec {
    /// 标准 Llama 形式:`out = x_normed * weight`(GLM-5.2、Llama)。
    Rms { eps: f32 },
    /// Gemma 形式:`out = x_normed * (1 + weight)`,权重是零中心 gamma(MiniMax-M3)。
    GemmaRms { eps: f32 },
    /// AdaLN 形式（Diffusion Transformer 专用）:`out = x_normed * (1 + scale) + shift`，
    /// scale/shift 由条件（时间步 + 文本）通过 modulation 网络生成（H3 DiT）。
    AdaLn { eps: f32 },
}
