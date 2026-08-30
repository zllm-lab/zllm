//! 权重加载领域：公共来源契约、文件容器、张量格式、模型映射与纯解码。

pub mod codec;
pub mod container;
pub mod expert_source;
pub mod format;
pub mod model;

// 向后兼容：旧 mistral_small32 模块名映射到 model::mistral。
pub mod mistral_small32 {
    pub use crate::weight::model::mistral::*;
}

pub use format::per_tensor_fp8::PerTensorFp8Matrix;
pub use format::quantization::Fp8Matrix;
pub use model::glm52::{DenseLayerF32, Glm52IndexerWeights, Glm52Weights, MoeLayerF32};

/// 运行时权重的模型无关驻留格式。Native 保留 checkpoint 原格式；量化格式由
/// runtime 公共准备路径生成，再交给各 backend 的标准权重能力处理。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidentWeightQuantization {
    #[default]
    Native,
    Q8g128,
}

/// 兼容现有模型和配置中的输出头命名；底层策略同样可用于 drafter 等独立 resident 权重。
pub type LmHeadQuantization = ResidentWeightQuantization;
