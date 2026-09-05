//! 平台与执行无关的模型架构配置。
//!
//! runtime 负责算法编排，weight 负责张量装配；二者共同依赖这里的单一规格定义。

pub mod deepseek_v4;
pub mod gemma4;
pub mod glm52;
pub mod glm53_flash;
pub mod h3;
pub mod kimi_k3;
pub mod laguna;
pub mod minimax_m3;
pub mod ornith;
pub mod qwen36;
pub mod qwen3_vl;
