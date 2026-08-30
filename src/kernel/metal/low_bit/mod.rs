//! 低比特量化 matmul 家族:NVFP4 / MXFP4·MXFP8 / W4A16·W8A16。
//!
//! 原单文件按量化家族拆成子模块;kernel 之间互不调用,家族私有 helper 在
//! 各子文件内先于 kernel 定义,跨家族共享 helper 都在全局 preamble,
//! 因此子模块 SHADERS 的拼接顺序不影响编译结果。

pub mod mxfp;
pub mod nvfp4;
pub mod w4a16;
pub use mxfp::*;
pub use nvfp4::*;
pub use w4a16::*;

use crate::kernel::metal::MetalTensorDType;
use std::sync::OnceLock;

/// 按家族聚合子模块 shader,供 `super::kernels_source()` 拼接。
pub fn shaders() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| [nvfp4::NVFP4_SHADERS, mxfp::MXFP_SHADERS, w4a16::W4A16_SHADERS].concat()).as_str()
}
