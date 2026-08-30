//! ROCm 算子能力入口。
//!
//! 真实现在 `hip/` 子模块(HIP 自研 kernel);host-slice 输入的算子直接调用
//! `hip::try_*_f32`,不再经过转发壳。

pub const ROCM_STATUS: &str = "ROCm backend kernel: HIP";

pub mod hip;
pub mod routing;
