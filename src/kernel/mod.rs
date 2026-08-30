//! 算子层。按平台分目录。
//! CPU(NEON/SSE)、Metal(Apple GPU)、CUDA(NVIDIA GPU)各自实现同名算子,runtime 按 cfg 选择。

pub mod cpu;
pub mod huawei;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(feature = "with-cuda")]
pub mod cuda;
#[cfg(feature = "with-rocm")]
pub mod rocm;
