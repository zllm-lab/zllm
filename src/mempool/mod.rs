//! backend 无关的显存分配内核：event 感知 TLSF arena。
//!
//! 只管理段内 offset，不接触任何平台对象；设备完成证明经 [`Completion`]
//! 注入，ROCm/CUDA/Metal 各自实现平台 event 后复用同一内核。
//! 设计见 docs/rocm-mempool-arena-design-20260914.md。

pub mod arena;

/// 设备完成证明：backend 注入的 event 能力。arena 只依赖非阻塞查询；
/// 实现必须保证查询无副作用（不同步设备、不提交工作、不取其他锁）。
pub trait Completion: Send {
    fn is_complete(&self) -> bool;
}
