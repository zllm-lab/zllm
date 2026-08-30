//! CUDA 算子层。
//!
//! 对称 `kernel/metal/`:内联 CUDA C++ 源码(各功能文件的 `SHADERS` + 下方共享
//! preamble)+ Rust 算子封装(`attention.rs` / `linear.rs` / `routing.rs` / `tensor.rs`)。
//! 源码运行时由 NVRTC 编译为 CUBIN(sm_86 本地机器码),由 `backend::cuda::context::CudaContext` 加载。

use cudarc::cublas::{GemmConfig, sys};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;

use crate::backend::cuda::context::{CudaContext, CudaTensor};

const THREADS: u32 = 256;

/// 权重 GPU 句柄的别名(简化 trait impl 里的引用)。
pub type CudaSliceF16 = cudarc::driver::safe::CudaSlice<half::f16>;

// 子文件只按算子领域分组，仍共享同一绑定命名空间和 launch helper。
pub mod attention;
pub mod diffusion;
pub mod fp8;
pub mod linear;
pub mod mlx_affine;
pub mod nvfp4;
pub mod routing;
pub mod tensor;
pub mod vae;
pub mod w4a16;
pub mod w8a16;

/// 整个 shader 翻译单元唯一的公共前导(原独立 preamble.rs)。
const PREAMBLE_SHADERS: &str = r#"
#include <cuda_fp16.h>
"#;

use std::sync::OnceLock;

static KERNELS: OnceLock<String> = OnceLock::new();

/// 把所有模块的 CUDA shader 拼成一个完整字符串,首次访问时一次性分配。
///
/// 每个模块自己负责自己的 SHADERS const(代码就近原则),这里只是把它们
/// 拼起来给 `CudaContext::new` 用。原 `kernels.rs` 已被删除。
pub fn kernels_source() -> &'static str {
    let parts: [&str; 12] = [PREAMBLE_SHADERS, attention::SHADERS, diffusion::SHADERS, fp8::SHADERS, linear::SHADERS, mlx_affine::SHADERS, nvfp4::SHADERS, routing::SHADERS, tensor::SHADERS, vae::SHADERS, w4a16::SHADERS, w8a16::SHADERS];
    KERNELS
        .get_or_init(|| {
            let total: usize = parts.iter().map(|s| s.len()).sum();
            let mut buf = String::with_capacity(total);
            for p in &parts {
                buf.push_str(p);
            }
            buf
        })
        .as_str()
}
