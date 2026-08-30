//! 标准张量编码与量化格式；保存布局、metadata 和加载来源，不包含模型执行。

pub mod block_fp8;
pub mod compressed_tensors;
pub mod compressed_tensors_hybrid;
pub mod mlx_affine;
pub mod mxfp4;
pub mod mxfp8;
pub mod nvfp4;
pub mod official_fp8;
pub mod per_tensor_fp8;
pub mod quantization;
