//! H3 与 CUDA GEMM 路径的组合；通用 CUDA kernel 不保存模型 shape。

use crate::{backend::cuda::CudaContext, kernel::cuda::linear::prewarm_hgemm};

pub fn prewarm_h3_cublas_kernels(ctx: &CudaContext, hidden_size: usize, ffn_hidden_size: usize, num_heads: usize, head_dim: usize, time_embed_dim: usize, seq_len: usize) {
    let attention_columns = num_heads * head_dim;
    let qkv_columns = attention_columns * 3;

    // 首层 packed hidden 为 F16；后续残差流为 F32。
    prewarm_hgemm(ctx, qkv_columns, seq_len, hidden_size, false);
    prewarm_hgemm(ctx, hidden_size, seq_len, attention_columns, false);
    prewarm_hgemm(ctx, qkv_columns, seq_len, hidden_size, true);
    prewarm_hgemm(ctx, hidden_size, seq_len, attention_columns, true);
    prewarm_hgemm(ctx, 2 * ffn_hidden_size, seq_len, hidden_size, true);
    prewarm_hgemm(ctx, hidden_size, seq_len, ffn_hidden_size, true);
    // curve checkpoint 的 final AdaLN:一行时间 basis → 2*hidden。
    prewarm_hgemm(ctx, 2 * hidden_size, 1, time_embed_dim, false);
}
