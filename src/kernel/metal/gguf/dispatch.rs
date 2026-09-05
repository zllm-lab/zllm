//! GGUF Metal host dispatch：shape 校验、pipeline 选择与 command 编码。

use super::*;

#[cfg(test)]
mod tests;

const GGUF_PREFILL_MPS_ROWS: usize = 32;

fn gguf_prefill_mps_rows() -> usize {
    // A/B 实验开关:同 binary 内切回物化+MPS 路径,消除跨重启热状态噪声。
    std::env::var("ZLLM_GGUF_MPS_ROWS").ok().and_then(|value| value.parse().ok()).unwrap_or(GGUF_PREFILL_MPS_ROWS)
}

pub fn gguf_matmul_tensor_resident(ctx: &MetalContext, input: &MetalTensor, blob: &metal::Buffer, tensor_type: u32, row_bytes: usize, weight_rows: usize, weight_cols: usize) -> Result<MetalTensor, String> {
    if input.cols != weight_cols {
        return Err(format!("GGUF input=[{},{}] 与 weight=[{weight_rows},{weight_cols}] 不兼容", input.rows, input.cols));
    }
    let (block_elements, block_bytes) = metal_block_layout(tensor_type)?;
    if !weight_cols.is_multiple_of(block_elements) {
        return Err(format!("GGUF weight columns={weight_cols} 未按 block {block_elements} 对齐"));
    }
    let expected_row_bytes = weight_cols / block_elements * block_bytes;
    if row_bytes != expected_row_bytes {
        return Err(format!("GGUF row bytes={row_bytes}，期望 {expected_row_bytes}"));
    }
    let expected = row_bytes.checked_mul(weight_rows).ok_or("GGUF resident buffer 大小溢出")?;
    validate_size("GGUF resident buffer", expected, blob.length() as usize)?;

    // IQ4_NL 多行(2..31 行,verify/短 prefill):8 行批 kernel 按 grid.y 分组,
    // 行间隔离经 rows=1..4 CPU 参照验证(iq4nl_multirow_rows_scan_cpu)。
    // mul_mv_ext 形态(移植 llama.cpp kernel_mul_mv_ext_iq4_nl_f32_r1_4,每 lane
    // 一行权重,反量化一次摊给 4 个 input 行)已实现并通过 3840 列数值对照;
    // M5 + F16 input + M=4 实测四形态:8 行批 155ms < ext 161ms < ext-half-dot
    // 171ms < per-y 254ms < MMA 538ms,8 行批最优,ext 不启用(保留作研究)。

    // Q3_K / Q6_K / IQ4_NL prefill 直接走 fused GEMM，避免 dequant + MPS 两遍。
    let use_q3k_fused = input.dtype == MetalTensorDType::F16 && input.rows >= 4 && tensor_type == 11;
    let use_q6k_fused = input.dtype == MetalTensorDType::F16 && input.rows >= 4 && tensor_type == 14;
    let use_iq4nl_fused = input.dtype == MetalTensorDType::F16 && input.rows >= 4 && tensor_type == 20;
    // IQ4_XS 64×64 tile 的行利用率在 32 行以上才划算;4..31 行多行 gemv
    // (权重读一次跨行共享)实测 ~80-100GB/s 更优。IQ3_S 同理(单矩阵版主要
    // 服务 gated dual 的分解路径)。
    let use_iq4xs_fused = input.dtype == MetalTensorDType::F16 && input.rows >= gguf_prefill_mps_rows() && tensor_type == 23;
    let use_iq3s_fused = input.dtype == MetalTensorDType::F16 && input.rows >= gguf_prefill_mps_rows() && tensor_type == 21;
    if use_q3k_fused || use_q6k_fused || use_iq4nl_fused || use_iq4xs_fused || use_iq3s_fused {
        // fused 路径不依赖 caller 的 dtype guard, 显式用 input dtype (F16)
        let m = validate_u32("GGUF M", input.rows)?;
        let n = validate_u32("GGUF N", weight_rows)?;
        let k = validate_u32("GGUF K", weight_cols)?;
        let row_bytes_u32 = validate_u32("GGUF row bytes", row_bytes)?;
        let iq2s_grid = ctx.resident_byte_weight_buffer(as_bytes(crate::weight::codec::ggml::iq2s_grid()));
        let kernel_name = if use_q3k_fused {
            "gguf_gemm_q3k_fused_f16"
        } else if use_q6k_fused {
            "gguf_gemm_q6k_fused_f16"
        } else if use_iq4xs_fused {
            "gguf_gemm_iq4xs_fused_f16"
        } else if use_iq3s_fused {
            "gguf_gemm_iq3s_fused_f16"
        } else if ctx.metal4_available() {
            "gguf_gemm_iq4nl_mpp_f16"
        } else {
            "gguf_gemm_iq4nl_fused_f16"
        };
        let output = if kernel_name == "gguf_gemm_iq4nl_mpp_f16" { ctx.tensor_pooled_f32("gguf_iq4nl_mpp_f32", input.rows, weight_rows) } else { ctx.tensor_kernel_output(input.rows, weight_rows) };
        let pipeline = ctx.pipeline(kernel_name)?;
        let threads: usize = 128;
        if pipeline.max_total_threads_per_threadgroup() < threads as u64 {
            return Err(format!("{kernel_name} 需要至少 {threads} threads/threadgroup"));
        }
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(blob), 0);
        encoder.set_buffer(2, Some(&iq2s_grid), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        set_bytes(&encoder, 4, &m);
        set_bytes(&encoder, 5, &n);
        set_bytes(&encoder, 6, &k);
        set_bytes(&encoder, 7, &row_bytes_u32);
        let groups = if kernel_name == "gguf_gemm_iq4nl_mpp_f16" {
            MTLSize::new(input.rows.div_ceil(128) as u64, weight_rows.div_ceil(64) as u64, 1)
        } else if use_iq4nl_fused || use_iq4xs_fused || use_iq3s_fused {
            // iq4nl 是 64×64 tile;iq4xs/iq3s 是 32×32 tile(MLX qmm 同款,
            // 小布局换并发 TG)。
            MTLSize::new(weight_rows.div_ceil(if use_iq4nl_fused { 64 } else { 32 }) as u64, input.rows.div_ceil(if use_iq4nl_fused { 64 } else { 32 }) as u64, 1)
        } else {
            MTLSize::new(weight_rows.div_ceil(4) as u64, input.rows.div_ceil(4) as u64, 1)
        };
        encoder.dispatch_thread_groups(groups, MTLSize::new(threads as u64, 1, 1));
        encoder.end_encoding();
        let shape = format!("input=[{m},{k}],weight=[{n},{k}],type={tensor_type}");
        ctx.commit_and_wait_profiled(&command, kernel_name, &shape, input.buffer.length() + expected as u64 + iq2s_grid.length(), output.buffer.length());
        return if output.dtype == MetalTensorDType::F32 {
            // pooled MPP scratch 的地址稳定但内容每次重写，不能命中按地址缓存的
            // immutable cast view（否则 gate/up 会错误共享上一轮 F16 结果）。
            ctx.invalidate_f16_cast(&output);
            to_f16_tensor(ctx, &output)
        } else {
            Ok(output)
        };
    }

    if input.rows >= gguf_prefill_mps_rows() {
        const DEQUANT_THREADS: usize = 256;
        let columns = validate_u32("GGUF columns", weight_cols)?;
        let rows = validate_u32("GGUF rows", weight_rows)?;
        let row_bytes_u32 = validate_u32("GGUF row bytes", row_bytes)?;
        let element_count = weight_rows.checked_mul(weight_cols).ok_or("GGUF dequant 元素数溢出")?;
        let weight_f16 = ctx.tensor_kernel_output(weight_rows, weight_cols);
        let iq2s_grid = ctx.resident_byte_weight_buffer(as_bytes(crate::weight::codec::ggml::iq2s_grid()));
        let pipeline_name = match tensor_type {
            11 => "gguf_dequant_q3k_matrix_f16",
            21 => "gguf_dequant_iq3s_matrix_f16",
            22 => "gguf_dequant_iq2s_matrix_f16",
            23 => "gguf_dequant_iq4xs_matrix_f16",
            _ => "gguf_dequant_matrix_f16",
        };
        let pipeline = ctx.pipeline(pipeline_name)?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(blob), 0);
        encoder.set_buffer(1, Some(&iq2s_grid), 0);
        encoder.set_buffer(2, Some(&weight_f16.buffer), 0);
        set_bytes(&encoder, 3, &columns);
        set_bytes(&encoder, 4, &rows);
        set_bytes(&encoder, 5, &tensor_type);
        set_bytes(&encoder, 6, &row_bytes_u32);
        if matches!(tensor_type, 11 | 22) {
            encoder.dispatch_thread_groups(MTLSize::new((weight_cols / 256) as u64, weight_rows as u64, 1), MTLSize::new(64, 1, 1));
        } else if matches!(tensor_type, 21 | 23) {
            encoder.dispatch_thread_groups(MTLSize::new((weight_cols / 32).div_ceil(64) as u64, weight_rows as u64, 1), MTLSize::new(64, 1, 1));
        } else {
            encoder.dispatch_thread_groups(MTLSize::new(element_count.div_ceil(DEQUANT_THREADS) as u64, 1, 1), MTLSize::new(DEQUANT_THREADS as u64, 1, 1));
        }
        encoder.end_encoding();
        let shape = format!("weight=[{weight_rows},{weight_cols}],type={tensor_type}");
        ctx.commit_and_wait_profiled(&command, pipeline_name, &shape, expected as u64 + iq2s_grid.length(), weight_f16.buffer.length());

        let output = if input.dtype == MetalTensorDType::F32 {
            let output = ctx.tensor_kernel_output_f32(input.rows, weight_rows);
            let command = ctx.command_buffer();
            crate::kernel::metal::mps::encode_f32_f16_matmul_transposed_f32(&command, &ctx.device, &input.buffer, input.rows, input.cols, &weight_f16.buffer, weight_rows, &output.buffer)?;
            let shape = format!("input=[{},{}],weight=[{weight_rows},{weight_cols}]", input.rows, input.cols);
            ctx.commit_and_wait_profiled(&command, "gguf_mps_matrix_multiplication_f32_f16_f32", &shape, input.buffer.length() + weight_f16.buffer.length(), output.buffer.length());
            output
        } else {
            let output = ctx.tensor_kernel_output(input.rows, weight_rows);
            launch_matmul_f16(ctx, &input.buffer, &weight_f16.buffer, &output.buffer, input.rows, weight_cols, weight_rows)?;
            output
        };
        // command buffer 会保留已编码的 dequant weight；层级 batch 结束时统一同步。
        return Ok(output);
    }

    let input_f16 = (input.dtype == MetalTensorDType::F32).then(|| to_f16_tensor(ctx, input)).transpose()?;
    let input = input_f16.as_ref().unwrap_or(input);
    // 多行 gemv(权重读一次跨行共享)覆盖到 MPS 阈值之前的全部行数;kernel 内
    // grid.y 按 8 行一组批处理。9..31 行曾落入 gemm_rows 标量路径(~2GB/s,
    // 短 prompt prefill 实测 6s/16 token),专用 gemv 同形状 ~80-100GB/s。
    const GGUF_THREADS: usize = 64;
    let multirow_rows = input.rows < gguf_prefill_mps_rows();
    let pipeline_name = if multirow_rows && tensor_type == 8 {
        "gguf_gemv_q8_0_f16"
    } else if multirow_rows && tensor_type == 2 {
        // q4_0 与 q8_0 同 kernel 结构(8 行 × 8 输入行);此前落通用 gguf_gemv_f16
        // 标量解码,decode 实测 ~1.5-4 tok/s,QAT GGUF 主力格式必须走专用路径
        "gguf_gemv_q4_0_f16"
    } else if input.rows == 1 && tensor_type == 11 {
        "gguf_gemv_q3k_f16"
    } else if input.rows < 4 && tensor_type == 14 {
        // rows>=4 走上面的 fused GEMM;1-3 行用单行 gemv(q6k per-y)
        "gguf_gemv_q6k_f16"
    } else if input.rows == 1 && tensor_type == 22 {
        "gguf_gemv_iq2s_f16"
    } else if input.rows == 1 && tensor_type == 18 {
        "gguf_gemv_iq3xxs_f16"
    } else if input.rows == 1 && tensor_type == 23 {
        // 单行(decode)专用:去掉 8 行累加器的寄存器压力,交替 A/B 实测 +7%
        "gguf_gemv_iq4xs_1r_f16"
    } else if multirow_rows && tensor_type == 23 {
        // iq4xs 多行 gemv(kernel 按 8 行/threadgroup 分组):小批(MTP verify /
        // DSpark k+1 行 verify / 短 prefill chunk)权重读一次算多行,避免
        // dequant+GEMM 的 prefill 路径
        "gguf_gemv_iq4xs_f16"
    } else if input.rows == 1 && tensor_type == 20 {
        "gguf_gemv_iq4nl_1r_f16"
    } else if multirow_rows && tensor_type == 20 {
        // iq4nl verify 4 行寄存器版(gguf_gemv_iq4nl_4r_f16,反量化 w[32] 常驻
        // + 4 行 FMA)实测 252ms:标量 dot 不敌 dot32 向量化,不启用;8 行批
        // 155ms 仍是六形态(MMA/ext/per-y/4r)穷尽后的最优。
        // iq4nl 多行 gemv,分组与复用结构同 iq4xs。三形态实测(M=4 verify):
        // 8 行批 155ms < per-y 254ms(L2 放不下 33MB/层,多行重复读) <
        // MMA 538ms(64 行 tile 只 4 行有效);8 行批最优保留。
        "gguf_gemv_iq4nl_f16"
    } else if input.rows == 1 && tensor_type == 21 {
        "gguf_gemv_iq3s_f16"
    } else if tensor_type == 12 && input.rows == 3 {
        "gguf_gemv_q4k_3m_f16"
    } else if tensor_type == 12 && (input.rows == 1 || multirow_rows) {
        // q4k 多行:per-y 单行 gemv(grid.y = input.rows),权重经 L2 跨行共享;
        // 原路径落 gemm_rows 标量(~2GB/s),短 prompt prefill 曾 172ms/15 token
        "gguf_gemv_q4k_f16"
    } else if input.rows == 1 && tensor_type == 13 {
        // 单行(decode)专用:交替 A/B 实测 +49%(lm head 形状 75 -> 112 GB/s)
        "gguf_gemv_q5k_1r_f16"
    } else if multirow_rows && tensor_type == 13 {
        "gguf_gemv_q5k_f16"
    } else if input.rows == 1 {
        "gguf_gemv_f16"
    } else {
        "gguf_gemm_rows_f16"
    };
    let columns = validate_u32("GGUF columns", weight_cols)?;
    let pipeline = ctx.pipeline(pipeline_name)?;
    let iq_packed_rows = matches!(
        pipeline_name,
        "gguf_gemv_iq4xs_f16" | "gguf_gemv_iq4xs_1r_f16" | "gguf_gemv_iq4nl_f16" | "gguf_gemv_iq4nl_1r_f16" | "gguf_gemv_iq4nl_4r_f16" | "gguf_gemv_iq3s_f16" | "gguf_gemv_q4_0_f16" | "gguf_gemv_q5k_f16" | "gguf_gemv_q5k_1r_f16"
    );
    let packed_simd_rows = matches!(pipeline_name, "gguf_gemv_q8_0_f16" | "gguf_gemv_q3k_f16" | "gguf_gemv_q4k_f16" | "gguf_gemv_q4k_3m_f16" | "gguf_gemv_q6k_f16" | "gguf_gemv_qk_f16" | "gguf_gemv_iq2s_f16" | "gguf_gemv_iq3xxs_f16");
    let threads = if matches!(pipeline_name, "gguf_gemv_q6k_f16" | "gguf_gemv_q3k_f16") {
        64
    } else if iq_packed_rows {
        128
    } else if packed_simd_rows {
        256
    } else {
        GGUF_THREADS
    };
    if pipeline.max_total_threads_per_threadgroup() < threads as u64 {
        return Err(format!("GGUF GEMV 需要至少 {threads} threads/threadgroup"));
    }
    let rows = validate_u32("GGUF rows", weight_rows)?;
    let row_bytes = validate_u32("GGUF row bytes", row_bytes)?;
    let output = ctx.tensor_kernel_output(input.rows, weight_rows);
    let iq2s_grid = ctx.resident_byte_weight_buffer(as_bytes(crate::weight::codec::ggml::iq2s_grid()));
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(blob), 0);
    encoder.set_buffer(2, Some(&iq2s_grid), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &columns);
    set_bytes(&encoder, 5, &rows);
    set_bytes(&encoder, 6, &tensor_type);
    set_bytes(&encoder, 7, &row_bytes);
    let input_rows_arg = validate_u32("GGUF input rows", input.rows.max(1))?;
    set_bytes(&encoder, 8, &input_rows_arg);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            if pipeline_name == "gguf_gemv_q6k_f16" {
                weight_rows.div_ceil(2)
            } else if pipeline_name == "gguf_gemv_iq4nl_1r_f16" || pipeline_name == "gguf_gemv_iq4nl_4r_f16" {
                // iq4nl 单行/4 行版:每 simdgroup 一行权重,128 threads = 4 行/threadgroup
                weight_rows.div_ceil(4)
            } else if pipeline_name == "gguf_gemv_iq4xs_f16" || pipeline_name == "gguf_gemv_iq4xs_1r_f16" || pipeline_name == "gguf_gemv_iq4nl_f16" || pipeline_name == "gguf_gemv_q4_0_f16" {
                // iq4xs/iq4nl/q4_0 是 16 K-lane x 2 行/simdgroup,128 threads = 8 行/threadgroup
                weight_rows.div_ceil(8)
            } else if pipeline_name == "gguf_gemv_q3k_f16" || iq_packed_rows {
                weight_rows.div_ceil(4)
            } else if packed_simd_rows {
                weight_rows.div_ceil(8)
            } else {
                weight_rows
            } as u64,
            // grid.y 是 kernel 的 input 行批数:multi-row kernel 批宽均为 8;
            // 但 q4k/q6k gemv 是 per-y 单行(y 直接索引输入行),多行时 grid.y=行数
            if matches!(pipeline_name, "gguf_gemv_q4k_f16" | "gguf_gemv_q6k_f16") && input.rows > 1 { input.rows as u64 } else { input.rows.div_ceil(8) as u64 },
            1,
        ),
        MTLSize::new(threads as u64, 1, 1),
    );
    encoder.end_encoding();
    let shape = format!("input=[{},{weight_cols}],weight=[{weight_rows},{weight_cols}],type={tensor_type}", input.rows);
    ctx.commit_and_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + expected as u64 + iq2s_grid.length(), output.buffer.length());
    Ok(output)
}

/// decode 单行 gemv + 残差 epilogue(q4k/q6k):与 gemv + add_f16 两步逐位一致,
/// 省一次 dispatch 与中间 tensor 往返。
#[allow(clippy::too_many_arguments)]
pub fn gguf_gemv_add_tensor(ctx: &MetalContext, input: &MetalTensor, blob: &metal::Buffer, tensor_type: u32, row_bytes: usize, rows: usize, cols: usize, residual: &MetalTensor) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != cols || input.dtype != MetalTensorDType::F16 || residual.dtype != MetalTensorDType::F16 || residual.len() != rows {
        return Err(format!("GGUF gemv+add 输入不符: input=[{},{}] {:?} residual={} rows={rows} cols={cols}", input.rows, input.cols, input.dtype, residual.len()));
    }
    let pipeline_name = match tensor_type {
        12 => "gguf_gemv_q4k_add_2r_f16",
        14 => "gguf_gemv_q6k_add_f16",
        _ => return Err(format!("GGUF gemv+add 不支持 type={tensor_type}")),
    };
    let output = ctx.tensor_uninit(1, rows);
    let pipeline = ctx.pipeline(pipeline_name)?;
    let columns_u32 = validate_u32("GGUF gemv+add columns", cols)?;
    let rows_u32 = validate_u32("GGUF gemv+add rows", rows)?;
    let row_bytes_u32 = validate_u32("GGUF gemv+add row bytes", row_bytes)?;
    let threads: u64 = if tensor_type == 14 { 64 } else { 128 };
    if pipeline.max_total_threads_per_threadgroup() < threads as u64 {
        return Err(format!("GGUF gemv+add 需要 {threads} threads/threadgroup"));
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(blob), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    encoder.set_buffer(3, Some(&residual.buffer), 0);
    set_bytes(&encoder, 4, &columns_u32);
    set_bytes(&encoder, 5, &rows_u32);
    set_bytes(&encoder, 6, &row_bytes_u32);
    let groups = if tensor_type == 14 { (rows as u64).div_ceil(2) } else { (rows as u64).div_ceil(8) };
    encoder.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{cols}],weight=[{rows},{cols}],type={tensor_type}");
    ctx.commit_and_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + (rows * row_bytes) as u64 + residual.buffer.length(), output.buffer.length());
    Ok(output)
}

/// Q/K 双路 q4k gemv(decode 单行):一次 dispatch 算两个矩阵,行数可不同。
/// 逐行数学与 gguf_gemv_q4k_f16 完全一致。
#[allow(clippy::too_many_arguments)]
pub fn gguf_dual_gemv_q4k_tensor(
    ctx: &MetalContext,
    input: &MetalTensor,
    first_blob: &metal::Buffer,
    first_row_bytes: usize,
    first_rows: usize,
    second_blob: &metal::Buffer,
    second_row_bytes: usize,
    second_rows: usize,
    columns: usize,
) -> Result<(MetalTensor, MetalTensor), String> {
    if input.rows != 1 || input.cols != columns || input.dtype != MetalTensorDType::F16 {
        return Err(format!("Q4_K dual gemv 输入不符: [{},{}] {:?} columns={columns}", input.rows, input.cols, input.dtype));
    }
    if !columns.is_multiple_of(256) || first_row_bytes != columns / 256 * 144 || second_row_bytes != columns / 256 * 144 {
        return Err(format!("Q4_K dual gemv row_bytes 与 columns={columns} 不符"));
    }
    let first_output = ctx.tensor_uninit(1, first_rows);
    let second_output = ctx.tensor_uninit(1, second_rows);
    let pipeline = ctx.pipeline("gguf_dual_gemv_q4k_2r_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("Q4_K dual gemv 需要 128 threads/threadgroup".to_owned());
    }
    let columns_u32 = validate_u32("Q4_K dual columns", columns)?;
    let first_rows_u32 = validate_u32("Q4_K dual first rows", first_rows)?;
    let second_rows_u32 = validate_u32("Q4_K dual second rows", second_rows)?;
    let first_row_bytes_u32 = validate_u32("Q4_K dual first row bytes", first_row_bytes)?;
    let second_row_bytes_u32 = validate_u32("Q4_K dual second row bytes", second_row_bytes)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(first_blob), 0);
    encoder.set_buffer(2, Some(second_blob), 0);
    encoder.set_buffer(3, Some(&first_output.buffer), 0);
    encoder.set_buffer(4, Some(&second_output.buffer), 0);
    set_bytes(&encoder, 5, &columns_u32);
    set_bytes(&encoder, 6, &first_rows_u32);
    set_bytes(&encoder, 7, &second_rows_u32);
    set_bytes(&encoder, 8, &first_row_bytes_u32);
    set_bytes(&encoder, 9, &second_row_bytes_u32);
    let total_rows = (first_rows + second_rows) as u64;
    encoder.dispatch_thread_groups(MTLSize::new(total_rows.div_ceil(8), 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{columns}],first={first_rows},second={second_rows},type=12");
    ctx.commit_and_wait_profiled(&command, "gguf_dual_gemv_q4k_2r_f16", &shape, input.buffer.length() + (first_rows * first_row_bytes + second_rows * second_row_bytes) as u64, first_output.buffer.length() + second_output.buffer.length());
    Ok((first_output, second_output))
}

/// 设备端 embedding gather:按 `token_id`(设备 u32 buffer)从常驻 Q6_K GGUF 矩阵
/// 抠一行 dequant 成 F16。decode 流水线(argmax → embedding → 下一轮)在 GPU 上闭环,
/// CPU 不必逐 token 同步取 id 再上传 embedding。
pub fn gguf_gather_row_q6k_tensor(ctx: &MetalContext, token_id: &metal::Buffer, blob: &metal::Buffer, weight_rows: usize, weight_cols: usize, row_bytes: usize) -> Result<MetalTensor, String> {
    gguf_gather_row_q6k_tensor_offset(ctx, token_id, 0, blob, weight_rows, weight_cols, row_bytes)
}

/// `id_offset` 是 token id 在 buffer 中的字节偏移(异步流水线 per-position 读回区)。
#[allow(clippy::too_many_arguments)]
pub fn gguf_gather_row_q6k_tensor_offset(ctx: &MetalContext, token_id: &metal::Buffer, id_offset: u64, blob: &metal::Buffer, weight_rows: usize, weight_cols: usize, row_bytes: usize) -> Result<MetalTensor, String> {
    if weight_cols == 0 || !weight_cols.is_multiple_of(256) {
        return Err(format!("Q6_K gather columns={weight_cols} 必须是 256 倍数"));
    }
    if row_bytes != weight_cols / 256 * 210 {
        return Err(format!("Q6_K gather row_bytes={row_bytes} 与 columns={weight_cols} 不符"));
    }
    let needed = (weight_rows as u64).checked_mul(row_bytes as u64).ok_or("Q6_K gather 大小溢出")?;
    if blob.length() < needed {
        return Err(format!("Q6_K gather weight buffer {}B 不足 {needed}B", blob.length()));
    }
    let output = ctx.tensor_uninit(1, weight_cols);
    let pipeline = ctx.pipeline("gguf_gather_row_q6k_f16")?;
    let columns = validate_u32("Q6_K gather columns", weight_cols)?;
    let row_bytes_u32 = validate_u32("Q6_K gather row bytes", row_bytes)?;
    let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(token_id), id_offset);
    encoder.set_buffer(1, Some(blob), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &row_bytes_u32);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, "gguf_gather_row_q6k_f16", &format!("columns={weight_cols}"), row_bytes as u64, output.buffer.length());
    Ok(output)
}

/// 设备端 embedding gather(Q4_K):按 `token_id`(设备 u32 buffer,`id_offset` 为其
/// 字节偏移,对应异步流水线 per-position 读回区)从常驻 Q4_K GGUF 矩阵抠一行,
/// dequant 后乘 `embedding_scale`,输出 F16。数值与 CPU `gemma4_embedding_rows`
/// 的双 BF16 舍入路径逐位一致(见 kernel 注释)。
#[allow(clippy::too_many_arguments)]
pub fn gguf_gather_row_q4k_tensor_offset(ctx: &MetalContext, token_id: &metal::Buffer, id_offset: u64, blob: &metal::Buffer, weight_rows: usize, weight_cols: usize, row_bytes: usize, embedding_scale: f32) -> Result<MetalTensor, String> {
    if weight_cols == 0 || !weight_cols.is_multiple_of(256) {
        return Err(format!("Q4_K gather columns={weight_cols} 必须是 256 倍数"));
    }
    if row_bytes != weight_cols / 256 * 144 {
        return Err(format!("Q4_K gather row_bytes={row_bytes} 与 columns={weight_cols} 不符"));
    }
    if !embedding_scale.is_finite() {
        return Err(format!("Q4_K gather embedding_scale={embedding_scale} 非法"));
    }
    let needed = (weight_rows as u64).checked_mul(row_bytes as u64).ok_or("Q4_K gather 大小溢出")?;
    if blob.length() < needed {
        return Err(format!("Q4_K gather weight buffer {}B 不足 {needed}B", blob.length()));
    }
    let output = ctx.tensor_uninit(1, weight_cols);
    let pipeline = ctx.pipeline("gguf_gather_row_q4k_f16")?;
    let columns = validate_u32("Q4_K gather columns", weight_cols)?;
    let row_bytes_u32 = validate_u32("Q4_K gather row bytes", row_bytes)?;
    let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(token_id), id_offset);
    encoder.set_buffer(1, Some(blob), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &row_bytes_u32);
    set_bytes(&encoder, 5, &embedding_scale);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, "gguf_gather_row_q4k_f16", &format!("columns={weight_cols}"), row_bytes as u64, output.buffer.length());
    Ok(output)
}

/// 设备端 E4B per-layer embedding gather；Q5_K 行解码和双 BF16 舍入与
/// `gemma4_metal_per_layer_inputs` 的 CPU 路径逐位一致。
#[allow(clippy::too_many_arguments)]
pub fn gguf_gather_row_q5k_tensor_offset(ctx: &MetalContext, token_id: &metal::Buffer, id_offset: u64, blob: &metal::Buffer, weight_rows: usize, weight_cols: usize, row_bytes: usize, embedding_scale: f32) -> Result<MetalTensor, String> {
    if weight_cols == 0 || !weight_cols.is_multiple_of(256) {
        return Err(format!("Q5_K gather columns={weight_cols} 必须是 256 倍数"));
    }
    if row_bytes != weight_cols / 256 * 176 {
        return Err(format!("Q5_K gather row_bytes={row_bytes} 与 columns={weight_cols} 不符"));
    }
    if !embedding_scale.is_finite() {
        return Err(format!("Q5_K gather embedding_scale={embedding_scale} 非法"));
    }
    let needed = (weight_rows as u64).checked_mul(row_bytes as u64).ok_or("Q5_K gather 大小溢出")?;
    if blob.length() < needed {
        return Err(format!("Q5_K gather weight buffer {}B 不足 {needed}B", blob.length()));
    }
    let output = ctx.tensor_uninit(1, weight_cols);
    let pipeline = ctx.pipeline("gguf_gather_row_q5k_f16")?;
    let columns = validate_u32("Q5_K gather columns", weight_cols)?;
    let row_bytes_u32 = validate_u32("Q5_K gather row bytes", row_bytes)?;
    let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(token_id), id_offset);
    encoder.set_buffer(1, Some(blob), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &row_bytes_u32);
    set_bytes(&encoder, 5, &embedding_scale);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, "gguf_gather_row_q5k_f16", &format!("columns={weight_cols}"), row_bytes as u64, output.buffer.length());
    Ok(output)
}

/// 把常驻 GGUF 量化矩阵整块 dequant 成 F16(与 prefill 大矩阵回退路径同一 kernel)。
/// 用于 untied embedding 的一次性常驻化:之后每 token 的 gather 退化为行拷贝。
pub fn gguf_dequant_matrix_f16_tensor(ctx: &MetalContext, blob: &metal::Buffer, tensor_type: u32, row_bytes: usize, weight_rows: usize, weight_cols: usize) -> Result<MetalTensor, String> {
    let columns = validate_u32("GGUF dequant columns", weight_cols)?;
    let rows = validate_u32("GGUF dequant rows", weight_rows)?;
    let row_bytes_u32 = validate_u32("GGUF dequant row bytes", row_bytes)?;
    let element_count = weight_rows.checked_mul(weight_cols).ok_or("GGUF dequant 元素数溢出")?;
    let weight_f16 = ctx.tensor_kernel_output(weight_rows, weight_cols);
    let iq2s_grid = gguf_iq2s_grid_buffer(ctx);
    let pipeline_name = match tensor_type {
        11 => "gguf_dequant_q3k_matrix_f16",
        21 => "gguf_dequant_iq3s_matrix_f16",
        22 => "gguf_dequant_iq2s_matrix_f16",
        23 => "gguf_dequant_iq4xs_matrix_f16",
        _ => "gguf_dequant_matrix_f16",
    };
    let pipeline = ctx.pipeline(pipeline_name)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(blob), 0);
    encoder.set_buffer(1, Some(&iq2s_grid), 0);
    encoder.set_buffer(2, Some(&weight_f16.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &rows);
    set_bytes(&encoder, 5, &tensor_type);
    set_bytes(&encoder, 6, &row_bytes_u32);
    const DEQUANT_THREADS: usize = 256;
    if matches!(tensor_type, 11 | 22) {
        encoder.dispatch_thread_groups(MTLSize::new((weight_cols / 256) as u64, weight_rows as u64, 1), MTLSize::new(64, 1, 1));
    } else if matches!(tensor_type, 21 | 23) {
        encoder.dispatch_thread_groups(MTLSize::new((weight_cols / 32).div_ceil(64) as u64, weight_rows as u64, 1), MTLSize::new(64, 1, 1));
    } else {
        encoder.dispatch_thread_groups(MTLSize::new(element_count.div_ceil(DEQUANT_THREADS) as u64, 1, 1), MTLSize::new(DEQUANT_THREADS as u64, 1, 1));
    }
    encoder.end_encoding();
    let shape = format!("weight=[{weight_rows},{weight_cols}],type={tensor_type}");
    ctx.commit_and_wait_profiled(&command, pipeline_name, &shape, blob.length() + iq2s_grid.length(), weight_f16.buffer.length());
    Ok(weight_f16)
}

pub fn gguf_gated_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate_blob: &metal::Buffer,
    gate_type: u32,
    gate_row_bytes: usize,
    gate_rows: usize,
    gate_cols: usize,
    up_blob: &metal::Buffer,
    up_type: u32,
    up_row_bytes: usize,
    up_rows: usize,
    up_cols: usize,
    activation: &Activation,
) -> Result<MetalTensor, String> {
    // fused gated gemv:单行全类型;iq3s 已多行化(grid.y 每 8 行一组,行数无上限)。
    // 其余类型 2 行以上走分解路径(2 gemv + activation)——iq4nl 多行 gemv 是
    // 8 行批(权重读一次),分解后两行各自共享,优于 fused 单行 kernel 的双倍权重流量。
    // (iq4nl 单行分解 A/B:kernel 时间 -10ms 但 +96 dispatch 净 -3ms 更差,fused 保留)
    if input.rows == 1 || (input.rows < gguf_prefill_mps_rows() && gate_type == 21 && up_type == 21) {
        return gguf_gated_gemv_tensor_resident(ctx, input, gate_blob, gate_type, gate_row_bytes, gate_rows, gate_cols, up_blob, up_type, up_row_bytes, up_rows, up_cols, activation);
    }
    if input.cols != gate_cols || gate_cols != up_cols || gate_rows != up_rows {
        return Err(format!("GGUF gated matmul shape 不兼容: input=[{},{}], gate=[{gate_rows},{gate_cols}], up=[{up_rows},{up_cols}]", input.rows, input.cols,));
    }
    // IQ4_NL 长 prefill 复用 tiled fused GEMM，避免 gate/up 各自先展开整份
    // F16 权重。activation 仍走统一 epilogue，保持与分解路径相同的语义。
    if input.rows >= 4 && gate_type == 20 && up_type == 20 {
        let gate = gguf_matmul_tensor_resident(ctx, input, gate_blob, gate_type, gate_row_bytes, gate_rows, gate_cols)?;
        let up = gguf_matmul_tensor_resident(ctx, input, up_blob, up_type, up_row_bytes, up_rows, up_cols)?;
        return gated_activation_tensor(ctx, &gate, &up, activation);
    }
    // IQ3_S gate/up 在 32 行以上同样走两次 fused GEMM + 激活 epilogue,
    // 替代 dual_dequant 物化 + MPS 两步(58-token prefill 占 GPU 38%)。
    if input.rows >= gguf_prefill_mps_rows() && gate_type == 21 && up_type == 21 {
        let gate = gguf_matmul_tensor_resident(ctx, input, gate_blob, gate_type, gate_row_bytes, gate_rows, gate_cols)?;
        let up = gguf_matmul_tensor_resident(ctx, input, up_blob, up_type, up_row_bytes, up_rows, up_cols)?;
        return gated_activation_tensor(ctx, &gate, &up, activation);
    }
    if input.rows < gguf_prefill_mps_rows() {
        let gate = gguf_matmul_tensor_resident(ctx, input, gate_blob, gate_type, gate_row_bytes, gate_rows, gate_cols)?;
        let up = gguf_matmul_tensor_resident(ctx, input, up_blob, up_type, up_row_bytes, up_rows, up_cols)?;
        return gated_activation_tensor(ctx, &gate, &up, activation);
    }
    for (name, blob, tensor_type, row_bytes, rows, columns) in [("gate", gate_blob, gate_type, gate_row_bytes, gate_rows, gate_cols), ("up", up_blob, up_type, up_row_bytes, up_rows, up_cols)] {
        let expected_row_bytes = gguf_expected_row_bytes(tensor_type, columns)?;
        if row_bytes != expected_row_bytes {
            return Err(format!("GGUF {name} row bytes={row_bytes}，期望 {expected_row_bytes}"));
        }
        validate_size(&format!("GGUF {name} buffer"), row_bytes.checked_mul(rows).ok_or_else(|| format!("GGUF {name} 大小溢出"))?, blob.length() as usize)?;
    }

    const DEQUANT_THREADS: usize = 256;
    let combined_rows = gate_rows.checked_add(up_rows).ok_or("GGUF gate/up rows 溢出")?;
    let combined_weight = ctx.tensor_kernel_output(combined_rows, gate_cols);
    let packed = ctx.tensor_kernel_output(input.rows, combined_rows);
    let grid = ctx.resident_byte_weight_buffer(as_bytes(crate::weight::codec::ggml::iq2s_grid()));
    let generic_pipeline = ctx.pipeline("gguf_dequant_matrix_f16")?;
    let q3k_pipeline = ctx.pipeline("gguf_dequant_q3k_matrix_f16")?;
    let iq2s_pipeline = ctx.pipeline("gguf_dequant_iq2s_matrix_f16")?;
    let iq3s_pipeline = ctx.pipeline("gguf_dequant_iq3s_matrix_f16")?;
    let iq4xs_pipeline = ctx.pipeline("gguf_dequant_iq4xs_matrix_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_buffer(1, Some(&grid), 0);
    for (blob, tensor_type, row_bytes, rows, output_offset) in [(gate_blob, gate_type, gate_row_bytes, gate_rows, 0usize), (up_blob, up_type, up_row_bytes, up_rows, gate_rows * gate_cols * mem::size_of::<u16>())] {
        let element_count = rows.checked_mul(gate_cols).ok_or("GGUF gate/up dequant 元素数溢出")?;
        encoder.set_compute_pipeline_state(match tensor_type {
            11 => &q3k_pipeline,
            21 => &iq3s_pipeline,
            22 => &iq2s_pipeline,
            23 => &iq4xs_pipeline,
            _ => &generic_pipeline,
        });
        encoder.set_buffer(0, Some(blob), 0);
        encoder.set_buffer(2, Some(&combined_weight.buffer), output_offset as u64);
        set_bytes(&encoder, 3, &validate_u32("GGUF columns", gate_cols)?);
        set_bytes(&encoder, 4, &validate_u32("GGUF rows", rows)?);
        set_bytes(&encoder, 5, &tensor_type);
        set_bytes(&encoder, 6, &validate_u32("GGUF row bytes", row_bytes)?);
        if matches!(tensor_type, 11 | 22) {
            encoder.dispatch_thread_groups(MTLSize::new((gate_cols / 256) as u64, rows as u64, 1), MTLSize::new(64, 1, 1));
        } else if matches!(tensor_type, 21 | 23) {
            encoder.dispatch_thread_groups(MTLSize::new((gate_cols / 32).div_ceil(64) as u64, rows as u64, 1), MTLSize::new(64, 1, 1));
        } else {
            encoder.dispatch_thread_groups(MTLSize::new(element_count.div_ceil(DEQUANT_THREADS) as u64, 1, 1), MTLSize::new(DEQUANT_THREADS as u64, 1, 1));
        }
    }
    encoder.end_encoding();
    let shape = format!("input=[{},{}],gate/up=[{gate_rows},{gate_cols}],types={gate_type}/{up_type}", input.rows, input.cols);
    ctx.commit_and_wait_profiled(&command, "gguf_dual_dequant_gated_f16", &shape, gate_blob.length() + up_blob.length() + grid.length(), combined_weight.buffer.length());
    launch_matmul_f16(ctx, &input.buffer, &combined_weight.buffer, &packed.buffer, input.rows, gate_cols, combined_rows)?;

    let params = GatedActivation::from_spec(activation)?;
    let output = ctx.tensor_kernel_output(input.rows, gate_rows);
    let count = validate_u32("GGUF packed activation count", output.len())?;
    let columns = validate_u32("GGUF packed activation columns", gate_rows)?;
    launch_1d(ctx, "gated_activation_packed_f16", &format!("rows={},columns={gate_rows}", input.rows), output.len(), packed.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&packed.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &params.kind);
        set_bytes(encoder, 5, &params.alpha);
        set_bytes(encoder, 6, &params.limit);
    })?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn gguf_q3k_gemv_accumulate_f32(ctx: &MetalContext, input: &MetalTensor, blob: &metal::Buffer, row_bytes: usize, weight_rows: usize, weight_cols: usize, output: &MetalF32Accumulator, route_weight: f32) -> Result<(), String> {
    if input.rows != 1 || input.cols != weight_cols || output.rows != 1 || output.cols != weight_rows {
        return Err(format!("Q3_K accumulate shape 不兼容: input=[{},{}], weight=[{weight_rows},{weight_cols}], output=[{},{}]", input.rows, input.cols, output.rows, output.cols,));
    }
    if !weight_cols.is_multiple_of(256) || row_bytes != weight_cols / 256 * 110 {
        return Err(format!("Q3_K accumulate row bytes={row_bytes} 与 columns={weight_cols} 不兼容"));
    }
    validate_size("Q3_K accumulate weight buffer", row_bytes.checked_mul(weight_rows).ok_or("Q3_K accumulate 权重大小溢出")?, blob.length() as usize)?;
    let input = to_f16_tensor(ctx, input)?;
    let columns = validate_u32("Q3_K accumulate columns", weight_cols)?;
    let rows = validate_u32("Q3_K accumulate rows", weight_rows)?;
    let row_bytes = validate_u32("Q3_K accumulate row bytes", row_bytes)?;
    let pipeline = ctx.pipeline("gguf_gemv_q3k_accumulate_f32")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(blob), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &columns);
    set_bytes(&encoder, 5, &rows);
    set_bytes(&encoder, 7, &row_bytes);
    set_bytes(&encoder, 8, &route_weight);
    encoder.dispatch_thread_groups(MTLSize::new(weight_rows.div_ceil(4) as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{weight_cols}],weight=[{weight_rows},{weight_cols}],type=11");
    ctx.commit_and_wait_profiled(&command, "gguf_gemv_q3k_accumulate_f32", &shape, input.buffer.length() + blob.length() + output.buffer.length(), output.buffer.length());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gguf_iq2s_gemv_accumulate_f32(ctx: &MetalContext, input: &MetalTensor, blob: &metal::Buffer, row_bytes: usize, weight_rows: usize, weight_cols: usize, output: &MetalF32Accumulator, route_weight: f32) -> Result<(), String> {
    if input.rows != 1 || input.cols != weight_cols || output.rows != 1 || output.cols != weight_rows {
        return Err(format!("IQ2_S accumulate shape 不兼容: input=[{},{}], weight=[{weight_rows},{weight_cols}], output=[{},{}]", input.rows, input.cols, output.rows, output.cols,));
    }
    if !weight_cols.is_multiple_of(256) || row_bytes != weight_cols / 256 * 82 {
        return Err(format!("IQ2_S accumulate row bytes={row_bytes} 与 columns={weight_cols} 不兼容"));
    }
    validate_size("IQ2_S accumulate weight buffer", row_bytes.checked_mul(weight_rows).ok_or("IQ2_S accumulate 权重大小溢出")?, blob.length() as usize)?;
    let input = to_f16_tensor(ctx, input)?;
    let columns = validate_u32("IQ2_S accumulate columns", weight_cols)?;
    let rows = validate_u32("IQ2_S accumulate rows", weight_rows)?;
    let row_bytes = validate_u32("IQ2_S accumulate row bytes", row_bytes)?;
    let grid = ctx.resident_byte_weight_buffer(as_bytes(crate::weight::codec::ggml::iq2s_grid()));
    let pipeline = ctx.pipeline("gguf_gemv_iq2s_accumulate_f32")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(blob), 0);
    encoder.set_buffer(2, Some(&grid), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &columns);
    set_bytes(&encoder, 5, &rows);
    set_bytes(&encoder, 7, &row_bytes);
    set_bytes(&encoder, 8, &route_weight);
    encoder.dispatch_thread_groups(MTLSize::new(weight_rows.div_ceil(8) as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{weight_cols}],weight=[{weight_rows},{weight_cols}],type=22");
    ctx.commit_and_wait_profiled(&command, "gguf_gemv_iq2s_accumulate_f32", &shape, input.buffer.length() + blob.length() + grid.length() + output.buffer.length(), output.buffer.length());
    Ok(())
}

pub(super) fn gguf_expected_row_bytes(tensor_type: u32, columns: usize) -> Result<usize, String> {
    let (block_elements, block_bytes) = metal_block_layout(tensor_type)?;
    if !columns.is_multiple_of(block_elements) {
        return Err(format!("GGUF expert columns={columns} 未按 block {block_elements} 对齐"));
    }
    Ok(columns / block_elements * block_bytes)
}

fn metal_block_layout(tensor_type: u32) -> Result<(usize, usize), String> {
    if !crate::weight::codec::ggml::supports_decode(tensor_type) {
        return Err(format!("Metal GGUF tensor type {tensor_type} 不受支持"));
    }
    crate::weight::codec::ggml::block_layout(tensor_type)
}

pub fn gguf_iq2s_grid_buffer(ctx: &MetalContext) -> metal::Buffer {
    ctx.resident_byte_weight_buffer(as_bytes(crate::weight::codec::ggml::iq2s_grid()))
}

pub fn gguf_gated_gemv_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate_blob: &metal::Buffer,
    gate_type: u32,
    gate_row_bytes: usize,
    gate_rows: usize,
    gate_cols: usize,
    up_blob: &metal::Buffer,
    up_type: u32,
    up_row_bytes: usize,
    up_rows: usize,
    up_cols: usize,
    activation: &Activation,
) -> Result<MetalTensor, String> {
    // iq3s gated gemv 已多行化(grid.y 每 8 行一组);iq4nl 支持 1-2 行(2 行共享
    // 权重反量化);其余 gated kernel 仍是单行
    let multirow_capable = gate_type == 21 && up_type == 21;
    let two_row_capable = gate_type == 20 && up_type == 20;
    if input.rows == 0 || (input.rows == 2 && !two_row_capable && !multirow_capable) || (input.rows > 2 && !multirow_capable) || input.cols != gate_cols || gate_cols != up_cols || gate_rows != up_rows {
        return Err(format!("GGUF gated GEMV shape 不兼容: input=[{},{}], gate=[{gate_rows},{gate_cols}], up=[{up_rows},{up_cols}]", input.rows, input.cols,));
    }
    if !crate::weight::codec::ggml::supports_decode(gate_type) || !crate::weight::codec::ggml::supports_decode(up_type) {
        return Err(format!("GGUF gated GEMV tensor type 不受支持: gate={gate_type}, up={up_type}"));
    }
    validate_size("GGUF gated gate buffer", gate_row_bytes.checked_mul(gate_rows).ok_or("GGUF gate 大小溢出")?, gate_blob.length() as usize)?;
    validate_size("GGUF gated up buffer", up_row_bytes.checked_mul(up_rows).ok_or("GGUF up 大小溢出")?, up_blob.length() as usize)?;
    let params = GatedActivation::from_spec(activation)?;
    let columns = validate_u32("GGUF gated columns", input.cols)?;
    let output_rows = validate_u32("GGUF gated output rows", gate_rows)?;
    let gate_row_bytes = validate_u32("GGUF gated gate row bytes", gate_row_bytes)?;
    let up_row_bytes = validate_u32("GGUF gated up row bytes", up_row_bytes)?;
    let output = ctx.tensor_kernel_output(input.rows, gate_rows);
    let iq2s_grid = ctx.resident_byte_weight_buffer(as_bytes(crate::weight::codec::ggml::iq2s_grid()));
    let pipeline_name = if gate_type == 11 && up_type == 11 {
        "gguf_gated_gemv_q3k_f16"
    } else if gate_type == 2 && up_type == 2 {
        // q4_0 gated:8 行/threadgroup,gate/up 同读一次输入;此前落通用
        // gguf_gated_gemv_f16 标量解码,decode 实测仅 ~2 tok/s
        "gguf_gated_gemv_q4_0_f16"
    } else if gate_type == 12 && up_type == 12 {
        // decode 单行用双行变体:96 个 work item 恰好摊满 32 lane(见 kernel 注释)
        if input.rows == 1 { "gguf_gated_gemv_q4k_2r_f16" } else { "gguf_gated_gemv_q4k_f16" }
    } else if gate_type == 22 && up_type == 22 {
        "gguf_gated_gemv_iq2s_f16"
    } else if gate_type == 18 && up_type == 18 {
        "gguf_gated_gemv_iq3xxs_f16"
    } else if gate_type == 23 && up_type == 23 {
        "gguf_gated_gemv_iq4xs_f16"
    } else if gate_type == 20 && up_type == 20 {
        "gguf_gated_gemv_iq4nl_f16"
    } else if gate_type == 21 && up_type == 21 {
        // 单行(decode)专用:去掉 8 行累加器的寄存器压力,交替 A/B 实测 +13%
        if input.rows == 1 { "gguf_gated_gemv_iq3s_1r_f16" } else { "gguf_gated_gemv_iq3s_f16" }
    } else {
        "gguf_gated_gemv_f16"
    };
    let pipeline = ctx.pipeline(pipeline_name)?;
    let packed_rows = matches!(gate_type, 2 | 11 | 12 | 18 | 22) && gate_type == up_type;
    let iq_packed_rows = matches!(gate_type, 20 | 21 | 23) && gate_type == up_type;
    let threads = if gate_type == 11 && up_type == 11 {
        64
    } else if gate_type == 2 && up_type == 2 {
        // q4_0 gated:16 K-lane x 2 行/simdgroup,4 simdgroups = 8 行/threadgroup
        128
    } else if iq_packed_rows {
        128
    } else if gate_type == 12 && up_type == 12 && input.rows == 1 {
        // 双行变体:4 simdgroups x 2 行 = 8 行/threadgroup
        128
    } else if packed_rows {
        256
    } else {
        64
    };
    if pipeline.max_total_threads_per_threadgroup() < threads as u64 {
        return Err(format!("GGUF gated GEMV 需要至少 {threads} threads/threadgroup"));
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(gate_blob), 0);
    encoder.set_buffer(2, Some(up_blob), 0);
    encoder.set_buffer(3, Some(&iq2s_grid), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &columns);
    set_bytes(&encoder, 6, &output_rows);
    set_bytes(&encoder, 7, &gate_type);
    set_bytes(&encoder, 8, &up_type);
    set_bytes(&encoder, 9, &gate_row_bytes);
    set_bytes(&encoder, 10, &up_row_bytes);
    set_bytes(&encoder, 11, &params.kind);
    set_bytes(&encoder, 12, &params.alpha);
    set_bytes(&encoder, 13, &params.limit);
    // iq3s 多行 gemv 读 input_rows;iq4nl 是 per-y 单行(grid.y = input 行数,
    // 权重经 L2 跨行共享);其余 gated kernel 不读 group.y,grid.y 必须保持 1
    let gated_input_rows = validate_u32("GGUF gated input rows", input.rows.max(1))?;
    set_bytes(&encoder, 14, &gated_input_rows);
    let grid_y = if gate_type == 21 && up_type == 21 {
        input.rows.div_ceil(8)
    } else if gate_type == 20 && up_type == 20 {
        input.rows
    } else {
        1
    };
    encoder.dispatch_thread_groups(
        MTLSize::new(
            if gate_type == 11 && up_type == 11 {
                gate_rows.div_ceil(4)
            } else if gate_type == 21 && up_type == 21 {
                // iq3s 多行版:16 K-lane x 2 行/simdgroup,128 threads = 8 行/threadgroup
                gate_rows.div_ceil(8)
            } else if iq_packed_rows {
                gate_rows.div_ceil(4)
            } else if packed_rows {
                gate_rows.div_ceil(8)
            } else {
                gate_rows
            } as u64,
            grid_y as u64,
            1,
        ),
        MTLSize::new(threads as u64, 1, 1),
    );
    encoder.end_encoding();
    let shape = format!("input=[{},{}],output={gate_rows},types={gate_type}/{up_type}", input.rows, input.cols);
    ctx.commit_and_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + gate_blob.length() + up_blob.length() + iq2s_grid.length(), output.buffer.length());
    Ok(output)
}
