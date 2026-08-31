//! Apple Metal 计算算法。设备资源和执行调度位于 `backend::metal`。

use crate::backend::metal::api as metal;

pub mod attention;
pub mod attn_res;
pub mod compressed_sparse;
pub mod dense;
pub mod diffusion;
pub mod fp8;
pub mod fused;
pub mod gguf;
pub mod hyper_connection;
pub mod kda;
pub mod low_bit;
pub mod mla;
pub mod mlx;
pub mod moe;
pub mod mps;
pub mod preamble;
pub mod shape;
pub mod tensor;
pub mod vae;
pub mod vision;

/// 把所有模块的 Metal shader 拼成一个完整字符串,首次访问时一次性分配。
///
/// 每个模块自己负责自己的 SHADERS const(代码就近原则),这里只是把它们
/// 拼起来给 `MetalContext::new` 用。原 `kernels.rs` 已被删除。
static KERNELS: OnceLock<String> = OnceLock::new();

/// 强制初始化并返回完整 shader 字符串,后续调用零拷贝。
pub fn kernels_source() -> &'static str {
    let parts: [&str; 19] = [
        preamble::SHADERS,
        dense::SHADERS,
        shape::SHADERS,
        attention::SHADERS,
        compressed_sparse::SHADERS,
        mla::shaders(),
        kda::SHADERS,
        mlx::SHADERS,
        gguf::shaders(),
        fp8::SHADERS,
        fused::SHADERS,
        low_bit::shaders(),
        moe::SHADERS,
        vision::SHADERS,
        vae::SHADERS,
        diffusion::SHADERS,
        tensor::SHADERS,
        attn_res::SHADERS,
        hyper_connection::SHADERS,
    ];
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

use crate::attention::{
    gated_delta_net::GatedDeltaNetSpec,
    gqa::{CausalWindow, GqaSpec},
    mla::MlaSpec,
};
use crate::backend::metal::api::MTLSize;
use crate::backend::metal::kv_cache::MetalGqaCacheView;
use crate::backend::metal::{MetalContext, MetalKvCache, MetalKvCacheFormat, MetalTensor, MetalTensorDType};
use crate::moe::Activation;
use crate::weight::Fp8Matrix;
use half::f16;
use std::{ffi::c_void, mem, sync::OnceLock};

const THREADS: usize = 256;
const FP8_PREFILL_MPS_ROWS: usize = 128;

fn nvfp4_direct_rows() -> usize {
    1
}

pub(crate) fn set_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(index, mem::size_of::<T>() as u64, value as *const T as *const c_void);
}

pub fn to_f16_tensor(ctx: &MetalContext, input: &MetalTensor) -> Result<MetalTensor, String> {
    if input.dtype == MetalTensorDType::F16 {
        return Ok(input.clone());
    }
    if let Some(output) = ctx.cached_f16_cast(input) {
        return Ok(output);
    }
    let output = ctx.tensor_kernel_output(input.rows, input.cols);
    let count = validate_u32("F32/BF16->F16 count", input.len())?;
    let shape = format!("elements={}", input.len());
    let (pipeline, total) = match input.dtype {
        MetalTensorDType::F16 => unreachable!(),
        MetalTensorDType::Bf16 => ("cast_bf16_f16", input.len()),
        MetalTensorDType::F32 => ("cast_f32_f16_x4", input.len().div_ceil(4)),
    };
    launch_1d(ctx, pipeline, &shape, total, input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
    })?;
    ctx.retain_f16_cast(input, &output);
    Ok(output)
}

pub fn to_f32_tensor(ctx: &MetalContext, input: &MetalTensor) -> Result<MetalTensor, String> {
    if input.dtype == MetalTensorDType::F32 {
        return Ok(input.clone());
    }
    let output = ctx.tensor_kernel_output_f32(input.rows, input.cols);
    let count = validate_u32("F16/BF16->F32 count", input.len())?;
    let pipeline = if input.dtype == MetalTensorDType::Bf16 { "cast_bf16_to_f32" } else { "cast_f16_to_f32" };
    let shape = format!("elements={}", input.len());
    launch_1d(ctx, pipeline, &shape, input.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
    })?;
    Ok(output)
}

pub fn to_bf16_tensor(ctx: &MetalContext, input: &MetalTensor) -> Result<MetalTensor, String> {
    if input.dtype == MetalTensorDType::Bf16 {
        return Ok(input.clone());
    }
    let output = ctx.tensor_kernel_output_bf16(input.rows, input.cols);
    let count = validate_u32("F16/F32->BF16 count", input.len())?;
    let pipeline = match input.dtype {
        MetalTensorDType::F16 => "cast_f16_bf16",
        MetalTensorDType::Bf16 => unreachable!(),
        MetalTensorDType::F32 => "cast_f32_bf16",
    };
    let shape = format!("elements={}", input.len());
    launch_1d(ctx, pipeline, &shape, input.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
    })?;
    Ok(output)
}

fn launch_nd<F>(ctx: &MetalContext, pipeline: &str, shape: &str, thread_groups: MTLSize, threads_per_group: MTLSize, estimated_read_bytes: u64, estimated_write_bytes: u64, setup: F) -> Result<(), String>
where
    F: FnOnce(&metal::ComputeCommandEncoderRef),
{
    let pipeline_state = ctx.pipeline(pipeline)?;
    let command_buffer = ctx.command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);

    setup(&encoder);

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command_buffer, pipeline, shape, estimated_read_bytes, estimated_write_bytes);
    Ok(())
}

fn launch_2d<F>(ctx: &MetalContext, pipeline: &str, shape: &str, groups_x: usize, groups_y: usize, threads: usize, estimated_read_bytes: u64, estimated_write_bytes: u64, setup: F) -> Result<(), String>
where
    F: FnOnce(&metal::ComputeCommandEncoderRef),
{
    if groups_x == 0 || groups_y == 0 {
        return Ok(());
    }
    launch_nd(ctx, pipeline, shape, MTLSize::new(groups_x as u64, groups_y as u64, 1), MTLSize::new(threads as u64, 1, 1), estimated_read_bytes, estimated_write_bytes, setup)
}

fn launch_1d<F>(ctx: &MetalContext, pipeline: &str, shape: &str, total: usize, estimated_read_bytes: u64, estimated_write_bytes: u64, setup: F) -> Result<(), String>
where
    F: FnOnce(&metal::ComputeCommandEncoderRef),
{
    if total == 0 {
        return Ok(());
    }
    let threads = total.min(THREADS);
    let groups = total.div_ceil(threads);
    launch_nd(ctx, pipeline, shape, MTLSize::new(groups as u64, 1, 1), MTLSize::new(threads as u64, 1, 1), estimated_read_bytes, estimated_write_bytes, setup)
}

fn launch_rows_with_pipeline<F>(ctx: &MetalContext, pipeline: &str, rows: usize, columns: usize, estimated_read_bytes: u64, estimated_write_bytes: u64, setup: F) -> Result<(), String>
where
    F: FnOnce(&metal::ComputeCommandEncoderRef),
{
    if rows == 0 {
        return Ok(());
    }
    let pipeline_state = ctx.pipeline(pipeline)?;
    let command_buffer = ctx.command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);

    setup(&encoder);

    // rms_norm/layernorm 家族的树归约要求 width 为 2 的幂；clamp 上限是 2 的幂，取幂后仍 ≤ THREADS。
    let threads = columns.clamp(1, THREADS).next_power_of_two();
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("[{rows},{columns}]");
    ctx.commit_and_wait_profiled(&command_buffer, pipeline, &shape, estimated_read_bytes, estimated_write_bytes);
    Ok(())
}

fn validate_u32(name: &str, value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} 超过 u32 上限: {value}"))
}

fn validate_size(name: &str, expected: usize, actual: usize) -> Result<(), String> {
    if expected == actual {
        return Ok(());
    }
    Err(format!("{name} 长度不匹配: 实际={actual}, 期望={expected}"))
}

fn f32_to_f16(ctx: &MetalContext, data: &[f32]) -> metal::Buffer {
    let half_data: Vec<f16> = data.iter().copied().map(f16::from_f32).collect();
    let bytes = unsafe { std::slice::from_raw_parts(half_data.as_ptr() as *const u8, half_data.len() * mem::size_of::<f16>()) };
    ctx.device.new_buffer_with_data(bytes.as_ptr() as *const c_void, bytes.len() as u64, metal::MTLResourceOptions::StorageModeShared)
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), mem::size_of_val(values)) }
}

// 子文件共享本模块的私有 launch/校验 helper；保持原 API 与可见性，不增加中间抽象。
// (mod 声明已在上方,这里不再重复。)

#[cfg(all(test, target_os = "macos"))]
mod icb_tests {
    use super::*;

    /// ICB compute 原语验证:录制 → 重放(改输入内容)→ ICB 内链式依赖按序生效。
    /// 注意:PSO 必须来自 ICB 兼容的 pipeline()(supportIndirectCommandBuffers=true,
    /// context 统一走 descriptor 路径)。
    #[test]
    fn icb_compute_records_replays_and_orders() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let icb = ctx.device.new_indirect_command_buffer(4).expect("创建 ICB");
        let pipeline = ctx.pipeline("add_f32").unwrap();
        let a = ctx.shared_buffer(&1.0f32.to_le_bytes());
        let b = ctx.shared_buffer(&2.0f32.to_le_bytes());
        let output = ctx.shared_buffer(&0.0f32.to_le_bytes());
        let count = ctx.shared_buffer(&1u32.to_le_bytes());

        let command = icb.compute_command(0);
        command.set_pipeline(&pipeline);
        command.set_kernel_buffer(0, &a, 0);
        command.set_kernel_buffer(1, &b, 0);
        command.set_kernel_buffer(2, &output, 0);
        command.set_kernel_buffer(3, &count, 0);
        command.dispatch(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));

        let run = |expected: f32| {
            let command = ctx.command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.execute_indirect(&icb, 1);
            encoder.end_encoding();
            ctx.commit_and_wait(&command);
            let value = f32::from_le_bytes(unsafe { std::slice::from_raw_parts(output.contents() as *const u8, 4) }.try_into().unwrap());
            assert!((value - expected).abs() < 1e-6, "ICB 输出 {value},期望 {expected}");
        };
        run(3.0);

        // 重放语义:改共享输入内容,再执行同一 ICB,输出应跟随
        unsafe { std::ptr::write(a.contents() as *mut f32, 10.0) };
        run(12.0);

        // 链式依赖:cmd0 写 c = a+b;cmd1 写 output = c+b。ICB concurrent 命令间
        // 无顺序保证(encoder 看不到 ICB 内资源依赖),依赖链必须分段 execute + barrier。
        icb.reset(2);
        let c = ctx.shared_buffer(&0.0f32.to_le_bytes());
        let command0 = icb.compute_command(0);
        command0.set_pipeline(&pipeline);
        command0.set_kernel_buffer(0, &a, 0);
        command0.set_kernel_buffer(1, &b, 0);
        command0.set_kernel_buffer(2, &c, 0);
        command0.set_kernel_buffer(3, &count, 0);
        command0.dispatch(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
        let command1 = icb.compute_command(1);
        command1.set_pipeline(&pipeline);
        command1.set_kernel_buffer(0, &c, 0);
        command1.set_kernel_buffer(1, &b, 0);
        command1.set_kernel_buffer(2, &output, 0);
        command1.set_kernel_buffer(3, &count, 0);
        command1.dispatch(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));

        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.execute_indirect_range(&icb, 0, 1);
        encoder.memory_barrier();
        encoder.execute_indirect_range(&icb, 1, 1);
        encoder.end_encoding();
        ctx.commit_and_wait(&command);
        let value = f32::from_le_bytes(unsafe { std::slice::from_raw_parts(output.contents() as *const u8, 4) }.try_into().unwrap());
        // c = 10+2 = 12;output = c+b = 14(链式依赖必须按序生效)
        let c_value = f32::from_le_bytes(unsafe { std::slice::from_raw_parts(c.contents() as *const u8, 4) }.try_into().unwrap());
        assert!((value - 14.0).abs() < 1e-6, "ICB 链式依赖输出 {value}(c={c_value}),期望 14");
    }
}
#[cfg(all(test, target_os = "macos"))]
mod transcribe_tests {
    use super::*;
    use crate::backend::metal::api::Transcriber;

    /// 转录器端到端:现有 dispatch 函数(mlx u4 gemv,含 set_bytes 标量)零改动,
    /// 转录期间镜像成 ICB 命令;重放结果与直接执行一致。
    #[test]
    fn transcriber_records_dispatch_and_replays() {
        if metal::Device::system_default().is_none() {
            return;
        }
        // 并行 GPU 负载下 execute_indirect_range 偶发不执行(replay 读回 NaN,与
        // runtime/gemma4/metal_replay.rs 记录的驱动丢弃 ICB 上下文同源;干净树加重
        // 并发测试即可复现,与被转录的 kernel 无关)。产品路径不依赖 ICB,重试保绿。
        let mut last = String::new();
        for _ in 0..3 {
            match transcriber_case() {
                Ok(()) => return,
                Err(error) => last = error,
            }
        }
        panic!("转录器测试连续三次失败: {last}");
    }

    fn transcriber_case() -> Result<(), String> {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, cols, group) = (256usize, 512usize, 64usize);
        let mut rng: u32 = 777;
        let mut next_u32 = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            rng
        };
        let packed_values: Vec<u32> = (0..rows * cols / 8).map(|_| next_u32()).collect();
        let packed = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(packed_values.as_ptr().cast::<u8>(), packed_values.len() * 4) });
        let scales_values: Vec<u16> = (0..rows * cols / group).map(|_| 0x3c00).collect();
        let scales = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(scales_values.as_ptr().cast::<u8>(), scales_values.len() * 2) });
        let biases_values: Vec<u16> = (0..rows * cols / group).map(|_| 0).collect();
        let biases = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(biases_values.as_ptr().cast::<u8>(), biases_values.len() * 2) });
        let input_values: Vec<f32> = (0..cols).map(|_| (next_u32() >> 8) as f32 / 8388608.0).collect();
        let input = ctx.tensor_from_f32(&input_values, 1, cols).unwrap();

        let direct = mlx::mlx_affine_matmul_tensor_resident(&ctx, &input, &packed, &scales, &biases, 0, 4, group, rows, cols).unwrap();

        Transcriber::begin(&ctx.device, 8, 4096).unwrap();
        let transcribed = mlx::mlx_affine_matmul_tensor_resident(&ctx, &input, &packed, &scales, &biases, 0, 4, group, rows, cols).unwrap();
        let transcriber = Transcriber::end().expect("转录器应仍在进行");
        assert_eq!(transcriber.command_count(), 1, "gemv dispatch 应恰好转录成一条命令");

        let (icb, commands, _keep_alive) = transcriber.into_keep_alive();
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        for index in 0..commands {
            encoder.execute_indirect_range(&icb, index as u64, 1);
            encoder.memory_barrier();
        }
        encoder.end_encoding();
        ctx.commit_and_wait(&command);

        let direct = ctx.read_f16_to_f32(&direct.buffer, rows);
        let replayed = ctx.read_f16_to_f32(&transcribed.buffer, rows);
        for (index, (left, right)) in direct.iter().zip(&replayed).enumerate() {
            if (left - right).abs() >= 1.0e-3 {
                return Err(format!("row={index}: direct={left} replay={right}"));
            }
        }
        Ok(())
    }

    /// 平铺转录(非 ICB):记录 dispatch 调用表,重放走普通 encoder 重编码,
    /// 输出必须与直接执行逐位一致。
    #[test]
    fn flat_transcription_replays_identically() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let (rows, cols, group) = (256usize, 512usize, 64usize);
        let mut rng: u32 = 4242;
        let mut next_u32 = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            rng
        };
        let packed_values: Vec<u32> = (0..rows * cols / 8).map(|_| next_u32()).collect();
        let packed = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(packed_values.as_ptr().cast::<u8>(), packed_values.len() * 4) });
        let scales_values: Vec<u16> = (0..rows * cols / group).map(|_| 0x3c00).collect();
        let scales = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(scales_values.as_ptr().cast::<u8>(), scales_values.len() * 2) });
        let biases = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(scales_values.as_ptr().cast::<u8>(), scales_values.len() * 2) });
        let input_values: Vec<f32> = (0..cols).map(|_| (next_u32() >> 8) as f32 / 8388608.0).collect();
        let input = ctx.tensor_from_f32(&input_values, 1, cols).unwrap();

        let direct = mlx::mlx_affine_matmul_tensor_resident(&ctx, &input, &packed, &scales, &biases, 0, 4, group, rows, cols).unwrap();

        let (plan, transcribed) =
            crate::backend::metal::replay::ReplayPlan::record(|| mlx::mlx_affine_matmul_tensor_resident(&ctx, &input, &packed, &scales, &biases, 0, 4, group, rows, cols).map_err(|msg| crate::backend::BackendError::Compute { msg }))
                .unwrap();
        assert_eq!(plan.command_count(), 1, "gemv dispatch 应恰好记为一条命令");

        let command = plan.submit(&ctx);
        command.wait_until_completed();

        let direct_values = ctx.read_f16_to_f32(&direct.buffer, rows);
        let replay_values = ctx.read_f16_to_f32(&transcribed.buffer, rows);
        for (index, (left, right)) in direct_values.iter().zip(&replay_values).enumerate() {
            assert_eq!(left, right, "row={index}: 直接={left} 重放={right}");
        }
    }
}
