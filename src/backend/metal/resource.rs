//! Metal 统一内存预算与 profile 聚合。

use super::{MetalContext, MetalGpuProfile};

const MIN_SYSTEM_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// 模型加载完成后的 Metal resident 容量口径:统一内存工作集减系统保留。
/// 引擎常驻(权重等 current_allocated_size)由 FixedSessionResidency 在
/// admission 侧统一扣除;这里再扣一次会把同一笔常驻算两遍,大权重
/// (如 26GB 机器装 27B)直接把 session 预算压成 0,节点永不调度。
pub fn available_residency_bytes(ctx: &MetalContext) -> u64 {
    let working_set = ctx.device.recommended_max_working_set_size();
    if working_set == 0 {
        return 0;
    }
    let reserve = (working_set / 8).max(MIN_SYSTEM_RESERVE_BYTES).min(working_set / 2);
    working_set.saturating_sub(reserve)
}

pub fn merge_gpu_profiles(totals: &mut Vec<MetalGpuProfile>, current: &[MetalGpuProfile]) {
    for profile in current {
        if let Some(total) = totals.iter_mut().find(|total| total.operator == profile.operator && total.shape == profile.shape) {
            total.calls += profile.calls;
            total.gpu_seconds += profile.gpu_seconds;
            total.estimated_read_bytes = total.estimated_read_bytes.saturating_add(profile.estimated_read_bytes);
            total.estimated_write_bytes = total.estimated_write_bytes.saturating_add(profile.estimated_write_bytes);
        } else {
            totals.push(profile.clone());
        }
    }
    totals.sort_by(|a, b| b.gpu_seconds.total_cmp(&a.gpu_seconds).then_with(|| a.operator.cmp(&b.operator)).then_with(|| a.shape.cmp(&b.shape)));
}

pub fn report_metal_resource_plan(ctx: &MetalContext, execution_layers: usize, max_seq_len: usize, kv_cache_bytes: usize, dsa_cache_bytes: usize, expert_arena_bytes: usize) -> Result<(), String> {
    let working_set = ctx.device.recommended_max_working_set_size();
    if working_set == 0 {
        println!("Metal resource plan: unified={} working_set=unknown", ctx.device.has_unified_memory());
        return Ok(());
    }
    let reserve = (working_set / 8).max(MIN_SYSTEM_RESERVE_BYTES).min(working_set / 2);
    let usable = working_set - reserve;
    let known = (kv_cache_bytes as u64).saturating_add(dsa_cache_bytes as u64).saturating_add(expert_arena_bytes as u64);
    println!(
        "Metal resource plan: unified={} working_set={:.1} GiB usable={:.1} GiB reserve={:.1} GiB known={:.1} GiB (KV {:.1} MiB + DSA {:.1} MiB + expert {:.1} MiB), layers={execution_layers}, max_seq_len={max_seq_len}",
        ctx.device.has_unified_memory(),
        working_set as f64 / (1024.0 * 1024.0 * 1024.0),
        usable as f64 / (1024.0 * 1024.0 * 1024.0),
        reserve as f64 / (1024.0 * 1024.0 * 1024.0),
        known as f64 / (1024.0 * 1024.0 * 1024.0),
        kv_cache_bytes as f64 / (1024.0 * 1024.0),
        dsa_cache_bytes as f64 / (1024.0 * 1024.0),
        expert_arena_bytes as f64 / (1024.0 * 1024.0),
    );
    if known > usable {
        return Err(format!("已知 Metal 分配 {:.1} GiB 超过可用预算 {:.1} GiB，请减小 --max-seq-len 或执行层数", known as f64 / (1024.0 * 1024.0 * 1024.0), usable as f64 / (1024.0 * 1024.0 * 1024.0),));
    }
    Ok(())
}

/// 系统物理内存字节数（sysctl hw.memsize）；节点 capabilities 上报用。
pub fn system_memory_bytes() -> Option<u64> {
    let output = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

/// 当前加速器名称对应的 GPU core 数（system_profiler）；节点 capabilities 上报用。
pub fn gpu_compute_units(accelerator: &str) -> Option<usize> {
    let output = std::process::Command::new("system_profiler").args(["SPDisplaysDataType", "-json"]).output().ok()?;
    let value = serde_json::from_slice::<serde_json::Value>(&output.stdout).ok()?;
    value.get("SPDisplaysDataType")?.as_array()?.iter().find_map(|display| {
        let name = display.get("sppci_model").or_else(|| display.get("_name")).and_then(serde_json::Value::as_str)?;
        if name != accelerator {
            return None;
        }
        display.get("sppci_cores").and_then(serde_json::Value::as_str)?.parse().ok()
    })
}
