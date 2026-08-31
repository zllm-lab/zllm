//! CPU 算子(aarch64→NEON, x86_64→SSE)。SIMD 实现,无 fallback。

pub mod attn_res;
pub mod blas;
pub mod block_fp8;
pub mod dsa;
pub mod ggml_quant;
pub mod gqa;
pub mod kda;
pub mod matmul;
pub mod mla;
pub mod moe;
pub mod nvfp4;
pub mod rmsnorm;
pub mod silu;
#[cfg(target_arch = "x86_64")]
pub mod team;
pub mod vae;
pub mod vision;
pub mod w4a16;
pub use rmsnorm::{gemma_rmsnorm, rmsnorm};
pub use silu::{gelu, gelu_tanh_mul, silu_mul, swiglu_oai_mul};

/// 返回当前进程 affinity mask 中实际可运行的 CPU 数。systemd/cgroup 的 quota
/// 会让 `available_parallelism()` 在多核 affinity 下仍报告 1，算子分片不能据此退化。
#[cfg(target_arch = "x86_64")]
pub(crate) fn allowed_parallelism() -> usize {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut std::ffi::c_void) -> i32;
        }
        let mut mask = [0_usize; 16];
        if sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr().cast()) == 0 {
            let count = mask.iter().map(|word| word.count_ones() as usize).sum();
            if count > 0 {
                return count;
            }
        }
    }
    std::thread::available_parallelism().map_or(1, usize::from)
}

#[cfg(target_os = "linux")]
fn parse_cpu_list_mask(cpu_list: &str) -> Result<[usize; 16], String> {
    let mut mask = [0_usize; 16];
    let mut count = 0usize;
    for part in cpu_list.split(',').map(str::trim) {
        if part.is_empty() {
            return Err(format!("CPU list 含空项: {cpu_list}"));
        }
        let (begin, end) = match part.split_once('-') {
            Some((begin, end)) => {
                let begin = begin.parse::<usize>().map_err(|_| format!("CPU list 起点非法: {part}"))?;
                let end = end.parse::<usize>().map_err(|_| format!("CPU list 终点非法: {part}"))?;
                if begin > end {
                    return Err(format!("CPU list 范围倒置: {part}"));
                }
                (begin, end)
            }
            None => {
                let cpu = part.parse::<usize>().map_err(|_| format!("CPU 编号非法: {part}"))?;
                (cpu, cpu)
            }
        };
        for cpu in begin..=end {
            let word = cpu / usize::BITS as usize;
            if word >= mask.len() {
                return Err(format!("CPU 编号 {cpu} 超出当前 1024 CPU mask"));
            }
            let bit = 1_usize << (cpu % usize::BITS as usize);
            count += usize::from(mask[word] & bit == 0);
            mask[word] |= bit;
        }
    }
    if count == 0 {
        return Err("CPU list 不能为空".to_owned());
    }
    Ok(mask)
}

/// 只绑定当前线程；它随后创建的固定 CPU team 会继承该 mask，进程内其他线程不受影响。
#[cfg(target_os = "linux")]
pub(crate) fn set_current_thread_affinity(cpu_list: &str) -> Result<(), String> {
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const std::ffi::c_void) -> i32;
    }
    let mask = parse_cpu_list_mask(cpu_list)?;
    let result = unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr().cast()) };
    if result != 0 {
        return Err(format!("设置 CPU affinity {cpu_list}: {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

/// 临时绑定当前线程执行初始化，使大块常驻内存按执行 CPU 的 NUMA 节点首次触碰；
/// 初始化完成后恢复调用线程原有 mask，避免把 GPU/runtime 主线程永久限在 CPU team。
#[cfg(target_os = "linux")]
pub(crate) fn with_current_thread_affinity<T>(cpu_list: &str, run: impl FnOnce() -> T) -> Result<T, String> {
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut std::ffi::c_void) -> i32;
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const std::ffi::c_void) -> i32;
    }
    let mut previous = [0_usize; 16];
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&previous), previous.as_mut_ptr().cast()) } != 0 {
        return Err(format!("读取当前 CPU affinity: {}", std::io::Error::last_os_error()));
    }
    set_current_thread_affinity(cpu_list)?;
    let output = run();
    if unsafe { sched_setaffinity(0, std::mem::size_of_val(&previous), previous.as_ptr().cast()) } != 0 {
        return Err(format!("恢复 CPU affinity: {}", std::io::Error::last_os_error()));
    }
    Ok(output)
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
pub(crate) fn set_current_thread_affinity(cpu_list: &str) -> Result<(), String> {
    Err(format!("当前平台不支持 CPU affinity: {cpu_list}"))
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
pub(crate) fn with_current_thread_affinity<T>(cpu_list: &str, _run: impl FnOnce() -> T) -> Result<T, String> {
    Err(format!("当前平台不支持 CPU affinity: {cpu_list}"))
}

pub(crate) fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub(crate) fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

/// 对 `[base, base+bytes)` 建议 THP。应在首次写入前调用,使缺页直接落在
/// 2MB 大页;madvise 模式内核下方生效,不可用则静默忽略。
#[cfg(target_os = "linux")]
pub(crate) fn advise_huge_pages_region(base: *mut u8, bytes: usize) {
    const HUGE_PAGE: usize = 2 * 1024 * 1024;
    const MADV_HUGEPAGE: i32 = 14;
    let begin = base as usize;
    let end = begin.saturating_add(bytes);
    let aligned_begin = begin.saturating_add(HUGE_PAGE - 1) & !(HUGE_PAGE - 1);
    let aligned_end = end & !(HUGE_PAGE - 1);
    if aligned_begin >= aligned_end {
        return;
    }
    unsafe extern "C" {
        fn madvise(address: *mut std::ffi::c_void, length: usize, advice: i32) -> i32;
    }
    unsafe {
        let _ = madvise(aligned_begin as *mut std::ffi::c_void, aligned_end - aligned_begin, MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn advise_huge_pages_region(_base: *mut u8, _bytes: usize) {}

/// CPU reference top-p；调用方负责提供 `[0, 1)` 的确定性随机数。
pub fn sample_top_p(input: &[f32], temperature: f32, top_p: f32, random: f32) -> Result<u32, String> {
    sample_top_p_excluding(input, temperature, top_p, random, &[])
}

pub fn sample_top_p_excluding(input: &[f32], temperature: f32, top_p: f32, random: f32, excluded: &[u32]) -> Result<u32, String> {
    if input.is_empty() || !temperature.is_finite() || temperature <= 0.0 || !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) || !random.is_finite() || !(0.0..1.0).contains(&random) {
        return Err("CPU top-p sampling 参数非法".to_owned());
    }
    let mut candidates: Vec<(usize, f32)> = input.iter().copied().enumerate().filter(|(index, _)| !excluded.contains(&(*index as u32))).collect();
    if candidates.is_empty() {
        return Err("CPU top-p sampling 没有可选 token".to_owned());
    }
    candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let max_logit = candidates[0].1 / temperature;
    let masses: Vec<f32> = candidates.iter().map(|(_, value)| (value / temperature - max_logit).exp()).collect();
    let cutoff = top_p * masses.iter().sum::<f32>();
    let mut prefix_mass = 0.0;
    let mut prefix_len = 0;
    for mass in &masses {
        prefix_mass += *mass;
        prefix_len += 1;
        if prefix_mass >= cutoff {
            break;
        }
    }
    let threshold = candidates[prefix_len - 1].1 / temperature;
    prefix_mass = input.iter().enumerate().filter(|(index, _)| !excluded.contains(&(*index as u32))).map(|(_, value)| value / temperature).filter(|value| *value >= threshold).map(|value| (value - max_logit).exp()).sum();
    let target = random * prefix_mass;
    let mut cumulative = 0.0;
    for (index, value) in input.iter().enumerate() {
        if excluded.contains(&(index as u32)) {
            continue;
        }
        let value = value / temperature;
        if value >= threshold {
            cumulative += (value - max_logit).exp();
            if cumulative >= target {
                return u32::try_from(index).map_err(|_| "CPU top-p sampling index 超出 u32".to_owned());
            }
        }
    }
    Err("CPU top-p sampling 未选择 token".to_owned())
}

/// 行优先 `[rows, cols]` f32 张量。CPU 平台的 tensor 表示。
#[derive(Debug, Clone)]
pub struct CpuTensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl CpuTensor {
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }
}

#[cfg(all(test, target_os = "linux"))]
mod affinity_tests {
    use super::parse_cpu_list_mask;

    #[test]
    fn cpu_list_supports_ranges_and_rejects_invalid_masks() {
        let mask = parse_cpu_list_mask("0-2,8,64").unwrap();
        assert_eq!(mask[0] & 0x107, 0x107);
        assert_eq!(mask[1] & 1, 1);
        assert!(parse_cpu_list_mask("3-1").is_err());
        assert!(parse_cpu_list_mask("").is_err());
        assert!(parse_cpu_list_mask("1024").is_err());
    }
}
