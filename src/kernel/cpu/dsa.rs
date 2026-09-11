//! DSA 全历史 CPU 评分与稳定 Top-K；长上下文 offload 先由 ROCm shadow 验证。

use half::bf16 as HalfBf16;
use rayon::prelude::*;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn dsa_cpu_ids() -> Vec<usize> {
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut std::ffi::c_void) -> i32;
    }
    let mut mask = [0_usize; 16];
    let mut allowed = Vec::new();
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr().cast()) } == 0 {
        for cpu in 0..mask.len() * usize::BITS as usize {
            if mask[cpu / usize::BITS as usize] & (1_usize << (cpu % usize::BITS as usize)) != 0 {
                allowed.push(cpu);
            }
        }
    }
    if allowed.is_empty() {
        return (0..std::thread::available_parallelism().map_or(1, usize::from)).collect();
    }
    let mut packages = std::collections::BTreeMap::<usize, Vec<(usize, usize)>>::new();
    let mut seen = std::collections::BTreeSet::new();
    for &cpu in &allowed {
        let package = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id")).ok().and_then(|value| value.trim().parse::<usize>().ok());
        let core = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_id")).ok().and_then(|value| value.trim().parse::<usize>().ok());
        if let (Some(package), Some(core)) = (package, core)
            && seen.insert((package, core))
        {
            packages.entry(package).or_default().push((core, cpu));
        }
    }
    for cores in packages.values_mut() {
        cores.sort_unstable();
        // 每个 NUMA package 预留最后一个物理核：decode 提交线程与 hot-MLA host
        // 工作不与绑核 worker 同核争用；worker 自旋时它们仍能全速运行。
        if cores.len() > 4 {
            cores.pop();
        }
    }
    let mut physical = Vec::new();
    for core_index in 0..packages.values().map(Vec::len).max().unwrap_or(0) {
        for cores in packages.values() {
            if let Some(&(_, cpu)) = cores.get(core_index) {
                physical.push(cpu);
            }
        }
    }
    if physical.is_empty() { allowed } else { physical }
}

fn dsa_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        {
            let cpu_ids = Arc::new(dsa_cpu_ids());
            let worker_cpu_ids = Arc::clone(&cpu_ids);
            return rayon::ThreadPoolBuilder::new()
                .num_threads(cpu_ids.len().min(64).max(1))
                .thread_name(|index| format!("zllm-dsa-{index}"))
                .start_handler(move |index| super::set_current_thread_affinity(&worker_cpu_ids[index].to_string()).expect("CPU DSA worker affinity 设置失败"))
                .build()
                .expect("CPU DSA 固定线程池创建失败");
        }
        #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
        {
            #[cfg(target_arch = "x86_64")]
            let parallelism = super::allowed_parallelism();
            #[cfg(not(target_arch = "x86_64"))]
            let parallelism = std::thread::available_parallelism().map_or(1, usize::from);
            rayon::ThreadPoolBuilder::new().num_threads(parallelism.min(64).max(1)).thread_name(|index| format!("zllm-dsa-{index}")).build().expect("CPU DSA 固定线程池创建失败")
        }
    })
}

type DsaTeamCallback = unsafe fn(usize, usize);

/// 自旋保活窗口：每次任务提交后续命。decode 期间 indexer 层间隔(约 5-8ms)远小于
/// 该值，64 个绑核 worker 全程自旋、唤醒延迟 ~1us；长 prefill 间隔下自然回落
/// futex 睡眠，只在长空闲后的首个任务付一次全量唤醒成本。
const DSA_TEAM_SPIN_KEEPALIVE_MICROS: u64 = 50_000;
/// 提交线程等待完成前的自旋预算；Q8 打分一轮约 0.1-0.5ms，自旋几乎总能命中。
const DSA_TEAM_SUBMIT_SPINS: u32 = 1 << 16;

fn dsa_team_micros() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

struct DsaTeamShared {
    /// 只作为完成通知的守卫；任务字段全部走原子发布。
    job: Mutex<()>,
    complete: Condvar,
    pending: std::sync::atomic::AtomicUsize,
    /// 每个 worker 独立发布 generation，只唤醒本轮真正参与计算的线程。
    published: Box<[std::sync::atomic::AtomicU64]>,
    callback: std::sync::atomic::AtomicUsize,
    context: std::sync::atomic::AtomicUsize,
}

struct DsaTeam {
    workers: usize,
    threads: Vec<std::thread::Thread>,
    submission: Mutex<()>,
    shared: Arc<DsaTeamShared>,
}

impl DsaTeam {
    fn new() -> Self {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        let cpu_ids = dsa_cpu_ids();
        #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
        let cpu_ids = (0..std::thread::available_parallelism().map_or(1, usize::from)).collect::<Vec<_>>();
        let cpu_ids = &cpu_ids[..cpu_ids.len().clamp(1, 64)];
        unsafe fn idle(_: usize, _: usize) {}
        let shared = Arc::new(DsaTeamShared {
            job: Mutex::new(()),
            complete: Condvar::new(),
            pending: std::sync::atomic::AtomicUsize::new(0),
            published: (0..cpu_ids.len()).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            callback: std::sync::atomic::AtomicUsize::new(idle as *const () as usize),
            context: std::sync::atomic::AtomicUsize::new(0),
        });
        let mut threads = Vec::with_capacity(cpu_ids.len());
        for (index, &_cpu_id) in cpu_ids.iter().enumerate() {
            let shared = Arc::clone(&shared);
            let worker = std::thread::Builder::new()
                .name(format!("zllm-dsa-direct-{index}"))
                .spawn(move || {
                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                    super::set_current_thread_affinity(&_cpu_id.to_string()).expect("CPU DSA direct worker affinity 设置失败");
                    let mut seen = 0_u64;
                    let mut spin_until = 0_u64;
                    let mut spin_probe = 0_u32;
                    loop {
                        // 近期参与过任务的 worker 自旋等待；超时后 park。unpark 的
                        // 单 bit permit 保证发布发生在 park 前也不会丢失唤醒。
                        'seek: loop {
                            let current = shared.published[index].load(std::sync::atomic::Ordering::Acquire);
                            if current != seen {
                                seen = current;
                                break 'seek;
                            }
                            spin_probe = spin_probe.wrapping_add(1);
                            if spin_probe & 31 == 0 && dsa_team_micros() >= spin_until {
                                std::thread::park();
                            }
                            std::hint::spin_loop();
                        }
                        spin_probe = 0;
                        let callback = unsafe { std::mem::transmute::<usize, DsaTeamCallback>(shared.callback.load(std::sync::atomic::Ordering::Relaxed)) };
                        let context = shared.context.load(std::sync::atomic::Ordering::Relaxed);
                        unsafe { callback(context, index) };
                        spin_until = dsa_team_micros() + DSA_TEAM_SPIN_KEEPALIVE_MICROS;
                        if shared.pending.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                            let _guard = shared.job.lock().expect("CPU DSA direct completion mutex 已损坏");
                            shared.complete.notify_one();
                        }
                    }
                })
                .expect("CPU DSA direct worker 启动失败");
            threads.push(worker.thread().clone());
        }
        Self { workers: cpu_ids.len(), threads, submission: Mutex::new(()), shared }
    }

    fn workers(&self) -> usize {
        self.workers
    }

    fn run<F: Fn(usize) + Sync>(&self, active_workers: usize, job: F) {
        unsafe fn invoke<F: Fn(usize) + Sync>(context: usize, index: usize) {
            unsafe { (&*(context as *const F))(index) };
        }

        assert!((1..=self.workers).contains(&active_workers));
        // context 指向调用栈上的 closure；run 在 pending 归零前不返回，且 submission
        // 串行化全部调用，因此 worker 不会观察到失效或被下一份 job 覆盖的地址。
        let _submission = self.submission.lock().expect("CPU DSA direct submission mutex 已损坏");
        assert_eq!(self.shared.pending.load(std::sync::atomic::Ordering::Acquire), 0);
        self.shared.callback.store(invoke::<F> as *const () as usize, std::sync::atomic::Ordering::Relaxed);
        self.shared.context.store((&job as *const _) as usize, std::sync::atomic::Ordering::Relaxed);
        self.shared.pending.store(active_workers, std::sync::atomic::Ordering::Release);
        for worker in 0..active_workers {
            self.shared.published[worker].fetch_add(1, std::sync::atomic::Ordering::Release);
            self.threads[worker].unpark();
        }
        let mut spins = 0_u32;
        while self.shared.pending.load(std::sync::atomic::Ordering::Acquire) != 0 {
            if spins < DSA_TEAM_SUBMIT_SPINS {
                std::hint::spin_loop();
                spins += 1;
                continue;
            }
            let mut guard = self.shared.job.lock().expect("CPU DSA direct completion mutex 已损坏");
            if self.shared.pending.load(std::sync::atomic::Ordering::Acquire) == 0 {
                break;
            }
            guard = self.shared.complete.wait(guard).expect("CPU DSA direct completion wait 失败");
        }
    }
}

fn dsa_team() -> &'static DsaTeam {
    static TEAM: OnceLock<DsaTeam> = OnceLock::new();
    TEAM.get_or_init(DsaTeam::new)
}

fn ordered_score(score: f32) -> u32 {
    let bits = score.to_bits();
    bits ^ if bits & 0x8000_0000 != 0 { 0xffff_ffff } else { 0x8000_0000 }
}

/// final Top-K 的 13-bit 高位 radix；并行版与串行版共用。
const RADIX_BITS: u32 = 13;
const RADIX_SIZE: usize = 1 << RADIX_BITS;

fn stable_topk_into(scores: &[u32], top_k: usize, selected: &mut Vec<u32>) {
    let mut histogram = [0_u32; RADIX_SIZE];
    for &score in scores {
        histogram[(score >> (32 - RADIX_BITS)) as usize] += 1;
    }
    let mut rank = top_k;
    let mut threshold = 0_usize;
    for bucket in (0..RADIX_SIZE).rev() {
        let count = histogram[bucket] as usize;
        if count < rank {
            rank -= count;
        } else {
            threshold = bucket;
            break;
        }
    }
    selected.clear();
    selected.extend(scores.iter().enumerate().filter_map(|(token, &score)| ((score >> (32 - RADIX_BITS)) as usize >= threshold).then_some(token as u32)));
    selected.sort_unstable_by(|&left, &right| scores[right as usize].cmp(&scores[left as usize]).then_with(|| left.cmp(&right)));
    selected.truncate(top_k);
    let threshold = scores[selected[top_k - 1] as usize];
    // GPU radix 先把严格高于 kth score 的成员按 token 顺序压紧，再追加
    // kth/tie 成员；MLA 的浮点归约依赖这个稳定顺序。
    selected.sort_unstable_by_key(|&token| (scores[token as usize] == threshold, token));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn collect_score_tokens_avx512<const EQUAL: bool>(scores: &[u32], threshold: u32, selected: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    selected.reserve(scores.len());
    let start = selected.len();
    let output = unsafe { selected.as_mut_ptr().add(start) };
    let threshold_vector = _mm512_set1_epi32(threshold as i32);
    let lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let mut written = 0_usize;
    let mut chunks = scores.chunks_exact(16);
    for (chunk_index, chunk) in (&mut chunks).enumerate() {
        let values = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
        let mask = if EQUAL { _mm512_cmpeq_epi32_mask(values, threshold_vector) } else { _mm512_cmpgt_epu32_mask(values, threshold_vector) };
        let tokens = _mm512_add_epi32(lanes, _mm512_set1_epi32((chunk_index * 16) as i32));
        unsafe { _mm512_mask_compressstoreu_epi32(output.add(written).cast(), mask, tokens) };
        written += mask.count_ones() as usize;
    }
    unsafe { selected.set_len(start + written) };
    let tail_first = scores.len() - chunks.remainder().len();
    selected.extend(chunks.remainder().iter().enumerate().filter_map(|(offset, &score)| {
        let keep = if EQUAL { score == threshold } else { score > threshold };
        keep.then_some((tail_first + offset) as u32)
    }));
}

/// 精确定位 kth score 后按 GPU 稳定语义收集成员，避免高位 bucket 内再做比较排序。
fn exact_radix_topk_into(scores: &[u32], top_k: usize, selected: &mut Vec<u32>, candidates: &mut Vec<u32>) {
    const MID_BITS: u32 = 11;
    const MID_SIZE: usize = 1 << MID_BITS;
    const LOW_BITS: u32 = 32 - RADIX_BITS - MID_BITS;
    const LOW_SIZE: usize = 1 << LOW_BITS;

    let mut high_histogram = [0_u32; RADIX_SIZE];
    for &score in scores {
        high_histogram[(score >> (32 - RADIX_BITS)) as usize] += 1;
    }
    let mut rank = top_k;
    let mut high = 0_usize;
    for bucket in (0..RADIX_SIZE).rev() {
        let count = high_histogram[bucket] as usize;
        if count < rank {
            rank -= count;
        } else {
            high = bucket;
            break;
        }
    }
    candidates.clear();
    candidates.extend(scores.iter().enumerate().filter_map(|(token, &score)| ((score >> (32 - RADIX_BITS)) as usize == high).then_some(token as u32)));

    let mut mid_histogram = [0_u32; MID_SIZE];
    for &token in candidates.iter() {
        mid_histogram[((scores[token as usize] >> LOW_BITS) as usize) & (MID_SIZE - 1)] += 1;
    }
    let mut mid = 0_usize;
    for bucket in (0..MID_SIZE).rev() {
        let count = mid_histogram[bucket] as usize;
        if count < rank {
            rank -= count;
        } else {
            mid = bucket;
            break;
        }
    }
    candidates.retain(|&token| ((scores[token as usize] >> LOW_BITS) as usize) & (MID_SIZE - 1) == mid);

    let mut low_histogram = [0_u32; LOW_SIZE];
    for &token in candidates.iter() {
        low_histogram[scores[token as usize] as usize & (LOW_SIZE - 1)] += 1;
    }
    let mut low = 0_usize;
    for bucket in (0..LOW_SIZE).rev() {
        let count = low_histogram[bucket] as usize;
        if count < rank {
            rank -= count;
        } else {
            low = bucket;
            break;
        }
    }
    let threshold = ((high as u32) << (32 - RADIX_BITS)) | ((mid as u32) << LOW_BITS) | low as u32;
    selected.clear();
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe {
            collect_score_tokens_avx512::<false>(scores, threshold, selected);
            collect_score_tokens_avx512::<true>(scores, threshold, selected);
        }
    } else {
        selected.extend(scores.iter().enumerate().filter_map(|(token, &score)| (score > threshold).then_some(token as u32)));
        selected.extend(scores.iter().enumerate().filter_map(|(token, &score)| (score == threshold).then_some(token as u32)));
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        selected.extend(scores.iter().enumerate().filter_map(|(token, &score)| (score > threshold).then_some(token as u32)));
        selected.extend(scores.iter().enumerate().filter_map(|(token, &score)| (score == threshold).then_some(token as u32)));
    }
    selected.truncate(top_k);
}

fn stable_topk(scores: &[u32], top_k: usize) -> Vec<u32> {
    let mut selected = Vec::new();
    stable_topk_into(scores, top_k, &mut selected);
    selected
}

/// 感知反向 barrier：team 单次 run 内做多阶段同步，全部 worker 都会到达。
struct TopkBarrier {
    arrivals: std::sync::atomic::AtomicU32,
    epoch: std::sync::atomic::AtomicU32,
}

impl Default for TopkBarrier {
    fn default() -> Self {
        Self { arrivals: std::sync::atomic::AtomicU32::new(0), epoch: std::sync::atomic::AtomicU32::new(0) }
    }
}

impl TopkBarrier {
    fn wait(&self, threads: u32) {
        let epoch = self.epoch.load(std::sync::atomic::Ordering::Acquire);
        if self.arrivals.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1 == threads {
            self.arrivals.store(0, std::sync::atomic::Ordering::Relaxed);
            self.epoch.store(epoch.wrapping_add(1), std::sync::atomic::Ordering::Release);
        } else {
            while self.epoch.load(std::sync::atomic::Ordering::Acquire) == epoch {
                std::hint::spin_loop();
            }
        }
    }
}

/// 并行 final Top-K 的复用空间；行区间与打分阶段同 worker，scores 读取保持 NUMA 本地。
#[derive(Default)]
pub struct Q8TopkScratch {
    partials: Vec<Vec<u32>>,
    reduced: Vec<u32>,
    staging: Vec<Vec<u32>>,
    threshold: std::sync::atomic::AtomicU32,
    barrier: TopkBarrier,
    /// worker 0 在打分 barrier 后记录时刻，供融合路径拆分 score/topk 耗时。
    phase_micros: std::sync::atomic::AtomicU64,
    /// 融合路径专用：共享原子直方图、(score,token) 对 staging 与尾排序缓冲。
    hist: Vec<std::sync::atomic::AtomicU32>,
    fused_staging: Vec<Vec<(u32, u32)>>,
    pairs: Vec<(u32, u32)>,
}

/// `stable_topk_into` 的 team 并行版：直方图按 worker 分块统计后列归约，
/// 阈值扫描与幸存者过滤语义同串行实现，输出逐字节一致。
fn parallel_stable_topk_into(scores: &[u32], top_k: usize, selected: &mut Vec<u32>, scratch: &mut Q8TopkScratch) {
    let rows = scores.len();
    let blocks = rows.div_ceil(DSA_ROW_BLOCK);
    let threads = dsa_team().workers().min(blocks.div_ceil(128).max(1));
    if threads <= 1 {
        stable_topk_into(scores, top_k, selected);
        return;
    }
    if scratch.partials.len() != threads || scratch.reduced.len() != RADIX_SIZE || scratch.staging.len() != threads {
        scratch.partials = (0..threads).map(|_| vec![0_u32; RADIX_SIZE]).collect();
        scratch.reduced = vec![0_u32; RADIX_SIZE];
        scratch.staging = (0..threads).map(|_| Vec::with_capacity(4096)).collect();
    }
    let chunk_blocks = blocks.div_ceil(threads);
    let chunk_rows = chunk_blocks * DSA_ROW_BLOCK;
    let scores_address = scores.as_ptr() as usize;
    let partials_address = scratch.partials.as_ptr() as usize;
    let reduced_address = scratch.reduced.as_mut_ptr() as usize;
    let staging_address = scratch.staging.as_mut_ptr() as usize;
    let barrier = &scratch.barrier;
    let threshold = &scratch.threshold;
    let buckets_per_worker = RADIX_SIZE.div_ceil(threads);
    dsa_team().run(threads, move |worker| {
        let scores = unsafe { std::slice::from_raw_parts(scores_address as *const u32, rows) };
        let first_row = worker * chunk_rows;
        // 阶段 1：清零并统计本 worker 的 partial 直方图；无行的 worker 也必须清零。
        let partial = unsafe { &mut *(partials_address as *mut Vec<u32>).add(worker) };
        partial.fill(0);
        if first_row < rows {
            let rows = chunk_rows.min(rows - first_row);
            for &score in &scores[first_row..first_row + rows] {
                partial[(score >> (32 - RADIX_BITS)) as usize] += 1;
            }
        }
        barrier.wait(threads as u32);
        // 阶段 2：列归约，各 worker 负责一段 bucket，跨 worker 求和。
        let bucket_first = worker * buckets_per_worker;
        let bucket_end = (bucket_first + buckets_per_worker).min(RADIX_SIZE);
        for bucket in bucket_first..bucket_end {
            let mut total = 0_u32;
            for partial in 0..threads {
                let partial = unsafe { &*(partials_address as *const Vec<u32>).add(partial) };
                total += partial[bucket];
            }
            unsafe { *(reduced_address as *mut u32).add(bucket) = total };
        }
        barrier.wait(threads as u32);
        // 阶段 3：worker 0 找 threshold bucket，语义与串行扫描一致。
        if worker == 0 {
            let mut rank = top_k;
            let mut found = 0_usize;
            for bucket in (0..RADIX_SIZE).rev() {
                let count = unsafe { *(reduced_address as *const u32).add(bucket) } as usize;
                if count < rank {
                    rank -= count;
                } else {
                    found = bucket;
                    break;
                }
            }
            threshold.store(found as u32, std::sync::atomic::Ordering::Release);
        }
        barrier.wait(threads as u32);
        let found = threshold.load(std::sync::atomic::Ordering::Acquire) as usize;
        // 阶段 4：幸存者按 token 顺序写入本 worker staging。
        let staging = unsafe { &mut *(staging_address as *mut Vec<u32>).add(worker) };
        staging.clear();
        if first_row < rows {
            let rows = chunk_rows.min(rows - first_row);
            for (offset, &score) in scores[first_row..first_row + rows].iter().enumerate() {
                if (score >> (32 - RADIX_BITS)) as usize >= found {
                    staging.push((first_row + offset) as u32);
                }
            }
        }
    });
    selected.clear();
    for worker in 0..threads {
        selected.extend_from_slice(&scratch.staging[worker]);
    }
    // 尾部与串行版完全一致：全序排序、截断、kth tie 重排。
    selected.sort_unstable_by(|&left, &right| scores[right as usize].cmp(&scores[left as usize]).then_with(|| left.cmp(&right)));
    selected.truncate(top_k);
    let threshold = scores[selected[top_k - 1] as usize];
    selected.sort_unstable_by_key(|&token| (scores[token as usize] == threshold, token));
}

/// 候选粗排不需要 CPU 内部精确排序：保留包含第 minimum 名的完整 21-bit radix bucket，
/// 结果天然按 token 递增，最终 score 与 tie-break 由 GPU rerank 决定。
fn radix_candidate_superset_into(scores: &[u32], minimum: usize, selected: &mut Vec<u32>) {
    const HIGH_BITS: u32 = 13;
    const HIGH_SIZE: usize = 1 << HIGH_BITS;
    const LOW_BITS: u32 = 8;
    const LOW_SIZE: usize = 1 << LOW_BITS;
    let mut histogram = [0_u32; HIGH_SIZE];
    for &score in scores {
        histogram[(score >> (32 - HIGH_BITS)) as usize] += 1;
    }
    let mut rank = minimum;
    let mut high_threshold = 0_usize;
    for bucket in (0..HIGH_SIZE).rev() {
        let count = histogram[bucket] as usize;
        if count < rank {
            rank -= count;
        } else {
            high_threshold = bucket;
            break;
        }
    }
    let mut low_histogram = [0_u32; LOW_SIZE];
    for &score in scores {
        if (score >> (32 - HIGH_BITS)) as usize == high_threshold {
            low_histogram[((score >> (32 - HIGH_BITS - LOW_BITS)) & (LOW_SIZE as u32 - 1)) as usize] += 1;
        }
    }
    let mut low_threshold = 0_usize;
    for bucket in (0..LOW_SIZE).rev() {
        let count = low_histogram[bucket] as usize;
        if count < rank {
            rank -= count;
        } else {
            low_threshold = bucket;
            break;
        }
    }
    let threshold = (high_threshold << LOW_BITS) | low_threshold;
    selected.clear();
    selected.extend(scores.iter().enumerate().filter_map(|(token, &score)| ((score >> (32 - HIGH_BITS - LOW_BITS)) as usize >= threshold).then_some(token as u32)));
}

const DSA_ROW_BLOCK: usize = 16;

/// CPU DSA 自有的 BF16 key 布局：每个维度对连续保存 16 个历史 token，供一条 DPBF16 指令并行累计。
pub struct Bf16KeyBlocks {
    rows: usize,
    head_dim: usize,
    values: Vec<HalfBf16>,
}

impl Bf16KeyBlocks {
    pub fn from_row_major(keys: &[HalfBf16], head_dim: usize) -> Result<Self, String> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) || !keys.len().is_multiple_of(head_dim) {
            return Err(format!("CPU DSA blocked key shape keys={} dim={head_dim} 非法", keys.len()));
        }
        let rows = keys.len() / head_dim;
        let blocks = rows.div_ceil(DSA_ROW_BLOCK);
        let block_values = DSA_ROW_BLOCK * head_dim;
        let threads = dsa_pool().current_num_threads().min(blocks.div_ceil(128).max(1));
        let chunk_blocks = blocks.div_ceil(threads);
        let mut values = Vec::<std::mem::MaybeUninit<HalfBf16>>::with_capacity(blocks * block_values);
        // 每个元素都会由下面的并行 first-touch 写入；避免单一调用线程先把全部页放到一个 NUMA node。
        unsafe { values.set_len(blocks * block_values) };
        dsa_pool().install(|| {
            values.par_chunks_mut(chunk_blocks * block_values).enumerate().for_each(|(chunk, output)| {
                let first_block = chunk * chunk_blocks;
                for (local_block, block_output) in output.chunks_mut(block_values).enumerate() {
                    let block = first_block + local_block;
                    for pair in 0..head_dim / 2 {
                        for lane in 0..DSA_ROW_BLOCK {
                            let row = block * DSA_ROW_BLOCK + lane;
                            let value = if row < rows { &keys[row * head_dim + pair * 2..][..2] } else { &[HalfBf16::ZERO; 2] };
                            block_output[pair * DSA_ROW_BLOCK * 2 + lane * 2].write(value[0]);
                            block_output[pair * DSA_ROW_BLOCK * 2 + lane * 2 + 1].write(value[1]);
                        }
                    }
                }
            });
        });
        let mut values = std::mem::ManuallyDrop::new(values);
        // 上面的循环完整覆盖含 padding 在内的全部元素，BF16 与 MaybeUninit<BF16> 布局相同。
        let values = unsafe { Vec::from_raw_parts(values.as_mut_ptr().cast::<HalfBf16>(), values.len(), values.capacity()) };
        Ok(Self { rows, head_dim, values })
    }

    /// 从 GPU cache 的 row-major Q8+BF16 scale 直接生成 blocked BF16，避免 32 MiB 中间 row-major 副本。
    pub fn from_q8_row_major(keys: &[i8], scale_bits: &[u16], head_dim: usize, group_size: usize) -> Result<Self, String> {
        if keys.is_empty() || head_dim == 0 || !head_dim.is_multiple_of(2) || group_size == 0 || !head_dim.is_multiple_of(group_size) || !keys.len().is_multiple_of(head_dim) {
            return Err(format!("CPU DSA Q8 blocked key shape keys={} dim={head_dim} group={group_size} 非法", keys.len()));
        }
        let rows = keys.len() / head_dim;
        let groups = head_dim / group_size;
        if scale_bits.len() != rows * groups {
            return Err(format!("CPU DSA Q8 scale={}，期望 {}", scale_bits.len(), rows * groups));
        }
        let blocks = rows.div_ceil(DSA_ROW_BLOCK);
        let block_values = DSA_ROW_BLOCK * head_dim;
        let threads = dsa_pool().current_num_threads().min(blocks.div_ceil(128).max(1));
        let chunk_blocks = blocks.div_ceil(threads);
        let mut values = Vec::<std::mem::MaybeUninit<HalfBf16>>::with_capacity(blocks * block_values);
        // 与 F32/BF16 构造路径相同，所有页都由最终负责评分的 NUMA worker first-touch。
        unsafe { values.set_len(blocks * block_values) };
        dsa_pool().install(|| {
            values.par_chunks_mut(chunk_blocks * block_values).enumerate().for_each(|(chunk, output)| {
                let first_block = chunk * chunk_blocks;
                for (local_block, block_output) in output.chunks_mut(block_values).enumerate() {
                    let block = first_block + local_block;
                    for pair in 0..head_dim / 2 {
                        for lane in 0..DSA_ROW_BLOCK {
                            let row = block * DSA_ROW_BLOCK + lane;
                            let values = if row < rows {
                                let first = row * head_dim + pair * 2;
                                let scale0 = HalfBf16::from_bits(scale_bits[row * groups + pair * 2 / group_size]).to_f32();
                                let scale1 = HalfBf16::from_bits(scale_bits[row * groups + (pair * 2 + 1) / group_size]).to_f32();
                                [HalfBf16::from_f32(keys[first] as f32 * scale0), HalfBf16::from_f32(keys[first + 1] as f32 * scale1)]
                            } else {
                                [HalfBf16::ZERO; 2]
                            };
                            block_output[pair * DSA_ROW_BLOCK * 2 + lane * 2].write(values[0]);
                            block_output[pair * DSA_ROW_BLOCK * 2 + lane * 2 + 1].write(values[1]);
                        }
                    }
                }
            });
        });
        let mut values = std::mem::ManuallyDrop::new(values);
        // 上面的循环完整覆盖含 padding 在内的全部元素。
        let values = unsafe { Vec::from_raw_parts(values.as_mut_ptr().cast::<HalfBf16>(), values.len(), values.capacity()) };
        Ok(Self { rows, head_dim, values })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// decode 每次只追加一行；已有尾块直接覆盖 padding，跨 16 行边界时才扩一页。
    pub fn append_row(&mut self, key: &[HalfBf16]) -> Result<(), String> {
        if key.len() != self.head_dim {
            return Err(format!("CPU DSA append key={}，期望 {}", key.len(), self.head_dim));
        }
        let lane = self.rows % DSA_ROW_BLOCK;
        if lane == 0 {
            self.values.resize(self.values.len() + DSA_ROW_BLOCK * self.head_dim, HalfBf16::ZERO);
        }
        let block = self.rows / DSA_ROW_BLOCK;
        let block_offset = block * DSA_ROW_BLOCK * self.head_dim;
        for pair in 0..self.head_dim / 2 {
            let output = block_offset + pair * DSA_ROW_BLOCK * 2 + lane * 2;
            self.values[output] = key[pair * 2];
            self.values[output + 1] = key[pair * 2 + 1];
        }
        self.rows += 1;
        Ok(())
    }

    pub fn append_q8_row(&mut self, key: &[i8], scale_bits: &[u16], group_size: usize) -> Result<(), String> {
        if key.len() != self.head_dim || group_size == 0 || !self.head_dim.is_multiple_of(group_size) || scale_bits.len() != self.head_dim / group_size {
            return Err(format!("CPU DSA append Q8 key={} scale={} dim={} group={group_size} 非法", key.len(), scale_bits.len(), self.head_dim));
        }
        let lane = self.rows % DSA_ROW_BLOCK;
        if lane == 0 {
            self.values.resize(self.values.len() + DSA_ROW_BLOCK * self.head_dim, HalfBf16::ZERO);
        }
        let block_offset = self.rows / DSA_ROW_BLOCK * DSA_ROW_BLOCK * self.head_dim;
        for pair in 0..self.head_dim / 2 {
            let column = pair * 2;
            let output = block_offset + pair * DSA_ROW_BLOCK * 2 + lane * 2;
            self.values[output] = HalfBf16::from_f32(key[column] as f32 * HalfBf16::from_bits(scale_bits[column / group_size]).to_f32());
            self.values[output + 1] = HalfBf16::from_f32(key[column + 1] as f32 * HalfBf16::from_bits(scale_bits[(column + 1) / group_size]).to_f32());
        }
        self.rows += 1;
        Ok(())
    }

    /// rollback 只回退逻辑长度；后续 append 会覆盖旧尾行。
    pub fn truncate(&mut self, rows: usize) -> Result<(), String> {
        if rows > self.rows {
            return Err(format!("CPU DSA truncate rows={rows} 超过当前长度 {}", self.rows));
        }
        self.rows = rows;
        Ok(())
    }
}

/// CPU DSA 候选粗排使用的原生 Q8 布局：每个维度四元组连续保存 16 个 token，
/// key 加 128 后直接供 AVX512-VNNI 的 unsigned×signed dot 使用。
pub struct Q8KeyBlocks {
    rows: usize,
    available_rows: usize,
    head_dim: usize,
    values: Vec<u8>,
    scales: Vec<f32>,
}

impl Q8KeyBlocks {
    pub fn from_q8_row_major(keys: &[i8], scale_bits: &[u16], head_dim: usize) -> Result<Self, String> {
        if keys.is_empty() || head_dim == 0 || !head_dim.is_multiple_of(4) || !keys.len().is_multiple_of(head_dim) {
            return Err(format!("CPU DSA Q8 key shape keys={} dim={head_dim} 非法", keys.len()));
        }
        let rows = keys.len() / head_dim;
        if scale_bits.len() != rows {
            return Err(format!("CPU DSA Q8 scale={}，期望 {rows}", scale_bits.len()));
        }
        let blocks = rows.div_ceil(DSA_ROW_BLOCK);
        let block_values = DSA_ROW_BLOCK * head_dim;
        let threads = dsa_team().workers().min(blocks.div_ceil(128).max(1));
        let chunk_blocks = blocks.div_ceil(threads);
        let mut values = Vec::<std::mem::MaybeUninit<u8>>::with_capacity(blocks * block_values);
        let mut scales = Vec::<std::mem::MaybeUninit<f32>>::with_capacity(blocks * DSA_ROW_BLOCK);
        // vec![..] 会先由 dispatcher 在单一 NUMA node 分配全部物理页；评分时固定 worker
        // 读取自己的 chunk，必须由同一 worker 完成首次写入，才能让 128K 历史留在本地内存。
        unsafe {
            values.set_len(blocks * block_values);
            scales.set_len(blocks * DSA_ROW_BLOCK);
        }
        let values_address = values.as_mut_ptr() as usize;
        let scales_address = scales.as_mut_ptr() as usize;
        dsa_team().run(threads, |chunk| {
            let first_block = chunk * chunk_blocks;
            let block_count = chunk_blocks.min(blocks - first_block);
            let value_output = unsafe { std::slice::from_raw_parts_mut((values_address as *mut std::mem::MaybeUninit<u8>).add(first_block * block_values), block_count * block_values) };
            let scale_output = unsafe { std::slice::from_raw_parts_mut((scales_address as *mut std::mem::MaybeUninit<f32>).add(first_block * DSA_ROW_BLOCK), block_count * DSA_ROW_BLOCK) };
            for (local_block, block_output) in value_output.chunks_mut(block_values).enumerate() {
                let block = first_block + local_block;
                for quad in 0..head_dim / 4 {
                    for lane in 0..DSA_ROW_BLOCK {
                        let row = block * DSA_ROW_BLOCK + lane;
                        for offset in 0..4 {
                            let value = if row < rows { (i16::from(keys[row * head_dim + quad * 4 + offset]) + 128) as u8 } else { 128 };
                            block_output[quad * DSA_ROW_BLOCK * 4 + lane * 4 + offset].write(value);
                        }
                    }
                }
            }
            for (local_row, output) in scale_output.iter_mut().enumerate() {
                let row = first_block * DSA_ROW_BLOCK + local_row;
                output.write(if row < rows { HalfBf16::from_bits(scale_bits[row]).to_f32() } else { 0.0 });
            }
        });
        let mut values = std::mem::ManuallyDrop::new(values);
        let values = unsafe { Vec::from_raw_parts(values.as_mut_ptr().cast::<u8>(), values.len(), values.capacity()) };
        let mut scales = std::mem::ManuallyDrop::new(scales);
        let scales = unsafe { Vec::from_raw_parts(scales.as_mut_ptr().cast::<f32>(), scales.len(), scales.capacity()) };
        Ok(Self { rows, available_rows: rows, head_dim, values, scales })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Prefill batch 按每个 query 的绝对 causal 终点递增暴露同一份完整 blocked history。
    pub fn set_logical_rows(&mut self, rows: usize) -> Result<(), String> {
        if rows == 0 || rows > self.available_rows {
            return Err(format!("CPU DSA Q8 logical rows={rows} 超过可用历史 {}", self.available_rows));
        }
        self.rows = rows;
        Ok(())
    }

    pub fn append_q8_row(&mut self, key: &[i8], scale_bits: u16) -> Result<(), String> {
        if key.len() != self.head_dim {
            return Err(format!("CPU DSA append Q8 key={}，期望 {}", key.len(), self.head_dim));
        }
        let lane = self.rows % DSA_ROW_BLOCK;
        let block = self.rows / DSA_ROW_BLOCK;
        // MTP rollback 保留已分配的尾块；再次跨过同一边界时直接覆盖，不能
        // 每轮拒绝 draft 都追加一块永远用不到的存储。
        let stored_rows = (block + 1) * DSA_ROW_BLOCK;
        if self.scales.len() < stored_rows {
            self.values.resize(stored_rows * self.head_dim, 128);
            self.scales.resize(stored_rows, 0.0);
        }
        let block_offset = block * DSA_ROW_BLOCK * self.head_dim;
        for quad in 0..self.head_dim / 4 {
            let output = block_offset + quad * DSA_ROW_BLOCK * 4 + lane * 4;
            for offset in 0..4 {
                self.values[output + offset] = (i16::from(key[quad * 4 + offset]) + 128) as u8;
            }
        }
        self.scales[block * DSA_ROW_BLOCK + lane] = HalfBf16::from_bits(scale_bits).to_f32();
        self.rows += 1;
        self.available_rows = self.available_rows.max(self.rows);
        Ok(())
    }

    /// rollback 只回退逻辑长度；后续 append 会覆盖旧尾行。
    pub fn truncate(&mut self, rows: usize) -> Result<(), String> {
        if rows > self.rows {
            return Err(format!("CPU DSA Q8 truncate rows={rows} 超过当前长度 {}", self.rows));
        }
        self.rows = rows;
        Ok(())
    }

    fn code(&self, row: usize, column: usize) -> i8 {
        let block = row / DSA_ROW_BLOCK;
        let lane = row % DSA_ROW_BLOCK;
        let quad = column / 4;
        let offset = column % 4;
        (self.values[block * DSA_ROW_BLOCK * self.head_dim + quad * DSA_ROW_BLOCK * 4 + lane * 4 + offset] as i16 - 128) as i8
    }
}

/// 单路 decode 逐层复用 query 转换、score 与 Top-K 空间，避免每层反复分配。
#[derive(Default)]
pub struct Bf16DsaWorkspace {
    query: Vec<HalfBf16>,
    #[cfg(target_arch = "x86_64")]
    query_pairs: Vec<u32>,
    scores: Vec<u32>,
    selection: Vec<u32>,
    topk_candidates: Vec<u32>,
}

impl Bf16DsaWorkspace {
    #[cfg(feature = "with-rocm")]
    pub(crate) fn scores(&self) -> &[u32] {
        &self.scores
    }
}

/// Q8 candidate path 逐层复用 query 量化、score 与 Top-K 空间。
#[derive(Default)]
pub struct Q8DsaWorkspace {
    query_codes: Vec<i8>,
    query_quads: Vec<u32>,
    query_scales: Vec<f32>,
    query_corrections: Vec<i32>,
    scores: Vec<u32>,
    selection: Vec<u32>,
    exact_keys: Vec<HalfBf16>,
    exact_workspace: Bf16DsaWorkspace,
    exact_selection: Vec<u32>,
    topk: Q8TopkScratch,
}

fn score_row_scalar(key: &[HalfBf16], query: &[HalfBf16], weights: &[f32], heads: usize, head_dim: usize) -> f32 {
    (0..heads)
        .map(|head| {
            let query = &query[head * head_dim..(head + 1) * head_dim];
            let dot = query.iter().zip(key).map(|(&left, &right)| left.to_f32() * right.to_f32()).sum::<f32>();
            weights[head] * dot.max(0.0)
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn score_row_avx512_bf16(key: &[HalfBf16], query: &[HalfBf16], weights: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let load = |values: &[HalfBf16]| unsafe { std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(values.as_ptr().cast())) };
    let keys = [load(&key[0..32]), load(&key[32..64]), load(&key[64..96]), load(&key[96..128])];
    let mut score = 0.0_f32;
    for head in 0..32 {
        let query = &query[head * 128..(head + 1) * 128];
        let mut sum = _mm512_setzero_ps();
        sum = _mm512_dpbf16_ps(sum, keys[0], load(&query[0..32]));
        sum = _mm512_dpbf16_ps(sum, keys[1], load(&query[32..64]));
        sum = _mm512_dpbf16_ps(sum, keys[2], load(&query[64..96]));
        sum = _mm512_dpbf16_ps(sum, keys[3], load(&query[96..128]));
        score += weights[head] * _mm512_reduce_add_ps(sum).max(0.0);
    }
    score
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
#[inline(never)]
unsafe fn score_head_tile_range_avx512_bf16<const FINISH: bool>(keys: *const HalfBf16, blocks: usize, valid_rows: usize, query_pairs: *const u32, weights: *const f32, scores: *mut u32) {
    use std::arch::x86_64::*;

    const HEAD_TILE: usize = 16;
    let pairs = 128 / 2;
    let zero = _mm512_setzero_ps();
    for block in 0..blocks {
        let keys = unsafe { keys.add(block * DSA_ROW_BLOCK * 128) };
        let mut sums = [zero; HEAD_TILE];
        for pair in 0..pairs {
            let key = unsafe { std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(keys.add(pair * DSA_ROW_BLOCK * 2).cast())) };
            let query = unsafe { query_pairs.add(pair * 32) };
            for lane in 0..HEAD_TILE {
                let query = unsafe { std::mem::transmute::<__m512i, __m512bh>(_mm512_set1_epi32(*query.add(lane) as i32)) };
                sums[lane] = _mm512_dpbf16_ps(sums[lane], key, query);
            }
        }
        // ROCm decode WMMA 用两个 parity lane 分别累加偶/奇 head，再做一次相加；
        // CPU 保持相同的 head 分组，剩余差异只来自 WMMA 与 AVX512 的点积归约顺序。
        let mut even = zero;
        let mut odd = zero;
        for lane in (0..HEAD_TILE).step_by(2) {
            even = _mm512_fmadd_ps(_mm512_max_ps(sums[lane], zero), _mm512_set1_ps(unsafe { *weights.add(lane) }), even);
            odd = _mm512_fmadd_ps(_mm512_max_ps(sums[lane + 1], zero), _mm512_set1_ps(unsafe { *weights.add(lane + 1) }), odd);
        }
        let mut total = _mm512_add_ps(even, odd);
        let output = unsafe { scores.add(block * DSA_ROW_BLOCK) };
        if FINISH {
            total = _mm512_add_ps(total, unsafe { _mm512_loadu_ps(output.cast()) });
            let bits = _mm512_castps_si512(total);
            let flip = _mm512_or_si512(_mm512_srai_epi32(bits, 31), _mm512_set1_epi32(i32::MIN));
            let ordered = _mm512_xor_si512(bits, flip);
            let rows = DSA_ROW_BLOCK.min(valid_rows - block * DSA_ROW_BLOCK);
            let mask = if rows == DSA_ROW_BLOCK { u16::MAX } else { (1_u16 << rows) - 1 };
            unsafe { _mm512_mask_storeu_epi32(output.cast(), mask, ordered) };
        } else {
            let rows = DSA_ROW_BLOCK.min(valid_rows - block * DSA_ROW_BLOCK);
            let mask = if rows == DSA_ROW_BLOCK { u16::MAX } else { (1_u16 << rows) - 1 };
            unsafe { _mm512_mask_storeu_ps(output.cast(), mask, total) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,fma")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn score_q8_head_tile_range_vnni<const FINISH: bool>(
    keys: *const u8,
    key_scales: *const f32,
    blocks: usize,
    valid_rows: usize,
    head_dim: usize,
    head_count: usize,
    query_quads: *const u32,
    query_scales: *const f32,
    query_corrections: *const i32,
    weights: *const f32,
    scores: *mut u32,
) {
    use std::arch::x86_64::*;

    const HEAD_TILE: usize = 16;
    let zero_f32 = _mm512_setzero_ps();
    let sign_flip = _mm512_set1_epi32(i32::MIN);
    for block in 0..blocks {
        let block_keys = unsafe { keys.add(block * DSA_ROW_BLOCK * head_dim) };
        let mut dots = [_mm512_setzero_si512(); HEAD_TILE];
        for quad in 0..head_dim / 4 {
            let key = unsafe { _mm512_loadu_si512(block_keys.add(quad * DSA_ROW_BLOCK * 4).cast()) };
            for head in 0..HEAD_TILE {
                let query = _mm512_set1_epi32(unsafe { *query_quads.add(quad * head_count + head) } as i32);
                dots[head] = _mm512_dpbusd_epi32(dots[head], key, query);
            }
        }
        let scales = unsafe { _mm512_loadu_ps(key_scales.add(block * DSA_ROW_BLOCK)) };
        let mut even = zero_f32;
        let mut odd = zero_f32;
        for head in (0..HEAD_TILE).step_by(2) {
            let correction = _mm512_set1_epi32(unsafe { *query_corrections.add(head) });
            let dot = _mm512_cvtepi32_ps(_mm512_sub_epi32(dots[head], correction));
            let scale = _mm512_mul_ps(scales, _mm512_set1_ps(unsafe { *query_scales.add(head) }));
            let value = _mm512_max_ps(_mm512_mul_ps(dot, scale), zero_f32);
            even = _mm512_fmadd_ps(value, _mm512_set1_ps(unsafe { *weights.add(head) }), even);

            let correction = _mm512_set1_epi32(unsafe { *query_corrections.add(head + 1) });
            let dot = _mm512_cvtepi32_ps(_mm512_sub_epi32(dots[head + 1], correction));
            let scale = _mm512_mul_ps(scales, _mm512_set1_ps(unsafe { *query_scales.add(head + 1) }));
            let value = _mm512_max_ps(_mm512_mul_ps(dot, scale), zero_f32);
            odd = _mm512_fmadd_ps(value, _mm512_set1_ps(unsafe { *weights.add(head + 1) }), odd);
        }
        let mut total = _mm512_add_ps(even, odd);
        let output = unsafe { scores.add(block * DSA_ROW_BLOCK) };
        if FINISH {
            total = _mm512_add_ps(total, unsafe { _mm512_loadu_ps(output.cast()) });
            let bits = _mm512_castps_si512(total);
            let flip = _mm512_or_si512(_mm512_srai_epi32(bits, 31), sign_flip);
            let ordered = _mm512_xor_si512(bits, flip);
            let rows = DSA_ROW_BLOCK.min(valid_rows - block * DSA_ROW_BLOCK);
            let mask = if rows == DSA_ROW_BLOCK { u16::MAX } else { (1_u16 << rows) - 1 };
            unsafe { _mm512_mask_storeu_epi32(output.cast(), mask, ordered) };
        } else {
            let rows = DSA_ROW_BLOCK.min(valid_rows - block * DSA_ROW_BLOCK);
            let mask = if rows == DSA_ROW_BLOCK { u16::MAX } else { (1_u16 << rows) - 1 };
            unsafe { _mm512_mask_storeu_ps(output.cast(), mask, total) };
        }
    }
}

fn quantize_q8_query(query: &[f32], heads: usize, head_dim: usize, workspace: &mut Q8DsaWorkspace) -> Result<(), String> {
    workspace.query_codes.resize(query.len(), 0);
    workspace.query_scales.resize(heads, 1.0);
    workspace.query_corrections.resize(heads, 0);
    for head in 0..heads {
        let source = &query[head * head_dim..][..head_dim];
        let maximum = source.iter().try_fold(0.0_f32, |maximum, &value| if value.is_finite() { Ok(maximum.max(HalfBf16::from_f32(value).to_f32().abs())) } else { Err(format!("CPU DSA query head={head} 包含非有限值")) })?;
        let scale = if maximum > 0.0 { maximum / 127.0 } else { 1.0 };
        let mut sum = 0_i32;
        for (column, &value) in source.iter().enumerate() {
            let value = HalfBf16::from_f32(value).to_f32();
            let code = (value / scale).round().clamp(-127.0, 127.0) as i8;
            workspace.query_codes[head * head_dim + column] = code;
            sum += i32::from(code);
        }
        workspace.query_scales[head] = scale;
        workspace.query_corrections[head] = 128 * sum;
    }
    workspace.query_quads.resize(heads * (head_dim / 4), 0);
    for quad in 0..head_dim / 4 {
        for head in 0..heads {
            let begin = head * head_dim + quad * 4;
            workspace.query_quads[quad * heads + head] = u32::from_le_bytes([workspace.query_codes[begin] as u8, workspace.query_codes[begin + 1] as u8, workspace.query_codes[begin + 2] as u8, workspace.query_codes[begin + 3] as u8]);
        }
    }
    Ok(())
}

/// 单个 chunk 的打分：VNNI 双 tile 或标量回退；score 与融合 top-k 共用。
#[allow(clippy::too_many_arguments)]
fn score_q8_chunk(
    use_vnni: bool,
    keys: &Q8KeyBlocks,
    _query_quads: &[u32],
    query_codes: &[i8],
    query_scales: &[f32],
    _query_corrections: &[i32],
    weights: &[f32],
    heads: usize,
    head_dim: usize,
    chunk_blocks: usize,
    chunk: usize,
    output: &mut [u32],
) {
    let first_block = chunk * chunk_blocks;
    if use_vnni {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let block_count = output.len().div_ceil(DSA_ROW_BLOCK);
            let block_keys = keys.values.as_ptr().add(first_block * head_dim * DSA_ROW_BLOCK);
            let block_scales = keys.scales.as_ptr().add(first_block * DSA_ROW_BLOCK);
            score_q8_head_tile_range_vnni::<false>(block_keys, block_scales, block_count, output.len(), head_dim, heads, _query_quads.as_ptr(), query_scales.as_ptr(), _query_corrections.as_ptr(), weights.as_ptr(), output.as_mut_ptr());
            score_q8_head_tile_range_vnni::<true>(
                block_keys,
                block_scales,
                block_count,
                output.len(),
                head_dim,
                heads,
                _query_quads.as_ptr().add(16),
                query_scales.as_ptr().add(16),
                _query_corrections.as_ptr().add(16),
                weights.as_ptr().add(16),
                output.as_mut_ptr(),
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        unreachable!()
    } else {
        for (local_row, output) in output.iter_mut().enumerate() {
            let row = first_block * DSA_ROW_BLOCK + local_row;
            let mut score = 0.0_f32;
            for head in 0..heads {
                let mut dot = 0_i32;
                for column in 0..head_dim {
                    dot += i32::from(keys.code(row, column)) * i32::from(query_codes[head * head_dim + column]);
                }
                score += weights[head] * (dot as f32 * keys.scales[row / DSA_ROW_BLOCK * DSA_ROW_BLOCK + row % DSA_ROW_BLOCK] * query_scales[head]).max(0.0);
            }
            *output = ordered_score(score);
        }
    }
}

pub fn score_q8_blocked_workspace(keys: &Q8KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, workspace: &mut Q8DsaWorkspace) -> Result<f64, String> {
    if heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(4) || keys.head_dim != head_dim || query.len() != heads.checked_mul(head_dim).ok_or("CPU DSA Q8 query 大小溢出")? || weights.len() != heads {
        return Err(format!("CPU DSA Q8 shape rows={} key_dim={} query={} weights={} heads={heads} dim={head_dim} 非法", keys.rows, keys.head_dim, query.len(), weights.len()));
    }
    let score_started = std::time::Instant::now();
    quantize_q8_query(query, heads, head_dim, workspace)?;
    #[cfg(target_arch = "x86_64")]
    let use_vnni = heads == 32 && std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") && std::arch::is_x86_feature_detected!("avx512vnni") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_vnni = false;
    let blocks = keys.rows.div_ceil(DSA_ROW_BLOCK);
    let threads = dsa_team().workers().min(blocks.div_ceil(128).max(1));
    let chunk_blocks = blocks.div_ceil(threads);
    let chunk_rows = chunk_blocks * DSA_ROW_BLOCK;
    workspace.scores.resize(keys.rows, 0);
    let scores = &mut workspace.scores;
    let scores_address = scores.as_mut_ptr() as usize;
    dsa_team().run(threads, |chunk| {
        let first_row = chunk * chunk_rows;
        let rows = chunk_rows.min(keys.rows - first_row);
        let output = unsafe { std::slice::from_raw_parts_mut((scores_address as *mut u32).add(first_row), rows) };
        score_q8_chunk(use_vnni, keys, &workspace.query_quads, &workspace.query_codes, &workspace.query_scales, &workspace.query_corrections, weights, heads, head_dim, chunk_blocks, chunk, output);
    });
    Ok(score_started.elapsed().as_secs_f64() * 1e3)
}

/// 生产 decode 的融合路径：全历史 Q8 打分与 final Top-K 在同一次 team run 内完成，
/// 打分与四个 top-k 阶段之间用 barrier 衔接，省去第二次分发同步。输出与
/// `score_q8_blocked_workspace` + `finish_q8_blocked_topk` 逐字节一致。
pub fn score_q8_blocked_topk<'a>(keys: &Q8KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize, workspace: &'a mut Q8DsaWorkspace) -> Result<(&'a [u32], f64, f64), String> {
    if heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(4) || keys.head_dim != head_dim || query.len() != heads.checked_mul(head_dim).ok_or("CPU DSA Q8 query 大小溢出")? || weights.len() != heads {
        return Err(format!("CPU DSA Q8 shape rows={} key_dim={} query={} weights={} heads={heads} dim={head_dim} 非法", keys.rows, keys.head_dim, query.len(), weights.len()));
    }
    if top_k == 0 || top_k > keys.rows {
        return Err(format!("CPU DSA Q8 final top_k={top_k}，rows={}", keys.rows));
    }
    let score_started = std::time::Instant::now();
    quantize_q8_query(query, heads, head_dim, workspace)?;
    #[cfg(target_arch = "x86_64")]
    let use_vnni = heads == 32 && std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") && std::arch::is_x86_feature_detected!("avx512vnni") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_vnni = false;
    let blocks = keys.rows.div_ceil(DSA_ROW_BLOCK);
    let threads = dsa_team().workers().min(blocks.div_ceil(128).max(1));
    let chunk_blocks = blocks.div_ceil(threads);
    let chunk_rows = chunk_blocks * DSA_ROW_BLOCK;
    workspace.scores.resize(keys.rows, 0);
    let topk = &mut workspace.topk;
    if topk.fused_staging.len() != threads {
        topk.fused_staging = (0..threads).map(|_| Vec::with_capacity(4096)).collect();
    }
    if topk.partials.len() != threads {
        topk.partials = (0..threads).map(|_| vec![0_u32; RADIX_SIZE]).collect();
    }
    if topk.hist.len() != RADIX_SIZE {
        topk.hist = (0..RADIX_SIZE).map(|_| std::sync::atomic::AtomicU32::new(0)).collect();
    }
    for bucket in &topk.hist {
        bucket.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    let rows = keys.rows;
    let query_quads = &workspace.query_quads;
    let query_codes = &workspace.query_codes;
    let query_scales = &workspace.query_scales;
    let query_corrections = &workspace.query_corrections;
    let scores_address = workspace.scores.as_mut_ptr() as usize;
    let hist_address = topk.hist.as_ptr() as usize;
    let partials_address = topk.partials.as_mut_ptr() as usize;
    let staging_address = topk.fused_staging.as_mut_ptr() as usize;
    let barrier = &topk.barrier;
    let threshold = &topk.threshold;
    let phase_micros = &topk.phase_micros;
    let start_micros = dsa_team_micros();
    dsa_team().run(threads, move |worker| {
        // 打分阶段：chunk 划分与 score_q8_blocked_workspace 完全一致。
        let first_row = worker * chunk_rows;
        if first_row < rows {
            let chunk_rows = chunk_rows.min(rows - first_row);
            let output = unsafe { std::slice::from_raw_parts_mut((scores_address as *mut u32).add(first_row), chunk_rows) };
            score_q8_chunk(use_vnni, keys, &query_quads, &query_codes, &query_scales, &query_corrections, weights, heads, head_dim, chunk_blocks, worker, output);
        }
        barrier.wait(threads as u32);
        if worker == 0 {
            phase_micros.store(dsa_team_micros(), std::sync::atomic::Ordering::Release);
        }
        // 分数的高位通常集中在很少几个桶；先在本核统计，再按非空桶发布，
        // 避免每一行都争用跨 NUMA 的原子 cache line。
        let scores = unsafe { std::slice::from_raw_parts(scores_address as *const u32, rows) };
        let partial = unsafe { &mut *(partials_address as *mut Vec<u32>).add(worker) };
        partial.fill(0);
        if first_row < rows {
            let chunk_rows = chunk_rows.min(rows - first_row);
            for &score in &scores[first_row..first_row + chunk_rows] {
                partial[(score >> (32 - RADIX_BITS)) as usize] += 1;
            }
            for (bucket, &count) in partial.iter().enumerate().filter(|(_, count)| **count != 0) {
                unsafe {
                    (*(hist_address as *mut std::sync::atomic::AtomicU32).add(bucket)).fetch_add(count, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        barrier.wait(threads as u32);
        if worker == 0 {
            let mut rank = top_k;
            let mut found = 0_usize;
            for bucket in (0..RADIX_SIZE).rev() {
                let count = unsafe { (*(hist_address as *const std::sync::atomic::AtomicU32).add(bucket)).load(std::sync::atomic::Ordering::Relaxed) } as usize;
                if count < rank {
                    rank -= count;
                } else {
                    found = bucket;
                    break;
                }
            }
            threshold.store(found as u32, std::sync::atomic::Ordering::Release);
        }
        barrier.wait(threads as u32);
        let found = threshold.load(std::sync::atomic::Ordering::Acquire) as usize;
        let staging = unsafe { &mut *(staging_address as *mut Vec<(u32, u32)>).add(worker) };
        staging.clear();
        if first_row < rows {
            let chunk_rows = chunk_rows.min(rows - first_row);
            for (offset, &score) in scores[first_row..first_row + chunk_rows].iter().enumerate() {
                if (score >> (32 - RADIX_BITS)) as usize >= found {
                    staging.push((score, (first_row + offset) as u32));
                }
            }
        }
    });
    let score_ms = phase_micros.load(std::sync::atomic::Ordering::Acquire).saturating_sub(start_micros) as f64 / 1000.0;
    // 尾部语义与 stable_topk_into 一致：(score desc, token asc) 全序排序后截断，
    // 再按 (==kth, token) 重排；对内比较全为寄存器比较，无 scores 间接寻址。
    let Q8DsaWorkspace { exact_selection, topk, .. } = workspace;
    let pairs = &mut topk.pairs;
    pairs.clear();
    for worker in 0..threads {
        pairs.extend_from_slice(&topk.fused_staging[worker]);
    }
    // 只求第 K 名；大量同分时也不对整段历史排序。全序比较保留 token tie-break。
    let (_, kth, _) = pairs.select_nth_unstable_by(top_k - 1, |left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let threshold = kth.0;
    pairs.truncate(top_k);
    pairs.sort_unstable_by(|left, right| (left.0 == threshold, left.1).cmp(&(right.0 == threshold, right.1)));
    exact_selection.clear();
    exact_selection.extend(pairs.iter().map(|&(_, token)| token));
    let total_ms = score_started.elapsed().as_secs_f64() * 1e3;
    Ok((exact_selection, score_ms, total_ms - score_ms))
}

fn select_q8_blocked_workspace_profile(keys: &Q8KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize, workspace: &mut Q8DsaWorkspace) -> Result<(f64, f64), String> {
    if top_k == 0 || top_k > keys.rows {
        return Err(format!("CPU DSA Q8 candidate top_k={top_k}，rows={}", keys.rows));
    }
    let score_ms = score_q8_blocked_workspace(keys, query, weights, heads, head_dim, workspace)?;
    let topk_started = std::time::Instant::now();
    radix_candidate_superset_into(&workspace.scores, top_k, &mut workspace.selection);
    let topk_ms = topk_started.elapsed().as_secs_f64() * 1e3;
    Ok((score_ms, topk_ms))
}

/// 两级 exact rerank 的 VNNI 第一阶段：返回按 token 递增的候选集合。
pub fn select_q8_blocked_candidates<'a>(keys: &Q8KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, candidate_count: usize, workspace: &'a mut Q8DsaWorkspace) -> Result<(&'a [u32], f64, f64), String> {
    let (score_ms, topk_ms) = select_q8_blocked_workspace_profile(keys, query, weights, heads, head_dim, candidate_count, workspace)?;
    Ok((&workspace.selection, score_ms, topk_ms))
}

/// 对刚完成的全历史 Q8 score 直接做稳定 Top-K，不再展开候选并重复评分。
pub fn finish_q8_blocked_topk(top_k: usize, workspace: &mut Q8DsaWorkspace) -> Result<&[u32], String> {
    if top_k == 0 || top_k > workspace.scores.len() {
        return Err(format!("CPU DSA Q8 final top_k={top_k}，score rows={}", workspace.scores.len()));
    }
    parallel_stable_topk_into(&workspace.scores, top_k, &mut workspace.exact_selection, &mut workspace.topk);
    Ok(&workspace.exact_selection)
}

/// 对粗排候选使用生产 Q8 key、BF16 query 重新评分，并按 global token id
/// 保持与 GPU radix 相同的稳定输出顺序。该路径只展开候选，不扫描或复制全历史。
pub fn rerank_q8_blocked_candidates<'a>(keys: &Q8KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, candidates: &[u32], top_k: usize, workspace: &'a mut Q8DsaWorkspace) -> Result<&'a [u32], String> {
    if heads == 0
        || head_dim == 0
        || keys.head_dim != head_dim
        || query.len() != heads.checked_mul(head_dim).ok_or("CPU DSA exact query 大小溢出")?
        || weights.len() != heads
        || top_k == 0
        || top_k > candidates.len()
        || candidates.windows(2).any(|pair| pair[0] >= pair[1])
        || candidates.last().is_some_and(|&token| token as usize >= keys.rows)
    {
        return Err(format!("CPU DSA exact shape rows={} dim={} query={} weights={} candidates={} top_k={top_k} 非法", keys.rows, keys.head_dim, query.len(), weights.len(), candidates.len()));
    }
    workspace.exact_keys.resize(candidates.len() * head_dim, HalfBf16::ZERO);
    for (candidate, &token) in candidates.iter().enumerate() {
        let token = token as usize;
        let scale = keys.scales[token];
        for column in 0..head_dim {
            workspace.exact_keys[candidate * head_dim + column] = HalfBf16::from_f32(keys.code(token, column) as f32 * scale);
        }
    }
    let exact_keys = Bf16KeyBlocks::from_row_major(&workspace.exact_keys, head_dim)?;
    workspace.exact_selection.clear();
    workspace.exact_selection.extend_from_slice(select_bf16_blocked(&exact_keys, query, weights, heads, head_dim, top_k, &mut workspace.exact_workspace)?);
    for token in &mut workspace.exact_selection {
        *token = candidates[*token as usize];
    }
    Ok(&workspace.exact_selection)
}

fn validate_shape(keys: &Bf16KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize) -> Result<(), String> {
    if heads == 0 || head_dim == 0 || top_k == 0 || keys.rows == 0 || keys.head_dim != head_dim || query.len() != heads.checked_mul(head_dim).ok_or("CPU DSA query 大小溢出")? || weights.len() != heads || top_k > keys.rows {
        return Err(format!("CPU DSA blocked shape rows={} key_dim={} query={} weights={} heads={heads} dim={head_dim} top_k={top_k} 非法", keys.rows, keys.head_dim, query.len(), weights.len()));
    }
    Ok(())
}

fn select_bf16_blocked_workspace_profile(keys: &Bf16KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize, workspace: &mut Bf16DsaWorkspace) -> Result<(f64, f64), String> {
    validate_shape(keys, query, weights, heads, head_dim, top_k)?;
    let score_started = std::time::Instant::now();
    workspace.query.clear();
    workspace.query.extend(query.iter().copied().map(HalfBf16::from_f32));
    let query = &workspace.query;
    #[cfg(target_arch = "x86_64")]
    let use_avx512 = heads == 32 && head_dim == 128 && std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bf16");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512 = false;
    let pairs = head_dim / 2;
    #[cfg(target_arch = "x86_64")]
    {
        workspace.query_pairs.clear();
        workspace.query_pairs.extend((0..pairs).flat_map(|pair| {
            let query = &query;
            (0..heads).map(move |head| query[head * head_dim + pair * 2].to_bits() as u32 | (query[head * head_dim + pair * 2 + 1].to_bits() as u32) << 16)
        }));
    }
    let blocks = keys.rows.div_ceil(DSA_ROW_BLOCK);
    let threads = dsa_team().workers().min(blocks.div_ceil(128).max(1));
    let chunk_blocks = blocks.div_ceil(threads);
    let chunk_rows = chunk_blocks * DSA_ROW_BLOCK;
    workspace.scores.resize(keys.rows, 0);
    let scores = &mut workspace.scores;
    let score_chunk = |chunk: usize, output: &mut [u32]| {
        let first_block = chunk * chunk_blocks;
        if use_avx512 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let block_count = output.len().div_ceil(DSA_ROW_BLOCK);
                let block_keys = keys.values.as_ptr().add(first_block * head_dim * DSA_ROW_BLOCK);
                score_head_tile_range_avx512_bf16::<false>(block_keys, block_count, output.len(), workspace.query_pairs.as_ptr(), weights.as_ptr(), output.as_mut_ptr());
                score_head_tile_range_avx512_bf16::<true>(block_keys, block_count, output.len(), workspace.query_pairs.as_ptr().add(16), weights.as_ptr().add(16), output.as_mut_ptr());
            }
            #[cfg(not(target_arch = "x86_64"))]
            unreachable!()
        } else {
            for (local_block, block_output) in output.chunks_mut(DSA_ROW_BLOCK).enumerate() {
                let block = first_block + local_block;
                let block_keys = &keys.values[block * head_dim * DSA_ROW_BLOCK..][..head_dim * DSA_ROW_BLOCK];
                for (lane, output) in block_output.iter_mut().enumerate() {
                    let mut key = vec![HalfBf16::ZERO; head_dim];
                    for pair in 0..pairs {
                        key[pair * 2] = block_keys[pair * DSA_ROW_BLOCK * 2 + lane * 2];
                        key[pair * 2 + 1] = block_keys[pair * DSA_ROW_BLOCK * 2 + lane * 2 + 1];
                    }
                    *output = ordered_score(score_row_scalar(&key, &query, weights, heads, head_dim));
                }
            }
        }
    };
    let scores_address = scores.as_mut_ptr() as usize;
    dsa_team().run(threads, |worker| {
        let first_row = worker * chunk_rows;
        let rows = chunk_rows.min(keys.rows - first_row);
        // 固定 worker 读取 first-touch 在同一物理核上的历史页，并只写自己的区间。
        let output = unsafe { std::slice::from_raw_parts_mut((scores_address as *mut u32).add(first_row), rows) };
        score_chunk(worker, output);
    });
    let score_ms = score_started.elapsed().as_secs_f64() * 1e3;
    let topk_started = std::time::Instant::now();
    exact_radix_topk_into(scores, top_k, &mut workspace.selection, &mut workspace.topk_candidates);
    let topk_ms = topk_started.elapsed().as_secs_f64() * 1e3;
    Ok((score_ms, topk_ms))
}

/// 生产 decode 路径：复用全部临时空间，只返回当前 global Top-K。
pub fn select_bf16_blocked<'a>(keys: &Bf16KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize, workspace: &'a mut Bf16DsaWorkspace) -> Result<&'a [u32], String> {
    select_bf16_blocked_workspace_profile(keys, query, weights, heads, head_dim, top_k, workspace)?;
    Ok(&workspace.selection)
}

/// 生产 decode 路径的带计时版本；与 `select_bf16_blocked` 复用同一份临时空间。
pub fn select_bf16_blocked_timed<'a>(keys: &Bf16KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize, workspace: &'a mut Bf16DsaWorkspace) -> Result<(&'a [u32], f64, f64), String> {
    let (score_ms, topk_ms) = select_bf16_blocked_workspace_profile(keys, query, weights, heads, head_dim, top_k, workspace)?;
    Ok((&workspace.selection, score_ms, topk_ms))
}

/// 两级 exact rerank 的第一阶段：返回按 token 递增的候选集合，供 GPU 在候选域内稳定选择。
pub fn select_bf16_blocked_candidates<'a>(keys: &Bf16KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, candidate_count: usize, workspace: &'a mut Bf16DsaWorkspace) -> Result<&'a [u32], String> {
    select_bf16_blocked_workspace_profile(keys, query, weights, heads, head_dim, candidate_count, workspace)?;
    workspace.selection.sort_unstable();
    Ok(&workspace.selection)
}

/// 使用 CPU 自有 blocked key 布局评分；返回 ordered score、稳定 Top-K 与两段耗时。
pub fn select_bf16_blocked_profile(keys: &Bf16KeyBlocks, query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize) -> Result<(Vec<u32>, Vec<u32>, f64, f64), String> {
    let mut workspace = Bf16DsaWorkspace::default();
    let (score_ms, topk_ms) = select_bf16_blocked_workspace_profile(keys, query, weights, heads, head_dim, top_k, &mut workspace)?;
    Ok((workspace.scores, workspace.selection, score_ms, topk_ms))
}

/// 输入 key/query 都必须已经舍入到生产 BF16 score 域；返回 ordered score、稳定 Top-K 与两段耗时。
pub fn select_bf16_profile(keys: &[HalfBf16], query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize) -> Result<(Vec<u32>, Vec<u32>, f64, f64), String> {
    if heads == 0 || head_dim == 0 || top_k == 0 || keys.is_empty() || !keys.len().is_multiple_of(head_dim) || query.len() != heads.checked_mul(head_dim).ok_or("CPU DSA query 大小溢出")? || weights.len() != heads {
        return Err(format!("CPU DSA shape keys={} query={} weights={} heads={heads} dim={head_dim} top_k={top_k} 非法", keys.len(), query.len(), weights.len()));
    }
    let rows = keys.len() / head_dim;
    if top_k > rows {
        return Err(format!("CPU DSA top_k={top_k} 超过 rows={rows}"));
    }
    let score_started = std::time::Instant::now();
    let query = query.iter().copied().map(HalfBf16::from_f32).collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    let use_avx512 = heads == 32 && head_dim == 128 && std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bf16");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512 = false;
    let threads = dsa_pool().current_num_threads().min(rows.div_ceil(2048).max(1));
    let chunk_rows = rows.div_ceil(threads);
    let mut scores = vec![0_u32; rows];
    dsa_pool().install(|| {
        scores.par_chunks_mut(chunk_rows).enumerate().for_each(|(chunk, output)| {
            let first = chunk * chunk_rows;
            let keys = &keys[first * head_dim..(first + output.len()) * head_dim];
            for (row, output) in output.iter_mut().enumerate() {
                let key = &keys[row * head_dim..(row + 1) * head_dim];
                let score = if use_avx512 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        score_row_avx512_bf16(key, &query, weights)
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    unreachable!()
                } else {
                    score_row_scalar(key, &query, weights, heads, head_dim)
                };
                *output = ordered_score(score);
            }
        });
    });
    let score_ms = score_started.elapsed().as_secs_f64() * 1e3;
    let topk_started = std::time::Instant::now();
    let selection = stable_topk(&scores, top_k);
    let topk_ms = topk_started.elapsed().as_secs_f64() * 1e3;
    Ok((scores, selection, score_ms, topk_ms))
}

/// 输入 key/query 都必须已经舍入到生产 BF16 score 域；返回 ordered score 与稳定 Top-K。
pub fn select_bf16(keys: &[HalfBf16], query: &[f32], weights: &[f32], heads: usize, head_dim: usize, top_k: usize) -> Result<(Vec<u32>, Vec<u32>), String> {
    let (scores, selection, _, _) = select_bf16_profile(keys, query, weights, heads, head_dim, top_k)?;
    Ok((scores, selection))
}

#[cfg(test)]
mod tests {
    #[test]
    fn q8_append_after_rollback_reuses_blocks_and_replaces_tail() {
        let dim = 4;
        let mut codes = vec![1_i8; 15 * dim];
        let mut scales = vec![HalfBf16::from_f32(0.5).to_bits(); 15];
        let mut keys = Q8KeyBlocks::from_q8_row_major(&codes, &scales, dim).unwrap();
        for round in 0..32 {
            keys.truncate(15).unwrap();
            codes.truncate(15 * dim);
            scales.truncate(15);
            for row in 0..4 {
                let key = [round as i8, row, -3, 7];
                let scale = HalfBf16::from_f32(0.25 + f32::from(row)).to_bits();
                keys.append_q8_row(&key, scale).unwrap();
                codes.extend_from_slice(&key);
                scales.push(scale);
            }
            let rebuilt = Q8KeyBlocks::from_q8_row_major(&codes, &scales, dim).unwrap();
            assert_eq!(keys.rows(), rebuilt.rows());
            assert_eq!(keys.values, rebuilt.values);
            assert_eq!(keys.scales, rebuilt.scales);
        }
    }

    #[test]
    fn direct_team_changes_active_workers_without_reusing_a_job() {
        let team = dsa_team();
        let visits = (0..team.workers()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect::<Vec<_>>();
        for round in 0..128 {
            let active = if round % 2 == 0 { 1 } else { team.workers() };
            team.run(active, |worker| {
                visits[worker].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
            for (worker, count) in visits.iter().enumerate() {
                assert_eq!(count.swap(0, std::sync::atomic::Ordering::Relaxed), usize::from(worker < active), "round={round} worker={worker}");
            }
        }
    }

    #[test]
    fn fused_q8_topk_preserves_ties_and_workspace_reuse() {
        let heads = 2;
        let dim = 4;
        let rows = 8209;
        let codes = (0..rows * dim).map(|index| ((index / dim) % 31) as i8 - 15).collect::<Vec<_>>();
        let scales = vec![HalfBf16::from_f32(0.25).to_bits(); rows];
        let mut keys = Q8KeyBlocks::from_q8_row_major(&codes, &scales, dim).unwrap();
        let mut workspace = Q8DsaWorkspace::default();
        for length in [rows, 2051, 17, rows] {
            keys.set_logical_rows(length).unwrap();
            for weights in [[0.0, 0.0], [0.5, -0.25]] {
                let query = [1.0, -2.0, 3.0, 0.0, -1.0, 0.0, 0.5, 2.0];
                for k in [1, 13, length] {
                    let actual = score_q8_blocked_topk(&keys, &query, &weights, heads, dim, k, &mut workspace).unwrap().0.to_vec();
                    assert_eq!(actual, stable_topk(&workspace.scores, k), "rows={length} k={k}");
                    assert_eq!(actual, finish_q8_blocked_topk(k, &mut workspace).unwrap());
                }
            }
        }
    }

    use super::*;

    #[test]
    fn fused_score_topk_matches_staged_path() {
        let heads = 32;
        let head_dim = 128;
        let rows = 150_000;
        let codes = (0..rows * head_dim).map(|index| ((index * 37 + 11) % 251) as i16 - 125).map(|value| value as i8).collect::<Vec<_>>();
        let scale_bits = (0..rows).map(|index| HalfBf16::from_f32(0.002 + (index % 97) as f32 * 0.00003).to_bits()).collect::<Vec<_>>();
        let keys = Q8KeyBlocks::from_q8_row_major(&codes, &scale_bits, head_dim).unwrap();
        let query = (0..heads * head_dim).map(|index| ((index as f32 * 0.013).cos() * 1.9).clamp(-1.8, 1.8)).collect::<Vec<_>>();
        let weights = (0..heads).map(|head| 0.008 + head as f32 * 0.00021).collect::<Vec<_>>();
        let mut staged = Q8DsaWorkspace::default();
        score_q8_blocked_workspace(&keys, &query, &weights, heads, head_dim, &mut staged).unwrap();
        let staged_selection = finish_q8_blocked_topk(2048, &mut staged).unwrap().to_vec();
        let mut fused = Q8DsaWorkspace::default();
        let (fused_selection, _, _) = score_q8_blocked_topk(&keys, &query, &weights, heads, head_dim, 2048, &mut fused).unwrap();
        assert_eq!(staged_selection, fused_selection);
        assert_eq!(staged.scores, fused.scores);
    }

    #[test]
    fn parallel_topk_matches_serial_stable_topk() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (rows, top_k, mode) in [(150_000, 2048, 0), (150_000, 2048, 1), (150_000, 2048, 2), (4_096, 512, 1)] {
            let scores = (0..rows)
                .map(|_index| match mode {
                    // 0: 均匀随机；1: 高位集中制造大 tie bucket；2: 全同分极端情形。
                    0 => (next() >> 40) as u32,
                    1 => ((next() >> 60) as u32) << 28,
                    _ => 0x1234_5600,
                })
                .collect::<Vec<_>>();
            let mut serial = Vec::new();
            stable_topk_into(&scores, top_k, &mut serial);
            let mut exact_radix = Vec::new();
            exact_radix_topk_into(&scores, top_k, &mut exact_radix, &mut Vec::new());
            assert_eq!(serial, exact_radix, "exact radix rows={rows} top_k={top_k} mode={mode}");
            let mut parallel = Vec::new();
            parallel_stable_topk_into(&scores, top_k, &mut parallel, &mut Q8TopkScratch::default());
            assert_eq!(serial, parallel, "rows={rows} top_k={top_k} mode={mode}");
        }
    }

    #[test]
    fn high_radix_topk_matches_full_order() {
        let scores = (0..10_003)
            .map(|index| {
                let mixed = (index as u32).wrapping_mul(2_654_435_761).rotate_left((index % 31) as u32);
                mixed & 0xffff_fff0
            })
            .collect::<Vec<_>>();
        let mut expected = (0..scores.len() as u32).collect::<Vec<_>>();
        expected.sort_unstable_by(|&left, &right| scores[right as usize].cmp(&scores[left as usize]).then_with(|| left.cmp(&right)));
        expected.truncate(2048);
        let threshold = scores[expected[2047] as usize];
        expected.sort_unstable_by_key(|&token| (scores[token as usize] == threshold, token));
        assert_eq!(stable_topk(&scores, 2048), expected);
        let mut candidates = Vec::new();
        radix_candidate_superset_into(&scores, 2048, &mut candidates);
        assert!(candidates.len() >= 2048);
        assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(expected.iter().all(|token| candidates.binary_search(token).is_ok()));
    }

    #[test]
    fn bf16_selection_keeps_score_order_and_stable_ties() {
        let heads = 32;
        let head_dim = 128;
        let rows = 128;
        let mut keys = vec![HalfBf16::ZERO; rows * head_dim];
        for row in 0..rows {
            keys[row * head_dim] = HalfBf16::from_f32((row / 2) as f32);
        }
        let mut query = vec![0.0_f32; heads * head_dim];
        query[0] = 1.0;
        let mut weights = vec![0.0_f32; heads];
        weights[0] = 1.0;
        let (_, selected) = select_bf16(&keys, &query, &weights, heads, head_dim, 4).unwrap();
        assert_eq!(selected, vec![126, 127, 124, 125]);
    }

    #[test]
    fn blocked_bf16_selection_keeps_score_order_and_stable_ties() {
        let heads = 32;
        let head_dim = 128;
        let rows = 128;
        let mut keys = vec![HalfBf16::ZERO; rows * head_dim];
        for row in 0..rows {
            keys[row * head_dim] = HalfBf16::from_f32((row / 2) as f32);
        }
        let keys = Bf16KeyBlocks::from_row_major(&keys, head_dim).unwrap();
        let mut query = vec![0.0_f32; heads * head_dim];
        query[0] = 1.0;
        let mut weights = vec![0.0_f32; heads];
        weights[0] = 1.0;
        let (_, selected, _, _) = select_bf16_blocked_profile(&keys, &query, &weights, heads, head_dim, 4).unwrap();
        assert_eq!(selected, vec![126, 127, 124, 125]);
    }

    #[test]
    fn blocked_bf16_q8_append_and_rollback_match_rebuild() {
        let head_dim = 128;
        let rows = 19;
        let codes = (0..rows * head_dim).map(|index| ((index * 17 + 5) % 251) as i16 - 125).map(|value| value as i8).collect::<Vec<_>>();
        let scales = (0..rows).map(|row| HalfBf16::from_f32(0.002 + row as f32 * 0.0001).to_bits()).collect::<Vec<_>>();
        let rebuilt = Bf16KeyBlocks::from_q8_row_major(&codes, &scales, head_dim, head_dim).unwrap();
        let mut incremental = Bf16KeyBlocks::from_q8_row_major(&codes[..16 * head_dim], &scales[..16], head_dim, head_dim).unwrap();
        for row in 16..rows {
            incremental.append_q8_row(&codes[row * head_dim..(row + 1) * head_dim], &scales[row..row + 1], head_dim).unwrap();
        }
        assert_eq!(incremental.rows, rebuilt.rows);
        assert_eq!(incremental.values, rebuilt.values);

        incremental.truncate(17).unwrap();
        for row in 17..rows {
            incremental.append_q8_row(&codes[row * head_dim..(row + 1) * head_dim], &scales[row..row + 1], head_dim).unwrap();
        }
        assert_eq!(incremental.values, rebuilt.values);
    }

    #[test]
    fn q8_candidate_rerank_returns_global_stable_tokens() {
        let heads = 32;
        let head_dim = 128;
        let rows = 32;
        let mut codes = vec![0_i8; rows * head_dim];
        for row in 0..rows {
            codes[row * head_dim] = (row / 2) as i8;
        }
        let scales = vec![HalfBf16::from_f32(1.0).to_bits(); rows];
        let keys = Q8KeyBlocks::from_q8_row_major(&codes, &scales, head_dim).unwrap();
        let mut query = vec![0.0_f32; heads * head_dim];
        query[0] = 1.0;
        let mut weights = vec![0.0_f32; heads];
        weights[0] = 1.0;
        let candidates = vec![2, 3, 8, 9, 20, 21, 30, 31];
        let mut workspace = Q8DsaWorkspace::default();
        let selected = rerank_q8_blocked_candidates(&keys, &query, &weights, heads, head_dim, &candidates, 4, &mut workspace).unwrap();
        assert_eq!(selected, [30, 31, 20, 21]);
        score_q8_blocked_workspace(&keys, &query, &weights, heads, head_dim, &mut workspace).unwrap();
        assert_eq!(finish_q8_blocked_topk(4, &mut workspace).unwrap(), [30, 31, 28, 29]);
        select_q8_blocked_candidates(&keys, &query, &weights, heads, head_dim, 8, &mut workspace).unwrap();
        assert_eq!(finish_q8_blocked_topk(4, &mut workspace).unwrap(), [30, 31, 28, 29]);
    }

    #[test]
    fn q8_blocked_build_append_and_truncate_match_row_major() {
        let head_dim = 128;
        let group_size = 64;
        let rows = 19;
        let codes = (0..rows * head_dim).map(|index| (index as i32 % 127 - 63) as i8).collect::<Vec<_>>();
        let scales = (0..rows * 2).map(|index| HalfBf16::from_f32(0.003 + index as f32 * 0.00001).to_bits()).collect::<Vec<_>>();
        let decoded = (0..rows - 1)
            .flat_map(|row| {
                let codes = &codes;
                let scales = &scales;
                (0..head_dim).map(move |column| HalfBf16::from_f32(codes[row * head_dim + column] as f32 * HalfBf16::from_bits(scales[row * 2 + column / group_size]).to_f32()))
            })
            .collect::<Vec<_>>();
        let expected = Bf16KeyBlocks::from_row_major(&decoded, head_dim).unwrap();
        let mut actual = Bf16KeyBlocks::from_q8_row_major(&codes[..(rows - 1) * head_dim], &scales[..(rows - 1) * 2], head_dim, group_size).unwrap();
        assert_eq!(actual.values, expected.values);
        actual.append_q8_row(&codes[(rows - 1) * head_dim..], &scales[(rows - 1) * 2..], group_size).unwrap();
        assert_eq!(actual.rows(), rows);
        actual.truncate(rows - 2).unwrap();
        actual.append_q8_row(&codes[(rows - 2) * head_dim..(rows - 1) * head_dim], &scales[(rows - 2) * 2..(rows - 1) * 2], group_size).unwrap();
        assert_eq!(actual.rows(), rows - 1);
    }

    #[test]
    fn q8_vnni_scores_match_quantized_scalar() {
        let heads = 32;
        let head_dim = 128;
        let rows = 35;
        let codes = (0..rows * head_dim).map(|index| ((index * 29 + 17) % 255) as i16 - 127).map(|value| value as i8).collect::<Vec<_>>();
        let scale_bits = (0..rows).map(|index| HalfBf16::from_f32(0.001 + index as f32 * 0.00001).to_bits()).collect::<Vec<_>>();
        let keys = Q8KeyBlocks::from_q8_row_major(&codes, &scale_bits, head_dim).unwrap();
        let query = (0..heads * head_dim).map(|index| ((index as f32 * 0.017).sin() * 2.7).clamp(-2.5, 2.5)).collect::<Vec<_>>();
        let weights = (0..heads).map(|head| 0.01 + head as f32 * 0.0003).collect::<Vec<_>>();
        let mut workspace = Q8DsaWorkspace::default();
        select_q8_blocked_workspace_profile(&keys, &query, &weights, heads, head_dim, 8, &mut workspace).unwrap();
        let actual = workspace.scores.clone();
        for row in 0..rows {
            let mut tiles = [0.0_f32; 2];
            for tile in 0..2 {
                let mut even = 0.0_f32;
                let mut odd = 0.0_f32;
                for local in (0..16).step_by(2) {
                    for (head, output) in [(tile * 16 + local, &mut even), (tile * 16 + local + 1, &mut odd)] {
                        let dot = (0..head_dim).map(|column| i32::from(codes[row * head_dim + column]) * i32::from(workspace.query_codes[head * head_dim + column])).sum::<i32>();
                        let scale = HalfBf16::from_bits(scale_bits[row]).to_f32() * workspace.query_scales[head];
                        let value = (dot as f32 * scale).max(0.0);
                        *output = value.mul_add(weights[head], *output);
                    }
                }
                tiles[tile] = even + odd;
            }
            let expected = ordered_score(tiles[1] + tiles[0]);
            assert!(actual[row].abs_diff(expected) <= 2, "row={row}: {} != {expected}", actual[row]);
        }
        assert_eq!(keys.code(17, 91), codes[17 * head_dim + 91]);
    }
}
