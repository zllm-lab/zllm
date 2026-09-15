//! DeepSeek-V4.1 engram n-gram 记忆的 CPU 算子。
//!
//! 用户决策:engram 的算法与数据(每层 384M 行 embed 表)都驻内存,不进显存。
//! 语义逐行对照官方 inference/{engram.py, model.py};静态表(multipliers/primes/offsets)
//! 由官方代码生成(tools 脚本 gen_engram_static.py),token_map 运行时从文件读。

use std::{
    path::Path,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use crate::weight::model::deepseek_v4::{DeepSeekV4EngramEmbedding, DeepSeekV4EngramWeights, DeepSeekV4Weights};

pub(crate) const ENGRAM_LAYER_IDS: [usize; 2] = [1, 14];
pub(crate) const ENGRAM_HASH_COLS: usize = 24;
pub(crate) const ENGRAM_HEAD_DIM: usize = 256;
const MAX_NGRAM: usize = 4;
const DEAD: i64 = -1;

/// 每层 4 个 hash 乘子(官方 compute_hash_multipliers,seed=10007*layer_id,PCG64 已离线展开)。
const MULTIPLIERS: [[i64; MAX_NGRAM]; 2] = [[76632096046245, 4839876093313, 35959672319349, 73987337458391], [67716810739261, 51510806800915, 30921347202721, 82619226485591]];

/// 每层 24 个桶素数(2/3/4-gram × 8 头,官方 find_next_prime 自 16M-1 顺序取)。
const PRIMES: [[i64; ENGRAM_HASH_COLS]; 2] = [
    [
        16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183, 16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321, 16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447,
        16000463,
    ],
    [
        16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627, 16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769, 16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879,
        16000889,
    ],
];

/// 每层 24 个桶起点(primes 前缀和)。
const OFFSETS: [[i64; ENGRAM_HASH_COLS]; 2] = [
    [
        0, 16000057, 32000136, 48000217, 64000314, 80000435, 96000564, 112000697, 128000880, 144001069, 160001276, 176001487, 192001740, 208002017, 224002306, 240002613, 256002934, 272003273, 288003654, 304004047, 320004446, 336004849,
        352005258, 368005705,
    ],
    [
        0, 16000477, 32000964, 48001463, 64001970, 80002481, 96003054, 112003663, 128004290, 144004957, 160005626, 176006319, 192007016, 208007727, 224008456, 240009215, 256009984, 272010765, 288011564, 304012377, 320013196, 336014037,
        352014914, 368015793,
    ],
];

/// token id → 压缩 id 映射(129280 × u32,官方 build_compressed_token_map 离线生成)。
#[derive(Clone)]
pub struct EngramTokenMap {
    pub(crate) map: Vec<u32>,
    pub(crate) pad_id: i64,
}

impl EngramTokenMap {
    /// 读取 engram_token_map.bin(129280 个 u32 LE);pad_id 为 engram_pad_token_id=2 的压缩 id。
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref()).map_err(|error| format!("读取 engram token map {:?}: {error}", path.as_ref()))?;
        if bytes.len() % 4 != 0 || bytes.is_empty() {
            return Err(format!("engram token map 字节数 {} 不是 u32 的倍数", bytes.len()));
        }
        let map = bytes.chunks_exact(4).map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 块"))).collect::<Vec<_>>();
        let pad_id = *map.get(2).ok_or("engram token map 缺少 pad token 2 的映射")? as i64;
        Ok(Self { map, pad_id })
    }

    fn compressed(&self, token: u32) -> i64 {
        self.map.get(token as usize).copied().unwrap_or(0) as i64
    }
}

/// n-gram hash 状态:cache 全序列压缩 id,prefill 批量 / decode 增量。
/// 语义对照官方 NgramHashState(文本路径,无 image mask;VL 接入时补 DEAD span)。
pub struct EngramHashState {
    map: EngramTokenMap,
    cache: Vec<i64>,
    transaction_start: Option<usize>,
}

impl EngramHashState {
    pub fn new(map: EngramTokenMap) -> Self {
        Self { map, cache: Vec::new(), transaction_start: None }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    fn begin_transaction(&mut self) -> Result<(), String> {
        if self.transaction_start.replace(self.cache.len()).is_some() {
            return Err("engram speculative transaction 已存在".to_owned());
        }
        Ok(())
    }

    fn commit_transaction(&mut self, retained_rows: usize) -> Result<(), String> {
        let start = self.transaction_start.take().ok_or_else(|| "engram speculative transaction 不存在".to_owned())?;
        let rows = self.cache.len().checked_sub(start).ok_or_else(|| "engram speculative transaction 起点超过当前长度".to_owned())?;
        if retained_rows > rows {
            return Err(format!("engram speculative retained_rows={retained_rows} 超过 transaction rows={rows}"));
        }
        self.cache.truncate(start + retained_rows);
        Ok(())
    }

    /// 追加一个 token 并返回其位置的 24 个表行号(decode 单步;两个 engram 层各自调用)。
    pub fn push(&mut self, token: u32, layer_slot: usize) -> [i64; ENGRAM_HASH_COLS] {
        assert!(layer_slot < 2, "engram layer_slot={layer_slot} 越界");
        self.cache.push(self.map.compressed(token));
        self.hash_at(self.cache.len() - 1, layer_slot)
    }

    /// 追加一个 DEAD 位(VL image span):自身与跨过它的 n-gram 全部阻断。
    pub fn push_dead(&mut self) {
        self.cache.push(DEAD);
    }

    /// prefill:从空状态算整段;返回每位置 24 行号。
    pub fn prefill(&mut self, tokens: &[u32], layer_slot: usize) -> Vec<[i64; ENGRAM_HASH_COLS]> {
        assert!(layer_slot < 2, "engram layer_slot={layer_slot} 越界");
        let start = self.cache.len();
        self.cache.extend(tokens.iter().map(|&token| self.map.compressed(token)));
        (start..start + tokens.len()).map(|pos| self.hash_at(pos, layer_slot)).collect()
    }

    pub fn hash_at(&self, pos: usize, layer_slot: usize) -> [i64; ENGRAM_HASH_COLS] {
        let mut tokens = [0_i64; MAX_NGRAM];
        let mut blocked = false;
        for (shift, slot) in tokens.iter_mut().enumerate() {
            let source = if pos >= shift { self.cache[pos - shift] } else { DEAD };
            blocked |= pos < shift || source == DEAD;
            *slot = if blocked { self.map.pad_id } else { source };
        }
        let multipliers = &MULTIPLIERS[layer_slot];
        let mut rolling = tokens[0] * multipliers[0];
        let mut out = [0_i64; ENGRAM_HASH_COLS];
        let primes = &PRIMES[layer_slot];
        let offsets = &OFFSETS[layer_slot];
        for step in 1..MAX_NGRAM {
            rolling ^= tokens[step] * multipliers[step];
            let base = (step - 1) * 8;
            for head in 0..8 {
                out[base + head] = rolling % primes[base + head] + offsets[base + head];
            }
        }
        out
    }
}

/// 单层 engram 的 CPU 执行体:wkv 转 BF16 驻内存(25600×6144 ≈ 314MB/层,带宽减半),
/// prefill 走 AVX512-BF16 批量矩阵乘；qk_weight 为 q_weight×k_weight 预乘([hc, dim])。
/// wkv/qk_weight 经 Arc 共享,会话 fork 零拷贝。
#[derive(Clone)]
pub struct EngramLayerCpu {
    /// `[output/16, input/2, 16]`；每个 u32 打包同一输出行的两个连续 BF16。
    /// 该布局让 AVX512-BF16 一次读取 16 个输出，并在一个输入 tile 内复用权重。
    wkv: Arc<Vec<u32>>,
    qk_weight: Arc<Vec<f32>>,
    team: Arc<EngramTeam>,
    workspace: Arc<Mutex<Vec<f32>>>,
    kv_cols: usize,
    kv_rows: usize,
    hc: usize,
    dim: usize,
    eps: f32,
}

const OUTPUT_TILE: usize = 16;

/// GPU stage 边界已经把 hidden 压成 BF16。直接下载原始 bits 后在 CPU 扩展，
/// 避免 GPU 先生成同规模 F32 临时量并把 PCIe 流量翻倍。
pub(crate) fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    let mut output = Vec::<f32>::with_capacity(bits.len());
    unsafe { output.set_len(bits.len()) };
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") {
        unsafe { bf16_bits_to_f32_avx512(bits, &mut output) };
        return output;
    }
    for (output, &bits) in output.iter_mut().zip(bits) {
        *output = f32::from_bits(u32::from(bits) << 16);
    }
    output
}

fn bf16_bits_to_f32_parallel(team: &EngramTeam, bits: &[u16]) -> Vec<f32> {
    let mut output = Vec::<f32>::with_capacity(bits.len());
    unsafe { output.set_len(bits.len()) };
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") {
        let vectors = bits.len() / 16;
        let workers = team.workers.min(vectors.div_ceil(4096).max(1));
        let input = bits.as_ptr() as usize;
        let target = output.as_mut_ptr() as usize;
        team.run(workers, |worker| unsafe {
            let first = vectors * worker / workers;
            let last = vectors * (worker + 1) / workers;
            bf16_bits_to_f32_avx512_range(input, target, first, last);
        });
        for index in vectors * 16..bits.len() {
            output[index] = f32::from_bits(u32::from(bits[index]) << 16);
        }
        return output;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = team;
    for (output, &bits) in output.iter_mut().zip(bits) {
        *output = f32::from_bits(u32::from(bits) << 16);
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn bf16_bits_to_f32_avx512(bits: &[u16], output: &mut [f32]) {
    use std::arch::x86_64::*;

    unsafe {
        let vectors = bits.len() / 16;
        for vector in 0..vectors {
            let packed = _mm256_loadu_si256(bits.as_ptr().add(vector * 16).cast());
            let expanded = _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16);
            _mm512_storeu_ps(output.as_mut_ptr().add(vector * 16), _mm512_castsi512_ps(expanded));
        }
        for index in vectors * 16..bits.len() {
            *output.get_unchecked_mut(index) = f32::from_bits(u32::from(*bits.get_unchecked(index)) << 16);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn bf16_bits_to_f32_avx512_range(input: usize, output: usize, first: usize, last: usize) {
    use std::arch::x86_64::*;

    unsafe {
        for vector in first..last {
            let packed = _mm256_loadu_si256((input as *const u16).add(vector * 16).cast());
            let expanded = _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16);
            _mm512_storeu_ps((output as *mut f32).add(vector * 16), _mm512_castsi512_ps(expanded));
        }
    }
}

type EngramTeamCallback = unsafe fn(usize, usize);

struct EngramTeamShared {
    complete_lock: Mutex<()>,
    complete: Condvar,
    pending: AtomicUsize,
    published: Box<[AtomicU64]>,
    callback: AtomicUsize,
    context: AtomicUsize,
}

/// 两层各自占一个 NUMA package 的固定 worker team。Engram 的矩阵乘单次会连续
/// 扫描约 314MB 权重，固定分片同时消除 Rayon 调度和跨 socket 权重读取。
struct EngramTeam {
    workers: usize,
    threads: Vec<std::thread::Thread>,
    submission: Mutex<()>,
    shared: Arc<EngramTeamShared>,
}

impl EngramTeam {
    fn new(cpu_ids: &[usize]) -> Self {
        unsafe fn idle(_: usize, _: usize) {}
        let workers = cpu_ids.len().max(1);
        let shared = Arc::new(EngramTeamShared {
            complete_lock: Mutex::new(()),
            complete: Condvar::new(),
            pending: AtomicUsize::new(0),
            published: (0..workers).map(|_| AtomicU64::new(0)).collect(),
            callback: AtomicUsize::new(idle as *const () as usize),
            context: AtomicUsize::new(0),
        });
        let mut threads = Vec::with_capacity(workers);
        for worker in 0..workers {
            let shared = Arc::clone(&shared);
            let cpu_id = cpu_ids.get(worker).copied();
            let handle = std::thread::Builder::new()
                .name(format!("zllm-engram-{cpu_id:?}"))
                .spawn(move || {
                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                    if let Some(cpu_id) = cpu_id {
                        crate::kernel::cpu::set_current_thread_affinity(&cpu_id.to_string()).expect("CPU Engram worker affinity 设置失败");
                    }
                    let mut seen = 0_u64;
                    loop {
                        while shared.published[worker].load(Ordering::Acquire) == seen {
                            std::thread::park();
                        }
                        seen = shared.published[worker].load(Ordering::Acquire);
                        let callback = unsafe { std::mem::transmute::<usize, EngramTeamCallback>(shared.callback.load(Ordering::Relaxed)) };
                        let context = shared.context.load(Ordering::Relaxed);
                        unsafe { callback(context, worker) };
                        if shared.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
                            let _guard = shared.complete_lock.lock().expect("CPU Engram completion mutex 已损坏");
                            shared.complete.notify_one();
                        }
                    }
                })
                .expect("CPU Engram 固定 worker 创建失败");
            threads.push(handle.thread().clone());
        }
        Self { workers, threads, submission: Mutex::new(()), shared }
    }

    fn run<F: Fn(usize) + Sync>(&self, active_workers: usize, job: F) {
        unsafe fn invoke<F: Fn(usize) + Sync>(context: usize, worker: usize) {
            unsafe { (&*(context as *const F))(worker) };
        }

        let active_workers = active_workers.clamp(1, self.workers);
        let _submission = self.submission.lock().expect("CPU Engram submission mutex 已损坏");
        assert_eq!(self.shared.pending.load(Ordering::Acquire), 0);
        self.shared.callback.store(invoke::<F> as *const () as usize, Ordering::Relaxed);
        self.shared.context.store((&job as *const F) as usize, Ordering::Relaxed);
        self.shared.pending.store(active_workers, Ordering::Release);
        for worker in 0..active_workers {
            self.shared.published[worker].fetch_add(1, Ordering::Release);
            self.threads[worker].unpark();
        }
        let mut guard = self.shared.complete_lock.lock().expect("CPU Engram completion mutex 已损坏");
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            guard = self.shared.complete.wait(guard).expect("CPU Engram completion wait 失败");
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn engram_cpu_sets() -> Vec<Vec<usize>> {
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
    let mut sets = packages
        .into_values()
        .map(|mut cores| {
            cores.sort_unstable();
            if cores.len() >= 16 {
                // AVX512 GEMM 在单 socket 内存带宽饱和后继续加核收益很小；保留
                // 四分之一物理核给 8 条 GPU stage 提交线程和 H2D/网络线程，
                // 避免 Engram 运行时把流水线的主机提交挤出 CPU。
                cores.truncate(cores.len() * 3 / 4);
            } else if cores.len() > 4 {
                cores.pop();
            }
            cores.into_iter().map(|(_, cpu)| cpu).collect::<Vec<_>>()
        })
        .filter(|cpus| !cpus.is_empty())
        .collect::<Vec<_>>();
    if sets.len() == 1 && sets[0].len() >= 8 {
        let all = sets.pop().expect("单 NUMA CPU set 存在");
        let middle = all.len().div_ceil(2);
        sets.push(all[..middle].to_vec());
        sets.push(all[middle..].to_vec());
    }
    if sets.is_empty() {
        sets.push(if allowed.is_empty() { vec![0] } else { allowed });
    }
    sets
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn engram_cpu_sets() -> Vec<Vec<usize>> {
    vec![(0..std::thread::available_parallelism().map_or(1, usize::from).min(16)).collect()]
}

fn pack_wkv_bf16(team: &EngramTeam, wkv: &[u16], rows: usize, cols: usize) -> Vec<u32> {
    assert!(cols.is_multiple_of(2));
    let pairs = cols / 2;
    let blocks = rows.div_ceil(OUTPUT_TILE);
    let len = blocks * pairs * OUTPUT_TILE;
    let mut packed = Vec::<u32>::with_capacity(len);
    unsafe { packed.set_len(len) };
    let source = wkv.as_ptr() as usize;
    let target = packed.as_mut_ptr() as usize;
    team.run(team.workers, |worker| {
        let first = blocks * worker / team.workers;
        let last = blocks * (worker + 1) / team.workers;
        for block in first..last {
            for pair in 0..pairs {
                for lane in 0..OUTPUT_TILE {
                    let row = block * OUTPUT_TILE + lane;
                    let value = if row < rows {
                        let index = row * cols + pair * 2;
                        unsafe { u32::from(*(source as *const u16).add(index)) | (u32::from(*(source as *const u16).add(index + 1)) << 16) }
                    } else {
                        0
                    };
                    unsafe { *(target as *mut u32).add((block * pairs + pair) * OUTPUT_TILE + lane) = value };
                }
            }
        }
    });
    packed
}

fn pack_input_bf16(team: &EngramTeam, input: &[f32], rows: usize, cols: usize) -> Vec<u32> {
    assert_eq!(input.len(), rows * cols);
    assert!(cols.is_multiple_of(2));
    let pairs = cols / 2;
    let mut packed = Vec::<u32>::with_capacity(rows * pairs);
    unsafe { packed.set_len(rows * pairs) };
    let source = input.as_ptr() as usize;
    let target = packed.as_mut_ptr() as usize;
    let workers = team.workers.min(rows.div_ceil(8).max(1));
    team.run(workers, |worker| {
        let first = rows * worker / workers;
        let last = rows * (worker + 1) / workers;
        for row in first..last {
            for pair in 0..pairs {
                let index = row * cols + pair * 2;
                let low = half::bf16::from_f32(unsafe { *(source as *const f32).add(index) }).to_bits();
                let high = half::bf16::from_f32(unsafe { *(source as *const f32).add(index + 1) }).to_bits();
                // GEMM 内层按 input pair 前进；把同一 pair 的 token 行放连续，
                // 使 16 行广播只读一条 cache line，并消除每行的大跨度寻址。
                unsafe { *(target as *mut u32).add(pair * rows + row) = u32::from(low) | (u32::from(high) << 16) };
            }
        }
    });
    packed
}

fn decode_mxfp8_bf16_packed(team: &EngramTeam, matrix: &crate::weight::format::mxfp8::Mxfp8Matrix) -> Vec<u32> {
    assert_eq!(matrix.rows % ENGRAM_HASH_COLS, 0);
    let values = std::array::from_fn::<_, 256, _>(|bits| crate::weight::codec::fp8::decode_f8_e4m3(bits as u8));
    let scales = std::array::from_fn::<_, 256, _>(|bits| crate::weight::codec::mxfp8::decode_e8m0(bits as u8));
    let rows = matrix.rows / ENGRAM_HASH_COLS;
    let head_pairs = matrix.cols / 2;
    let pairs = ENGRAM_HASH_COLS * head_pairs;
    let scale_cols = matrix.cols / crate::weight::codec::mxfp8::MXFP8_BLOCK;
    let mut packed = Vec::<u32>::with_capacity(rows * pairs);
    unsafe { packed.set_len(rows * pairs) };
    let codes = matrix.codes().as_ptr() as usize;
    let scale_inv = matrix.scale_inv().as_ptr() as usize;
    let output = packed.as_mut_ptr() as usize;
    let workers = team.workers.min(rows.div_ceil(8).max(1));
    team.run(workers, |worker| {
        let first = rows * worker / workers;
        let last = rows * (worker + 1) / workers;
        for row in first..last {
            for head in 0..ENGRAM_HASH_COLS {
                let source_row = row * ENGRAM_HASH_COLS + head;
                for pair in 0..head_pairs {
                    let column = pair * 2;
                    let scale = scales[unsafe { *(scale_inv as *const u8).add(source_row * scale_cols + column / crate::weight::codec::mxfp8::MXFP8_BLOCK) } as usize];
                    let low = values[unsafe { *(codes as *const u8).add(source_row * matrix.cols + column) } as usize] * scale;
                    let high = values[unsafe { *(codes as *const u8).add(source_row * matrix.cols + column + 1) } as usize] * scale;
                    let target_pair = head * head_pairs + pair;
                    unsafe { *(output as *mut u32).add(target_pair * rows + row) = u32::from(half::bf16::from_f32(low).to_bits()) | (u32::from(half::bf16::from_f32(high).to_bits()) << 16) };
                }
            }
        }
    });
    packed
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn gemm_bf16_tile<const INPUT_ROWS: usize>(
    wkv_addr: usize,
    input_addr: usize,
    output_addr: usize,
    row_start: usize,
    input_rows: usize,
    rows: usize,
    cols: usize,
    first_block: usize,
    last_block: usize,
) {
    use std::arch::x86_64::*;

    unsafe {
        let pairs = cols / 2;
        for block in first_block..last_block {
            let mut sums = [_mm512_setzero_ps(); INPUT_ROWS];
            for pair in 0..pairs {
                let weights = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512((wkv_addr as *const u32).add((block * pairs + pair) * OUTPUT_TILE).cast()));
                for row in 0..INPUT_ROWS {
                    let values = std::mem::transmute::<__m512i, __m512bh>(_mm512_set1_epi32(*(input_addr as *const u32).add(pair * input_rows + row_start + row) as i32));
                    sums[row] = _mm512_dpbf16_ps(sums[row], weights, values);
                }
            }
            let output_start = block * OUTPUT_TILE;
            let output_count = OUTPUT_TILE.min(rows - output_start);
            for row in 0..INPUT_ROWS {
                let target = (output_addr as *mut f32).add((row_start + row) * rows + output_start);
                if output_count == OUTPUT_TILE {
                    _mm512_storeu_ps(target, sums[row]);
                } else {
                    let mut tail = [0.0_f32; OUTPUT_TILE];
                    _mm512_storeu_ps(tail.as_mut_ptr(), sums[row]);
                    std::ptr::copy_nonoverlapping(tail.as_ptr(), target, output_count);
                }
            }
        }
    }
}

fn gemm_bf16_packed(team: &EngramTeam, wkv: &[u32], input: &[u32], output: &mut [f32], input_rows: usize, output_rows: usize, cols: usize) {
    assert_eq!(output.len(), input_rows * output_rows);
    assert_eq!(input.len(), input_rows * cols / 2);
    #[cfg(target_arch = "x86_64")]
    if cols.is_multiple_of(2) && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
        let output_blocks = output_rows.div_ceil(OUTPUT_TILE);
        let workers = team.workers.min(output_blocks.div_ceil(16).max(1));
        let wkv_addr = wkv.as_ptr() as usize;
        let input_addr = input.as_ptr() as usize;
        let output_addr = output.as_mut_ptr() as usize;
        team.run(workers, |worker| unsafe {
            let first_block = output_blocks * worker / workers;
            let last_block = output_blocks * (worker + 1) / workers;
            let mut row = 0;
            while row + 30 <= input_rows {
                gemm_bf16_tile::<30>(wkv_addr, input_addr, output_addr, row, input_rows, output_rows, cols, first_block, last_block);
                row += 30;
            }
            while row + 16 <= input_rows {
                gemm_bf16_tile::<16>(wkv_addr, input_addr, output_addr, row, input_rows, output_rows, cols, first_block, last_block);
                row += 16;
            }
            while row + 8 <= input_rows {
                gemm_bf16_tile::<8>(wkv_addr, input_addr, output_addr, row, input_rows, output_rows, cols, first_block, last_block);
                row += 8;
            }
            while row + 4 <= input_rows {
                gemm_bf16_tile::<4>(wkv_addr, input_addr, output_addr, row, input_rows, output_rows, cols, first_block, last_block);
                row += 4;
            }
            while row + 2 <= input_rows {
                gemm_bf16_tile::<2>(wkv_addr, input_addr, output_addr, row, input_rows, output_rows, cols, first_block, last_block);
                row += 2;
            }
            if row < input_rows {
                gemm_bf16_tile::<1>(wkv_addr, input_addr, output_addr, row, input_rows, output_rows, cols, first_block, last_block);
            }
        });
        return;
    }

    let target = output.as_mut_ptr() as usize;
    let workers = team.workers.min(input_rows.max(1));
    team.run(workers, |worker| {
        let first = input_rows * worker / workers;
        let last = input_rows * (worker + 1) / workers;
        let pairs = cols / 2;
        for input_row in first..last {
            for output_row in 0..output_rows {
                let block = output_row / OUTPUT_TILE;
                let lane = output_row % OUTPUT_TILE;
                let mut sum = 0.0_f32;
                for pair in 0..pairs {
                    let weight = wkv[(block * pairs + pair) * OUTPUT_TILE + lane];
                    let values = input[pair * input_rows + input_row];
                    sum += half::bf16::from_bits(weight as u16).to_f32() * half::bf16::from_bits(values as u16).to_f32();
                    sum += half::bf16::from_bits((weight >> 16) as u16).to_f32() * half::bf16::from_bits((values >> 16) as u16).to_f32();
                }
                unsafe { *(target as *mut f32).add(input_row * output_rows + output_row) = sum };
            }
        }
    });
}

#[cfg(test)]
fn gemm_bf16(team: &EngramTeam, wkv: &[u32], input: &[f32], output: &mut [f32], input_rows: usize, output_rows: usize, cols: usize) {
    let input = pack_input_bf16(team, input, input_rows, cols);
    gemm_bf16_packed(team, wkv, &input, output, input_rows, output_rows, cols);
}

impl EngramLayerCpu {
    pub fn prepare(weights: &DeepSeekV4EngramWeights, hc: usize, dim: usize, eps: f32) -> Result<Self, String> {
        let cpu_sets = engram_cpu_sets();
        Self::prepare_with_team(weights, Arc::new(EngramTeam::new(&cpu_sets[0])), hc, dim, eps)
    }

    fn prepare_with_team(weights: &DeepSeekV4EngramWeights, team: Arc<EngramTeam>, hc: usize, dim: usize, eps: f32) -> Result<Self, String> {
        let q = tensor_f32(&weights.q, "engram.q_weight")?;
        let k = tensor_f32(&weights.k, "engram.k_weight")?;
        if q.len() != hc * dim || k.len() != hc * dim {
            return Err(format!("engram q/k_weight 元素数 {}/{} 与 hc={hc} dim={dim} 不匹配", q.len(), k.len()));
        }
        let qk_weight = q.iter().zip(k.iter()).map(|(q, k)| q * k).collect::<Vec<_>>();
        let (kv_rows, kv_cols) = (weights.wkv.rows(), weights.wkv.cols());
        if kv_rows != dim * (hc + 1) || kv_cols != ENGRAM_HASH_COLS * ENGRAM_HEAD_DIM {
            return Err(format!("engram wkv [{kv_rows},{kv_cols}] 期望 [{},{}]", dim * (hc + 1), ENGRAM_HASH_COLS * ENGRAM_HEAD_DIM));
        }
        let wkv_f32 = core_matrix_f32(&weights.wkv)?;
        let wkv = wkv_f32.into_iter().map(|v| half::bf16::from_f32(v).to_bits()).collect::<Vec<_>>();
        let wkv = Arc::new(pack_wkv_bf16(&team, &wkv, kv_rows, kv_cols));
        let qk_weight = Arc::new(qk_weight);
        Ok(Self { wkv, qk_weight, team, workspace: Arc::new(Mutex::new(Vec::new())), kv_cols, kv_rows, hc, dim, eps })
    }

    /// 单 token 应用:embed_rows 为 24 行解量后 F32(24×256 行主序),h 为 [hc, dim] 就地修正。
    pub fn apply(&self, embed_rows: &[f32], h: &mut [f32]) -> Result<(), String> {
        self.apply_batch(embed_rows, 1, h)
    }

    fn apply_batch(&self, embed_rows: &[f32], rows: usize, h: &mut [f32]) -> Result<(), String> {
        if embed_rows.len() != rows * self.kv_cols || h.len() != rows * self.hc * self.dim {
            return Err(format!("engram apply 维度: rows={} embed={} h={} 期望 {}/{}", rows, embed_rows.len(), h.len(), rows * self.kv_cols, rows * self.hc * self.dim));
        }
        let embed_rows = pack_input_bf16(&self.team, embed_rows, rows, self.kv_cols);
        self.apply_batch_packed(&embed_rows, rows, h)
    }

    fn apply_batch_packed(&self, embed_rows: &[u32], rows: usize, h: &mut [f32]) -> Result<(), String> {
        if embed_rows.len() != rows * self.kv_cols / 2 || h.len() != rows * self.hc * self.dim {
            return Err(format!("engram packed apply 维度: rows={} embed={} h={} 期望 {}/{}", rows, embed_rows.len(), h.len(), rows * self.kv_cols / 2, rows * self.hc * self.dim));
        }
        let kv_len = rows * self.kv_rows;
        let mut workspace = self.workspace.lock().map_err(|_| "CPU Engram workspace mutex 已损坏".to_owned())?;
        if workspace.capacity() < kv_len {
            let additional = kv_len - workspace.len();
            workspace.reserve_exact(additional);
        }
        unsafe { workspace.set_len(kv_len) };
        gemm_bf16_packed(&self.team, &self.wkv, embed_rows, &mut workspace, rows, self.kv_rows, self.kv_cols);
        self.apply_projected(&workspace, rows, h)
    }

    fn project_batch_packed(&self, embed_rows: &[u32], rows: usize) -> Result<Vec<f32>, String> {
        if embed_rows.len() != rows * self.kv_cols / 2 {
            return Err(format!("engram packed projection 维度: rows={} embed={} 期望 {}", rows, embed_rows.len(), rows * self.kv_cols / 2));
        }
        let mut projected = Vec::<f32>::with_capacity(rows * self.kv_rows);
        unsafe { projected.set_len(rows * self.kv_rows) };
        gemm_bf16_packed(&self.team, &self.wkv, embed_rows, &mut projected, rows, self.kv_rows, self.kv_cols);
        Ok(projected)
    }

    fn apply_projected(&self, kv: &[f32], rows: usize, h: &mut [f32]) -> Result<(), String> {
        if kv.len() != rows * self.kv_rows || h.len() != rows * self.hc * self.dim {
            return Err(format!("engram projected apply 维度: rows={} kv={} h={} 期望 {}/{}", rows, kv.len(), h.len(), rows * self.kv_rows, rows * self.hc * self.dim));
        }
        let stride = self.hc * self.dim;
        let h_addr = h.as_mut_ptr() as usize;
        let kv_addr = kv.as_ptr() as usize;
        let workers = self.team.workers.min(rows.div_ceil(8).max(1));
        self.team.run(workers, |worker| {
            let first = rows * worker / workers;
            let last = rows * (worker + 1) / workers;
            for row in first..last {
                let h = unsafe { std::slice::from_raw_parts_mut((h_addr as *mut f32).add(row * stride), stride) };
                let kv = unsafe { std::slice::from_raw_parts((kv_addr as *const f32).add(row * self.kv_rows), self.kv_rows) };
                let (key, value) = kv.split_at(stride);
                // gate:sigmoid(copysign(sqrt(max(|dot|,1e-6)), dot));rstd 每 (copy) 独立
                for copy in 0..self.hc {
                    let h_copy = &mut h[copy * self.dim..(copy + 1) * self.dim];
                    let key = &key[copy * self.dim..(copy + 1) * self.dim];
                    let weight = &self.qk_weight[copy * self.dim..(copy + 1) * self.dim];
                    let h_energy = h_copy.iter().map(|v| v * v).sum::<f32>() / self.dim as f32;
                    let k_energy = key.iter().map(|v| v * v).sum::<f32>() / self.dim as f32;
                    let rstd = (h_energy + self.eps).sqrt().recip() * (k_energy + self.eps).sqrt().recip();
                    let dot = h_copy.iter().zip(key.iter()).zip(weight.iter()).map(|((h, k), w)| h * w * k).sum::<f32>() * rstd / (self.dim as f32).sqrt();
                    let gate = 1.0 / (1.0 + (-dot.abs().max(1e-6).sqrt().copysign(dot)).exp());
                    for (h, &v) in h_copy.iter_mut().zip(value.iter()) {
                        *h += gate * v;
                    }
                }
            }
        });
        Ok(())
    }
}

fn tensor_f32(tensor: &crate::weight::container::safetensor::TensorData, name: &str) -> Result<Vec<f32>, String> {
    crate::weight::container::safetensor::decode_to_f32(name, &tensor.dtype, &tensor.data)
}

fn core_matrix_f32(matrix: &crate::weight::model::deepseek_v4::DeepSeekV4CoreMatrix) -> Result<Vec<f32>, String> {
    use crate::weight::model::deepseek_v4::DeepSeekV4CoreMatrix;
    match matrix {
        DeepSeekV4CoreMatrix::BlockFp8(matrix) => Ok(matrix.decode()),
        DeepSeekV4CoreMatrix::Mxfp8(matrix) => Ok(matrix.decode()),
        DeepSeekV4CoreMatrix::Dense(tensor) => tensor_f32(tensor, &tensor.name),
    }
}

/// engram 总控:token map + 两层 CPU 执行体 + hash 状态。
///
/// 调用约定:每个 token 经 `push_token` 入 hash 恰好一次(prefill 用 `push_prefill`);
/// 之后在层 1/14 的 hook 点各调 `apply_current`/`apply_prefill_rows`。
pub struct DeepSeekV4EngramCpu {
    hash: EngramHashState,
    layers: [EngramLayerCpu; 2],
    weights: DeepSeekV4Weights,
}

pub struct DeepSeekV4EngramBatch {
    layer: EngramLayerCpu,
    embed: Vec<u32>,
    projected: Option<Vec<f32>>,
    rows: usize,
}

impl DeepSeekV4EngramBatch {
    pub fn expand_hidden_bf16(&self, bits: &[u16]) -> Vec<f32> {
        bf16_bits_to_f32_parallel(&self.layer.team, bits)
    }

    /// WKV 与 hidden 无关，可在 GPU 到达 engram 层前由对应 NUMA team 预计算。
    pub fn precompute(mut self) -> Result<Self, String> {
        self.projected = Some(self.layer.project_batch_packed(&self.embed, self.rows)?);
        self.embed.clear();
        Ok(self)
    }

    pub fn apply(self, h: &mut [f32]) -> Result<(), String> {
        match self.projected {
            Some(projected) => self.layer.apply_projected(&projected, self.rows, h),
            None => self.layer.apply_batch_packed(&self.embed, self.rows, h),
        }
    }

    /// 把提前投影结果交给 GPU 门控；Engram 大表仍只驻留 CPU。
    pub fn into_projected(self) -> Result<(Vec<f32>, Arc<Vec<f32>>, usize, usize, f32), String> {
        let projected = self.projected.ok_or_else(|| "engram batch 尚未提前投影".to_owned())?;
        Ok((projected, self.layer.qk_weight.clone(), self.layer.hc, self.layer.dim, self.layer.eps))
    }
}

impl DeepSeekV4EngramCpu {
    /// `token_map_path`:engram_token_map.bin;`weights`:已打开的 checkpoint(engram 行按需读)。
    pub fn load(token_map_path: impl AsRef<Path>, weights: DeepSeekV4Weights, hc: usize, dim: usize, eps: f32) -> Result<Self, String> {
        let map = EngramTokenMap::load(token_map_path)?;
        let cpu_sets = engram_cpu_sets();
        eprintln!("[load] engram CPU teams: L1={:?} L14={:?}", cpu_sets[0], cpu_sets[1 % cpu_sets.len()]);
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        for (slot, &layer) in ENGRAM_LAYER_IDS.iter().enumerate() {
            let started = std::time::Instant::now();
            let cpu = cpu_sets[slot % cpu_sets.len()][0];
            let bytes = crate::kernel::cpu::with_current_thread_affinity(&cpu.to_string(), || weights.prefetch_engram_embedding(layer))??;
            eprintln!("[load] engram L{layer} CPU resident {:.2}GiB numa_cpu={cpu} wall={:.1}s", bytes as f64 / (1_u64 << 30) as f64, started.elapsed().as_secs_f64());
        }
        let mut layers = Vec::with_capacity(2);
        for (slot, &layer) in ENGRAM_LAYER_IDS.iter().enumerate() {
            let layer_weights = weights.load_engram(layer)?.ok_or_else(|| format!("DeepSeek-V4 layer {layer} 缺 engram 权重"))?;
            let team = Arc::new(EngramTeam::new(&cpu_sets[slot % cpu_sets.len()]));
            layers.push(EngramLayerCpu::prepare_with_team(&layer_weights, team, hc, dim, eps)?);
        }
        let [first, second] = layers.try_into().map_err(|_| "engram 层数必须为 2".to_owned())?;
        Ok(Self { hash: EngramHashState::new(map), layers: [first, second], weights })
    }

    /// 会话 fork:权重/wkv 共享(Arc),hash 状态重置(新会话从空序列开始)。
    pub fn fork(&self) -> Self {
        Self { hash: EngramHashState::new(self.hash.map.clone()), layers: self.layers.clone(), weights: self.weights.clone() }
    }

    /// 清空 hash 缓存(会话 reset;共享实例只清序列状态,不重建)。
    pub fn reset_hash(&mut self) {
        self.hash = EngramHashState::new(self.hash.map.clone());
    }

    /// hash 已摄入的 token 数(位置偏移用)。
    pub fn hash_len(&self) -> usize {
        self.hash.len()
    }

    /// decode/prefill:新 token 入 hash(每 token 一次,与层无关)。
    /// `image_span` 为 true 时推入 DEAD(image span 不参与 n-gram,官方 engram_mask)。
    pub fn push_token(&mut self, token: u32, image_span: bool) {
        if image_span {
            self.hash.push_dead();
        } else {
            self.hash.push(token, 0);
        }
    }

    pub fn begin_speculative(&mut self) -> Result<(), String> {
        self.hash.begin_transaction()
    }

    pub fn commit_speculative(&mut self, retained_rows: usize) -> Result<(), String> {
        self.hash.commit_transaction(retained_rows)
    }

    /// decode:在当前(最新)位置应用层 `layer_slot` 的 engram,就地修正 h [hc, dim]。
    pub fn apply_current(&self, layer_slot: usize, h: &mut [f32]) -> Result<(), String> {
        if self.hash.len() == 0 {
            return Err("engram apply_current 前没有 token".to_owned());
        }
        let rows = self.hash.hash_at(self.hash.len() - 1, layer_slot);
        let embed = self.embed_rows_packed(layer_slot, &rows)?;
        self.layers[layer_slot].apply_batch_packed(&embed, 1, h)
    }

    /// prefill:整段入 hash,返回起始位置(供 apply_prefill_rows 的 pos 偏移)。
    pub fn push_prefill(&mut self, tokens: &[u32]) -> usize {
        let start = self.hash.len();
        self.hash.prefill(tokens, 0);
        start
    }

    /// prefill:在锁内只算 hash、读取并解码稀疏 embed 行；大矩阵计算由返回值在锁外执行，
    /// 避免 L1/L14 两个 pipeline stage 因共享 hash 状态而整段串行。
    pub fn prepare_prefill_rows(&self, layer_slot: usize, pos_start: usize, rows: usize) -> Result<DeepSeekV4EngramBatch, String> {
        let hash_rows = (0..rows).flat_map(|row| self.hash.hash_at(pos_start + row, layer_slot)).collect::<Vec<_>>();
        let embed = self.embed_rows_packed(layer_slot, &hash_rows)?;
        if embed.len() != rows * self.layers[layer_slot].kv_cols / 2 {
            return Err(format!("engram prefill packed embed={} 期望 {}×{}/2", embed.len(), rows, self.layers[layer_slot].kv_cols));
        }
        Ok(DeepSeekV4EngramBatch { layer: self.layers[layer_slot].clone(), embed, projected: None, rows })
    }

    pub fn apply_prefill_rows(&self, layer_slot: usize, pos_start: usize, rows: usize, h: &mut [f32]) -> Result<(), String> {
        self.prepare_prefill_rows(layer_slot, pos_start, rows)?.apply(h)
    }

    fn embed_rows_packed(&self, layer_slot: usize, rows: &[i64]) -> Result<Vec<u32>, String> {
        let layer = ENGRAM_LAYER_IDS[layer_slot];
        let row_ids = rows.iter().map(|&row| usize::try_from(row).map_err(|_| format!("engram 行号 {row} 非法"))).collect::<Result<Vec<_>, _>>()?;
        let embedding = self.weights.engram_embedding_rows(layer, &row_ids)?;
        match embedding {
            DeepSeekV4EngramEmbedding::Mxfp8(matrix) => Ok(decode_mxfp8_bf16_packed(&self.layers[layer_slot].team, &matrix)),
            DeepSeekV4EngramEmbedding::MlxAffine(matrix) => {
                let values = matrix.decode()?;
                Ok(pack_input_bf16(&self.layers[layer_slot].team, &values, values.len() / self.layers[layer_slot].kv_cols, self.layers[layer_slot].kv_cols))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_hidden_download_expands_exact_bits() {
        let bits = [0x0000, 0x3f80, 0xc020, 0x7f7f, 0x0080, 0x8000, 0x4049, 0x3eaa, 0x4120, 0xbf00, 0x0101, 0x7e00, 0x3f00, 0x4000, 0xc000, 0x42c8, 0x3dcd];
        let output = bf16_bits_to_f32(&bits);
        assert_eq!(output.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), bits.iter().map(|&bits| u32::from(bits) << 16).collect::<Vec<_>>());
    }

    fn test_map() -> EngramTokenMap {
        // 简化映射:compressed = token % 1000;pad(2)= 2
        EngramTokenMap { map: (0..129_280u32).map(|token| token % 1000).collect(), pad_id: 2 }
    }

    #[test]
    fn hash_matches_official_formula() {
        // 手算:numpy 公式的 Rust 镜像;用已知 token 序列验证 XOR/取模/offset 布局
        let mut state = EngramHashState::new(test_map());
        let rows = state.prefill(&[5, 7, 11, 13], 0);
        assert_eq!(rows.len(), 4);
        // 位置 3:tokens=[13,11,7,5](shift 0..3)
        let tokens = [13_i64, 11, 7, 5];
        let multipliers = &MULTIPLIERS[0];
        let mut rolling = tokens[0] * multipliers[0];
        let mut expected = [0_i64; 24];
        for step in 1..4 {
            rolling ^= tokens[step] * multipliers[step];
            for head in 0..8 {
                expected[(step - 1) * 8 + head] = rolling % PRIMES[0][(step - 1) * 8 + head] + OFFSETS[0][(step - 1) * 8 + head];
            }
        }
        assert_eq!(rows[3], expected);
        // 位置 0:shift=0 是当前 token=5,shift>0 blocked → pad
        let rows0 = rows[0];
        let pad = 2_i64;
        let tokens0 = [5_i64, pad, pad, pad];
        let mut rolling0 = tokens0[0] * multipliers[0];
        let mut expected0 = [0_i64; 24];
        for step in 1..4 {
            rolling0 ^= tokens0[step] * multipliers[step];
            for head in 0..8 {
                expected0[(step - 1) * 8 + head] = rolling0 % PRIMES[0][(step - 1) * 8 + head] + OFFSETS[0][(step - 1) * 8 + head];
            }
        }
        assert_eq!(rows0, expected0);
        // 行号落在表内
        assert!(rows[3].iter().all(|&row| (0..384_006_168).contains(&row)));
    }

    #[test]
    fn speculative_hash_discards_rejected_suffix() {
        let mut state = EngramHashState::new(test_map());
        state.prefill(&[10, 11, 12], 0);
        state.begin_transaction().unwrap();
        state.prefill(&[13, 14, 15, 16], 0);
        state.commit_transaction(2).unwrap();
        assert_eq!(state.len(), 5);

        let mut expected = EngramHashState::new(test_map());
        expected.prefill(&[10, 11, 12, 13, 14], 0);
        assert_eq!(state.hash_at(4, 0), expected.hash_at(4, 0));
    }

    #[test]
    fn decode_matches_prefill() {
        let tokens = [3_u32, 1, 4, 1, 5, 9, 2, 6];
        let mut batch = EngramHashState::new(test_map());
        let prefilled = batch.prefill(&tokens, 1);
        let mut step = EngramHashState::new(test_map());
        for (pos, &token) in tokens.iter().enumerate() {
            assert_eq!(step.push(token, 1), prefilled[pos], "pos {pos}");
        }
    }

    #[test]
    fn batched_bf16_matches_scalar_reference() {
        let team = EngramTeam::new(&[0, 1, 2, 3]);
        let cols = 6144_usize;
        let rows = 96_usize;
        let w_f32 = (0..rows * cols).map(|i| ((i * 2654435761) % 2000) as f32 / 1000.0 - 1.0).map(half::bf16::from_f32).map(|v| v.to_f32()).collect::<Vec<_>>();
        let w_bf16 = w_f32.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect::<Vec<_>>();
        let w_packed = pack_wkv_bf16(&team, &w_bf16, rows, cols);
        let input_rows = 3;
        let x = (0..input_rows * cols).map(|i| ((i * 40503) % 100) as f32 / 50.0 - 1.0).map(half::bf16::from_f32).map(|v| v.to_f32()).collect::<Vec<_>>();
        let mut out = vec![0.0_f32; input_rows * rows];
        gemm_bf16(&team, &w_packed, &x, &mut out, input_rows, rows, cols);
        for (index, &got) in out.iter().enumerate() {
            let input_row = index / rows;
            let row = index % rows;
            let reference = w_f32[row * cols..(row + 1) * cols].iter().zip(&x[input_row * cols..(input_row + 1) * cols]).map(|(&w, &x)| w * x).sum::<f32>();
            let tolerance = reference.abs().max(1.0) * 1e-3;
            assert!((got - reference).abs() < tolerance, "input={input_row} row={row}: got={got} ref={reference}");
        }
    }

    #[test]
    fn precomputed_projection_matches_inline_apply() {
        let team = Arc::new(EngramTeam::new(&[0]));
        let (rows, hc, dim, cols) = (3, 1, 16, 32);
        let kv_rows = dim * (hc + 1);
        let weights = (0..kv_rows * cols).map(|index| half::bf16::from_f32((index % 17) as f32 / 32.0 - 0.25).to_bits()).collect::<Vec<_>>();
        let layer = EngramLayerCpu {
            wkv: Arc::new(pack_wkv_bf16(&team, &weights, kv_rows, cols)),
            qk_weight: Arc::new(vec![0.5; hc * dim]),
            team,
            workspace: Arc::new(Mutex::new(Vec::new())),
            kv_cols: cols,
            kv_rows,
            hc,
            dim,
            eps: 1e-6,
        };
        let embed_f32 = (0..rows * cols).map(|index| (index % 11) as f32 / 16.0 - 0.3).collect::<Vec<_>>();
        let embed = pack_input_bf16(&layer.team, &embed_f32, rows, cols);
        let batch = || DeepSeekV4EngramBatch { layer: layer.clone(), embed: embed.clone(), projected: None, rows };
        let mut inline = vec![0.25; rows * hc * dim];
        let mut precomputed = inline.clone();
        batch().apply(&mut inline).unwrap();
        batch().precompute().unwrap().apply(&mut precomputed).unwrap();
        assert_eq!(precomputed, inline);
    }

    #[test]
    fn packed_mxfp8_decode_matches_f32_path() {
        let logical_rows = 2;
        let rows = logical_rows * ENGRAM_HASH_COLS;
        let cols = 256;
        let codes = (0..rows * cols).map(|index| (index % 0x77) as u8).collect::<Vec<_>>();
        let scales = (0..rows * cols / 32).map(|index| 124 + index as u8 % 7).collect::<Vec<_>>();
        let matrix = crate::weight::format::mxfp8::Mxfp8Matrix::new(codes, scales, rows, cols).unwrap();
        let expected = matrix.decode().into_iter().map(half::bf16::from_f32).map(|value| value.to_bits()).collect::<Vec<_>>();
        let team = EngramTeam::new(&[0, 1]);
        let packed = decode_mxfp8_bf16_packed(&team, &matrix);
        let pairs = ENGRAM_HASH_COLS * cols / 2;
        let actual = (0..logical_rows)
            .flat_map(|row| {
                let packed = &packed;
                (0..pairs).flat_map(move |pair| {
                    let value = packed[pair * logical_rows + row];
                    [value as u16, (value >> 16) as u16]
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    /// 单 token apply 的速度探针(release 模式):`cargo test --release engram_apply_bench -- --ignored --nocapture`。
    #[test]
    #[ignore]
    fn engram_apply_bench() {
        let team = Arc::new(EngramTeam::new(&engram_cpu_sets()[0]));
        let dir = "/tmp/dsv41-engram-oracle";
        let read_f32 = |name: &str| -> Vec<f32> { std::fs::read(format!("{dir}/{name}.bin")).unwrap().chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect() };
        let embed_rows = read_f32("embed_rows");
        let wkv = read_f32("wkv");
        let mut h = read_f32("h");
        let q = read_f32("q_weight");
        let k = read_f32("k_weight");
        let (hc, dim, kv_cols) = (4_usize, 5120_usize, 6144_usize);
        let wkv = wkv.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect::<Vec<_>>();
        let cpu = EngramLayerCpu {
            wkv: std::sync::Arc::new(pack_wkv_bf16(&team, &wkv, dim * (hc + 1), kv_cols)),
            qk_weight: std::sync::Arc::new(q.iter().zip(k.iter()).map(|(q, k)| q * k).collect::<Vec<_>>()),
            team,
            workspace: Arc::new(Mutex::new(Vec::new())),
            kv_cols,
            kv_rows: dim * (hc + 1),
            hc,
            dim,
            eps: 1e-20,
        };
        // 预热 + 计时
        for _ in 0..3 {
            cpu.apply(&embed_rows, &mut h).unwrap();
        }
        let start = std::time::Instant::now();
        let iters = 20;
        for _ in 0..iters {
            cpu.apply(&embed_rows, &mut h).unwrap();
        }
        let elapsed = start.elapsed();
        eprintln!("engram apply: {:.2} ms/iter ({} iters, threads={})", elapsed.as_secs_f64() * 1000.0 / iters as f64, iters, rayon::current_num_threads());
    }
    #[test]
    #[ignore]
    fn engram_oracle_matches_official() {
        fn read_f32(path: &str) -> Vec<f32> {
            let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("读取 {path}: {error}(先生成 oracle 数据)"));
            bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect()
        }
        let dir = "/tmp/dsv41-engram-oracle";
        let embed_rows = read_f32(&format!("{dir}/embed_rows.bin"));
        let wkv = read_f32(&format!("{dir}/wkv.bin"));
        let mut h = read_f32(&format!("{dir}/h.bin"));
        let q = read_f32(&format!("{dir}/q_weight.bin"));
        let k = read_f32(&format!("{dir}/k_weight.bin"));
        let expected = read_f32(&format!("{dir}/expected_h.bin"));
        let (hc, dim, kv_cols) = (4_usize, 5120_usize, 6144_usize);
        let kv_rows = dim * (hc + 1);
        assert_eq!(embed_rows.len(), kv_cols);
        assert_eq!(wkv.len(), kv_rows * kv_cols);
        assert_eq!(h.len(), hc * dim);
        let wkv = wkv.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect::<Vec<_>>();
        let team = Arc::new(EngramTeam::new(&engram_cpu_sets()[0]));
        let cpu = EngramLayerCpu {
            wkv: std::sync::Arc::new(pack_wkv_bf16(&team, &wkv, kv_rows, kv_cols)),
            qk_weight: std::sync::Arc::new(q.iter().zip(k.iter()).map(|(q, k)| q * k).collect::<Vec<_>>()),
            team,
            workspace: Arc::new(Mutex::new(Vec::new())),
            kv_cols,
            kv_rows,
            hc,
            dim,
            eps: 1e-20,
        };
        cpu.apply(&embed_rows, &mut h).unwrap();
        let mut worst = 0.0_f32;
        for (i, (&got, &reference)) in h.iter().zip(expected.iter()).enumerate() {
            let scale = reference.abs().max(1.0);
            let relative = (got - reference).abs() / scale;
            worst = worst.max(relative);
            assert!(relative < 2e-2, "元素 {i}: got={got} ref={reference} rel={relative}");
        }
        eprintln!("engram oracle 最大相对偏差 {worst:.2e}");
    }
}
