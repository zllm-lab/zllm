//! 热路径固定 worker team(Linux):替代逐算子 `std::thread::scope` 的重复
//! 建线程。每个"驱动线程"(bench 主线程 / DSpark executor worker)持有自己
//! 的 team,由该线程串行发起的全部算子共享;team 内只建一层并行,算子闭包
//! 不得再嵌套 team_execute(运行时守卫会退回临时 scope,防死锁)。
//! 唤醒协议:generation counter + 自旋后 futex 私有等待;主线程作为最后
//! 一个 worker 参与计算,不空等。非 Linux 或嵌套调用一律退回 scope 语义。

use std::{
    cell::RefCell,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

#[cfg(target_os = "linux")]
unsafe fn futex_wait(address: &AtomicU32, expected: u32) {
    unsafe extern "C" {
        fn syscall(number: i64, ...) -> i64;
    }
    const SYS_FUTEX: i64 = 202;
    const FUTEX_WAIT_PRIVATE: i64 = 128;
    unsafe {
        let _ = syscall(SYS_FUTEX, address as *const AtomicU32, FUTEX_WAIT_PRIVATE, expected, 0_usize, 0_usize, 0_usize);
    }
}

#[cfg(target_os = "linux")]
unsafe fn futex_wake_all(address: &AtomicU32) {
    unsafe extern "C" {
        fn syscall(number: i64, ...) -> i64;
    }
    const SYS_FUTEX: i64 = 202;
    const FUTEX_WAKE_PRIVATE: i64 = 129;
    unsafe {
        let _ = syscall(SYS_FUTEX, address as *const AtomicU32, FUTEX_WAKE_PRIVATE, i32::MAX, 0_usize, 0_usize, 0_usize);
    }
}

#[cfg(target_os = "linux")]
const MAX_THREADS: usize = 16;
#[cfg(target_os = "linux")]
const SHUTDOWN: u32 = u32::MAX;

/// 64 字节对齐,避免 generation(主线程写)与 arrived(worker 写)共享行。
#[cfg(target_os = "linux")]
#[repr(align(64))]
struct Pad<T>(T);

#[cfg(target_os = "linux")]
struct TeamInner {
    generation: Pad<AtomicU32>,
    arrived: Pad<AtomicU32>,
    /// 本轮参与 worker 数(含主线程,主线程承担编号 participants-1)。
    participants: Pad<AtomicU32>,
    job: Mutex<Option<Arc<dyn Fn(usize) + Send + Sync>>>,
}

#[cfg(target_os = "linux")]
impl TeamInner {
    fn publish(&self, participants: u32, job: Arc<dyn Fn(usize) + Send + Sync>) {
        self.arrived.0.store(0, Ordering::Relaxed);
        *self.job.lock().unwrap() = Some(job);
        self.participants.0.store(participants, Ordering::Release);
        self.generation.0.fetch_add(1, Ordering::Release);
        unsafe { futex_wake_all(&self.generation.0) };
    }

    fn wait_arrived(&self, participants: u32) {
        let target = participants - 1;
        let mut spins = 0_u32;
        let mut observed = self.arrived.0.load(Ordering::Acquire);
        while observed != target {
            if spins < 4_000 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                // 到达计数由最后一名 worker 的 futex_wake 唤醒(见 team_worker)。
                unsafe { futex_wait(&self.arrived.0, observed) };
            }
            observed = self.arrived.0.load(Ordering::Acquire);
        }
    }
}

#[cfg(target_os = "linux")]
fn team_worker(inner: Arc<TeamInner>, worker: u32) {
    let mut seen = 0_u32;
    let mut spins = 0_u32;
    loop {
        let generation = inner.generation.0.load(Ordering::Acquire);
        if generation == SHUTDOWN {
            return;
        }
        if generation > seen {
            seen = generation;
            spins = 0;
            // 任务/参与数必须与触发的 generation 同轮快照:在持锁期间复核
            // generation 未变,防止把下一轮任务当本轮执行导致 arrive
            // 计数失衡死锁。
            let snapshot = {
                let slot = inner.job.lock().unwrap();
                let participants = inner.participants.0.load(Ordering::Acquire);
                if inner.generation.0.load(Ordering::Acquire) != generation {
                    continue;
                }
                slot.clone().map(|job| (participants, job))
            };
            // 编号 participants-1 由主线程承担;多余 worker 直接等待下一轮。
            if let Some((participants, job)) = snapshot.filter(|(participants, _)| worker + 1 < *participants) {
                job(worker as usize);
                let target = participants - 1;
                let previous = inner.arrived.0.fetch_add(1, Ordering::Release);
                // 最后一名到达者唤醒在 arrived 上等待的主线程;主线程自旋窗口
                // 内的唤醒丢失由其回环 load 兜底。
                if previous + 1 == target {
                    unsafe { futex_wake_all(&inner.arrived.0) };
                }
            }
            continue;
        }
        // 自旋一段时间再睡眠;唤醒值不匹配说明已进入新轮,回到循环头部。
        spins += 1;
        if spins >= 20_000 {
            spins = 0;
            unsafe { futex_wait(&inner.generation.0, seen) };
        }
        std::hint::spin_loop();
    }
}

#[cfg(target_os = "linux")]
struct Team {
    inner: Arc<TeamInner>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(target_os = "linux")]
impl Team {
    fn new() -> Self {
        let inner = Arc::new(TeamInner { generation: Pad(AtomicU32::new(0)), arrived: Pad(AtomicU32::new(0)), participants: Pad(AtomicU32::new(0)), job: Mutex::new(None) });
        // 预建 MAX_THREADS-1 个 worker,算子按需声明参与数,剩余 worker 空转等待。
        let handles = (0..MAX_THREADS as u32 - 1)
            .map(|worker| {
                let inner = Arc::clone(&inner);
                std::thread::Builder::new().name(format!("zllm-cpu-team-{worker}")).spawn(move || team_worker(inner, worker)).expect("CPU team worker 启动失败")
            })
            .collect();
        Self { inner, handles }
    }

    fn run(&self, threads: u32, job: Arc<dyn Fn(usize) + Send + Sync>) {
        if threads <= 1 {
            job(0);
            return;
        }
        self.inner.publish(threads, Arc::clone(&job));
        // 主线程承担最后一个编号,和 worker 同批执行。
        job(threads as usize - 1);
        self.inner.wait_arrived(threads);
    }
}

#[cfg(target_os = "linux")]
impl Drop for Team {
    fn drop(&mut self) {
        self.inner.generation.0.store(SHUTDOWN, Ordering::Release);
        unsafe { futex_wake_all(&self.inner.generation.0) };
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

thread_local! {
    /// 每个驱动线程一个 team;线程退出时 Drop 唤醒并回收 worker。
    #[cfg(target_os = "linux")]
    static TEAM: RefCell<Option<Team>> = const { RefCell::new(None) };
    /// 算子闭包内嵌套调用时退回临时 scope,防止单线程死锁。
    static IN_TEAM: AtomicBool = const { AtomicBool::new(false) };
}

struct ReentrancyGuard;

/// 跨线程分发的裸指针载体:分片互斥由调用方保证,以此声明 Send/Sync。
/// 与 attention 的 SendPtr 同理:闭包不得直接捕获裸指针(RFC 2229 会绕过
/// 包装按字段捕获,需经方法/整体使用)。
pub(crate) struct SendCell<T>(pub(crate) T);

unsafe impl<T> Send for SendCell<T> {}
unsafe impl<T> Sync for SendCell<T> {}

impl<T: Copy> SendCell<T> {
    pub(crate) fn get(&self) -> T {
        self.0
    }
}

impl ReentrancyGuard {
    fn enter() -> Self {
        IN_TEAM.with(|flag| flag.store(true, Ordering::Relaxed));
        Self
    }
}

impl Drop for ReentrancyGuard {
    fn drop(&mut self) {
        IN_TEAM.with(|flag| flag.store(false, Ordering::Relaxed));
    }
}

/// 以固定 team 执行 `job(worker)`,`worker` 取 [0, threads);
/// 主线程参与最后一个编号。非 Linux 或嵌套调用退回 `std::thread::scope`。
pub(crate) fn team_execute(threads: usize, job: impl Fn(usize) + Send + Sync) {
    if threads <= 1 {
        job(0);
        return;
    }
    if IN_TEAM.with(|flag| flag.load(Ordering::Relaxed)) || !cfg!(target_os = "linux") {
        std::thread::scope(|scope| {
            for worker in 0..threads - 1 {
                let job = &job;
                scope.spawn(move || job(worker));
            }
            job(threads - 1);
        });
        return;
    }
    #[cfg(target_os = "linux")]
    {
        let _guard = ReentrancyGuard::enter();
        // run() 的 barrier 保证所有 worker 在返回前完成并结束对任务的调用,
        // 任务引用实际生命周期即本调用;主线程持有的 Arc 最后释放。
        // 据此把闭包擦成 'static 存入常驻 team(与 std::thread::scope 的
        // 作用域线程同一安全论证)。
        let job: Arc<dyn Fn(usize) + Send + Sync + 'static> = unsafe { std::mem::transmute::<Arc<dyn Fn(usize) + Send + Sync + '_>, Arc<dyn Fn(usize) + Send + Sync + 'static>>(Arc::new(job)) };
        TEAM.with(|team| {
            let mut team = team.borrow_mut();
            let team = team.get_or_insert_with(Team::new);
            team.run(threads as u32, job);
        });
    }
}
