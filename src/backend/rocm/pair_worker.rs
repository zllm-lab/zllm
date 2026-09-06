use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex, mpsc},
    thread,
};

use crate::backend::{BackendError, compute_error};

use super::RocmContext;

type PairJob = Box<dyn FnOnce(RocmContext, bool) -> bool + Send + 'static>;

enum PairCommand {
    Run(PairJob),
    Retain(Vec<Arc<super::ops::hip::DeviceBuffer>>),
    Finish { retained: Vec<Arc<super::ops::hip::DeviceBuffer>>, sender: mpsc::SyncSender<Result<Option<RocmPairStageCompletion>, BackendError>> },
    Shutdown,
}

struct RocmPairWorkerInner {
    sender: mpsc::Sender<PairCommand>,
    handle: Mutex<Option<thread::JoinHandle<()>>>,
}

impl Drop for RocmPairWorkerInner {
    fn drop(&mut self) {
        let _ = self.sender.send(PairCommand::Shutdown);
        if let Some(handle) = self.handle.get_mut().ok().and_then(Option::take) {
            let _ = handle.join();
        }
    }
}

/// 每张 peer 卡独占的常驻 host 提交线程。
///
/// pair 的两半必须由两个线程同时排入各自设备流；若 owner 线程轮流
/// `set_device` 并提交两张卡，decode 会把省下的 GPU 时间重新交成 host
/// submission 税。任务闭包只负责排队，GPU 依赖仍由 stream event 表达。
#[derive(Clone)]
pub(crate) struct RocmPairWorker {
    inner: Arc<RocmPairWorkerInner>,
}

impl RocmPairWorker {
    pub(crate) fn new(context: RocmContext) -> Result<Self, BackendError> {
        let (sender, receiver) = mpsc::channel::<PairCommand>();
        let handle = thread::Builder::new()
            .name(format!("zllm-rocm-pair-dev{}", context.device_id()))
            .spawn(move || {
                let mut stage_active = false;
                let mut stage_retained = Vec::new();
                while let Ok(command) = receiver.recv() {
                    match command {
                        PairCommand::Run(job) => {
                            stage_active = job(context, stage_active);
                            if !stage_active {
                                stage_retained.clear();
                            }
                        }
                        PairCommand::Retain(retained) => {
                            if stage_active {
                                stage_retained.extend(retained);
                            }
                        }
                        PairCommand::Finish { mut retained, sender } => {
                            retained.append(&mut stage_retained);
                            let result = if stage_active {
                                match context.activate().and_then(|()| super::ops::hip::DeviceCompletion::record(context.device_id())) {
                                    Ok(completion) => Ok(Some(RocmPairStageCompletion { completion, _retained: retained })),
                                    Err(error) => {
                                        let _ = super::ops::hip::abort_stage_buffer_recycle(context.device_id());
                                        Err(compute_error(error))
                                    }
                                }
                            } else {
                                Ok(None)
                            };
                            stage_active = false;
                            let _ = sender.send(result);
                        }
                        PairCommand::Shutdown => {
                            if stage_active {
                                let _ = super::ops::hip::abort_stage_buffer_recycle(context.device_id());
                            }
                            stage_retained.clear();
                            break;
                        }
                    }
                }
            })
            .map_err(|error| compute_error(format!("启动 ROCm pair device={} 提交线程失败: {error}", context.device_id())))?;
        Ok(Self { inner: Arc::new(RocmPairWorkerInner { sender, handle: Mutex::new(Some(handle)) }) })
    }

    pub(crate) fn submit<T, F>(&self, job: F) -> Result<RocmPairTicket<T>, BackendError>
    where
        T: Send + 'static,
        F: FnOnce(RocmContext) -> Result<T, BackendError> + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.inner
            .sender
            .send(PairCommand::Run(Box::new(move |context, stage_active| {
                if !super::ops::hip::is_hip_available() {
                    let _ = sender.send(job(context));
                    return false;
                }
                if let Err(error) = context.activate().map_err(compute_error).and_then(|()| if stage_active { Ok(()) } else { super::ops::hip::begin_stage_buffer_recycle(context.device_id()).map_err(compute_error) }) {
                    let _ = sender.send(Err(error));
                    return false;
                }
                let result = job(context);
                let success = result.is_ok();
                if !success {
                    let _ = super::ops::hip::abort_stage_buffer_recycle(context.device_id());
                }
                let _ = sender.send(result);
                success
            })))
            .map_err(|_| compute_error("ROCm pair 提交线程已经退出"))?;
        Ok(RocmPairTicket { receiver })
    }

    /// owner 上生产、peer stream 异步消费的 buffer 必须跟随 peer completion
    /// 保活；owner completion 只能证明反向传输和 owner join 已经完成。
    pub(crate) fn retain_for_stage(&self, retained: Vec<Arc<super::ops::hip::DeviceBuffer>>) -> Result<(), BackendError> {
        if retained.is_empty() {
            return Ok(());
        }
        self.inner.sender.send(PairCommand::Retain(retained)).map_err(|_| compute_error("ROCm pair 提交线程已经退出"))
    }

    /// 一个逻辑 stage 只记录一次 peer completion，覆盖该 stream 上全部前序
    /// job；逐层不额外建 event，也不让 owner stream 等 peer replica。
    pub(crate) fn finish_stage(&self, retained: Vec<Arc<super::ops::hip::DeviceBuffer>>) -> Result<Option<RocmPairStageCompletion>, BackendError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.inner.sender.send(PairCommand::Finish { retained, sender }).map_err(|_| compute_error("ROCm pair 提交线程已经退出"))?;
        receiver.recv().map_err(|_| compute_error("ROCm pair completion 通道提前关闭"))?
    }
}

pub(crate) struct RocmPairTicket<T> {
    receiver: mpsc::Receiver<Result<T, BackendError>>,
}

impl<T> RocmPairTicket<T> {
    /// 这里只等待 peer 的 host enqueue 完成，不能在任务闭包中等待 GPU。
    pub(crate) fn wait(self) -> Result<T, BackendError> {
        self.receiver.recv().map_err(|_| compute_error("ROCm pair 提交结果通道提前关闭"))?
    }
}

pub(crate) struct RocmPairStageCompletion {
    completion: super::ops::hip::DeviceCompletion,
    /// 最后一层 replica 可能在 stage 输出切片时先于 completion 被丢弃；
    /// completion 直接持有它，避免未完成的 peer cast 输出提前回池。
    _retained: Vec<Arc<super::ops::hip::DeviceBuffer>>,
}

impl RocmPairStageCompletion {
    pub(crate) fn is_complete(&self) -> Result<bool, BackendError> {
        self.completion.is_complete().map_err(compute_error)
    }

    pub(crate) fn wait(&self) -> Result<(), BackendError> {
        self.completion.wait().map_err(compute_error)
    }

    pub(crate) fn retire_ordered(&self) -> Result<(), BackendError> {
        // 通用单链 runner 只等待最后一个 owner stage；不同逻辑 stage 的
        // peer 不在 owner hidden 依赖链上，不能据此假定更早 peer 已完成。
        // 这里最多发生在 stage/请求退休边界，不会重新引入逐层等待。
        self.wait()
    }
}

thread_local! {
    static PENDING_PAIR_COMPLETIONS: RefCell<HashMap<i32, Vec<RocmPairStageCompletion>>> = RefCell::new(HashMap::new());
}

pub(crate) fn attach_pair_stage_completion(owner_device: i32, completion: RocmPairStageCompletion) {
    PENDING_PAIR_COMPLETIONS.with(|pending| pending.borrow_mut().entry(owner_device).or_default().push(completion));
}

pub(crate) fn take_pair_stage_completions(owner_device: i32) -> Vec<RocmPairStageCompletion> {
    PENDING_PAIR_COMPLETIONS.with(|pending| pending.borrow_mut().remove(&owner_device).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_worker_uses_a_persistent_peer_thread() {
        let caller = thread::current().id();
        let context = RocmContext { device_id: 7, allow_cpu_reference_fallback: false, compute_stream: 0 };
        let worker = RocmPairWorker::new(context).unwrap();
        let first = worker.submit(|context| Ok((context.device_id(), thread::current().id()))).unwrap().wait().unwrap();
        let second = worker.submit(|context| Ok((context.device_id(), thread::current().id()))).unwrap().wait().unwrap();
        assert_eq!(first.0, 7);
        assert_eq!(second.0, 7);
        assert_ne!(first.1, caller);
        assert_eq!(first.1, second.1);
    }

    #[test]
    fn pair_worker_returns_job_errors() {
        let context = RocmContext { device_id: 3, allow_cpu_reference_fallback: false, compute_stream: 0 };
        let worker = RocmPairWorker::new(context).unwrap();
        let error = worker.submit::<(), _>(|_| Err(compute_error("peer enqueue failed"))).unwrap().wait().unwrap_err();
        assert!(format!("{error:?}").contains("peer enqueue failed"));
    }
}
