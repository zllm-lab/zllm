//! CUDA device、stream、kernel module 和 pipeline 生命周期。
//!
//! 对称 `backend/metal/context.rs`:`CudaContext` 持有 cudarc 的 device context、
//! 默认 stream、NVRTC 编译出的 module、按名惰性缓存的 `CudaFunction`。
//! 各算子通过 `function(name)` 拿 `CudaFunction` 再 launch。

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use cudarc::cublas::CudaBlas;
use cudarc::driver::safe::{CudaContext as CudaCtx, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;
use half::f16;

#[derive(Clone)]
pub struct CudaContextOptions {
    pub device: usize,
    pub include_dir: std::path::PathBuf,
    pub arch: Option<String>,
}

impl Default for CudaContextOptions {
    fn default() -> Self {
        Self { device: 0, include_dir: std::path::PathBuf::from("/usr/local/cuda/include"), arch: None }
    }
}

// NVRTC 底层 FFI。直接输出当前 GPU 的 CUBIN，避免 Toolkit 比 driver 新时
// PTX JIT 报 CUDA_ERROR_UNSUPPORTED_PTX_VERSION。
/// 用 NVRTC 编译 CUDA C++ 源码为目标 GPU 的 CUBIN。
fn compile_cubin(src: &str, arch: &str, include: &std::path::Path) -> Result<Vec<u8>, String> {
    use cudarc::nvrtc::sys::{self, nvrtcResult::NVRTC_SUCCESS};
    let src_c = CString::new(src).map_err(|e| format!("源码含内嵌 NUL: {e}"))?;
    let arch_opt = CString::new(format!("-arch={arch}")).map_err(|e| format!("CUDA arch 含内嵌 NUL: {e}"))?;
    let include_opt = CString::new(format!("-I{}", include.display())).map_err(|e| format!("CUDA include 路径含内嵌 NUL: {e}"))?;
    let mut prog: sys::nvrtcProgram = std::ptr::null_mut();
    unsafe {
        let r = sys::nvrtcCreateProgram(&mut prog, src_c.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null());
        if r != NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram 失败: {r:?}"));
        }
        // -I 指向 CUDA 头文件目录(NVRTC 默认不搜索系统头文件路径,需显式指定)。
        let opts = [arch_opt.as_ptr(), include_opt.as_ptr()];
        let r = sys::nvrtcCompileProgram(prog, opts.len() as i32, opts.as_ptr());
        if r != NVRTC_SUCCESS {
            // 失败时取编译日志辅助排查。
            let mut log_size: usize = 0;
            sys::nvrtcGetProgramLogSize(prog, &mut log_size);
            let mut log = vec![0u8; log_size];
            sys::nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut _);
            let log_str = String::from_utf8_lossy(&log);
            sys::nvrtcDestroyProgram(&mut prog);
            return Err(format!("nvrtcCompileProgram 失败: {r:?}\n编译日志:\n{log_str}"));
        }
        let mut cubin_size: usize = 0;
        let r = sys::nvrtcGetCUBINSize(prog, &mut cubin_size);
        if r != NVRTC_SUCCESS || cubin_size == 0 {
            sys::nvrtcDestroyProgram(&mut prog);
            return Err(format!("nvrtcGetCUBINSize 失败: status={r:?}, size={cubin_size}"));
        }
        let mut cubin = vec![0u8; cubin_size];
        let r = sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut _);
        if r != NVRTC_SUCCESS {
            sys::nvrtcDestroyProgram(&mut prog);
            return Err(format!("nvrtcGetCUBIN 失败: {r:?}"));
        }
        let r = sys::nvrtcDestroyProgram(&mut prog);
        if r != NVRTC_SUCCESS {
            return Err(format!("nvrtcDestroyProgram 失败: {r:?}"));
        }
        Ok(cubin)
    }
}

/// 行优先 `[rows, cols]` GPU buffer。层间只传这个句柄,不回读 CPU。
///
/// 对称 `MetalTensor`。`slice` 是 cudarc 的设备内存句柄,Drop 时自动释放。
///
/// 默认 f16 存储。DiT 残差流(可达 ~6e4,超过 f16 上限 65504)额外用 `slice_f32`
/// 保存 f32 权威数据:当 `slice_f32` 为 `Some`,`slice`(f16)是 1 元素占位、不可读,
/// 只有 `rmsnorm`/`gated_residual_segmented`/`trace_tensor`/`tensor_to_f32` 会读 f32。
/// 这镜像 Metal/ROCm 的扩散 f32 激活(残差流必须 f32;块内 attention/MLP 输出 ~1e4 仍走 f16)。
#[derive(Clone)]
pub struct CudaTensor {
    pub slice: CudaSlice<f16>,
    pub slice_f32: Option<CudaSlice<f32>>,
    pub rows: usize,
    pub cols: usize,
}

impl CudaTensor {
    pub fn new(slice: CudaSlice<f16>, rows: usize, cols: usize) -> Self {
        Self { slice, slice_f32: None, rows, cols }
    }

    /// 构造 f32 残差张量:`slice_f32` 为权威数据,`slice` 是 1 元素占位(不可读)。
    pub fn new_f32_residual(slice_f32: CudaSlice<f32>, placeholder: CudaSlice<f16>, rows: usize, cols: usize) -> Self {
        Self { slice: placeholder, slice_f32: Some(slice_f32), rows, cols }
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

/// CUDA 执行上下文。一个 device + 一个 stream + kernel function 缓存。
///
/// 对称 `MetalContext`。统计字段在后续步骤接入 cudaEvent_t 计时。
///
/// 设计说明:cudarc 的 `CudaContext` 用 `Arc` 引用计数(多数方法签名是 `self: &Arc<Self>`),
/// 所以本结构持有 `Arc<CudaCtx>`,并在构造时顺便取一份 `Arc<CudaStream>` 缓存,
/// 避免每次算子调用都走 `default_stream`(那会原子地查/建默认流)。
/// expert 流式上传的轮转槽:buffer 按需增长,event 为上一次 DMA 的完成标记。
struct PinnedUploadSlot {
    buffer: cudarc::driver::safe::PinnedHostSlice<u8>,
    event: Option<cudarc::driver::safe::CudaEvent>,
}

const PINNED_UPLOAD_SLOTS: usize = 8;

#[derive(Default)]
struct PinnedUploadRing {
    slots: Vec<Option<PinnedUploadSlot>>,
    next: usize,
}

impl PinnedUploadRing {
    /// 取出下一槽;返回槽在环内的原位置(release 时放回,保留已增长的容量)。
    fn take_slot(&mut self) -> (usize, Option<PinnedUploadSlot>) {
        if self.slots.len() < PINNED_UPLOAD_SLOTS {
            self.slots.push(None);
            let index = self.slots.len() - 1;
            self.next = 0;
            return (index, None);
        }
        let index = self.next;
        self.next = (self.next + 1) % self.slots.len();
        (index, self.slots[index].take())
    }

    fn release_slot(&mut self, index: usize, slot: Option<PinnedUploadSlot>) {
        if index < self.slots.len() {
            self.slots[index] = slot;
        }
    }
}

pub struct CudaContext {
    ctx: Arc<CudaCtx>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    module: Arc<CudaModule>,
    functions: Mutex<HashMap<String, CudaFunction>>,
    row_maps: Mutex<HashMap<Vec<u32>, Arc<CudaSlice<u32>>>>,
    /// rope 表窗口缓存:键 = (源切片地址, 元素数)。rope 表由引擎长期持有,地址稳定;
    /// 同一窗口(query/key × 各层)跨层命中,避免逐层 pageable 上传的重型负载 stall。
    /// 容量封顶:decode 逐 token 前移窗口,超过上限后退化为直接 pinned 上传。
    rope_windows: Mutex<HashMap<(usize, usize), Arc<CudaSlice<f16>>>>,
    pinned_u32: Mutex<Option<cudarc::driver::safe::PinnedHostSlice<u32>>>,
    pinned_f16: Mutex<Option<cudarc::driver::safe::PinnedHostSlice<f16>>>,
    pinned_f32_src: Mutex<Option<cudarc::driver::safe::PinnedHostSlice<f32>>>,
    /// expert 流式上传的 pinned 槽环:槽复用前只等自己的 DMA 事件,不排空计算流。
    pinned_u8_ring: Mutex<PinnedUploadRing>,
    /// expert 流式上传专用 copy 流:分配/DMA 不进计算流 backlog,消费侧
    /// wait copy fence 事件拿跨流依赖。
    copy_stream: Arc<cudarc::driver::safe::CudaStream>,
    copy_fence: Mutex<Option<cudarc::driver::safe::CudaEvent>>,
    /// 计算流上 expert buffer 分配完成的事件;copy 流 DMA 前等待,
    /// 保证 mallocAsync 指针在跨流使用前已物化。
    copy_fence_dirty: AtomicBool,
    gpu_nanoseconds: AtomicU64,
    command_buffers: AtomicU64,
}

impl CudaContext {
    /// 用默认内联 kernel 源码初始化:创建 device 0 的 context,NVRTC 编译 kernels_source()。
    pub fn new_default() -> Result<Self, String> {
        Self::new_with_options(crate::kernel::cuda::kernels_source(), CudaContextOptions::default())
    }

    pub fn new_default_with_options(options: CudaContextOptions) -> Result<Self, String> {
        Self::new_with_options(crate::kernel::cuda::kernels_source(), options)
    }

    /// 用指定 kernel 源码初始化。对称 `MetalContext::new(kernel_source)`。
    pub fn new(kernel_source: &str) -> Result<Self, String> {
        Self::new_with_options(kernel_source, CudaContextOptions::default())
    }

    pub fn new_with_options(kernel_source: &str, options: CudaContextOptions) -> Result<Self, String> {
        // 1. 创建配置指定设备的 context(对应 Metal 的 Device::system_default)。
        //    cudarc 返回 Arc<CudaCtx>(主 context,进程内共享)。
        let ctx = CudaCtx::new(options.device).map_err(|e| format!("CUDA device {} 初始化失败: {e:?}", options.device))?;
        // 关闭 cudarc 默认的 event tracking:每张 CudaSlice 都会 create 两个 CUDA event,
        // 而在单 stream + 重型 stream 负载路径下,event tracking + pageable HTOD 路径会
        // 导致 ~30s 的 stall(实测 mlp.gated_residual 的 row_map upload 卡 32s)。
        // 我们自己用 cuMemcpyHtoDAsync 直接处理 row_map,不需要 cudarc 的事件簿记。
        unsafe {
            ctx.disable_event_tracking();
        }

        // 2. NVRTC 为当前设备架构编译 CUBIN(而非 PTX)。
        //    用 CUBIN 绕过 PTX JIT 版本不匹配:NVRTC 产出的 PTX 版本号可能高于
        //    驱动 595.80(CUDA 13.2)的 JIT 支持范围,CUBIN 是本地机器码无需 JIT。
        let major = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).map_err(|e| format!("读取 CUDA compute capability major 失败: {e:?}"))?;
        let minor = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).map_err(|e| format!("读取 CUDA compute capability minor 失败: {e:?}"))?;
        if major <= 0 || minor < 0 {
            return Err(format!("CUDA compute capability 非法: {major}.{minor}"));
        }
        let arch = options.arch.clone().unwrap_or_else(|| format!("sm_{major}{minor}"));
        if !arch.starts_with("sm_") {
            return Err(format!("CUDA CUBIN architecture 必须是 sm_XX，实际为 {arch}"));
        }
        let cubin = compile_cubin(kernel_source, &arch, &options.include_dir)?;

        // 3. 加载已经针对当前 device 编译的 CUBIN。
        //    Ptx::from_binary 包成 PtxKind::Binary,load_module 内部走 cuModuleLoadData。
        let module = ctx.load_module(Ptx::from_binary(cubin)).map_err(|e| format!("加载 CUDA module 失败: {e:?}"))?;

        // 4. 取默认 stream 缓存。default_stream 需要 &Arc<Self>,返回 Arc<CudaStream>。
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("CUDA cuBLAS 初始化失败: {e:?}"))?;
        // expert 上传专用 copy 流;同时把 mallocAsync 池的 release threshold 归零,
        // freeAsync 即时归还设备内存——12GB 级显存上池峰值保留会挤掉 expert cache
        // (实测 6GiB cache 因池保留峰值 OOM)。
        let copy_stream = ctx.new_stream().map_err(|e| format!("CUDA copy stream 创建失败: {e:?}"))?;
        unsafe {
            let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
            if cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, options.device as cudarc::driver::sys::CUdevice) == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS && !pool.is_null() {
                // eager 归还:freeAsync 即时归还设备内存(expert cache 让出峰值余量)。
                if std::env::var_os("ZLLM_CUDA_POOL_EAGER").is_some() {
                    let threshold: usize = 0;
                    cudarc::driver::sys::cuMemPoolSetAttribute(pool, cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold as *const usize as *mut std::ffi::c_void);
                }
                // 严格复用(诊断开关):关闭全部跨流/机会主义复用捷径,验证
                // "池跨流复用竞态"假设——L46 损坏的领先根因候选。
                if std::env::var_os("ZLLM_CUDA_POOL_STRICT").is_some() {
                    for attr in [
                        cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                        cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                        cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
                    ] {
                        let disabled: i32 = 0;
                        cudarc::driver::sys::cuMemPoolSetAttribute(pool, attr, &disabled as *const i32 as *mut std::ffi::c_void);
                    }
                }
            }
        }

        Ok(Self {
            ctx,
            stream,
            blas,
            module,
            functions: Mutex::new(HashMap::new()),
            row_maps: Mutex::new(HashMap::new()),
            rope_windows: Mutex::new(HashMap::new()),
            pinned_u32: Mutex::new(None),
            pinned_f16: Mutex::new(None),
            pinned_f32_src: Mutex::new(None),
            pinned_u8_ring: Mutex::new(PinnedUploadRing::default()),
            copy_stream,
            copy_fence: Mutex::new(None),
            copy_fence_dirty: AtomicBool::new(false),
            gpu_nanoseconds: AtomicU64::new(0),
            command_buffers: AtomicU64::new(0),
        })
    }

    pub fn copy_stream(&self) -> &Arc<cudarc::driver::safe::CudaStream> {
        &self.copy_stream
    }

    /// 在 copy 流上记录栅栏;消费侧通过 wait_copy_fence 建立跨流依赖。
    fn record_copy_fence(&self) -> Result<(), String> {
        let mut guard = self.copy_fence.lock().map_err(|_| "CUDA copy fence 锁已中毒".to_owned())?;
        let event = guard.get_or_insert_with(|| self.ctx.new_event(None).unwrap_or_else(|_| unreachable!("copy fence 事件创建失败")));
        event.record(&self.copy_stream).map_err(|e| format!("CUDA copy fence 记录失败: {e:?}"))?;
        self.copy_fence_dirty.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// 计算流等待 copy 栅栏(有新上传时);幂等。
    pub fn wait_copy_fence(&self) -> Result<(), String> {
        if !self.copy_fence_dirty.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return Ok(());
        }
        let guard = self.copy_fence.lock().map_err(|_| "CUDA copy fence 锁已中毒".to_owned())?;
        if let Some(event) = guard.as_ref() {
            // 二分诊断:ZLLM_CUDA_FENCE_SYNC=1 时用 host 同步替代跨流 stream wait,
            // 区分"event 跨流依赖未生效"与"内存被并发踩"。
            if std::env::var_os("ZLLM_CUDA_FENCE_SYNC").is_some() {
                event.synchronize().map_err(|e| format!("CUDA copy fence host 同步失败: {e:?}"))?;
                return Ok(());
            }
            self.stream.wait(event).map_err(|e| format!("CUDA copy fence 等待失败: {e:?}"))?;
        }
        Ok(())
    }

    /// copy-on 下 expert 块宿主在 copy 流;驱逐 freeAsync 前让 copy 流等计算流,
    /// 否则读块的 kernel(计算流)未完成块就被池复用,后续写入踩到在途读取。
    pub fn fence_compute_before_copy(&self) -> Result<(), String> {
        let event = self.ctx.new_event(None).map_err(|e| format!("CUDA 驱逐 fence 事件创建失败: {e:?}"))?;
        event.record(&self.stream).map_err(|e| format!("CUDA 驱逐 fence 记录失败: {e:?}"))?;
        self.copy_stream.wait(&event).map_err(|e| format!("CUDA 驱逐 fence 等待失败: {e:?}"))?;
        Ok(())
    }

    /// 设备名(用于启动时打印确认)。
    pub fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "unknown".to_string())
    }

    /// 默认 stream 引用(对应 Metal 的 command_queue)。所有算子在它上面排队。
    /// 返回 `&Arc<CudaStream>` 而非 `&CudaStream`:cudarc 的 clone_htod / alloc_zeros /
    /// clone_dtoh 等方法签名是 `self: &Arc<Self>`,需要保留 Arc。
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// 直接拿 cuBLAS 句柄(用于 cublasGemmEx 等 cudarc::safe::Gemm 之外的 API,
    /// 比如 f16 weight + f16 input → f32 output 的 mixed precision gemm_ex)。
    pub fn blas_handle(&self) -> &cudarc::cublas::sys::cublasHandle_t {
        self.blas.handle()
    }

    /// Arc 引用的底层 context(算子需要调 alloc_zeros / load_module 时用)。
    pub fn device(&self) -> &Arc<CudaCtx> {
        &self.ctx
    }

    /// 同步 stream,确保之前提交的所有 kernel 完工(对应 Metal 的 synchronize)。
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.stream.synchronize()
    }

    /// 惰性按名取 `CudaFunction` 并缓存(对应 Metal 的 `pipeline(name)`)。
    /// `CudaFunction` 是 Clone 的(内部是 cuFunction 句柄的 Arc 包裹)。
    pub fn function(&self, name: &str) -> Result<CudaFunction, String> {
        let mut functions = self.functions.lock().map_err(|_| "CUDA function cache 锁已中毒".to_owned())?;
        if let Some(f) = functions.get(name) {
            return Ok(f.clone());
        }
        let f = self.module.load_function(name).map_err(|e| format!("加载 kernel {name} 失败: {e:?}"))?;
        functions.insert(name.to_string(), f.clone());
        Ok(f)
    }

    /// 分配零填充的 f16 GPU buffer。对称 `MetalContext::tensor_zeros`。
    pub fn tensor_zeros(&self, rows: usize, cols: usize) -> Result<CudaTensor, String> {
        let len = rows.checked_mul(cols).ok_or_else(|| format!("tensor_zeros 维度溢出: {rows}×{cols}"))?;
        self.trace_large_alloc(len * std::mem::size_of::<f16>());
        let slice = self.stream.alloc_zeros::<f16>(len).map_err(|e| format!("CUDA alloc_zeros 失败: {e:?}"))?;
        Ok(CudaTensor::new(slice, rows, cols))
    }

    /// 分配由后续 kernel 完整覆写的 device buffer。
    pub fn buffer_uninit<T: cudarc::driver::DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, String> {
        self.trace_large_alloc(len.saturating_mul(std::mem::size_of::<T>()));
        unsafe { self.stream.alloc::<T>(len) }.map_err(|e| format!("CUDA alloc 失败: {e:?}"))
    }

    /// 大块分配诊断:定位显存峰值/泄漏时打印 >=32MiB 的请求与剩余显存。
    /// 默认关闭(每次 cuMemGetInfo 是 driver 调用,线上会写成日志洪水);
    /// ZLLM_CUDA_ALLOC_TRACE=1 显式开启。
    fn trace_large_alloc(&self, bytes: usize) {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if bytes < 32 * 1024 * 1024 || !ENABLED.get_or_init(|| std::env::var_os("ZLLM_CUDA_ALLOC_TRACE").is_some()) {
            return;
        }
        if let Ok((free, total)) = self.mem_info() {
            eprintln!("[cuda-alloc] {bytes} bytes (free={free}/total={total})");
        }
    }

    /// 当前设备 (free, total) 显存字节;分配失败诊断用。
    pub fn mem_info(&self) -> Result<(usize, usize), String> {
        cudarc::driver::result::mem_get_info().map_err(|e| format!("CUDA mem_info 失败: {e:?}"))
    }

    /// rope 表窗口的设备驻留副本(跨层/跨 chunk 复用)。键 = (源切片地址, 元素数),
    /// 语义与 Metal cast cache 相同:源表长期存活且内容不可变。缓存满后退化为
    /// 每次 pinned 上传,不阻塞调用方。
    pub fn rope_window_f16(&self, values: &[f32]) -> Result<Arc<CudaSlice<f16>>, String> {
        const MAX_WINDOWS: usize = 8192;
        let key = (values.as_ptr() as usize, values.len());
        if let Some(hit) = self.rope_windows.lock().map_err(|_| "CUDA rope 窗口缓存锁已中毒".to_owned())?.get(&key) {
            return Ok(hit.clone());
        }
        let converted: Vec<f16> = values.iter().map(|value| f16::from_f32(*value)).collect();
        let mut device = self.buffer_uninit::<f16>(converted.len())?;
        self.upload_f16_pinned(&converted, &mut device)?;
        let entry = Arc::new(device);
        let mut cache = self.rope_windows.lock().map_err(|_| "CUDA rope 窗口缓存锁已中毒".to_owned())?;
        if cache.len() < MAX_WINDOWS {
            cache.insert(key, entry.clone());
        }
        Ok(entry)
    }

    /// 把 CPU u32 row_map 上传到 GPU；相同内容复用不可变设备 buffer。
    /// 不能只按长度复用，否则并发提交会在前一个 kernel launch 前覆写内容。
    pub fn upload_row_map(&self, row_map: &[u32]) -> Result<Arc<CudaSlice<u32>>, String> {
        let len = row_map.len();
        if len == 0 {
            return Err("CUDA row_map 不能为空".to_owned());
        }
        let mut cache = self.row_maps.lock().map_err(|_| "CUDA row_map cache 锁已中毒".to_owned())?;
        if let Some(buf) = cache.get(row_map) {
            return Ok(buf.clone());
        }
        let buf = Arc::new(unsafe { self.stream.alloc::<u32>(len) }.map_err(|e| format!("CUDA row_map alloc 失败: {e:?}"))?);
        // 把源 Vec<u32> 复制到 pinned 内存再 cuMemcpyHtoDAsync。
        // 直接用 Vec<u32> 的 pageable 指针调用 cuMemcpyHtoDAsync 在重型 stream 负载下会
        // 触发 ~30s stall(CUDA runtime 在内部 sync 等 GPU)。pinned 路径绕过该 stall。
        let mut pinned_guard = self.pinned_u32.lock().map_err(|_| "CUDA pinned u32 cache 锁已中毒".to_owned())?;
        if pinned_guard.as_ref().is_none_or(|pinned| pinned.len() < len) {
            let p = unsafe { self.ctx.alloc_pinned::<u32>(len.max(4096)) }.map_err(|e| format!("CUDA pinned alloc 失败: {e:?}"))?;
            *pinned_guard = Some(p);
        }
        let pinned = pinned_guard.as_mut().ok_or_else(|| "CUDA pinned u32 cache 初始化失败".to_owned())?;
        let host_ptr = pinned.as_mut_ptr().map_err(|e| format!("pinned as_mut_ptr 失败: {e:?}"))?;
        let pinned_slice = unsafe { std::slice::from_raw_parts_mut(host_ptr, pinned.len()) };
        pinned_slice[..len].copy_from_slice(row_map);
        let src_ptr = pinned.as_ptr().map_err(|e| format!("pinned as_ptr 失败: {e:?}"))?;
        // 直接走 cuMemcpyHtoDAsync,绕开 cudarc 的 memcpy_htod 的 _HTOD 同步 stall 包装。
        use cudarc::driver::safe::DevicePtr;
        let (dev_ptr, _sync) = buf.device_ptr(&self.stream);
        let nbytes = len.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| "CUDA row_map 字节数溢出".to_owned())?;
        self.ctx.bind_to_thread().map_err(|e| format!("绑定 CUDA context 失败: {e:?}"))?;
        let status = unsafe { cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dev_ptr, src_ptr as *const std::ffi::c_void, nbytes, self.stream.cu_stream()) };
        // pinned 是跨调用复用的共享 staging;必须等 DMA 完成后才能让
        // Mutex 释放(下一次调用会覆写同一块 host 内存),否则静默数据竞态。
        self.ctx.synchronize().map_err(|e| format!("CUDA row_map pinned 同步失败: {e:?}"))?;
        drop(_sync);
        if let Err(e) = status.result() {
            return Err(format!("CUDA row_map cuMemcpyHtoDAsync 失败: {e:?}"));
        }
        cache.insert(row_map.to_vec(), buf.clone());
        Ok(buf)
    }

    /// f16 主机数组 → 设备 buffer,经 pinned 中转。
    /// cudarc 的 `stream.clone_htod(&Vec<f16>)` 在重型 stream 上 SyncOnDrop 会 stall
    /// ~30s(pageable 主机指针触发内部 stream sync)。pinned 路径绕开。
    pub fn upload_f16_pinned(&self, src: &[f16], dst: &mut CudaSlice<f16>) -> Result<(), String> {
        if dst.len() < src.len() {
            return Err(format!("CUDA upload_f16_pinned dst.len={} < src.len={}", dst.len(), src.len()));
        }
        if src.is_empty() {
            return Ok(());
        }
        let mut guard = self.pinned_f16.lock().map_err(|_| "CUDA pinned f16 cache 锁已中毒".to_owned())?;
        if guard.as_ref().is_none_or(|pinned| pinned.len() < src.len()) {
            // 按实际权重向上取 2 的幂，避免小模型也固定占用 512 MiB 锁页内存。
            let target = src.len().checked_next_power_of_two().ok_or_else(|| "CUDA pinned f16 容量溢出".to_owned())?.max(4096);
            let p = unsafe { self.ctx.alloc_pinned::<f16>(target) }.map_err(|e| format!("CUDA pinned_f16 alloc 失败: {e:?}"))?;
            *guard = Some(p);
        }
        let pinned = guard.as_mut().ok_or_else(|| "CUDA pinned f16 cache 初始化失败".to_owned())?;
        let host_ptr = pinned.as_mut_ptr().map_err(|e| format!("pinned_f16 as_mut_ptr 失败: {e:?}"))?;
        let pinned_slice = unsafe { std::slice::from_raw_parts_mut(host_ptr, pinned.len()) };
        pinned_slice[..src.len()].copy_from_slice(src);
        let src_ptr = pinned.as_ptr().map_err(|e| format!("pinned_f16 as_ptr 失败: {e:?}"))?;
        use cudarc::driver::safe::DevicePtrMut;
        let (dev_ptr, _sync) = dst.device_ptr_mut(&self.stream);
        let nbytes = src.len().checked_mul(std::mem::size_of::<f16>()).ok_or_else(|| "CUDA pinned f16 字节数溢出".to_owned())?;
        self.ctx.bind_to_thread().map_err(|e| format!("绑定 CUDA context 失败: {e:?}"))?;
        let status = unsafe { cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dev_ptr, src_ptr as *const std::ffi::c_void, nbytes, self.stream.cu_stream()) };
        // 同上:共享 pinned staging,覆写前必须完成 DMA。
        self.ctx.synchronize().map_err(|e| format!("CUDA pinned_f16 同步失败: {e:?}"))?;
        drop(_sync);
        if let Err(e) = status.result() {
            return Err(format!("CUDA pinned_f16 cuMemcpyHtoDAsync 失败: {e:?}"));
        }
        Ok(())
    }

    /// u32 主机数组 → 设备 buffer,pinned 中转(row_map 同款;同步保 staging 复用安全)。
    pub fn upload_u32_pinned(&self, src: &[u32], dst: &mut CudaSlice<u32>) -> Result<(), String> {
        if dst.len() < src.len() {
            return Err(format!("CUDA upload_u32_pinned dst.len={} < src.len={}", dst.len(), src.len()));
        }
        if src.is_empty() {
            return Ok(());
        }
        let mut guard = self.pinned_u32.lock().map_err(|_| "CUDA pinned u32 cache 锁已中毒".to_owned())?;
        if guard.as_ref().is_none_or(|pinned| pinned.len() < src.len()) {
            let pinned = unsafe { self.ctx.alloc_pinned::<u32>(src.len().checked_next_power_of_two().ok_or("CUDA pinned u32 容量溢出")?.max(4096)) }.map_err(|e| format!("CUDA pinned_u32 alloc 失败: {e:?}"))?;
            *guard = Some(pinned);
        }
        let pinned = guard.as_mut().ok_or("CUDA pinned u32 cache 初始化失败")?;
        let host_ptr = pinned.as_mut_ptr().map_err(|e| format!("pinned_u32 as_mut_ptr 失败: {e:?}"))?;
        let staging = unsafe { std::slice::from_raw_parts_mut(host_ptr, pinned.len()) };
        staging[..src.len()].copy_from_slice(src);
        let src_ptr = pinned.as_ptr().map_err(|e| format!("pinned_u32 as_ptr 失败: {e:?}"))?;
        use cudarc::driver::safe::DevicePtrMut;
        let (dev_ptr, _sync) = dst.device_ptr_mut(&self.stream);
        self.ctx.bind_to_thread().map_err(|e| format!("绑定 CUDA context 失败: {e:?}"))?;
        let nbytes = src.len() * std::mem::size_of::<u32>();
        let status = unsafe { cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dev_ptr, src_ptr as *const std::ffi::c_void, nbytes, self.stream.cu_stream()) };
        // 共享 staging 覆写安全:等 DMA 完成再放锁。
        self.ctx.synchronize().map_err(|e| format!("CUDA pinned_u32 同步失败: {e:?}"))?;
        drop(_sync);
        if let Err(e) = status.result() {
            return Err(format!("CUDA pinned_u32 cuMemcpyHtoDAsync 失败: {e:?}"));
        }
        Ok(())
    }

    /// f32 主机数组 → 设备 buffer,pinned 中转(路由权重等小上传)。
    pub fn upload_f32_pinned(&self, src: &[f32], dst: &mut CudaSlice<f32>) -> Result<(), String> {
        if dst.len() < src.len() {
            return Err(format!("CUDA upload_f32_pinned dst.len={} < src.len={}", dst.len(), src.len()));
        }
        if src.is_empty() {
            return Ok(());
        }
        let mut guard = self.pinned_f32_src.lock().map_err(|_| "CUDA pinned f32 src 锁已中毒".to_owned())?;
        if guard.as_ref().is_none_or(|pinned| pinned.len() < src.len()) {
            let pinned = unsafe { self.ctx.alloc_pinned::<f32>(src.len().checked_next_power_of_two().ok_or("CUDA pinned f32 容量溢出")?.max(4096)) }.map_err(|e| format!("CUDA pinned_f32 alloc 失败: {e:?}"))?;
            *guard = Some(pinned);
        }
        let pinned = guard.as_mut().ok_or("CUDA pinned f32 src 初始化失败")?;
        let host_ptr = pinned.as_mut_ptr().map_err(|e| format!("pinned_f32 as_mut_ptr 失败: {e:?}"))?;
        let staging = unsafe { std::slice::from_raw_parts_mut(host_ptr, pinned.len()) };
        staging[..src.len()].copy_from_slice(src);
        let src_ptr = pinned.as_ptr().map_err(|e| format!("pinned_f32 as_ptr 失败: {e:?}"))?;
        use cudarc::driver::safe::DevicePtrMut;
        let (dev_ptr, _sync) = dst.device_ptr_mut(&self.stream);
        self.ctx.bind_to_thread().map_err(|e| format!("绑定 CUDA context 失败: {e:?}"))?;
        let nbytes = src.len() * std::mem::size_of::<f32>();
        let status = unsafe { cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dev_ptr, src_ptr as *const std::ffi::c_void, nbytes, self.stream.cu_stream()) };
        self.ctx.synchronize().map_err(|e| format!("CUDA pinned_f32 同步失败: {e:?}"))?;
        drop(_sync);
        if let Err(e) = status.result() {
            return Err(format!("CUDA pinned_f32 cuMemcpyHtoDAsync 失败: {e:?}"));
        }
        Ok(())
    }

    /// u8 主机数组 → 设备 buffer,经 pinned 槽环 + 事件异步中转。
    /// 复用槽前只等待该槽上一次 DMA 的事件,计算流不被排空;pageable clone_htod
    /// 在 PCIe3 上带宽仅 ~1/3 且有 SyncOnDrop stall,逐次 ctx.synchronize 又会
    /// 消灭预取重叠,事件环是两者的折中。
    pub fn upload_u8_pinned(&self, src: &[u8], dst: &mut CudaSlice<u8>) -> Result<(), String> {
        if dst.len() < src.len() {
            return Err(format!("CUDA upload_u8_pinned dst.len={} < src.len={}", dst.len(), src.len()));
        }
        if src.is_empty() {
            return Ok(());
        }
        let mut ring = self.pinned_u8_ring.lock().map_err(|_| "CUDA pinned u8 ring 锁已中毒".to_owned())?;
        let (slot_index, mut slot) = ring.take_slot();
        let result = self.upload_u8_slot(&mut slot, src, dst);
        ring.release_slot(slot_index, slot);
        result
    }

    /// 槽内执行一次上传;`slot` 为 None 时按需创建新槽。
    fn upload_u8_slot(&self, slot: &mut Option<PinnedUploadSlot>, src: &[u8], dst: &mut CudaSlice<u8>) -> Result<(), String> {
        if slot.is_none() {
            let capacity = src.len().checked_next_power_of_two().ok_or_else(|| "CUDA pinned u8 容量溢出".to_owned())?;
            let buffer = unsafe { self.ctx.alloc_pinned::<u8>(capacity) }.map_err(|e| format!("CUDA pinned_u8 alloc 失败: {e:?}"))?;
            *slot = Some(PinnedUploadSlot { buffer, event: None });
        }
        let slot = slot.as_mut().expect("槽已就位");
        if slot.buffer.len() < src.len() {
            let capacity = src.len().checked_next_power_of_two().ok_or_else(|| "CUDA pinned u8 容量溢出".to_owned())?;
            slot.buffer = unsafe { self.ctx.alloc_pinned::<u8>(capacity) }.map_err(|e| format!("CUDA pinned_u8 扩容失败: {e:?}"))?;
            slot.event = None;
        }
        if let Some(event) = slot.event.as_ref() {
            event.synchronize().map_err(|e| format!("CUDA pinned_u8 槽事件等待失败: {e:?}"))?;
        }
        let host_ptr = slot.buffer.as_mut_ptr().map_err(|e| format!("pinned_u8 as_mut_ptr 失败: {e:?}"))?;
        let pinned_slice = unsafe { std::slice::from_raw_parts_mut(host_ptr, slot.buffer.len()) };
        pinned_slice[..src.len()].copy_from_slice(src);
        let src_ptr = slot.buffer.as_ptr().map_err(|e| format!("pinned_u8 as_ptr 失败: {e:?}"))?;
        use cudarc::driver::safe::DevicePtrMut;
        // 正确性二分:默认走计算流(旧行为);ZLLM_CUDA_COPY_STREAM=1 时 DMA 进 copy 流
        // 并记录消费栅栏。
        let use_copy_stream = std::env::var_os("ZLLM_CUDA_COPY_STREAM").is_some();
        let stream: &Arc<cudarc::driver::safe::CudaStream> = if use_copy_stream { &self.copy_stream } else { &self.stream };
        let (dev_ptr, _sync) = dst.device_ptr_mut(stream);
        self.ctx.bind_to_thread().map_err(|e| format!("绑定 CUDA context 失败: {e:?}"))?;
        let nbytes = src.len();
        let status = unsafe { cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dev_ptr, src_ptr as *const std::ffi::c_void, nbytes, stream.cu_stream()) };
        let event = match slot.event.take() {
            Some(event) => event,
            None => self.ctx.new_event(None).map_err(|e| format!("CUDA pinned_u8 事件创建失败: {e:?}"))?,
        };
        event.record(stream).map_err(|e| format!("CUDA pinned_u8 事件记录失败: {e:?}"))?;
        slot.event = Some(event);
        drop(_sync);
        if let Err(e) = status.result() {
            return Err(format!("CUDA pinned_u8 cuMemcpyHtoDAsync 失败: {e:?}"));
        }
        if use_copy_stream {
            // 二分诊断:ZLLM_CUDA_COPY_DRAIN=1 时逐次 DMA 后等完成(host 同步),
            // 消灭 copy 流与计算流的一切并发——用于区分并发时序竞态与结构性别名。
            if std::env::var_os("ZLLM_CUDA_COPY_DRAIN").is_some() {
                slot.event.as_ref().expect("DMA 后槽事件必在").synchronize().map_err(|e| format!("CUDA copy drain 同步失败: {e:?}"))?;
            }
            self.record_copy_fence()?;
        }
        Ok(())
    }

    /// 分配由后续 kernel 完整覆写的 f16 tensor，避免无意义的 memset。
    pub fn tensor_uninit(&self, rows: usize, cols: usize) -> Result<CudaTensor, String> {
        let len = rows.checked_mul(cols).ok_or_else(|| format!("tensor_uninit 维度溢出: {rows}×{cols}"))?;
        let slice = self.buffer_uninit(len)?;
        Ok(CudaTensor::new(slice, rows, cols))
    }

    /// 调试/防御:ZLLM_CUDA_ZERO_UNINIT 时 tensor_uninit 走清零分配,用于判定
    /// "kernel 未写区域 + 池旧字节"类污染。
    pub fn tensor_alloc(&self, rows: usize, cols: usize) -> Result<CudaTensor, String> {
        if std::env::var_os("ZLLM_CUDA_ZERO_UNINIT").is_some() {
            let len = rows.checked_mul(cols).ok_or_else(|| format!("tensor_alloc 维度溢出: {rows}×{cols}"))?;
            let slice = self.stream().alloc_zeros::<half::f16>(len).map_err(|e| format!("tensor_alloc 清零失败: {e:?}"))?;
            return Ok(CudaTensor::new(slice, rows, cols));
        }
        self.tensor_uninit(rows, cols)
    }

    /// 分配由后续 kernel 完整覆写的 f32 device buffer(DiT 残差流用)。
    pub fn buffer_uninit_f32(&self, len: usize) -> Result<CudaSlice<f32>, String> {
        self.buffer_uninit(len)
    }

    /// 1 元素 f16 占位(f32 残差张量的 `slice` 字段不可读,只需合法句柄)。
    pub fn placeholder_f16(&self) -> Result<CudaSlice<f16>, String> {
        self.stream.alloc_zeros::<f16>(1).map_err(|e| format!("CUDA placeholder alloc 失败: {e:?}"))
    }

    /// f32 host 数据 → f16 GPU tensor。对称 `MetalContext::tensor_from_f32`。
    pub fn tensor_from_f32(&self, data: &[f32], rows: usize, cols: usize) -> Result<CudaTensor, String> {
        if data.len() != rows * cols {
            return Err(format!("CudaTensor 上传长度不匹配: 实际={}, 期望={}", data.len(), rows * cols));
        }
        // host 侧 f32→f16,再整体上传。
        let f16_data: Vec<f16> = data.iter().map(|v| f16::from_f32(*v)).collect();
        let slice = self.stream.clone_htod::<f16, _>(&f16_data).map_err(|e| format!("CUDA clone_htod 失败: {e:?}"))?;
        Ok(CudaTensor::new(slice, rows, cols))
    }

    /// f16 GPU tensor → f32 host 数据。对称 `MetalContext::tensor_to_f32`。
    /// f32 残差张量(`slice_f32` 为 Some)直接回读 f32。
    pub fn tensor_to_f32(&self, tensor: &CudaTensor) -> Result<Vec<f32>, String> {
        if let Some(slice_f32) = &tensor.slice_f32 {
            return self.stream.clone_dtoh::<f32, _>(slice_f32).map_err(|e| format!("CUDA clone_dtoh f32 失败: {e:?}"));
        }
        let f16_data = self.stream.clone_dtoh::<f16, _>(&tensor.slice).map_err(|e| format!("CUDA clone_dtoh 失败: {e:?}"))?;
        Ok(f16_data.iter().map(|v| v.to_f32()).collect())
    }

    /// 权重 f32 host → f16 GPU(MVP 统一转 f16;后续 FP8/压缩权重 走 CudaWeight 枚举)。
    pub fn weight_from_f32(&self, data: &[f32], rows: usize, cols: usize) -> Result<CudaSlice<f16>, String> {
        if data.len() != rows * cols {
            return Err(format!("CudaWeight 上传长度不匹配: 实际={}, 期望={}", data.len(), rows * cols));
        }
        let f16_data: Vec<f16> = data.iter().map(|v| f16::from_f32(*v)).collect();
        self.stream.clone_htod::<f16, _>(&f16_data).map_err(|e| format!("CUDA weight clone_htod 失败: {e:?}"))
    }

    /// 统计:累计的 GPU 纳秒(后续步骤接入 cudaEvent_t)。
    pub fn gpu_seconds(&self) -> f64 {
        self.gpu_nanoseconds.load(Ordering::Relaxed) as f64 / 1e9
    }

    /// 统计:累计的 command buffer / launch 数。
    pub fn command_buffers(&self) -> u64 {
        self.command_buffers.load(Ordering::Relaxed)
    }
}
