//! `objc2-metal` 的薄封装，集中管理 Objective-C 对象所有权与 ABI 细节。

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder as NativeBlitCommandEncoder, MTLBuffer as NativeBuffer, MTLCommandBuffer as NativeCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder as NativeCommandEncoder, MTLCommandQueue as NativeCommandQueue,
    MTLCompileOptions as NativeCompileOptions, MTLComputeCommandEncoder as NativeComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState as NativeComputePipelineState, MTLDevice as NativeDevice,
    MTLFence as NativeFence, MTLFunction as NativeFunction, MTLFunctionConstantValues as NativeFunctionConstantValues, MTLIndirectCommandBuffer as NativeIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectComputeCommand as NativeIndirectComputeCommand, MTLLanguageVersion, MTLLibrary as NativeLibrary, MTLPipelineOption, MTLResource as NativeResource,
};

pub use objc2_metal::{MTLDataType, MTLResourceOptions};

// `MTLCreateSystemDefaultDevice` 的实现依赖 CoreGraphics。
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

#[derive(Clone, Copy)]
pub struct MTLSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

impl MTLSize {
    pub const fn new(width: u64, height: u64, depth: u64) -> Self {
        Self { width, height, depth }
    }

    fn native(self) -> objc2_metal::MTLSize {
        objc2_metal::MTLSize { width: self.width as usize, height: self.height as usize, depth: self.depth as usize }
    }
}

#[derive(Clone)]
pub struct Buffer(Retained<ProtocolObject<dyn NativeBuffer>>);

pub type BufferRef = Buffer;

// Metal buffer 可跨线程共享；并发写入的非重叠范围由上层加载器保证。
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl AsRef<BufferRef> for Buffer {
    fn as_ref(&self) -> &BufferRef {
        self
    }
}

impl Buffer {
    pub fn length(&self) -> u64 {
        self.0.length() as u64
    }

    pub fn contents(&self) -> *mut c_void {
        self.0.contents().as_ptr()
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn NativeBuffer> {
        &self.0
    }

    /// 底层 MTLBuffer 是否同一实例(录制表重映射的句柄判等)。
    pub fn same_handle(&self, other: &Buffer) -> bool {
        std::ptr::eq(self.raw() as *const ProtocolObject<dyn NativeBuffer>, other.raw() as *const ProtocolObject<dyn NativeBuffer>)
    }
}

#[derive(Clone)]
pub struct Device(Retained<ProtocolObject<dyn NativeDevice>>);

pub type DeviceRef = Device;

// MTLDevice 是系统级线程安全对象，heartbeat 只读取资源统计。
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// decode 重放用的间接命令缓冲:一次性录制 N 条 compute 命令,逐 token 重放。
    pub fn new_indirect_command_buffer(&self, max_commands: usize) -> Result<IndirectCommandBuffer, String> {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        // 逐命令显式 setComputePipelineState,不继承执行期 encoder 的 pipeline。
        descriptor.setInheritPipelineState(false);
        // 默认只允许 4 个 kernel buffer;decode 链每 kernel 最多绑 ~11 个。
        descriptor.setMaxKernelBufferBindCount(16);
        let icb = unsafe { self.0.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(&descriptor, max_commands, MTLResourceOptions::StorageModePrivate) };
        icb.map(IndirectCommandBuffer).ok_or_else(|| "创建 indirect command buffer 失败(设备可能不支持 compute ICB)".to_owned())
    }
    pub fn system_default() -> Option<Self> {
        objc2_metal::MTLCreateSystemDefaultDevice().map(Self)
    }

    pub fn new_command_queue(&self) -> CommandQueue {
        CommandQueue(self.0.newCommandQueue().expect("Metal 设备无法创建 command queue"))
    }

    pub fn name(&self) -> String {
        self.0.name().to_string()
    }

    pub fn current_allocated_size(&self) -> u64 {
        self.0.currentAllocatedSize() as u64
    }

    pub fn new_buffer(&self, length: u64, options: MTLResourceOptions) -> Buffer {
        self.try_new_buffer(length, options).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_new_buffer(&self, length: u64, options: MTLResourceOptions) -> Result<Buffer, String> {
        let length = usize::try_from(length).map_err(|_| format!("Metal buffer 长度超过 usize: {length}"))?;
        self.0.newBufferWithLength_options(length, options).map(Buffer).ok_or_else(|| format!("Metal 设备无法分配 buffer: length={length} bytes, options={options:?}"))
    }

    pub fn new_buffer_with_data(&self, bytes: *const c_void, length: u64, options: MTLResourceOptions) -> Buffer {
        self.try_new_buffer_with_data(bytes, length, options).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_new_buffer_with_data(&self, bytes: *const c_void, length: u64, options: MTLResourceOptions) -> Result<Buffer, String> {
        let bytes = NonNull::new(bytes.cast_mut()).ok_or("Metal buffer 数据指针不能为空")?;
        let length = usize::try_from(length).map_err(|_| format!("Metal buffer 长度超过 usize: {length}"))?;
        unsafe { self.0.newBufferWithBytes_length_options(bytes, length, options) }.map(Buffer).ok_or_else(|| format!("Metal 设备无法从数据分配 buffer: length={length} bytes, options={options:?}"))
    }

    pub fn new_library_with_source(&self, source: &str, options: &CompileOptions) -> Result<Library, String> {
        let source = NSString::from_str(source);
        self.0.newLibraryWithSource_options_error(&source, Some(&options.0)).map(Library).map_err(|error| format!("{error:?}"))
    }

    /// ICB 兼容 PSO:必须走 descriptor 路径并打开 supportIndirectCommandBuffers。
    /// 便捷路径创建的 PSO 传给 setComputePipelineState 时 Apple GPU 驱动直接段错误。
    pub fn new_compute_pipeline_state_icb(&self, function: &Function) -> Result<ComputePipelineState, String> {
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function.0));
        descriptor.setSupportIndirectCommandBuffers(true);
        self.0.newComputePipelineStateWithDescriptor_options_reflection_error(&descriptor, MTLPipelineOption::None, None).map(ComputePipelineState).map_err(|error| format!("{error:?}"))
    }

    pub fn new_fence(&self) -> Fence {
        Fence(self.0.newFence().expect("Metal 设备无法创建 fence"))
    }

    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.0.recommendedMaxWorkingSetSize()
    }

    pub fn has_unified_memory(&self) -> bool {
        self.0.hasUnifiedMemory()
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn NativeDevice> {
        &self.0
    }
}

#[derive(Clone)]
pub struct CommandQueue(Retained<ProtocolObject<dyn NativeCommandQueue>>);

impl CommandQueue {
    pub fn new_command_buffer(&self) -> CommandBuffer {
        CommandBuffer(self.0.commandBuffer().expect("Metal queue 无法创建 command buffer"))
    }
}

#[derive(Clone)]
pub struct CommandBuffer(Retained<ProtocolObject<dyn NativeCommandBuffer>>);

pub type CommandBufferRef = CommandBuffer;

impl AsRef<CommandBufferRef> for CommandBuffer {
    fn as_ref(&self) -> &CommandBufferRef {
        self
    }
}

impl CommandBuffer {
    pub fn new_compute_command_encoder(&self) -> ComputeCommandEncoder {
        ComputeCommandEncoder(self.0.computeCommandEncoder().expect("Metal command buffer 无法创建 compute encoder"))
    }

    pub fn new_blit_command_encoder(&self) -> BlitCommandEncoder {
        assert!(!Transcriber::is_active(), "转录路径遇到 blit encoder:该算子需要 position 化 compute kernel 替代后才能进入 ICB 重放");
        BlitCommandEncoder(self.0.blitCommandEncoder().expect("Metal command buffer 无法创建 blit encoder"))
    }

    pub fn commit(&self) {
        if Transcriber::is_active() {
            return;
        }
        self.0.commit();
    }

    pub fn wait_until_completed(&self) {
        if Transcriber::is_active() {
            return;
        }
        self.0.waitUntilCompleted();
    }

    pub fn gpu_start_time(&self) -> f64 {
        self.0.GPUStartTime()
    }

    pub fn gpu_end_time(&self) -> f64 {
        self.0.GPUEndTime()
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.0.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error)
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn NativeCommandBuffer> {
        &self.0
    }
}

#[derive(Clone)]
pub struct ComputeCommandEncoder(Retained<ProtocolObject<dyn NativeComputeCommandEncoder>>);

pub type ComputeCommandEncoderRef = ComputeCommandEncoder;

impl ComputeCommandEncoder {
    pub fn set_buffer(&self, index: u64, buffer: Option<&BufferRef>, offset: u64) {
        if Transcriber::with(|transcriber| {
            if let Some(buffer) = buffer {
                transcriber.bind_buffer(index, buffer, offset);
            }
        })
        .is_some()
        {
            return;
        }
        unsafe { self.0.setBuffer_offset_atIndex(buffer.map(Buffer::raw), offset as usize, index as usize) };
    }

    pub fn set_bytes(&self, index: u64, length: u64, bytes: *const c_void) {
        if Transcriber::with(|transcriber| transcriber.bind_bytes(index, length, bytes)).is_some() {
            return;
        }
        let bytes = NonNull::new(bytes.cast_mut()).expect("Metal 参数指针不能为空");
        unsafe { self.0.setBytes_length_atIndex(bytes, length as usize, index as usize) };
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipelineState) {
        if Transcriber::with(|transcriber| transcriber.begin_command(pipeline)).is_some() {
            return;
        }
        self.0.setComputePipelineState(&pipeline.0);
    }

    pub fn dispatch_thread_groups(&self, groups: MTLSize, threads: MTLSize) {
        if Transcriber::with(|transcriber| transcriber.dispatch(groups, threads)).is_some() {
            return;
        }
        self.0.dispatchThreadgroups_threadsPerThreadgroup(groups.native(), threads.native());
    }

    pub fn dispatch_threads(&self, threads: MTLSize, threads_per_group: MTLSize) {
        if Transcriber::with(|transcriber| transcriber.dispatch(dispatch_groups(threads, threads_per_group), threads_per_group)).is_some() {
            return;
        }
        self.0.dispatchThreads_threadsPerThreadgroup(threads.native(), threads_per_group.native());
    }

    pub fn update_fence(&self, fence: &FenceRef) {
        self.0.updateFence(&fence.0);
    }

    pub fn wait_for_fence(&self, fence: &FenceRef) {
        self.0.waitForFence(&fence.0);
    }

    pub fn memory_barrier_with_resources(&self, resources: &[&BufferRef]) {
        if resources.is_empty() {
            return;
        }
        let mut resources = resources.iter().map(|buffer| NonNull::from(ProtocolObject::<dyn NativeResource>::from_ref(buffer.raw()))).collect::<Vec<_>>();
        unsafe { self.0.memoryBarrierWithResources_count(NonNull::new_unchecked(resources.as_mut_ptr()), resources.len()) };
    }

    /// 重放 ICB 的 [0, count) 条命令;hazard tracking 仍按 buffer 依赖排序。
    pub fn execute_indirect(&self, icb: &IndirectCommandBuffer, count: u64) {
        unsafe { self.0.executeCommandsInBuffer_withRange(&icb.0, NSRange::new(0, count as usize)) }
    }

    /// 重放一条平铺记录:直接原始编码,绕开转录钩子(重放期转录器必然未激活)。
    pub fn encode_recorded(&self, op: &RecordedComputeOp) {
        self.0.setComputePipelineState(&op.pipeline.0);
        for (index, buffer, offset) in &op.buffers {
            unsafe { self.0.setBuffer_offset_atIndex(Some(buffer.raw()), *offset as usize, *index as usize) };
        }
        for (index, bytes) in &op.constants {
            unsafe { self.0.setBytes_length_atIndex(NonNull::new_unchecked(bytes.as_ptr() as *mut c_void), bytes.len(), *index as usize) };
        }
        self.0.dispatchThreadgroups_threadsPerThreadgroup(op.groups.native(), op.threads.native());
    }

    pub fn end_encoding(&self) {
        // 转录模式下 encoder 未收到任何真实编码,但 Metal 对象生命周期仍要求 endEncoding
        self.0.endEncoding();
    }

    /// 重放 ICB 的 [offset, offset+count) 条命令;与 barrier 配合给无依赖判断的
    /// ICB concurrent 命令补顺序(encoder 看不到 ICB 内资源依赖,自动 tracking 不生效)。
    pub fn execute_indirect_range(&self, icb: &IndirectCommandBuffer, offset: u64, count: u64) {
        unsafe { self.0.executeCommandsInBuffer_withRange(&icb.0, NSRange::new(offset as usize, count as usize)) }
    }

    /// buffer 范围的内存屏障:前一段 dispatch 的写入对后续 dispatch 可见。
    pub fn memory_barrier(&self) {
        self.0.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
    }
}

#[derive(Clone)]
pub struct BlitCommandEncoder(Retained<ProtocolObject<dyn NativeBlitCommandEncoder>>);

impl BlitCommandEncoder {
    pub fn copy_from_buffer(&self, source: &BufferRef, source_offset: u64, destination: &BufferRef, destination_offset: u64, size: u64) {
        unsafe { self.0.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(source.raw(), source_offset as usize, destination.raw(), destination_offset as usize, size as usize) };
    }

    pub fn end_encoding(&self) {
        self.0.endEncoding();
    }
}

#[derive(Clone)]
pub struct IndirectCommandBuffer(Retained<ProtocolObject<dyn NativeIndirectCommandBuffer>>);

impl IndirectCommandBuffer {
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// 取第 index 条 compute 命令槽进行录制;index 必须在创建容量内。
    pub fn compute_command(&self, index: usize) -> IndirectComputeCommand {
        unsafe { IndirectComputeCommand(self.0.indirectComputeCommandAtIndex(index)) }
    }

    /// 清空 [0, count) 的录制内容,重新录制前调用。
    pub fn reset(&self, count: usize) {
        unsafe { self.0.resetWithRange(NSRange::new(0, count)) }
    }
}

#[derive(Clone)]
pub struct IndirectComputeCommand(Retained<ProtocolObject<dyn NativeIndirectComputeCommand>>);

impl IndirectComputeCommand {
    pub fn set_pipeline(&self, pipeline: &ComputePipelineState) {
        self.0.setComputePipelineState(&pipeline.0)
    }

    /// ICB 没有 setBytes:所有标量必须经由 buffer 传入(uniform buffer 模式)。
    pub fn set_kernel_buffer(&self, index: u64, buffer: &Buffer, offset: u64) {
        unsafe { self.0.setKernelBuffer_offset_atIndex(&buffer.0, offset as usize, index as usize) }
    }

    pub fn dispatch(&self, groups: MTLSize, threads: MTLSize) {
        self.0.concurrentDispatchThreadgroups_threadsPerThreadgroup(groups.native(), threads.native())
    }
}

#[derive(Clone)]
pub struct ComputePipelineState(Retained<ProtocolObject<dyn NativeComputePipelineState>>);

impl ComputePipelineState {
    pub fn max_total_threads_per_threadgroup(&self) -> u64 {
        self.0.maxTotalThreadsPerThreadgroup() as u64
    }

    pub fn thread_execution_width(&self) -> u64 {
        self.0.threadExecutionWidth() as u64
    }

    /// pipeline 声明的静态 threadgroup memory；用于真机资源审计，不代表动态分配。
    pub fn static_threadgroup_memory_length(&self) -> u64 {
        self.0.staticThreadgroupMemoryLength() as u64
    }

    /// 是否同一 pipeline 实例(ctx 按名缓存,同名 kernel 共享实例;消融分组用)。
    pub fn same_handle(&self, other: &ComputePipelineState) -> bool {
        std::ptr::eq(&*self.0 as *const ProtocolObject<dyn NativeComputePipelineState>, &*other.0 as *const ProtocolObject<dyn NativeComputePipelineState>)
    }
}

pub struct CompileOptions(Retained<NativeCompileOptions>);

impl CompileOptions {
    pub fn new() -> Self {
        Self(NativeCompileOptions::new())
    }

    pub fn metal4() -> Self {
        let options = Self::new();
        options.0.setLanguageVersion(MTLLanguageVersion::Version4_0);
        options
    }
}

pub struct FunctionConstantValues(Retained<NativeFunctionConstantValues>);

impl FunctionConstantValues {
    pub fn new() -> Self {
        Self(NativeFunctionConstantValues::new())
    }

    pub fn set_constant_value_at_index(&self, value: *const c_void, data_type: MTLDataType, index: u64) {
        let value = NonNull::new(value.cast_mut()).expect("Metal 常量指针不能为空");
        unsafe { self.0.setConstantValue_type_atIndex(value, data_type, index as usize) };
    }
}

#[derive(Clone)]
pub struct Library(Retained<ProtocolObject<dyn NativeLibrary>>);

impl Library {
    pub fn get_function(&self, name: &str, constants: Option<FunctionConstantValues>) -> Result<Function, String> {
        let name = NSString::from_str(name);
        match constants {
            Some(constants) => self.0.newFunctionWithName_constantValues_error(&name, &constants.0).map(Function).map_err(|error| format!("{error:?}")),
            None => self.0.newFunctionWithName(&name).map(Function).ok_or_else(|| format!("Metal library 中没有函数 {name:?}")),
        }
    }
}

#[derive(Clone)]
pub struct Function(Retained<ProtocolObject<dyn NativeFunction>>);

#[derive(Clone)]
pub struct Fence(Retained<ProtocolObject<dyn NativeFence>>);

pub type FenceRef = Fence;

impl AsRef<FenceRef> for Fence {
    fn as_ref(&self) -> &FenceRef {
        self
    }
}

/// dispatch_threads 的 total/threads_per_group 换算成 threadgroup 数。
fn dispatch_groups(threads: MTLSize, per_group: MTLSize) -> MTLSize {
    let ceil = |total: u64, per: u64| total.div_ceil(per.max(1));
    MTLSize::new(ceil(threads.width, per_group.width), ceil(threads.height, per_group.height), ceil(threads.depth, per_group.depth))
}

/// ICB 转录器:激活期间 [`ComputeCommandEncoder`] 的绑定调用镜像成 ICB 命令,
/// 真实 encoder/command buffer 空转(commit 是 no-op),之后逐 token 重放 ICB。
/// 依赖既有 dispatch 函数零改动;decode 单线程执行,thread-local 足够。
pub struct Transcriber {
    transcription: Transcription,
    current: Option<CurrentCommand>,
    command_count: usize,
    keep_alive: Vec<Buffer>,
}

thread_local! {
    static TRANSCRIBER: std::cell::RefCell<Option<Transcriber>> = const { std::cell::RefCell::new(None) };
}

/// 平铺记录的一条 compute dispatch(非 ICB):重放时用普通 encoder 按序重编码。
#[derive(Clone)]
pub struct RecordedComputeOp {
    pub pipeline: ComputePipelineState,
    pub buffers: Vec<(u64, Buffer, u64)>,
    pub constants: Vec<(u64, Vec<u8>)>,
    pub groups: MTLSize,
    pub threads: MTLSize,
}

/// 平铺命令表:整步 kernel 序列 + 录制期钉住的 buffer(decode 形状恒定,即简化内存规划)。
pub struct CommandList {
    pub ops: Vec<RecordedComputeOp>,
    pub keep_alive: Vec<Buffer>,
}

impl CommandList {
    /// 请求级 buffer 重映射:把录制期钉住的 buffer 指针替换为新实例(KV cache 等
    /// 每请求重建的资源);offset/常量/线程网格不变,命令语义保持。
    pub fn remap_buffers(&mut self, replacements: &[(Buffer, Buffer)]) {
        for op in &mut self.ops {
            for (_, buffer, _) in &mut op.buffers {
                for (old, new) in replacements {
                    if buffer.same_handle(old) {
                        *buffer = new.clone();
                        break;
                    }
                }
            }
        }
        for buffer in &mut self.keep_alive {
            for (old, new) in replacements {
                if buffer.same_handle(old) {
                    *buffer = new.clone();
                    break;
                }
            }
        }
    }
}

enum Transcription {
    #[allow(dead_code)] // ICB 仅供显式诊断测试，生产路径使用 Flat。
    Icb {
        icb: IndirectCommandBuffer,
        blob: Buffer,
        blob_next: usize,
    },
    Flat(Vec<RecordedComputeOp>),
}

enum CurrentCommand {
    Icb(IndirectComputeCommand),
    Flat(RecordedComputeOp),
}

impl Transcriber {
    /// 开始 ICB 转录:镜像到 max_commands 条间接命令(Apple 驱动实测开销病态,仅诊断用)。
    #[allow(dead_code)] // ICB 驱动成本诊断入口。
    pub fn begin(device: &Device, max_commands: usize, blob_capacity: usize) -> Result<(), String> {
        let icb = device.new_indirect_command_buffer(max_commands)?;
        let blob = device.new_buffer(blob_capacity as u64, MTLResourceOptions::StorageModeShared);
        TRANSCRIBER.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_some() {
                return Err("转录器已在进行,不支持嵌套".to_owned());
            }
            *slot = Some(Self { transcription: Transcription::Icb { icb, blob, blob_next: 0 }, current: None, command_count: 0, keep_alive: Vec::new() });
            Ok(())
        })
    }

    /// 开始平铺转录:dispatch 记为普通调用表,重放走 encoder 重编码(无 ICB 依赖)。
    pub fn begin_flat() -> Result<(), String> {
        TRANSCRIBER.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_some() {
                return Err("转录器已在进行,不支持嵌套".to_owned());
            }
            *slot = Some(Self { transcription: Transcription::Flat(Vec::new()), current: None, command_count: 0, keep_alive: Vec::new() });
            Ok(())
        })
    }

    /// 结束平铺转录并取回命令表;返回 None 表示当前没有进行中的转录。
    #[allow(dead_code)] // 平铺转录诊断入口。
    pub fn end_flat() -> Option<Self> {
        Self::end()
    }

    /// 取出平铺命令表(仅 Flat 模式)。
    pub fn into_command_list(self) -> Result<CommandList, String> {
        match self.transcription {
            Transcription::Flat(ops) => Ok(CommandList { ops, keep_alive: self.keep_alive }),
            Transcription::Icb { .. } => Err("ICB 转录不产出命令表,用 into_keep_alive".to_owned()),
        }
    }

    /// 结束转录并取回结果;返回 None 表示当前没有进行中的转录。
    pub fn end() -> Option<Self> {
        TRANSCRIBER.with(|cell| cell.borrow_mut().take())
    }

    fn is_active() -> bool {
        TRANSCRIBER.with(|cell| cell.borrow().is_some())
    }

    /// 激活时对转录器执行 f;未激活返回 None(调用方走真实 Metal 路径)。
    fn with<T>(f: impl FnOnce(&mut Transcriber) -> T) -> Option<T> {
        TRANSCRIBER.with(|cell| {
            let mut slot = cell.borrow_mut();
            slot.as_mut().map(f)
        })
    }

    fn begin_command(&mut self, pipeline: &ComputePipelineState) {
        assert!(self.current.is_none(), "转录:上一条命令缺少 dispatch,或一次 encoder 编码了多个 kernel");
        match &mut self.transcription {
            Transcription::Icb { icb, .. } => {
                assert!(self.command_count < icb.size(), "转录:命令数超过 ICB 容量 {}", icb.size());
                let command = icb.compute_command(self.command_count);
                command.set_pipeline(pipeline);
                self.current = Some(CurrentCommand::Icb(command));
            }
            Transcription::Flat(ops) => {
                let _ = ops;
                self.current = Some(CurrentCommand::Flat(RecordedComputeOp { pipeline: pipeline.clone(), buffers: Vec::new(), constants: Vec::new(), groups: MTLSize::new(0, 1, 1), threads: MTLSize::new(0, 1, 1) }));
            }
        }
    }

    fn bind_buffer(&mut self, index: u64, buffer: &Buffer, offset: u64) {
        match self.current.as_mut().expect("转录:set_buffer 出现在 set_compute_pipeline_state 之前") {
            CurrentCommand::Icb(command) => command.set_kernel_buffer(index, buffer, offset),
            CurrentCommand::Flat(op) => op.buffers.push((index, buffer.clone(), offset)),
        }
        self.keep_alive.push(buffer.clone());
    }

    fn bind_bytes(&mut self, index: u64, length: u64, bytes: *const c_void) {
        match self.current.as_mut().expect("转录:set_bytes 出现在 set_compute_pipeline_state 之前") {
            CurrentCommand::Icb(command) => {
                // 16 字节对齐追加进常驻标量区;录制期内容不变,重放时无需更新
                let Transcription::Icb { blob, blob_next, .. } = &mut self.transcription else { unreachable!("ICB 命令只属于 ICB 转录") };
                let offset = (*blob_next + 15) & !15;
                let end = offset + length as usize;
                assert!(end <= blob.length() as usize, "转录:标量区 {} 字节耗尽", blob.length());
                unsafe { std::ptr::copy_nonoverlapping(bytes as *const u8, (blob.contents() as *mut u8).add(offset), length as usize) };
                *blob_next = end;
                command.set_kernel_buffer(index, blob, offset as u64);
            }
            CurrentCommand::Flat(op) => op.constants.push((index, unsafe { std::slice::from_raw_parts(bytes as *const u8, length as usize) }.to_vec())),
        }
    }

    fn dispatch(&mut self, groups: MTLSize, threads: MTLSize) {
        match self.current.take().expect("转录:dispatch 出现在 set_compute_pipeline_state 之前") {
            CurrentCommand::Icb(command) => command.dispatch(groups, threads),
            CurrentCommand::Flat(mut op) => {
                op.groups = groups;
                op.threads = threads;
                let Transcription::Flat(ops) = &mut self.transcription else { unreachable!("平铺命令只属于平铺转录") };
                ops.push(op);
            }
        }
        self.command_count += 1;
    }

    #[cfg(test)]
    pub fn command_count(&self) -> usize {
        self.command_count
    }

    /// 取出 keep-alive 列表与自身:绑定的 buffer 必须与录制产物同寿命,防止录制时
    /// 的临时张量被池回收导致重放读到悬空地址。
    #[allow(dead_code)] // ICB 驱动成本诊断入口。
    pub fn into_keep_alive(self) -> (IndirectCommandBuffer, usize, Vec<Buffer>) {
        match self.transcription {
            Transcription::Icb { icb, .. } => (icb, self.command_count, self.keep_alive),
            Transcription::Flat(_) => panic!("平铺转录用 into_command_list"),
        }
    }
}
