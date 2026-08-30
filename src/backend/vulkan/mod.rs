//! Android Vulkan Compute 后端。

mod dense;
mod gated_delta_net;
mod gqa;
mod iq4_xs;
mod layout;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8_0;
mod q8_1;
mod quant_gemm;
mod rope;
mod sample;
mod tensor;
mod w8a16;

pub use gated_delta_net::VulkanGatedDeltaNetStorage;
pub use gqa::VulkanKvCache;
pub use q4_k::Q4K_BLOCK_BYTES;

use std::{
    collections::HashMap,
    ops::Deref,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use wgpu::util::DeviceExt;

use super::{Backend, BackendError, BackendResources, GqaPrefillBackend, LinearWeight};
use crate::moe::Activation;
use crate::weight::format::quantization::QuantizedMatrixRef;

pub struct VulkanContext {
    device: wgpu::Device,
    queue: VulkanQueue,
    decode_resources: Mutex<VulkanDecodeResources>,
    decode_embedding_buffers: Mutex<HashMap<u64, wgpu::Buffer>>,
    rope_buffers: Mutex<HashMap<(usize, usize, usize, (u32, u32), (u32, u32)), (wgpu::Buffer, wgpu::Buffer)>>,
    adapter_info: wgpu::AdapterInfo,
    packed_i8_dot: bool,
    iq4_xs_pipeline: wgpu::ComputePipeline,
    iq4_xs_gated_pipeline: wgpu::ComputePipeline,
    iq4_xs_prefill_pipeline: Option<wgpu::ComputePipeline>,
    iq4_xs_gated_prefill_pipeline: Option<wgpu::ComputePipeline>,
    q4_k_pipeline: wgpu::ComputePipeline,
    q4_k_gated_pipeline: wgpu::ComputePipeline,
    q4_k_prefill_pipeline: Option<wgpu::ComputePipeline>,
    q4_k_gated_prefill_pipeline: Option<wgpu::ComputePipeline>,
    quant_gemm: Option<VulkanQuantGemm>,
    q5_k_pipeline: wgpu::ComputePipeline,
    q5_k_gated_pipeline: wgpu::ComputePipeline,
    q5_k_prefill_pipeline: Option<wgpu::ComputePipeline>,
    q5_k_gated_prefill_pipeline: Option<wgpu::ComputePipeline>,
    q6_k_pipeline: wgpu::ComputePipeline,
    q6_k_prefill_pipeline: Option<wgpu::ComputePipeline>,
    q6_k_embedding_pipeline: wgpu::ComputePipeline,
    q8_0_pipeline: wgpu::ComputePipeline,
    q8_1_pipeline: wgpu::ComputePipeline,
    q5_k_q8_1_pipeline: wgpu::ComputePipeline,
    w8a16_pipeline: wgpu::ComputePipeline,
    w8a16_embedding_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    add_gemma_rmsnorm_pipeline: wgpu::ComputePipeline,
    add_pipeline: wgpu::ComputePipeline,
    add_scaled_pipeline: wgpu::ComputePipeline,
    sigmoid_gate_pipeline: wgpu::ComputePipeline,
    silu_mul_pipeline: wgpu::ComputePipeline,
    f32_pipeline: wgpu::ComputePipeline,
    concat_pipeline: wgpu::ComputePipeline,
    slice_pipeline: wgpu::ComputePipeline,
    interleaved_pipeline: wgpu::ComputePipeline,
    rope_pipeline: wgpu::ComputePipeline,
    argmax_pipeline: wgpu::ComputePipeline,
    gqa_pipeline: wgpu::ComputePipeline,
    gqa_q8_pipeline: wgpu::ComputePipeline,
    gqa_q8_append_pipeline: wgpu::ComputePipeline,
    gdn_conv_pipeline: wgpu::ComputePipeline,
    gdn_norm_qk_pipeline: wgpu::ComputePipeline,
    gdn_recurrent_pipeline: wgpu::ComputePipeline,
    gdn_norm_gate_pipeline: wgpu::ComputePipeline,
}

struct VulkanQuantGemm {
    pack_pipeline: wgpu::ComputePipeline,
    q6_k_pack_pipeline: wgpu::ComputePipeline,
    iq4_xs_pipeline: wgpu::ComputePipeline,
    q4_k_pipeline: wgpu::ComputePipeline,
    q4_k_gated_pipeline: wgpu::ComputePipeline,
    q5_k_pipeline: wgpu::ComputePipeline,
    q6_k_pipeline: wgpu::ComputePipeline,
    scratch: Mutex<[Option<wgpu::Buffer>; 2]>,
}

#[derive(Default)]
struct VulkanDecodeResources {
    active: bool,
    layer: usize,
    // Decode 图每层固定；按“层 + 用途 + 同用途序号”保留资源，既避免跨算子
    // 误复用，也允许相邻 token 直接复用相同 buffer/bind group。
    output_slots: HashMap<(&'static str, u64), usize>,
    uniform_slots: HashMap<&'static str, usize>,
    bind_group_slots: HashMap<&'static str, usize>,
    outputs: HashMap<(usize, &'static str, u64, usize), wgpu::Buffer>,
    uniforms: HashMap<(usize, &'static str, usize), (wgpu::Buffer, Vec<u8>)>,
    bind_groups: HashMap<(usize, &'static str, usize), wgpu::BindGroup>,
    output_hits: u64,
    output_misses: u64,
    uniform_hits: u64,
    uniform_misses: u64,
    bind_group_hits: u64,
    bind_group_misses: u64,
}

impl VulkanDecodeResources {
    fn begin_layer(&mut self) {
        if self.active {
            self.layer += 1;
        } else {
            self.active = true;
            self.layer = 0;
        }
        self.output_slots.clear();
        self.uniform_slots.clear();
        self.bind_group_slots.clear();
    }

    fn finish(&mut self) {
        self.active = false;
        self.layer = 0;
    }

    fn take_profile(&mut self) -> [u64; 6] {
        let profile = [self.output_hits, self.output_misses, self.uniform_hits, self.uniform_misses, self.bind_group_hits, self.bind_group_misses];
        self.output_hits = 0;
        self.output_misses = 0;
        self.uniform_hits = 0;
        self.uniform_misses = 0;
        self.bind_group_hits = 0;
        self.bind_group_misses = 0;
        profile
    }
}

struct VulkanQueue {
    inner: wgpu::Queue,
    batch: Mutex<VulkanQueueBatch>,
    logical_submit_count: AtomicU64,
    driver_submit_count: AtomicU64,
    submit_nanos: AtomicU64,
}

#[derive(Default)]
struct VulkanQueueBatch {
    active: bool,
    max_pending: Option<usize>,
    pending: Vec<wgpu::CommandBuffer>,
}

const DECODE_BATCH_MAX_COMMAND_BUFFERS: usize = 2;

impl VulkanQueue {
    fn new(inner: wgpu::Queue) -> Self {
        Self { inner, batch: Mutex::new(VulkanQueueBatch::default()), logical_submit_count: AtomicU64::new(0), driver_submit_count: AtomicU64::new(0), submit_nanos: AtomicU64::new(0) }
    }

    fn submit<I>(&self, command_buffers: I)
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        let command_buffers = command_buffers.into_iter().collect::<Vec<_>>();
        self.logical_submit_count.fetch_add(1, Ordering::Relaxed);
        let mut batch = self.batch.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if batch.active {
            batch.pending.extend(command_buffers);
            if batch.max_pending.is_none_or(|limit| batch.pending.len() < limit) {
                return;
            }
            let pending = std::mem::take(&mut batch.pending);
            drop(batch);
            self.submit_now(pending);
            return;
        }
        drop(batch);
        self.submit_now(command_buffers);
    }

    fn begin_batch(&self) {
        let mut batch = self.batch.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if !batch.active {
            batch.active = true;
            batch.max_pending = None;
        }
    }

    fn begin_decode_batch(&self) {
        let mut batch = self.batch.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if !batch.active {
            batch.active = true;
            batch.max_pending = Some(DECODE_BATCH_MAX_COMMAND_BUFFERS);
        }
    }

    fn finish_batch(&self) {
        self.flush_batch();
        let mut batch = self.batch.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        batch.active = false;
        batch.max_pending = None;
    }

    fn flush_batch(&self) {
        let pending = {
            let mut batch = self.batch.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut batch.pending)
        };
        if !pending.is_empty() {
            self.submit_now(pending);
        }
    }

    fn submit_now(&self, command_buffers: Vec<wgpu::CommandBuffer>) {
        let started = Instant::now();
        self.inner.submit(command_buffers);
        self.driver_submit_count.fetch_add(1, Ordering::Relaxed);
        self.submit_nanos.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    fn take_profile(&self) -> (u64, u64, Duration) {
        let logical = self.logical_submit_count.swap(0, Ordering::Relaxed);
        let driver = self.driver_submit_count.swap(0, Ordering::Relaxed);
        let nanos = self.submit_nanos.swap(0, Ordering::Relaxed);
        (logical, driver, Duration::from_nanos(nanos))
    }
}

impl Deref for VulkanQueue {
    type Target = wgpu::Queue;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Clone)]
pub struct VulkanTensor {
    buffer: wgpu::Buffer,
    pub rows: usize,
    pub cols: usize,
}

pub struct VulkanQ8_1 {
    codes: wgpu::Buffer,
    scales: wgpu::Buffer,
    sums: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

pub enum VulkanWeight {
    Iq4Xs { buffer: wgpu::Buffer, rows: usize, cols: usize },
    Q4K { buffer: wgpu::Buffer, rows: usize, cols: usize },
    Q5K { buffer: wgpu::Buffer, rows: usize, cols: usize },
    Q6K { chunks: Vec<(wgpu::Buffer, usize)>, rows: usize, cols: usize },
    Q8_0 { buffer: wgpu::Buffer, rows: usize, cols: usize },
    W8A16 { chunks: Vec<(wgpu::Buffer, wgpu::Buffer, usize)>, rows: usize, cols: usize, group_size: usize },
    F32 { buffer: wgpu::Buffer, rows: usize, cols: usize },
}

impl VulkanWeight {
    pub fn allocated_bytes(&self) -> u64 {
        match self {
            Self::Iq4Xs { buffer, .. } | Self::Q4K { buffer, .. } | Self::Q5K { buffer, .. } | Self::Q8_0 { buffer, .. } | Self::F32 { buffer, .. } => buffer.size(),
            Self::Q6K { chunks, .. } => chunks.iter().map(|(buffer, _)| buffer.size()).sum(),
            Self::W8A16 { chunks, .. } => chunks.iter().map(|(packed, scales, _)| packed.size() + scales.size()).sum(),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Iq4Xs { rows, .. } | Self::Q4K { rows, .. } | Self::Q5K { rows, .. } | Self::Q6K { rows, .. } | Self::Q8_0 { rows, .. } | Self::W8A16 { rows, .. } | Self::F32 { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Iq4Xs { cols, .. } | Self::Q4K { cols, .. } | Self::Q5K { cols, .. } | Self::Q6K { cols, .. } | Self::Q8_0 { cols, .. } | Self::W8A16 { cols, .. } | Self::F32 { cols, .. } => *cols,
        }
    }
}

impl VulkanContext {
    pub fn new() -> Result<Self, BackendError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, BackendError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..wgpu::InstanceDescriptor::new_without_display_handle() });
        let packed_i8_dot = instance.wgsl_language_features().contains(wgpu::WgslLanguageFeatures::Packed4x8IntegerDotProduct);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: None, apply_limit_buckets: false })
            .await
            .map_err(|error| compute(format!("找不到 Android Vulkan adapter: {error}")))?;
        let adapter_info = adapter.get_info();
        // 64 个 invocation 的 workgroup 在 Adreno 上正好覆盖一个 subgroup；其他宽度
        // 不能直接用 subgroupAdd 代替整个 workgroup 的归约，因此保留原 GEMV 回退。
        let subgroup64 = adapter.features().contains(wgpu::Features::SUBGROUP) && adapter_info.subgroup_min_size == 64;
        let required_features = if subgroup64 { wgpu::Features::SUBGROUP } else { wgpu::Features::empty() };
        let required_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("zLLM Android Vulkan"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(|error| compute(format!("创建 Android Vulkan device 失败: {error}")))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("IQ4_XS GEMV"), source: wgpu::ShaderSource::Wgsl(iq4_xs::SHADER.into()) });
        let iq4_xs_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("IQ4_XS GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("IQ4_XS gated GEMV"), source: wgpu::ShaderSource::Wgsl(iq4_xs::GATED_SHADER.into()) });
        let iq4_xs_gated_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("IQ4_XS gated GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let iq4_xs_prefill_pipeline = subgroup64.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("IQ4_XS subgroup prefill GEMV"), source: wgpu::ShaderSource::Wgsl(iq4_xs::PREFILL_SHADER.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("IQ4_XS subgroup prefill GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        });
        let iq4_xs_gated_prefill_pipeline = subgroup64.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("IQ4_XS gated subgroup prefill GEMV"), source: wgpu::ShaderSource::Wgsl(iq4_xs::PREFILL_GATED_SHADER.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("IQ4_XS gated subgroup prefill GEMV"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q4_K GEMV"), source: wgpu::ShaderSource::Wgsl(q4_k::SHADER.into()) });
        let q4_k_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q4_K GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q4_K gated GEMV"), source: wgpu::ShaderSource::Wgsl(q4_k::GATED_SHADER.into()) });
        let q4_k_gated_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q4_K gated GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let q4_k_prefill_pipeline = subgroup64.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q4_K subgroup prefill GEMV"), source: wgpu::ShaderSource::Wgsl(q4_k::PREFILL_SHADER.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q4_K subgroup prefill GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        });
        let q4_k_gated_prefill_pipeline = subgroup64.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q4_K gated subgroup prefill GEMV"), source: wgpu::ShaderSource::Wgsl(q4_k::PREFILL_GATED_SHADER.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q4_K gated subgroup prefill GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        });
        let quant_gemm = subgroup64.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("quantized GEMM pack"), source: wgpu::ShaderSource::Wgsl(quant_gemm::PACK_SHADER.into()) });
            let pack_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("quantized GEMM pack"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q6_K GEMM byte pack"), source: wgpu::ShaderSource::Wgsl(q6_k::PACK_GEMM_SHADER.into()) });
            let q6_k_pack_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q6_K GEMM byte pack"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("IQ4_XS 8x8 GEMM"), source: wgpu::ShaderSource::Wgsl(iq4_xs::GEMM_SHADER.into()) });
            let iq4_xs_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("IQ4_XS 8x8 GEMM"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q4_K 8x8 GEMM"), source: wgpu::ShaderSource::Wgsl(q4_k::GEMM_SHADER.into()) });
            let q4_k_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q4_K 8x8 GEMM"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q4_K gated 8x8 GEMM"), source: wgpu::ShaderSource::Wgsl(q4_k::GEMM_GATED_SHADER.into()) });
            let q4_k_gated_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q4_K gated 8x8 GEMM"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q5_K 8x8 GEMM"), source: wgpu::ShaderSource::Wgsl(q5_k::GEMM_SHADER.into()) });
            let q5_k_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q5_K 8x8 GEMM"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q6_K 8x8 GEMM"), source: wgpu::ShaderSource::Wgsl(q6_k::GEMM_SHADER.into()) });
            let q6_k_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q6_K 8x8 GEMM"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            VulkanQuantGemm { pack_pipeline, q6_k_pack_pipeline, iq4_xs_pipeline, q4_k_pipeline, q4_k_gated_pipeline, q5_k_pipeline, q6_k_pipeline, scratch: Mutex::new([None, None]) }
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q5_K GEMV"), source: wgpu::ShaderSource::Wgsl(q5_k::SHADER.into()) });
        let q5_k_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q5_K GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q5_K gated GEMV"), source: wgpu::ShaderSource::Wgsl(q5_k::GATED_SHADER.into()) });
        let q5_k_gated_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q5_K gated GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let q5_k_prefill_pipeline = subgroup64.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q5_K subgroup prefill GEMV"), source: wgpu::ShaderSource::Wgsl(q5_k::PREFILL_SHADER.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q5_K subgroup prefill GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        });
        let q5_k_gated_prefill_pipeline = subgroup64.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q5_K gated subgroup prefill GEMV"), source: wgpu::ShaderSource::Wgsl(q5_k::PREFILL_GATED_SHADER.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q5_K gated subgroup prefill GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q6_K GEMV"), source: wgpu::ShaderSource::Wgsl(q6_k::SHADER.into()) });
        let q6_k_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q6_K GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let q6_k_prefill_pipeline = subgroup64.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q6_K subgroup prefill GEMV"), source: wgpu::ShaderSource::Wgsl(q6_k::PREFILL_SHADER.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q6_K subgroup prefill GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q6_K embedding"), source: wgpu::ShaderSource::Wgsl(q6_k::EMBEDDING_SHADER.into()) });
        let q6_k_embedding_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q6_K embedding"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q8_0 GEMV"), source: wgpu::ShaderSource::Wgsl(q8_0::SHADER.into()) });
        let q8_0_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q8_0 GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q8_1 activation"), source: wgpu::ShaderSource::Wgsl(q8_1::SHADER.into()) });
        let q8_1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q8_1 activation"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Q5_K × Q8_1 GEMV"), source: wgpu::ShaderSource::Wgsl(q5_k::Q8_1_SHADER.into()) });
        let q5_k_q8_1_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Q5_K × Q8_1 GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("W8A16 GEMV"), source: wgpu::ShaderSource::Wgsl(w8a16::SHADER.into()) });
        let w8a16_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("W8A16 GEMV"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("W8A16 embedding"), source: wgpu::ShaderSource::Wgsl(w8a16::EMBEDDING_SHADER.into()) });
        let w8a16_embedding_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("W8A16 embedding"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("RMSNorm"), source: wgpu::ShaderSource::Wgsl(tensor::RMSNORM_SHADER.into()) });
        let rmsnorm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("RMSNorm"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Add GemmaRMSNorm pair"), source: wgpu::ShaderSource::Wgsl(tensor::ADD_GEMMA_RMSNORM_SHADER.into()) });
        let add_gemma_rmsnorm_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Add GemmaRMSNorm pair"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Vulkan elementwise"), source: wgpu::ShaderSource::Wgsl(tensor::ELEMENTWISE_SHADER.into()) });
        let pipeline =
            |label, entry_point| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: None, module: &shader, entry_point: Some(entry_point), compilation_options: Default::default(), cache: None });
        let add_pipeline = pipeline("Vulkan add", "add");
        let add_scaled_pipeline = pipeline("Vulkan add_scaled", "add_scaled");
        let sigmoid_gate_pipeline = pipeline("Vulkan sigmoid_gate", "sigmoid_gate");
        let silu_mul_pipeline = pipeline("Vulkan silu_mul", "silu_mul");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("F32 linear"), source: wgpu::ShaderSource::Wgsl(dense::F32_SHADER.into()) });
        let f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("F32 linear"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let pipeline = |label: &'static str, source: &'static str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(source.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        };
        let concat_pipeline = pipeline("Vulkan concat", layout::CONCAT_SHADER);
        let slice_pipeline = pipeline("Vulkan slice", layout::SLICE_SHADER);
        let interleaved_pipeline = pipeline("Vulkan interleaved", layout::INTERLEAVED_SHADER);
        let rope_pipeline = pipeline("Vulkan RoPE", rope::SHADER);
        let argmax_pipeline = pipeline("Vulkan argmax", sample::ARGMAX_SHADER);
        let gqa_pipeline = pipeline("Vulkan GQA", gqa::SHADER);
        let gqa_q8_pipeline = pipeline("Vulkan GQA Q8G64", gqa::Q8_SHADER);
        let gqa_q8_append_pipeline = pipeline("Vulkan GQA Q8G64 append", gqa::Q8_APPEND_SHADER);
        let gdn_conv_pipeline = pipeline("Vulkan Gated DeltaNet conv", gated_delta_net::CONV_SHADER);
        let gdn_norm_qk_pipeline = pipeline("Vulkan Gated DeltaNet QK norm", gated_delta_net::NORM_QK_SHADER);
        let gdn_recurrent_pipeline = pipeline("Vulkan Gated DeltaNet recurrent", gated_delta_net::RECURRENT_SHADER);
        let gdn_norm_gate_pipeline = pipeline("Vulkan Gated DeltaNet output", gated_delta_net::NORM_GATE_SHADER);
        Ok(Self {
            device,
            queue: VulkanQueue::new(queue),
            decode_resources: Mutex::new(VulkanDecodeResources::default()),
            decode_embedding_buffers: Mutex::new(HashMap::new()),
            rope_buffers: Mutex::new(HashMap::new()),
            adapter_info,
            packed_i8_dot,
            iq4_xs_pipeline,
            iq4_xs_gated_pipeline,
            iq4_xs_prefill_pipeline,
            iq4_xs_gated_prefill_pipeline,
            q4_k_pipeline,
            q4_k_gated_pipeline,
            q4_k_prefill_pipeline,
            q4_k_gated_prefill_pipeline,
            quant_gemm,
            q5_k_pipeline,
            q5_k_gated_pipeline,
            q5_k_prefill_pipeline,
            q5_k_gated_prefill_pipeline,
            q6_k_pipeline,
            q6_k_prefill_pipeline,
            q6_k_embedding_pipeline,
            q8_0_pipeline,
            q8_1_pipeline,
            q5_k_q8_1_pipeline,
            w8a16_pipeline,
            w8a16_embedding_pipeline,
            rmsnorm_pipeline,
            add_gemma_rmsnorm_pipeline,
            add_pipeline,
            add_scaled_pipeline,
            sigmoid_gate_pipeline,
            silu_mul_pipeline,
            f32_pipeline,
            concat_pipeline,
            slice_pipeline,
            interleaved_pipeline,
            rope_pipeline,
            argmax_pipeline,
            gqa_pipeline,
            gqa_q8_pipeline,
            gqa_q8_append_pipeline,
            gdn_conv_pipeline,
            gdn_norm_qk_pipeline,
            gdn_recurrent_pipeline,
            gdn_norm_gate_pipeline,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    pub fn supports_packed_i8_dot(&self) -> bool {
        self.packed_i8_dot
    }

    pub fn take_submit_profile(&self) -> (u64, u64, Duration) {
        self.queue.take_profile()
    }

    pub fn take_decode_resource_profile(&self) -> [u64; 6] {
        self.decode_resources.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take_profile()
    }

    fn begin_decode_resources(&self) {
        self.decode_resources.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).begin_layer();
    }

    fn finish_decode_resources(&self) {
        self.decode_resources.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).finish();
    }

    fn rope_buffers(&self, cosine: &[f32], sine: &[f32]) -> (wgpu::Buffer, wgpu::Buffer) {
        // 裸指针作 key 在宿主表释放、新表复用同一地址时会静默命中旧 buffer;
        // 加入首尾元素位模式指纹后,误命中需要地址复用且内容边界全同。
        let key = (
            cosine.as_ptr() as usize,
            sine.as_ptr() as usize,
            cosine.len(),
            (cosine.first().map_or(0, |v| v.to_bits()), cosine.last().map_or(0, |v| v.to_bits())),
            (sine.first().map_or(0, |v| v.to_bits()), sine.last().map_or(0, |v| v.to_bits())),
        );
        let mut buffers = self.rope_buffers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((cosine, sine)) = buffers.get(&key) {
            return (cosine.clone(), sine.clone());
        }
        let cosine_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan RoPE cosine"), contents: f32_bytes(cosine), usage: wgpu::BufferUsages::STORAGE });
        let sine_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan RoPE sine"), contents: f32_bytes(sine), usage: wgpu::BufferUsages::STORAGE });
        buffers.insert(key, (cosine_buffer.clone(), sine_buffer.clone()));
        (cosine_buffer, sine_buffer)
    }

    fn decode_embedding_buffer(&self, elements: usize) -> Result<wgpu::Buffer, BackendError> {
        let bytes = elements.checked_mul(size_of::<f32>()).ok_or_else(|| compute("Vulkan decode embedding 大小溢出"))? as u64;
        let mut buffers = self.decode_embedding_buffers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(buffer) = buffers.get(&bytes) {
            return Ok(buffer.clone());
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Vulkan fixed decode embedding"), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        buffers.insert(bytes, buffer.clone());
        Ok(buffer)
    }

    pub fn tensor_from_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<VulkanTensor, BackendError> {
        validate_elements(values.len(), rows, cols, "Vulkan tensor")?;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan F32 tensor"), contents: f32_bytes(values), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });
        Ok(VulkanTensor { buffer, rows, cols })
    }

    pub fn prepare_q4_k(&self, packed: &[u8], rows: usize, cols: usize) -> Result<VulkanWeight, BackendError> {
        q4_k::validate_weight(packed, rows, cols)?;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan resident Q4_K weight"), contents: packed, usage: wgpu::BufferUsages::STORAGE });
        Ok(VulkanWeight::Q4K { buffer, rows, cols })
    }

    pub fn prepare_iq4_xs(&self, packed: &[u8], rows: usize, cols: usize) -> Result<VulkanWeight, BackendError> {
        iq4_xs::validate_weight(packed, rows, cols)?;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan resident IQ4_XS weight"), contents: packed, usage: wgpu::BufferUsages::STORAGE });
        Ok(VulkanWeight::Iq4Xs { buffer, rows, cols })
    }

    pub fn prepare_q5_k(&self, packed: &[u8], rows: usize, cols: usize) -> Result<VulkanWeight, BackendError> {
        q5_k::validate_weight(packed, rows, cols)?;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan resident Q5_K weight"), contents: packed, usage: wgpu::BufferUsages::STORAGE });
        Ok(VulkanWeight::Q5K { buffer, rows, cols })
    }

    pub fn prepare_q6_k(&self, packed: &[u8], rows: usize, cols: usize) -> Result<VulkanWeight, BackendError> {
        q6_k::validate_weight(packed, rows, cols)?;
        let row_bytes = packed.len() / rows;
        let binding_limit = self.device.limits().max_storage_buffer_binding_size as usize;
        let rows_per_chunk = (binding_limit / row_bytes).max(1);
        let chunks = packed
            .chunks(row_bytes * rows_per_chunk)
            .map(|bytes| {
                let chunk_rows = bytes.len() / row_bytes;
                let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan resident Q6_K weight chunk"), contents: bytes, usage: wgpu::BufferUsages::STORAGE });
                (buffer, chunk_rows)
            })
            .collect();
        Ok(VulkanWeight::Q6K { chunks, rows, cols })
    }

    pub fn prepare_q8_0(&self, packed: &[u8], rows: usize, cols: usize) -> Result<VulkanWeight, BackendError> {
        q8_0::validate_weight(packed, rows, cols)?;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan resident Q8_0 weight"), contents: packed, usage: wgpu::BufferUsages::STORAGE });
        Ok(VulkanWeight::Q8_0 { buffer, rows, cols })
    }

    fn prepare_w8a16(&self, matrix: &crate::weight::format::quantization::W8A16Matrix, rows: usize, cols: usize) -> Result<VulkanWeight, BackendError> {
        w8a16::validate_weight(matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), rows, cols)?;
        let packed_row_bytes = cols;
        let scale_row_bytes = cols / matrix.group_size() * 2;
        let binding_limit = self.device.limits().max_storage_buffer_binding_size as usize;
        let rows_per_chunk = (binding_limit / packed_row_bytes).max(1);
        let chunks = matrix
            .packed()
            .chunks(packed_row_bytes * rows_per_chunk)
            .zip(matrix.scales().chunks(scale_row_bytes * rows_per_chunk))
            .map(|(packed, scales)| {
                let chunk_rows = packed.len() / packed_row_bytes;
                let packed = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan resident W8A16 weight"), contents: packed, usage: wgpu::BufferUsages::STORAGE });
                let scales = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan resident W8A16 scales"), contents: scales, usage: wgpu::BufferUsages::STORAGE });
                (packed, scales, chunk_rows)
            })
            .collect();
        Ok(VulkanWeight::W8A16 { chunks, rows, cols, group_size: matrix.group_size() })
    }

    /// Decode GEMV：输入、权重和输出都留在设备，只有显式 readback 才同步到 host。
    pub fn linear_q4_k(&self, input: &VulkanTensor, weight: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Q4K { buffer, rows, cols } = weight else {
            return Err(compute("Vulkan linear_q4_k 收到非 Q4_K 权重"));
        };
        if input.rows >= quant_gemm::MIN_TOKEN_ROWS && self.quant_gemm.is_some() {
            return self.linear_q4_k_gemm(input, buffer, *rows, *cols);
        }
        if input.rows > 1 {
            if let Some(pipeline) = &self.q4_k_prefill_pipeline {
                return linear_quantized_tiled(self, pipeline, input, buffer, *rows, *cols, q4_k::PREFILL_TOKEN_TILE, "Q4_K subgroup prefill");
            }
        }
        linear_quantized(self, &self.q4_k_pipeline, input, buffer, *rows, *cols, "Q4_K")
    }

    fn linear_q4_k_gemm(&self, input: &VulkanTensor, weight: &wgpu::Buffer, rows: usize, cols: usize) -> Result<VulkanTensor, BackendError> {
        let pipeline = &self.quant_gemm.as_ref().ok_or_else(|| compute("Vulkan Q4_K GEMM capability 不可用"))?.q4_k_pipeline;
        self.linear_quantized_gemm(input, weight, rows, cols, Q4K_BLOCK_BYTES / 4, pipeline, "Vulkan Q4_K GEMM output")
    }

    fn linear_quantized_gemm(&self, input: &VulkanTensor, weight: &wgpu::Buffer, rows: usize, cols: usize, words_per_block: usize, pipeline: &wgpu::ComputePipeline, label: &'static str) -> Result<VulkanTensor, BackendError> {
        if input.rows == 0 || input.cols != cols {
            return Err(compute(format!("{label} shape 无效: input=[{},{}], weight=[{rows},{cols}]", input.rows, input.cols)));
        }
        let output = self.output_buffer(input.rows, rows, label)?;
        let mut pack_encoder = self.device.create_command_encoder(&Default::default());
        let packed = self.encode_quant_gemm_pack(&mut pack_encoder, weight, rows, cols, words_per_block, 0)?;
        self.queue.submit([pack_encoder.finish()]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let params = self.uniform_buffer("Vulkan quantized GEMM 参数", u32_bytes(&[rows as u32, cols as u32, input.rows as u32, 0]));
        let layout = pipeline.get_bind_group_layout(0);
        let group = self.bind_group("Vulkan quantized GEMM", &layout, &[binding(0, &packed), binding(1, &input.buffer), binding(2, &output), binding(3, &params)]);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(quant_gemm::ROW_TILE) as u32, input.rows.div_ceil(quant_gemm::TOKEN_TILE) as u32, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols: rows })
    }

    fn encode_quant_gemm_pack(&self, encoder: &mut wgpu::CommandEncoder, weight: &wgpu::Buffer, rows: usize, cols: usize, words_per_block: usize, slot: usize) -> Result<wgpu::Buffer, BackendError> {
        let gemm = self.quant_gemm.as_ref().ok_or_else(|| compute("Vulkan quantized GEMM pack capability 不可用"))?;
        let blocks = cols / 256;
        let padded_rows = rows.div_ceil(quant_gemm::ROW_TILE) * quant_gemm::ROW_TILE;
        let words_per_row = blocks.checked_mul(words_per_block).ok_or_else(|| compute("Vulkan quantized GEMM row 大小溢出"))?;
        let words = padded_rows.checked_mul(words_per_row).ok_or_else(|| compute("Vulkan quantized GEMM scratch 大小溢出"))?;
        let bytes = words.checked_mul(4).ok_or_else(|| compute("Vulkan quantized GEMM scratch 字节数溢出"))? as u64;
        let packed = self.quant_gemm_scratch(slot, bytes)?;
        let params = self.uniform_buffer("Vulkan quantized GEMM pack 参数", u32_bytes(&[rows as u32, words_per_row as u32, padded_rows as u32, 0]));
        let layout = gemm.pack_pipeline.get_bind_group_layout(0);
        let group = self.bind_group("Vulkan Q4_K GEMM pack", &layout, &[binding(0, weight), binding(1, &packed), binding(2, &params)]);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&gemm.pack_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(words.div_ceil(256) as u32, 1, 1);
        }
        Ok(packed)
    }

    fn quant_gemm_scratch(&self, slot: usize, bytes: u64) -> Result<wgpu::Buffer, BackendError> {
        let gemm = self.quant_gemm.as_ref().ok_or_else(|| compute("Vulkan quantized GEMM scratch capability 不可用"))?;
        let mut scratch = gemm.scratch.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = scratch.get_mut(slot).ok_or_else(|| compute(format!("Vulkan quantized GEMM scratch slot={slot} 越界")))?;
        if entry.as_ref().is_none_or(|buffer| buffer.size() < bytes) {
            *entry = Some(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Vulkan quantized GEMM packed scratch"), size: bytes, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }));
        }
        Ok(entry.as_ref().unwrap().clone())
    }

    pub fn linear_iq4_xs(&self, input: &VulkanTensor, weight: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Iq4Xs { buffer, rows, cols } = weight else {
            return Err(compute("Vulkan linear_iq4_xs 收到非 IQ4_XS 权重"));
        };
        if input.rows >= quant_gemm::MIN_TOKEN_ROWS {
            if let Some(gemm) = &self.quant_gemm {
                return self.linear_quantized_gemm(input, buffer, *rows, *cols, iq4_xs::BLOCK_BYTES / 4, &gemm.iq4_xs_pipeline, "Vulkan IQ4_XS GEMM output");
            }
        }
        if input.rows > 1 {
            if let Some(pipeline) = &self.iq4_xs_prefill_pipeline {
                return linear_quantized_tiled(self, pipeline, input, buffer, *rows, *cols, iq4_xs::PREFILL_TOKEN_TILE, "IQ4_XS subgroup prefill");
            }
        }
        linear_quantized(self, &self.iq4_xs_pipeline, input, buffer, *rows, *cols, "IQ4_XS")
    }

    pub fn linear_q5_k(&self, input: &VulkanTensor, weight: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Q5K { buffer, rows, cols } = weight else {
            return Err(compute("Vulkan linear_q5_k 收到非 Q5_K 权重"));
        };
        if input.rows >= quant_gemm::MIN_TOKEN_ROWS {
            if let Some(gemm) = &self.quant_gemm {
                return self.linear_quantized_gemm(input, buffer, *rows, *cols, q5_k::BLOCK_BYTES / 4, &gemm.q5_k_pipeline, "Vulkan Q5_K GEMM output");
            }
        }
        if input.rows > 1 {
            if let Some(pipeline) = &self.q5_k_prefill_pipeline {
                return linear_quantized_tiled(self, pipeline, input, buffer, *rows, *cols, q5_k::PREFILL_TOKEN_TILE, "Q5_K subgroup prefill");
            }
        }
        linear_quantized(self, &self.q5_k_pipeline, input, buffer, *rows, *cols, "Q5_K")
    }

    pub fn quantize_q8_1(&self, input: &VulkanTensor) -> Result<VulkanQ8_1, BackendError> {
        if input.rows == 0 || !input.cols.is_multiple_of(q8_1::GROUP_SIZE) {
            return Err(compute(format!("Vulkan Q8_1 input shape 无效: [{},{}] group={}", input.rows, input.cols, q8_1::GROUP_SIZE)));
        }
        let elements = input.rows.checked_mul(input.cols).ok_or_else(|| compute("Vulkan Q8_1 元素数溢出"))?;
        let groups = elements / q8_1::GROUP_SIZE;
        let codes = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Vulkan Q8_1 codes"), size: (elements / 4 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let scales = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Vulkan Q8_1 scales"), size: (groups * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let sums = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Vulkan Q8_1 sums"), size: (groups * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vulkan Q8_1"),
            layout: &self.q8_1_pipeline.get_bind_group_layout(0),
            entries: &[binding(0, &input.buffer), binding(1, &codes), binding(2, &scales), binding(3, &sums)],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.q8_1_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups as u32, 1, 1);
        drop(pass);
        self.queue.submit([encoder.finish()]);
        Ok(VulkanQ8_1 { codes, scales, sums, rows: input.rows, cols: input.cols })
    }

    pub fn linear_q5_k_q8_1(&self, input: &VulkanQ8_1, weight: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Q5K { buffer, rows, cols } = weight else {
            return Err(compute("Vulkan Q5_K × Q8_1 收到非 Q5_K 权重"));
        };
        if input.rows == 0 || input.cols != *cols {
            return Err(compute(format!("Vulkan Q5_K × Q8_1 shape 不一致: input=[{},{}] weight=[{},{}]", input.rows, input.cols, rows, cols)));
        }
        let output = self.output_buffer(input.rows, *rows, "Vulkan Q5_K × Q8_1 output")?;
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan Q5_K × Q8_1 参数"), contents: u32_bytes(&[*rows as u32, *cols as u32, 0, 0]), usage: wgpu::BufferUsages::UNIFORM });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vulkan Q5_K × Q8_1"),
            layout: &self.q5_k_q8_1_pipeline.get_bind_group_layout(0),
            entries: &[binding(0, buffer), binding(1, &input.codes), binding(2, &input.scales), binding(3, &input.sums), binding(4, &output), binding(5, &params)],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.q5_k_q8_1_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(*rows as u32, input.rows as u32, 1);
        drop(pass);
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols: *rows })
    }

    fn gated_linear_q5_k(&self, input: &VulkanTensor, gate: &VulkanWeight, up: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Q5K { buffer: gate, rows, cols } = gate else {
            return Err(compute("Vulkan gated Q5_K gate 权重类型错误"));
        };
        let VulkanWeight::Q5K { buffer: up, rows: up_rows, cols: up_cols } = up else {
            return Err(compute("Vulkan gated Q5_K up 权重类型错误"));
        };
        if (*rows, *cols) != (*up_rows, *up_cols) || input.rows == 0 || input.cols != *cols {
            return Err(compute(format!("Vulkan gated Q5_K shape 不一致: input=[{},{}] gate=[{},{}] up=[{},{}]", input.rows, input.cols, rows, cols, up_rows, up_cols)));
        }
        if input.rows >= quant_gemm::MIN_TOKEN_ROWS {
            if let Some(gemm) = &self.quant_gemm {
                let gate = self.linear_quantized_gemm(input, gate, *rows, *cols, q5_k::BLOCK_BYTES / 4, &gemm.q5_k_pipeline, "Vulkan Q5_K gate GEMM output")?;
                let up = self.linear_quantized_gemm(input, up, *rows, *cols, q5_k::BLOCK_BYTES / 4, &gemm.q5_k_pipeline, "Vulkan Q5_K up GEMM output")?;
                return self.silu_mul(&gate, &up);
            }
        }
        if input.rows > 1 {
            if let Some(pipeline) = &self.q5_k_gated_prefill_pipeline {
                return self.gated_linear_quantized(input, gate, up, *rows, *cols, pipeline, q5_k::PREFILL_TOKEN_TILE, "Vulkan Q5_K gated subgroup prefill");
            }
        }
        self.gated_linear_quantized(input, gate, up, *rows, *cols, &self.q5_k_gated_pipeline, 1, "Vulkan Q5_K gated")
    }

    fn gated_linear_q4_k(&self, input: &VulkanTensor, gate: &VulkanWeight, up: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Q4K { buffer: gate, rows, cols } = gate else {
            return Err(compute("Vulkan gated Q4_K gate 权重类型错误"));
        };
        let VulkanWeight::Q4K { buffer: up, rows: up_rows, cols: up_cols } = up else {
            return Err(compute("Vulkan gated Q4_K up 权重类型错误"));
        };
        if (*rows, *cols) != (*up_rows, *up_cols) || input.rows == 0 || input.cols != *cols {
            return Err(compute(format!("Vulkan gated Q4_K shape 不一致: input=[{},{}] gate=[{},{}] up=[{},{}]", input.rows, input.cols, rows, cols, up_rows, up_cols)));
        }
        if input.rows >= quant_gemm::MIN_TOKEN_ROWS && self.quant_gemm.is_some() {
            let gemm = self.quant_gemm.as_ref().unwrap();
            let output = self.output_buffer(input.rows, *rows, "Vulkan Q4_K gated GEMM output")?;
            let mut pack_encoder = self.device.create_command_encoder(&Default::default());
            let gate = self.encode_quant_gemm_pack(&mut pack_encoder, gate, *rows, *cols, Q4K_BLOCK_BYTES / 4, 0)?;
            let up = self.encode_quant_gemm_pack(&mut pack_encoder, up, *rows, *cols, Q4K_BLOCK_BYTES / 4, 1)?;
            self.queue.submit([pack_encoder.finish()]);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let params = self.uniform_buffer("Vulkan Q4_K gated GEMM 参数", u32_bytes(&[*rows as u32, *cols as u32, input.rows as u32, 0]));
            let layout = gemm.q4_k_gated_pipeline.get_bind_group_layout(0);
            let group = self.bind_group("Vulkan Q4_K gated GEMM", &layout, &[binding(0, &gate), binding(1, &up), binding(2, &input.buffer), binding(3, &output), binding(4, &params)]);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&gemm.q4_k_gated_pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(rows.div_ceil(quant_gemm::ROW_TILE) as u32, input.rows.div_ceil(quant_gemm::TOKEN_TILE) as u32, 1);
            }
            self.queue.submit([encoder.finish()]);
            return Ok(VulkanTensor { buffer: output, rows: input.rows, cols: *rows });
        }
        if input.rows > 1 {
            if let Some(pipeline) = &self.q4_k_gated_prefill_pipeline {
                return self.gated_linear_quantized(input, gate, up, *rows, *cols, pipeline, q4_k::PREFILL_TOKEN_TILE, "Vulkan Q4_K gated subgroup prefill");
            }
        }
        self.gated_linear_quantized(input, gate, up, *rows, *cols, &self.q4_k_gated_pipeline, 1, "Vulkan Q4_K gated")
    }

    fn gated_linear_iq4_xs(&self, input: &VulkanTensor, gate: &VulkanWeight, up: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Iq4Xs { buffer: gate, rows, cols } = gate else {
            return Err(compute("Vulkan gated IQ4_XS gate 权重类型错误"));
        };
        let VulkanWeight::Iq4Xs { buffer: up, rows: up_rows, cols: up_cols } = up else {
            return Err(compute("Vulkan gated IQ4_XS up 权重类型错误"));
        };
        if (*rows, *cols) != (*up_rows, *up_cols) || input.rows == 0 || input.cols != *cols {
            return Err(compute(format!("Vulkan gated IQ4_XS shape 不一致: input=[{},{}] gate=[{},{}] up=[{},{}]", input.rows, input.cols, rows, cols, up_rows, up_cols)));
        }
        if input.rows >= quant_gemm::MIN_TOKEN_ROWS {
            if let Some(gemm) = &self.quant_gemm {
                let gate = self.linear_quantized_gemm(input, gate, *rows, *cols, iq4_xs::BLOCK_BYTES / 4, &gemm.iq4_xs_pipeline, "Vulkan IQ4_XS gate GEMM output")?;
                let up = self.linear_quantized_gemm(input, up, *rows, *cols, iq4_xs::BLOCK_BYTES / 4, &gemm.iq4_xs_pipeline, "Vulkan IQ4_XS up GEMM output")?;
                return self.silu_mul(&gate, &up);
            }
        }
        if input.rows > 1 {
            if let Some(pipeline) = &self.iq4_xs_gated_prefill_pipeline {
                return self.gated_linear_quantized(input, gate, up, *rows, *cols, pipeline, iq4_xs::PREFILL_TOKEN_TILE, "Vulkan IQ4_XS gated subgroup prefill");
            }
        }
        self.gated_linear_quantized(input, gate, up, *rows, *cols, &self.iq4_xs_gated_pipeline, 1, "Vulkan IQ4_XS gated")
    }

    fn gated_linear_quantized(&self, input: &VulkanTensor, gate: &wgpu::Buffer, up: &wgpu::Buffer, rows: usize, cols: usize, pipeline: &wgpu::ComputePipeline, token_tile: usize, label: &'static str) -> Result<VulkanTensor, BackendError> {
        let output = self.output_buffer(input.rows, rows, label)?;
        let params = [rows as u32, cols as u32, input.rows as u32, 0];
        let params_buffer = self.uniform_buffer(label, u32_bytes(&params));
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group(label, &layout, &[binding(0, gate), binding(1, up), binding(2, &input.buffer), binding(3, &output), binding(4, &params_buffer)]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows as u32, input.rows.div_ceil(token_tile) as u32, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols: rows })
    }

    fn dual_linear_simple(&self, input: &VulkanTensor, first: &VulkanWeight, second: &VulkanWeight) -> Result<(VulkanTensor, VulkanTensor), BackendError> {
        let Some((first_pipeline, first_weight, first_rows, first_cols, first_token_tile)) = self.simple_linear_parts(first, input.rows) else {
            return Err(compute("Vulkan dual linear 不支持分块 Q6_K"));
        };
        let Some((second_pipeline, second_weight, second_rows, second_cols, second_token_tile)) = self.simple_linear_parts(second, input.rows) else {
            return Err(compute("Vulkan dual linear 不支持分块 Q6_K"));
        };
        if input.rows == 0 || input.cols != first_cols || input.cols != second_cols {
            return Err(compute(format!("Vulkan dual linear shape 不一致: input=[{},{}], first=[{first_rows},{first_cols}], second=[{second_rows},{second_cols}]", input.rows, input.cols)));
        }
        let first_output = self.output_buffer(input.rows, first_rows, "Vulkan dual linear first")?;
        let second_output = self.output_buffer(input.rows, second_rows, "Vulkan dual linear second")?;
        let make_params = |rows: usize, cols: usize| self.uniform_buffer("Vulkan dual linear 参数", u32_bytes(&[rows as u32, cols as u32, input.rows as u32, 0]));
        let first_params = make_params(first_rows, first_cols);
        let second_params = make_params(second_rows, second_cols);
        let first_layout = first_pipeline.get_bind_group_layout(0);
        let first_group = self.bind_group("Vulkan dual linear first", &first_layout, &[binding(0, first_weight), binding(1, &input.buffer), binding(2, &first_output), binding(3, &first_params)]);
        let second_layout = second_pipeline.get_bind_group_layout(0);
        let second_group = self.bind_group("Vulkan dual linear second", &second_layout, &[binding(0, second_weight), binding(1, &input.buffer), binding(2, &second_output), binding(3, &second_params)]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for (pipeline, group, rows, token_tile) in [(first_pipeline, &first_group, first_rows, first_token_tile), (second_pipeline, &second_group, second_rows, second_token_tile)] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(rows as u32, input.rows.div_ceil(token_tile) as u32, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok((VulkanTensor { buffer: first_output, rows: input.rows, cols: first_rows }, VulkanTensor { buffer: second_output, rows: input.rows, cols: second_rows }))
    }

    fn simple_linear_parts<'a>(&'a self, weight: &'a VulkanWeight, token_rows: usize) -> Option<(&'a wgpu::ComputePipeline, &'a wgpu::Buffer, usize, usize, usize)> {
        if token_rows > 1 {
            if let (VulkanWeight::Iq4Xs { buffer, rows, cols }, Some(pipeline)) = (weight, self.iq4_xs_prefill_pipeline.as_ref()) {
                return Some((pipeline, buffer, *rows, *cols, iq4_xs::PREFILL_TOKEN_TILE));
            }
            if let (VulkanWeight::Q4K { buffer, rows, cols }, Some(pipeline)) = (weight, self.q4_k_prefill_pipeline.as_ref()) {
                return Some((pipeline, buffer, *rows, *cols, q4_k::PREFILL_TOKEN_TILE));
            }
            if let (VulkanWeight::Q5K { buffer, rows, cols }, Some(pipeline)) = (weight, self.q5_k_prefill_pipeline.as_ref()) {
                return Some((pipeline, buffer, *rows, *cols, q5_k::PREFILL_TOKEN_TILE));
            }
        }
        match weight {
            VulkanWeight::Iq4Xs { buffer, rows, cols } => Some((&self.iq4_xs_pipeline, buffer, *rows, *cols, 1)),
            VulkanWeight::Q4K { buffer, rows, cols } => Some((&self.q4_k_pipeline, buffer, *rows, *cols, 1)),
            VulkanWeight::Q5K { buffer, rows, cols } => Some((&self.q5_k_pipeline, buffer, *rows, *cols, 1)),
            VulkanWeight::Q8_0 { buffer, rows, cols } => Some((&self.q8_0_pipeline, buffer, *rows, *cols, 1)),
            VulkanWeight::F32 { buffer, rows, cols } => Some((&self.f32_pipeline, buffer, *rows, *cols, 1)),
            VulkanWeight::Q6K { .. } | VulkanWeight::W8A16 { .. } => None,
        }
    }

    pub fn linear_q6_k(&self, input: &VulkanTensor, weight: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Q6K { chunks, rows, cols } = weight else {
            return Err(compute("Vulkan linear_q6_k 收到非 Q6_K 权重"));
        };
        if input.rows == 0 || input.cols != *cols {
            return Err(compute(format!("Vulkan Q6_K linear shape 无效: input=[{},{}], weight=[{rows},{cols}]", input.rows, input.cols)));
        }
        if chunks.is_empty() {
            return Err(compute("Vulkan Q6_K 权重没有 chunk"));
        }
        if input.rows >= quant_gemm::MIN_TOKEN_ROWS && self.quant_gemm.is_some() {
            return self.linear_q6_k_gemm(input, chunks, *rows, *cols);
        }
        let output = self.output_buffer(input.rows, *rows, "Vulkan Q6_K output")?;
        let (pipeline, token_tile) = if input.rows > 1 { self.q6_k_prefill_pipeline.as_ref().map_or((&self.q6_k_pipeline, 1), |pipeline| (pipeline, q6_k::PREFILL_TOKEN_TILE)) } else { (&self.q6_k_pipeline, 1) };
        let layout = pipeline.get_bind_group_layout(0);
        let mut output_offset = 0usize;
        let groups = chunks
            .iter()
            .map(|(weight, chunk_rows)| {
                let params = [input.rows as u32, *cols as u32, output_offset as u32, *rows as u32];
                output_offset += *chunk_rows;
                let params = self.uniform_buffer("Vulkan Q6_K chunk 参数", u32_bytes(&params));
                self.bind_group("Vulkan Q6_K chunk", &layout, &[binding(0, weight), binding(1, &input.buffer), binding(2, &output), binding(3, &params)])
            })
            .collect::<Vec<_>>();
        if output_offset != *rows {
            return Err(compute(format!("Vulkan Q6_K chunk rows={output_offset}，期望 {rows}")));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for ((_, chunk_rows), group) in chunks.iter().zip(&groups) {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(*chunk_rows as u32, input.rows.div_ceil(token_tile) as u32, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols: *rows })
    }

    fn linear_q6_k_gemm(&self, input: &VulkanTensor, chunks: &[(wgpu::Buffer, usize)], rows: usize, cols: usize) -> Result<VulkanTensor, BackendError> {
        let gemm = self.quant_gemm.as_ref().ok_or_else(|| compute("Vulkan Q6_K GEMM capability 不可用"))?;
        let output = self.output_buffer(input.rows, rows, "Vulkan Q6_K GEMM output")?;
        let blocks = cols / 256;
        let bytes_per_row = blocks.checked_mul(q6_k::BLOCK_BYTES).ok_or_else(|| compute("Vulkan Q6_K GEMM row 大小溢出"))?;
        let mut output_offset = 0usize;
        for (weight, chunk_rows) in chunks {
            let padded_rows = chunk_rows.div_ceil(quant_gemm::ROW_TILE) * quant_gemm::ROW_TILE;
            let packed_bytes = padded_rows.checked_mul(bytes_per_row).ok_or_else(|| compute("Vulkan Q6_K GEMM scratch 大小溢出"))? as u64;
            let packed = self.quant_gemm_scratch(0, packed_bytes)?;
            let pack_params = self.uniform_buffer("Vulkan Q6_K GEMM pack 参数", u32_bytes(&[*chunk_rows as u32, bytes_per_row as u32, padded_rows as u32, 0]));
            let pack_layout = gemm.q6_k_pack_pipeline.get_bind_group_layout(0);
            let pack_group = self.bind_group("Vulkan Q6_K GEMM pack", &pack_layout, &[binding(0, weight), binding(1, &packed), binding(2, &pack_params)]);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&gemm.q6_k_pack_pipeline);
                pass.set_bind_group(0, &pack_group, &[]);
                pass.dispatch_workgroups((packed_bytes as usize / 4).div_ceil(256) as u32, 1, 1);
            }
            self.queue.submit([encoder.finish()]);

            let params = [*chunk_rows as u32, cols as u32, input.rows as u32, output_offset as u32, rows as u32, 0, 0, 0];
            let params = self.uniform_buffer("Vulkan Q6_K GEMM 参数", u32_bytes(&params));
            let layout = gemm.q6_k_pipeline.get_bind_group_layout(0);
            let group = self.bind_group("Vulkan Q6_K GEMM", &layout, &[binding(0, &packed), binding(1, &input.buffer), binding(2, &output), binding(3, &params)]);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&gemm.q6_k_pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(chunk_rows.div_ceil(quant_gemm::ROW_TILE) as u32, input.rows.div_ceil(quant_gemm::TOKEN_TILE) as u32, 1);
            }
            self.queue.submit([encoder.finish()]);
            output_offset += *chunk_rows;
        }
        if output_offset != rows {
            return Err(compute(format!("Vulkan Q6_K GEMM chunk rows={output_offset}，期望 {rows}")));
        }
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols: rows })
    }

    pub fn embedding_q6_k(&self, weight: &VulkanWeight, row: u32) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Q6K { chunks, rows, cols } = weight else {
            return Err(compute("Vulkan embedding lookup 需要 Q6_K 权重"));
        };
        if row as usize >= *rows {
            return Err(compute(format!("Vulkan embedding row={row} 越界 rows={rows}")));
        }
        let mut first_row = 0usize;
        let (buffer, chunk_row) = chunks
            .iter()
            .find_map(|(buffer, chunk_rows)| {
                let end = first_row + chunk_rows;
                if (row as usize) < end {
                    Some((buffer, row as usize - first_row))
                } else {
                    first_row = end;
                    None
                }
            })
            .ok_or_else(|| compute(format!("Vulkan embedding row={row} 没有对应 Q6_K chunk")))?;
        let output = self.decode_embedding_buffer(*cols)?;
        let params = [chunk_row as u32, *cols as u32, 0, 0];
        let params_buffer = self.uniform_buffer("Vulkan Q6_K embedding 参数", u32_bytes(&params));
        let layout = self.q6_k_embedding_pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group("Vulkan Q6_K embedding", &layout, &[binding(0, buffer), binding(1, &output), binding(2, &params_buffer)]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.q6_k_embedding_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(cols.div_ceil(256) as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: 1, cols: *cols })
    }

    pub fn linear_q8_0(&self, input: &VulkanTensor, weight: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::Q8_0 { buffer, rows, cols } = weight else {
            return Err(compute("Vulkan linear_q8_0 收到非 Q8_0 权重"));
        };
        linear_quantized(self, &self.q8_0_pipeline, input, buffer, *rows, *cols, "Q8_0")
    }

    fn linear_w8a16(&self, input: &VulkanTensor, weight: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::W8A16 { chunks, rows, cols, group_size } = weight else {
            return Err(compute("Vulkan W8A16 GEMV 权重类型错误"));
        };
        if input.rows == 0 || input.cols != *cols {
            return Err(compute(format!("Vulkan W8A16 GEMV shape 不一致: input=[{},{}] weight=[{},{}]", input.rows, input.cols, rows, cols)));
        }
        if chunks.is_empty() {
            return Err(compute("Vulkan W8A16 权重没有 chunk"));
        }
        let output = self.output_buffer(input.rows, *rows, "Vulkan W8A16 output")?;
        let mut output_offset = 0_u32;
        let params = chunks
            .iter()
            .map(|(_, _, chunk_rows)| {
                let values = [*chunk_rows as u32, *cols as u32, *group_size as u32, output_offset, *rows as u32, 0, 0, 0];
                output_offset += *chunk_rows as u32;
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan W8A16 参数"), contents: u32_bytes(&values), usage: wgpu::BufferUsages::UNIFORM })
            })
            .collect::<Vec<_>>();
        let groups = chunks
            .iter()
            .zip(&params)
            .map(|((packed, scales, _), params)| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Vulkan W8A16 GEMV"),
                    layout: &self.w8a16_pipeline.get_bind_group_layout(0),
                    entries: &[binding(0, packed), binding(1, scales), binding(2, &input.buffer), binding(3, &output), binding(4, params)],
                })
            })
            .collect::<Vec<_>>();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for (((_, _, chunk_rows), group), _) in chunks.iter().zip(&groups).zip(&params) {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.w8a16_pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(*chunk_rows as u32, input.rows as u32, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols: *rows })
    }

    pub fn embedding_row(&self, weight: &VulkanWeight, row: u32) -> Result<VulkanTensor, BackendError> {
        if matches!(weight, VulkanWeight::Q6K { .. }) {
            return self.embedding_q6_k(weight, row);
        }
        let VulkanWeight::W8A16 { chunks, rows, cols, group_size } = weight else {
            return Err(compute("Vulkan embedding lookup 仅支持 Q6_K/W8A16 权重"));
        };
        if row as usize >= *rows {
            return Err(compute(format!("Vulkan embedding row={row} 越界 rows={rows}")));
        }
        let mut first_row = 0usize;
        let (packed, scales, chunk_row) = chunks
            .iter()
            .find_map(|(packed, scales, chunk_rows)| {
                let end = first_row + chunk_rows;
                if (row as usize) < end {
                    Some((packed, scales, row as usize - first_row))
                } else {
                    first_row = end;
                    None
                }
            })
            .ok_or_else(|| compute(format!("Vulkan embedding row={row} 没有对应 W8A16 chunk")))?;
        let output = self.decode_embedding_buffer(*cols)?;
        let params = [chunk_row as u32, *cols as u32, *group_size as u32, 0];
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan W8A16 embedding 参数"), contents: u32_bytes(&params), usage: wgpu::BufferUsages::UNIFORM });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vulkan W8A16 embedding"),
            layout: &self.w8a16_embedding_pipeline.get_bind_group_layout(0),
            entries: &[binding(0, packed), binding(1, scales), binding(2, &output), binding(3, &params)],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.w8a16_embedding_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(cols.div_ceil(256) as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: 1, cols: *cols })
    }

    pub fn linear_gguf(&self, input: &VulkanTensor, weight: &VulkanWeight) -> Result<VulkanTensor, BackendError> {
        match weight {
            VulkanWeight::Iq4Xs { .. } => self.linear_iq4_xs(input, weight),
            VulkanWeight::Q4K { .. } => self.linear_q4_k(input, weight),
            VulkanWeight::Q5K { .. } => self.linear_q5_k(input, weight),
            VulkanWeight::Q6K { .. } => self.linear_q6_k(input, weight),
            VulkanWeight::Q8_0 { .. } => self.linear_q8_0(input, weight),
            VulkanWeight::W8A16 { .. } => self.linear_w8a16(input, weight),
            VulkanWeight::F32 { buffer, rows, cols } => linear_quantized(self, &self.f32_pipeline, input, buffer, *rows, *cols, "F32"),
        }
    }

    pub fn tensor_to_f32(&self, tensor: &VulkanTensor) -> Result<Vec<f32>, BackendError> {
        let output_bytes = tensor.rows.checked_mul(tensor.cols).and_then(|elements| elements.checked_mul(size_of::<f32>())).ok_or_else(|| compute("Vulkan readback 大小溢出"))? as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Vulkan tensor readback"), size: output_bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &readback, 0, output_bytes);
        self.queue.submit([encoder.finish()]);
        read_f32(&self.device, &readback)
    }

    pub fn rmsnorm(&self, input: &VulkanTensor, weight: &VulkanWeight, eps: f32, add_one: bool) -> Result<VulkanTensor, BackendError> {
        let VulkanWeight::F32 { buffer: weight, rows, cols } = weight else {
            return Err(compute("Vulkan RMSNorm 需要 F32 权重"));
        };
        if *rows != 1 || *cols != input.cols || input.cols == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(compute(format!("Vulkan RMSNorm shape/eps 无效: input=[{},{}], weight=[{},{}], eps={eps}", input.rows, input.cols, rows, cols)));
        }
        let output = self.output_buffer(input.rows, input.cols, "Vulkan RMSNorm output")?;
        let params = [input.cols as u32, eps.to_bits(), add_one as u32, 0];
        let params_buffer = self.uniform_buffer("RMSNorm 参数", u32_bytes(&params));
        let layout = self.rmsnorm_pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group("RMSNorm", &layout, &[binding(0, &input.buffer), binding(1, weight), binding(2, &output), binding(3, &params_buffer)]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rmsnorm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(input.rows as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols: input.cols })
    }

    fn add_gemma_rmsnorm_pair_f32(&self, left: &VulkanTensor, right: &VulkanTensor, weight: &VulkanWeight, eps: f32) -> Result<(VulkanTensor, VulkanTensor), BackendError> {
        let VulkanWeight::F32 { buffer: weight, rows, cols } = weight else {
            return Err(compute("Vulkan fused Add GemmaRMSNorm 需要 F32 权重"));
        };
        if left.rows == 0 || left.rows != right.rows || left.cols != right.cols || *rows != 1 || *cols != left.cols || !eps.is_finite() || eps <= 0.0 {
            return Err(compute(format!("Vulkan fused Add GemmaRMSNorm shape/eps 无效: left=[{},{}] right=[{},{}] weight=[{},{}] eps={eps}", left.rows, left.cols, right.rows, right.cols, rows, cols)));
        }
        let residual = self.output_buffer(left.rows, left.cols, "Vulkan fused residual")?;
        let normalized = self.output_buffer(left.rows, left.cols, "Vulkan fused GemmaRMSNorm")?;
        let params = [left.cols as u32, eps.to_bits(), 0, 0];
        let params_buffer = self.uniform_buffer("Vulkan fused Add GemmaRMSNorm 参数", u32_bytes(&params));
        let layout = self.add_gemma_rmsnorm_pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group("Vulkan fused Add GemmaRMSNorm", &layout, &[binding(0, &left.buffer), binding(1, &right.buffer), binding(2, weight), binding(3, &residual), binding(4, &normalized), binding(5, &params_buffer)]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.add_gemma_rmsnorm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(left.rows as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok((VulkanTensor { buffer: residual, rows: left.rows, cols: left.cols }, VulkanTensor { buffer: normalized, rows: left.rows, cols: left.cols }))
    }

    pub fn add(&self, left: &VulkanTensor, right: &VulkanTensor) -> Result<VulkanTensor, BackendError> {
        self.elementwise(&self.add_pipeline, left, right, 0.0, "add")
    }

    pub fn add_scaled(&self, left: &VulkanTensor, right: &VulkanTensor, scale: f32) -> Result<VulkanTensor, BackendError> {
        self.elementwise(&self.add_scaled_pipeline, left, right, scale, "add_scaled")
    }

    pub fn sigmoid_gate(&self, input: &VulkanTensor, gate: &VulkanTensor) -> Result<VulkanTensor, BackendError> {
        self.elementwise(&self.sigmoid_gate_pipeline, input, gate, 0.0, "sigmoid_gate")
    }

    pub fn silu_mul(&self, gate: &VulkanTensor, up: &VulkanTensor) -> Result<VulkanTensor, BackendError> {
        self.elementwise(&self.silu_mul_pipeline, gate, up, 0.0, "silu_mul")
    }

    fn elementwise(&self, pipeline: &wgpu::ComputePipeline, left: &VulkanTensor, right: &VulkanTensor, value: f32, name: &str) -> Result<VulkanTensor, BackendError> {
        if left.rows != right.rows || left.cols != right.cols || left.rows == 0 || left.cols == 0 {
            return Err(compute(format!("Vulkan {name} shape 不一致: left=[{},{}], right=[{},{}]", left.rows, left.cols, right.rows, right.cols)));
        }
        let elements = left.rows.checked_mul(left.cols).ok_or_else(|| compute(format!("Vulkan {name} 元素数溢出")))?;
        let output = self.output_buffer(left.rows, left.cols, "Vulkan elementwise output")?;
        let params = [elements as u32, value.to_bits(), 0, 0];
        let params_buffer = self.uniform_buffer("Vulkan elementwise 参数", u32_bytes(&params));
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group("Vulkan elementwise", &layout, &[binding(0, &left.buffer), binding(1, &right.buffer), binding(2, &output), binding(3, &params_buffer)]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(elements.div_ceil(256) as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: left.rows, cols: left.cols })
    }

    pub fn concat_columns(&self, left: &VulkanTensor, right: &VulkanTensor) -> Result<VulkanTensor, BackendError> {
        if left.rows == 0 || left.rows != right.rows {
            return Err(compute(format!("Vulkan concat rows 不一致: left=[{},{}], right=[{},{}]", left.rows, left.cols, right.rows, right.cols)));
        }
        let cols = left.cols.checked_add(right.cols).ok_or_else(|| compute("Vulkan concat cols 溢出"))?;
        let output = self.output_buffer(left.rows, cols, "Vulkan concat output")?;
        let params = [left.rows as u32, left.cols as u32, right.cols as u32, 0];
        self.dispatch_layout(&self.concat_pipeline, &[&left.buffer, &right.buffer], &output, &params, left.rows * cols)?;
        Ok(VulkanTensor { buffer: output, rows: left.rows, cols })
    }

    pub fn split_columns(&self, input: &VulkanTensor, left_cols: usize) -> Result<(VulkanTensor, VulkanTensor), BackendError> {
        if left_cols == 0 || left_cols >= input.cols {
            return Err(compute(format!("Vulkan split_columns left={left_cols}，input_cols={}", input.cols)));
        }
        Ok((self.slice_columns(input, 0, left_cols)?, self.slice_columns(input, left_cols, input.cols - left_cols)?))
    }

    pub fn split_interleaved_columns(&self, input: &VulkanTensor, block_cols: usize) -> Result<(VulkanTensor, VulkanTensor), BackendError> {
        if block_cols == 0 || !input.cols.is_multiple_of(block_cols * 2) {
            return Err(compute(format!("Vulkan interleaved block={block_cols}，input_cols={}", input.cols)));
        }
        let run = |parity: u32| -> Result<VulkanTensor, BackendError> {
            let cols = input.cols / 2;
            let output = self.output_buffer(input.rows, cols, "Vulkan interleaved output")?;
            let params = [input.rows as u32, input.cols as u32, block_cols as u32, parity];
            self.dispatch_layout(&self.interleaved_pipeline, &[&input.buffer], &output, &params, input.rows * cols)?;
            Ok(VulkanTensor { buffer: output, rows: input.rows, cols })
        };
        Ok((run(0)?, run(1)?))
    }

    pub fn select_row(&self, input: &VulkanTensor, row: usize) -> Result<VulkanTensor, BackendError> {
        if row >= input.rows {
            return Err(compute(format!("Vulkan select_row {row} 越界 rows={}", input.rows)));
        }
        self.slice_rows(input, row, 1)
    }

    pub fn rope_prefix(&self, input: &VulkanTensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cosine: &[f32], sine: &[f32]) -> Result<VulkanTensor, BackendError> {
        if head_count == 0 || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || !input.cols.is_multiple_of(head_count) || rotary_dim > input.cols / head_count || cosine.len() != sine.len() {
            return Err(compute(format!("Vulkan RoPE 参数无效: input=[{},{}] heads={head_count} rotary={rotary_dim} cos={} sin={}", input.rows, input.cols, cosine.len(), sine.len())));
        }
        let half = rotary_dim / 2;
        let required = position.checked_add(input.rows).and_then(|rows| rows.checked_mul(half)).ok_or_else(|| compute("Vulkan RoPE table 大小溢出"))?;
        if cosine.len() < required {
            return Err(compute(format!("Vulkan RoPE table={}，至少需要 {required}", cosine.len())));
        }
        let (cosine, sine) = self.rope_buffers(cosine, sine);
        let output = self.output_buffer(input.rows, input.cols, "Vulkan RoPE output")?;
        let layout = match layout {
            crate::attention::rope::RotaryLayout::SplitHalf => 0,
            crate::attention::rope::RotaryLayout::Interleaved => 1,
        };
        let params = [input.rows as u32, input.cols as u32, head_count as u32, layout, rotary_dim as u32, position as u32, 0, 0];
        let params_buffer = self.uniform_buffer("Vulkan RoPE 参数", u32_bytes(&params));
        let bind_layout = self.rope_pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group("Vulkan RoPE", &bind_layout, &[binding(0, &input.buffer), binding(1, &cosine), binding(2, &sine), binding(3, &output), binding(4, &params_buffer)]);
        let elements = input.rows * input.cols;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rope_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(elements.div_ceil(256) as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols: input.cols })
    }

    pub fn argmax_excluding(&self, input: &VulkanTensor, excluded: &[u32]) -> Result<u32, BackendError> {
        let elements = input.rows.checked_mul(input.cols).ok_or_else(|| compute("Vulkan argmax 元素数溢出"))?;
        if elements == 0 || excluded.len() >= elements {
            return Err(compute(format!("Vulkan argmax 输入/排除数量无效: elements={elements} excluded={}", excluded.len())));
        }
        let excluded_values = if excluded.is_empty() { &[0_u32][..] } else { excluded };
        let excluded_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan argmax excluded"), contents: u32_bytes(excluded_values), usage: wgpu::BufferUsages::STORAGE });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Vulkan argmax output"), size: 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let params = [elements as u32, excluded.len() as u32, 0, 0];
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan argmax 参数"), contents: u32_bytes(&params), usage: wgpu::BufferUsages::UNIFORM });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vulkan argmax"),
            layout: &self.argmax_pipeline.get_bind_group_layout(0),
            entries: &[binding(0, &input.buffer), binding(1, &excluded_buffer), binding(2, &output), binding(3, &params_buffer)],
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Vulkan argmax readback"), size: 4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.argmax_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 4);
        self.queue.submit([encoder.finish()]);
        let bytes = read_bytes(&self.device, &readback)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("argmax readback 4 bytes")))
    }

    fn slice_columns(&self, input: &VulkanTensor, start: usize, cols: usize) -> Result<VulkanTensor, BackendError> {
        let output = self.output_buffer(input.rows, cols, "Vulkan slice output")?;
        let params = [input.rows as u32, input.cols as u32, start as u32, cols as u32];
        self.dispatch_layout(&self.slice_pipeline, &[&input.buffer], &output, &params, input.rows * cols)?;
        Ok(VulkanTensor { buffer: output, rows: input.rows, cols })
    }

    fn slice_rows(&self, input: &VulkanTensor, start: usize, rows: usize) -> Result<VulkanTensor, BackendError> {
        let output = self.output_buffer(rows, input.cols, "Vulkan row slice output")?;
        let bytes = rows.checked_mul(input.cols).and_then(|elements| elements.checked_mul(size_of::<f32>())).ok_or_else(|| compute("Vulkan row slice 大小溢出"))? as u64;
        let offset = start.checked_mul(input.cols).and_then(|elements| elements.checked_mul(size_of::<f32>())).ok_or_else(|| compute("Vulkan row slice offset 溢出"))? as u64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&input.buffer, offset, &output, 0, bytes);
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows, cols: input.cols })
    }

    fn output_buffer(&self, rows: usize, cols: usize, label: &'static str) -> Result<wgpu::Buffer, BackendError> {
        let bytes = rows.checked_mul(cols).and_then(|elements| elements.checked_mul(size_of::<f32>())).ok_or_else(|| compute(format!("{label} 大小溢出")))? as u64;
        let mut resources = self.decode_resources.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if resources.active {
            let slot = *resources.output_slots.entry((label, bytes)).and_modify(|slot| *slot += 1).or_insert(0);
            let key = (resources.layer, label, bytes, slot);
            if let Some(buffer) = resources.outputs.get(&key) {
                let buffer = buffer.clone();
                resources.output_hits += 1;
                return Ok(buffer);
            }
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            resources.outputs.insert(key, buffer.clone());
            resources.output_misses += 1;
            return Ok(buffer);
        }
        drop(resources);
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }))
    }

    fn uniform_buffer(&self, label: &'static str, contents: &[u8]) -> wgpu::Buffer {
        let mut resources = self.decode_resources.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if resources.active {
            let slot = *resources.uniform_slots.entry(label).and_modify(|slot| *slot += 1).or_insert(0);
            let key = (resources.layer, label, slot);
            if let Some((buffer, cached)) = resources.uniforms.get_mut(&key)
                && cached.len() == contents.len()
            {
                let buffer = buffer.clone();
                let changed = cached.as_slice() != contents;
                if changed {
                    cached.copy_from_slice(contents);
                }
                resources.uniform_hits += 1;
                drop(resources);
                if changed {
                    self.queue.write_buffer(&buffer, 0, contents);
                }
                return buffer;
            }
            let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
            resources.uniforms.insert(key, (buffer.clone(), contents.to_vec()));
            resources.uniform_misses += 1;
            return buffer;
        }
        drop(resources);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::UNIFORM })
    }

    fn bind_group(&self, label: &'static str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry<'_>]) -> wgpu::BindGroup {
        let mut resources = self.decode_resources.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if resources.active {
            let slot = *resources.bind_group_slots.entry(label).and_modify(|slot| *slot += 1).or_insert(0);
            let key = (resources.layer, label, slot);
            if let Some(group) = resources.bind_groups.get(&key) {
                let group = group.clone();
                resources.bind_group_hits += 1;
                return group;
            }
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries });
            resources.bind_groups.insert(key, group.clone());
            resources.bind_group_misses += 1;
            return group;
        }
        drop(resources);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries })
    }

    fn dispatch_layout(&self, pipeline: &wgpu::ComputePipeline, inputs: &[&wgpu::Buffer], output: &wgpu::Buffer, params: &[u32; 4], elements: usize) -> Result<(), BackendError> {
        let params_buffer = self.uniform_buffer("Vulkan layout 参数", u32_bytes(params));
        let mut entries = inputs.iter().enumerate().map(|(index, buffer)| binding(index as u32, buffer)).collect::<Vec<_>>();
        entries.push(binding(inputs.len() as u32, output));
        entries.push(binding(inputs.len() as u32 + 1, &params_buffer));
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group("Vulkan layout", &layout, &entries);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(elements.div_ceil(256) as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// 独立 kernel 诊断入口；正式 runtime 使用 resident tensor/weight API。
    pub fn q4_k_matvec(&self, packed: &[u8], input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, BackendError> {
        let input = self.tensor_from_f32(input, 1, cols)?;
        let weight = self.prepare_q4_k(packed, rows, cols)?;
        let output = self.linear_q4_k(&input, &weight)?;
        self.tensor_to_f32(&output)
    }

    pub fn iq4_xs_matvec(&self, packed: &[u8], input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, BackendError> {
        let input = self.tensor_from_f32(input, 1, cols)?;
        let weight = self.prepare_iq4_xs(packed, rows, cols)?;
        let output = self.linear_iq4_xs(&input, &weight)?;
        self.tensor_to_f32(&output)
    }
}

impl BackendResources for VulkanContext {
    type Tensor = VulkanTensor;
    type Weight = VulkanWeight;
    type Cache = VulkanKvCache;
    type LayerScope<'a>
        = ()
    where
        Self: 'a;

    fn layer_scope(&self) -> Self::LayerScope<'_> {}

    fn token_rows(&self, tensor: &Self::Tensor) -> usize {
        tensor.rows
    }

    fn token_cols(&self, tensor: &Self::Tensor) -> usize {
        tensor.cols
    }

    fn tensor_allocated_bytes(&self, tensor: &Self::Tensor) -> u64 {
        tensor.buffer.size()
    }

    fn begin_batch(&self) {
        self.queue.begin_batch();
    }

    fn begin_decode_batch(&self) {
        self.queue.begin_decode_batch();
        self.begin_decode_resources();
    }

    fn submit_batch(&self) {
        self.queue.flush_batch();
    }

    fn finish_batch(&self) {
        self.queue.finish_batch();
        self.finish_decode_resources();
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|error| compute(format!("等待 Vulkan 队列失败: {error}")))?;
        Ok(())
    }

    fn prepare_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        match weight {
            LinearWeight::Quantized(QuantizedMatrixRef::Gguf(matrix)) if matrix.tensor_type.0 == 23 => {
                if matrix.rows != rows || matrix.columns != cols {
                    return Err(compute(format!("Vulkan GGUF weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
                }
                let packed = read_matrix_packed(matrix)?;
                self.prepare_iq4_xs(&packed, rows, cols)
            }
            LinearWeight::Quantized(QuantizedMatrixRef::Gguf(matrix)) if matrix.tensor_type.0 == 12 => {
                if matrix.rows != rows || matrix.columns != cols {
                    return Err(compute(format!("Vulkan GGUF weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
                }
                let packed = read_matrix_packed(matrix)?;
                self.prepare_q4_k(&packed, rows, cols)
            }
            LinearWeight::Quantized(QuantizedMatrixRef::Gguf(matrix)) if matrix.tensor_type.0 == 13 => {
                if matrix.rows != rows || matrix.columns != cols {
                    return Err(compute(format!("Vulkan GGUF weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
                }
                let packed = read_matrix_packed(matrix)?;
                self.prepare_q5_k(&packed, rows, cols)
            }
            LinearWeight::Quantized(QuantizedMatrixRef::Gguf(matrix)) if matrix.tensor_type.0 == 14 => {
                if matrix.rows != rows || matrix.columns != cols {
                    return Err(compute(format!("Vulkan GGUF weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
                }
                let packed = read_matrix_packed(matrix)?;
                self.prepare_q6_k(&packed, rows, cols)
            }
            LinearWeight::Quantized(QuantizedMatrixRef::Gguf(matrix)) if matrix.tensor_type.0 == 8 => {
                if matrix.rows != rows || matrix.columns != cols {
                    return Err(compute(format!("Vulkan GGUF weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
                }
                let packed = read_matrix_packed(matrix)?;
                self.prepare_q8_0(&packed, rows, cols)
            }
            LinearWeight::F32(values) => self.prepare_f32(values, rows, cols),
            LinearWeight::F16(values) => {
                let values = values.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
                self.prepare_f32(&values, rows, cols)
            }
            LinearWeight::Quantized(QuantizedMatrixRef::W8A16(matrix)) => {
                if matrix.rows != rows || matrix.cols != cols {
                    return Err(compute(format!("Vulkan W8A16 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
                }
                self.prepare_w8a16(matrix, rows, cols)
            }
            _ => Err(compute("Vulkan 当前只支持 IQ4_XS/Q4_K/Q5_K/Q6_K/Q8_0 GGUF、W8A16 和 F32 常量权重")),
        }
    }

    fn prepare_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        validate_elements(values.len(), rows, cols, "Vulkan F32 weight")?;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vulkan resident F32 weight"), contents: f32_bytes(values), usage: wgpu::BufferUsages::STORAGE });
        Ok(VulkanWeight::F32 { buffer, rows, cols })
    }
}

impl Backend for VulkanContext {
    fn linear(&self, input: &Self::Tensor, weight: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        self.linear_gguf(input, weight)
    }

    fn dual_linear(&self, input: &Self::Tensor, first: &Self::Weight, second: &Self::Weight) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let packed_gemm = input.rows >= quant_gemm::MIN_TOKEN_ROWS
            && self.quant_gemm.is_some()
            && matches!((first, second), (VulkanWeight::Iq4Xs { .. } | VulkanWeight::Q4K { .. } | VulkanWeight::Q5K { .. }, _) | (_, VulkanWeight::Iq4Xs { .. } | VulkanWeight::Q4K { .. } | VulkanWeight::Q5K { .. }));
        if packed_gemm || matches!((first, second), (VulkanWeight::Q6K { .. }, _) | (_, VulkanWeight::Q6K { .. })) { Ok((self.linear(input, first)?, self.linear(input, second)?)) } else { self.dual_linear_simple(input, first, second) }
    }

    fn rmsnorm(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        VulkanContext::rmsnorm(self, input, weight, eps, false)
    }

    fn gemma_rmsnorm(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        VulkanContext::rmsnorm(self, input, weight, eps, true)
    }

    fn add_gemma_rmsnorm_pair(&self, input: &Self::Tensor, residual: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        self.add_gemma_rmsnorm_pair_f32(input, residual, weight, eps)
    }

    fn layernorm_bias(&self, _input: &Self::Tensor, _weight: &Self::Weight, _bias: &Self::Weight, _eps: f32) -> Result<Self::Tensor, BackendError> {
        Err(compute("Vulkan 尚未实现 LayerNorm；Qwen3.5 文本路径不使用该算子"))
    }

    fn split_columns(&self, input: &Self::Tensor, left_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        VulkanContext::split_columns(self, input, left_columns)
    }

    fn split_interleaved_columns(&self, input: &Self::Tensor, block_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        VulkanContext::split_interleaved_columns(self, input, block_columns)
    }

    fn concat_columns(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        VulkanContext::concat_columns(self, left, right)
    }

    fn rope(&self, _input: &Self::Tensor, _head_count: usize, _rotary_dim: usize, _layout: crate::attention::rope::RotaryLayout, _position: usize, _cos: &[f32], _sin: &[f32]) -> Result<Self::Tensor, BackendError> {
        Err(compute("Vulkan 尚未实现 suffix RoPE；Qwen3.5 使用 prefix RoPE"))
    }

    fn rope_prefix(&self, input: &Self::Tensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<Self::Tensor, BackendError> {
        VulkanContext::rope_prefix(self, input, head_count, rotary_dim, layout, position, cos, sin)
    }

    fn add(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        VulkanContext::add(self, left, right)
    }

    fn add_scaled(&self, left: &Self::Tensor, right: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        VulkanContext::add_scaled(self, left, right, scale)
    }

    fn sigmoid_gate(&self, input: &Self::Tensor, gate: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        VulkanContext::sigmoid_gate(self, input, gate)
    }

    fn select_row(&self, input: &Self::Tensor, row: usize) -> Result<Self::Tensor, BackendError> {
        VulkanContext::select_row(self, input, row)
    }

    fn argmax(&self, input: &Self::Tensor) -> Result<u32, BackendError> {
        VulkanContext::argmax_excluding(self, input, &[])
    }

    fn argmax_excluding(&self, input: &Self::Tensor, excluded: &[u32]) -> Result<u32, BackendError> {
        VulkanContext::argmax_excluding(self, input, excluded)
    }

    fn sample_top_p(&self, input: &Self::Tensor, temperature: f32, _top_p: f32, _random: f32) -> Result<u32, BackendError> {
        if temperature == 0.0 { self.argmax(input) } else { Err(compute("Vulkan top-p 尚未实现；当前正式入口使用 greedy")) }
    }

    fn gated_activation(&self, gate: &Self::Tensor, up: &Self::Tensor, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        match activation {
            Activation::Silu => self.silu_mul(gate, up),
            _ => Err(compute("Vulkan 当前仅实现 Qwen3.5 使用的 SiLU gated activation")),
        }
    }

    fn gated_linear(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Weight, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        if matches!(activation, Activation::Silu) {
            match (gate, up) {
                (VulkanWeight::Iq4Xs { .. }, VulkanWeight::Iq4Xs { .. }) => return self.gated_linear_iq4_xs(input, gate, up),
                (VulkanWeight::Q4K { .. }, VulkanWeight::Q4K { .. }) => return self.gated_linear_q4_k(input, gate, up),
                (VulkanWeight::Q5K { .. }, VulkanWeight::Q5K { .. }) => return self.gated_linear_q5_k(input, gate, up),
                _ => {}
            }
        }
        let (gate, up) = self.dual_linear(input, gate, up)?;
        self.gated_activation(&gate, &up, activation)
    }
}

impl GqaPrefillBackend for VulkanContext {
    fn gemma_rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        if input.cols != head_count.checked_mul(head_dim).ok_or_else(|| compute("Vulkan head norm 维度溢出"))? {
            return Err(compute(format!("Vulkan head norm input_cols={}，期望 {head_count}×{head_dim}", input.cols)));
        }
        let view = VulkanTensor { buffer: input.buffer.clone(), rows: input.rows * head_count, cols: head_dim };
        let output = VulkanContext::rmsnorm(self, &view, weight, eps, true)?;
        Ok(VulkanTensor { buffer: output.buffer, rows: input.rows, cols: input.cols })
    }

    fn gqa_prefill_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, spec: &crate::attention::gqa::GqaSpec) -> Result<Self::Tensor, BackendError> {
        self.gqa_uncached(query, key, value, spec)
    }

    fn gqa_prefill_attention_cached(
        &self,
        cache: &mut Self::Cache,
        layer: usize,
        position: usize,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        spec: &crate::attention::gqa::GqaSpec,
        _retain_full_cache: bool,
    ) -> Result<Self::Tensor, BackendError> {
        self.append_gqa_cache(cache, layer, position, key, value)?;
        self.gqa_cached(cache, layer, position, query, spec)
    }

    fn gqa_prefill_attention_cached_from(&self, cache: &Self::Cache, source_layer: usize, position: usize, query: &Self::Tensor, spec: &crate::attention::gqa::GqaSpec) -> Result<Self::Tensor, BackendError> {
        self.gqa_cached(cache, source_layer, position, query, spec)
    }
}

fn validate_elements(actual: usize, rows: usize, cols: usize, name: &str) -> Result<(), BackendError> {
    let expected = rows.checked_mul(cols).ok_or_else(|| compute(format!("{name} shape 大小溢出")))?;
    if actual != expected {
        return Err(compute(format!("{name} 元素数 {actual}，期望 {expected}")));
    }
    Ok(())
}

fn read_matrix_packed(matrix: &crate::weight::container::gguf::GgufMatrix) -> Result<Vec<u8>, BackendError> {
    let mut packed = vec![0_u8; matrix.storage_len()];
    matrix.read_into(&mut packed).map_err(compute)?;
    Ok(packed)
}

fn linear_quantized(context: &VulkanContext, pipeline: &wgpu::ComputePipeline, input: &VulkanTensor, weight: &wgpu::Buffer, rows: usize, cols: usize, name: &str) -> Result<VulkanTensor, BackendError> {
    linear_quantized_tiled(context, pipeline, input, weight, rows, cols, 1, name)
}

fn linear_quantized_tiled(context: &VulkanContext, pipeline: &wgpu::ComputePipeline, input: &VulkanTensor, weight: &wgpu::Buffer, rows: usize, cols: usize, token_tile: usize, name: &str) -> Result<VulkanTensor, BackendError> {
    if input.rows == 0 || input.cols != cols {
        return Err(compute(format!("Vulkan {name} linear shape 无效: input=[{},{}], weight=[{rows},{cols}]", input.rows, input.cols)));
    }
    let output = context.output_buffer(input.rows, rows, "Vulkan quantized output")?;
    let params = [rows as u32, cols as u32, input.rows as u32, 0];
    let params_buffer = context.uniform_buffer("Vulkan quantized 参数", u32_bytes(&params));
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = context.bind_group("Vulkan quantized GEMV", &layout, &[binding(0, weight), binding(1, &input.buffer), binding(2, &output), binding(3, &params_buffer)]);
    let mut encoder = context.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows as u32, input.rows.div_ceil(token_tile) as u32, 1);
    }
    context.queue.submit([encoder.finish()]);
    Ok(VulkanTensor { buffer: output, rows: input.rows, cols: rows })
}

fn read_f32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<f32>, BackendError> {
    Ok(read_bytes(device, buffer)?.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect())
}

fn read_bytes(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, BackendError> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|error| compute(format!("等待 Vulkan 队列失败: {error}")))?;
    receiver.recv().map_err(|error| compute(format!("接收 Vulkan readback 失败: {error}")))?.map_err(|error| compute(format!("映射 Vulkan readback 失败: {error}")))?;
    let mapped = slice.get_mapped_range().map_err(|error| compute(format!("读取 Vulkan readback 失败: {error}")))?;
    let result = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(result)
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

fn compute(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}
