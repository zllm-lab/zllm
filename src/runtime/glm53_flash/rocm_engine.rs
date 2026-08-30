//! GLM-5.3-Flash ROCm 分段引擎:8 卡链式装载与 prefill/decode。
//!
//! head 从 token 建立 mHC 展开态,tail 直接继续这份展开态。只有完整
//! L0..L44 成功后才折叠 copies 并执行 final norm / LM head。

use std::{
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{Receiver, SyncSender, sync_channel},
    },
};

use rayon::prelude::*;

use crate::{
    attention::{AttentionSpec, kda::KdaState},
    backend::{
        Backend, BackendError, StageExecutionBackend, StageTensorBackend,
        rocm::{RocmContext, RocmDsaState, RocmKvCache, RocmPrefillExperts, RocmTensor, RocmWeight},
    },
    runtime::{
        Model,
        glm53_flash::{
            Glm53Flash, Glm53FlashConfig,
            layer::{Glm53AttentionWeights, glm53_collapse_hidden, glm53_dsa_mla_layer, glm53_expand_hidden, glm53_kda_attention, glm53_kda_layer_with_observer},
            prepare::{self, Glm53PreparedLayer},
            vision::{Glm53FlashVisionConfig, glm53_vision_encode},
        },
        rocm_chain::RocmDeviceChain,
    },
    vision::{ImageTensor, ImageTokenRange, splice_image_embeddings_bf16},
};

pub struct Options {
    pub weights_directory: PathBuf,
    pub devices: Vec<i32>,
    /// 每卡层区间开区间末尾;末项 = layer_end。
    pub layer_ends: Vec<usize>,
    /// 本段负责的绝对层区间 `layer_start..layer_end`。
    pub layer_start: usize,
    pub layer_end: usize,
    pub max_sequence_length: usize,
    pub prefill_chunk_size: usize,
    /// 尾段 MTP speculative decode;装载 layers.{layer_count} 的 MTP 头。
    pub mtp: bool,
}

/// 尾段 MTP runtime:eh_proj + 一层 DSA-MLA + MoE FFN + shared_head norm。
/// 复用主干 output head 采样;MLA/DSA 状态挂在尾卡 cache 的 layer_count 槽。
struct MtpRuntime {
    weights: crate::runtime::glm53_flash::layer::Glm53Mtp<RocmWeight>,
    router_weight: RocmWeight,
    router_bias: RocmWeight,
    shared: crate::moe::DenseFfn<RocmWeight>,
    experts: RocmPrefillExperts,
    layer: usize,
}

/// verify 轮的 KDA 回退快照:推进前的递归状态(conv/recurrent 按层序)+
/// 各层 position 游标 + 本卡各 KDA 层完整 verify 块的 attention 输入。
/// 拒绝时恢复状态后按序重放保留前缀,数学上与只推进 retained rows 一致
/// (mHC/FFN 无状态,重放无需重跑)。
struct KdaSnapshot {
    conv: Vec<crate::kernel::rocm::hip::DeviceBuffer>,
    recurrent: Vec<crate::kernel::rocm::hip::DeviceBuffer>,
    layers: Vec<usize>,
    positions: Vec<usize>,
    layer_inputs: Vec<(usize, RocmTensor)>,
}

/// 同卡 D2D 整 buffer 深拷贝(快照)。
fn clone_resident_bytes(context: &RocmContext, buffer: &crate::kernel::rocm::hip::DeviceBuffer) -> Result<crate::kernel::rocm::hip::DeviceBuffer, BackendError> {
    use crate::kernel::rocm::hip::DeviceBuffer;
    if buffer.bytes() == 0 || buffer.bytes() % 16 != 0 {
        return Err(BackendError::Compute { msg: format!("KDA 快照要求 16 字节对齐大小,实际 {}", buffer.bytes()) });
    }
    let copy = DeviceBuffer::allocate(context.device_id(), buffer.bytes()).map_err(crate::runtime::compute_error)?;
    crate::kernel::rocm::hip::try_peer_copy_kernel_ordered(context.device_id(), copy.device_pointer() as *mut std::ffi::c_void, buffer.device_pointer() as *mut std::ffi::c_void, buffer.bytes()).map_err(crate::runtime::compute_error)?;
    Ok(copy)
}

/// 快照写回原 buffer(恢复)。
fn restore_resident_bytes(context: &RocmContext, destination: &crate::kernel::rocm::hip::DeviceBuffer, source: &crate::kernel::rocm::hip::DeviceBuffer) -> Result<(), BackendError> {
    if destination.bytes() != source.bytes() {
        return Err(BackendError::Compute { msg: format!("KDA 恢复大小不一致: {} vs {}", destination.bytes(), source.bytes()) });
    }
    crate::kernel::rocm::hip::try_peer_copy_kernel_ordered(context.device_id(), destination.device_pointer() as *mut std::ffi::c_void, source.device_pointer() as *mut std::ffi::c_void, source.bytes())
        .map_err(crate::runtime::compute_error)?;
    Ok(())
}

/// 从 resident tensor 深拷贝 [row, row+rows) 区段;行首偏移由 peer copy
/// 指针算术完成(行 0 无偏移,16 字节对齐由 cols*4 保证)。
fn clone_resident_rows(context: &RocmContext, tensor: &RocmTensor, row: usize, rows: usize) -> Result<RocmTensor, BackendError> {
    let device = tensor.device.as_ref().ok_or_else(|| BackendError::Compute { msg: "clone_resident_rows 需要 resident tensor".to_owned() })?;
    let bytes = rows.checked_mul(tensor.cols).and_then(|elements| elements.checked_mul(4)).ok_or_else(|| BackendError::Compute { msg: "clone_resident_rows 大小溢出".to_owned() })?;
    if bytes % 16 != 0 || row + rows > tensor.rows {
        return Err(BackendError::Compute { msg: format!("clone_resident_rows 区间 [{row},{}) 越界或未 16 对齐(bytes={bytes})", row + rows) });
    }
    let copy = crate::kernel::rocm::hip::DeviceBuffer::allocate(context.device_id(), bytes).map_err(crate::runtime::compute_error)?;
    let source = (device.device_pointer() + row * tensor.cols * 4) as *mut std::ffi::c_void;
    crate::kernel::rocm::hip::try_peer_copy_kernel_ordered(context.device_id(), copy.device_pointer() as *mut std::ffi::c_void, source, bytes).map_err(crate::runtime::compute_error)?;
    Ok(RocmTensor { data: Vec::new(), rows, cols: tensor.cols, dtype: crate::backend::rocm::RocmTensorDType::F32, layout: crate::backend::rocm::RocmTensorLayout::RowMajor, device: Some(std::sync::Arc::new(copy)) })
}

/// 拼接单行 tail 与 chunk 前 length-1 行,构成 MTP 移位 hidden [length, cols]。
fn concat_two_rows(context: &RocmContext, tail: &RocmTensor, collapsed: &RocmTensor, length: usize) -> Result<RocmTensor, BackendError> {
    if tail.rows != 1 || tail.cols != collapsed.cols || length == 0 || length - 1 > collapsed.rows {
        return Err(BackendError::Compute { msg: format!("MTP 移位拼接形状非法 tail=[{},{}] collapsed=[{},{}] length={length}", tail.rows, tail.cols, collapsed.rows, collapsed.cols) });
    }
    let bytes = length.checked_mul(tail.cols).and_then(|n| n.checked_mul(4)).ok_or_else(|| BackendError::Compute { msg: "MTP 移位拼接大小溢出".to_owned() })?;
    if bytes % 16 != 0 {
        return Err(BackendError::Compute { msg: format!("MTP 移位拼接未 16 对齐(bytes={bytes})") });
    }
    let target = crate::kernel::rocm::hip::DeviceBuffer::allocate(context.device_id(), bytes).map_err(crate::runtime::compute_error)?;
    crate::kernel::rocm::hip::try_peer_copy_kernel_ordered(context.device_id(), target.device_pointer() as *mut std::ffi::c_void, tail.device.as_ref().expect("tail resident").device_pointer() as *mut std::ffi::c_void, tail.cols * 4)
        .map_err(crate::runtime::compute_error)?;
    let rest = (length - 1) * tail.cols * 4;
    if rest > 0 {
        let destination = (target.device_pointer() + tail.cols * 4) as *mut std::ffi::c_void;
        let source = collapsed.device.as_ref().expect("collapsed resident").device_pointer() as *mut std::ffi::c_void;
        crate::kernel::rocm::hip::try_peer_copy_kernel_ordered(context.device_id(), destination, source, rest).map_err(crate::runtime::compute_error)?;
    }
    Ok(RocmTensor { data: Vec::new(), rows: length, cols: tail.cols, dtype: crate::backend::rocm::RocmTensorDType::F32, layout: crate::backend::rocm::RocmTensorLayout::RowMajor, device: Some(std::sync::Arc::new(target)) })
}

/// target token `start` 使用主干 `h[start-1]`，因此 MTP cache 从
/// `start-1` 开始写；首 token 没有前驱 hidden，不进入 MTP。
fn mtp_prefill_window(position: usize, rows: usize) -> Result<Option<(usize, usize, usize)>, BackendError> {
    let end = position.checked_add(rows).ok_or_else(|| BackendError::Compute { msg: "MTP prefill position 溢出".to_owned() })?;
    let start = position.max(1);
    Ok((end > start).then_some((start, start - 1, end - start)))
}

/// run_device_stage 的 KDA observer:保存完整 verify 块的 attention 输入。
fn clone_resident_tensor(context: &RocmContext, tensor: &RocmTensor) -> Result<RocmTensor, BackendError> {
    if tensor.rows == 0 {
        return Err(BackendError::Compute { msg: "KDA observer 收到空 tensor".to_owned() });
    }
    clone_resident_rows(context, tensor, 0, tensor.rows)
}

pub struct Engine {
    config: Glm53FlashConfig,
    model: Glm53Flash,
    /// 视觉塔配置;dims 校验与图像编码共用这一份。
    vision: Glm53FlashVisionConfig,
    contexts: Vec<RocmContext>,
    weights: prepare_weights::Weights,
    layers: Vec<(usize, usize, Glm53PreparedLayer<RocmWeight>)>,
    kda: Vec<KdaState<crate::backend::rocm::RocmKdaStorage>>,
    caches: Vec<RocmKvCache>,
    dsa: Vec<RocmDsaState>,
    experts: Vec<RocmPrefillExperts>,
    /// 只有包含最后一层的段在末卡驻留输出头。
    output: Option<prepare::Glm53PreparedOutput<RocmWeight>>,
    /// 尾段 MTP;head 段与未启用时为 None。
    mtp: Option<Box<MtpRuntime>>,
    /// 每卡 KDA 快照;只在 speculative verify forward 前刷新。
    kda_snapshot: Vec<Option<KdaSnapshot>>,
    position: usize,
}

enum PipelineInput {
    Tokens {
        position: usize,
        tokens: Vec<u32>,
    },
    /// 多模态 chunk:文本行查表后,按 overlay 覆盖 image token 行。
    VisionTokens {
        position: usize,
        tokens: Vec<u32>,
        overlay: Arc<VisionOverlay>,
    },
    Hidden {
        position: usize,
        rows: usize,
        cols: usize,
        values: Vec<u16>,
    },
}

/// 一次多模态请求的视觉行覆盖:每张图视觉塔输出的 BF16 行
/// (hidden_size 列)与其 token 占位区间;由 pipeline 各 chunk 共享。
pub struct VisionOverlay {
    images: Vec<Vec<u16>>,
    ranges: Vec<ImageTokenRange>,
}

struct PipelineValue {
    position: usize,
    rows: usize,
    hidden: RocmTensor,
}

mod prepare_weights {
    pub type Weights = crate::weight::model::glm53_flash::Glm53FlashWeights;
}

impl Engine {
    pub fn load(options: &Options) -> Result<Self, String> {
        let config = Glm53FlashConfig::standard();
        if options.layer_start >= options.layer_end || options.layer_end > config.layer_count {
            return Err(format!("GLM-5.3-Flash 分段必须满足 0 <= start < end <= {}", config.layer_count));
        }
        let model = Glm53Flash::new(config.clone()).map_err(|error| error.to_string())?;
        let vision = Glm53FlashVisionConfig::standard();
        if vision.out_hidden_size != config.hidden_size {
            return Err(format!("GLM-5.3-Flash 视觉输出宽 {} 与文本 hidden_size {} 不一致", vision.out_hidden_size, config.hidden_size));
        }
        if options.mtp && options.layer_end != config.layer_count {
            return Err("GLM-5.3-Flash MTP 只能装载在包含最后一层的尾段".to_owned());
        }
        // MTP 的 DSA-MLA 状态挂在尾卡 cache 的 layer_count 槽,容量随之 +1。
        let mtp_enabled = options.mtp && options.layer_end == config.layer_count;
        let state_layers = config.layer_count + usize::from(mtp_enabled);
        if options.layer_ends.len() != options.devices.len()
            || options.layer_ends.last().copied() != Some(options.layer_end)
            || options.layer_ends.windows(2).any(|pair| pair[0] >= pair[1])
            || options.layer_ends.first().is_some_and(|end| *end <= options.layer_start)
        {
            return Err(format!("layer_ends={:?} 与 devices={:?} / layers={}..{} 不匹配", options.layer_ends, options.devices, options.layer_start, options.layer_end));
        }
        // config 的 layer_ends 是开区间末尾;设备链要求闭区间。
        let chain_layer_ends = options.layer_ends.iter().map(|end| end - 1).collect::<Vec<_>>();
        let chain = RocmDeviceChain::new(&options.devices, chain_layer_ends, options.layer_end - 1, false).map_err(|error| error.to_string())?;
        let contexts = chain.contexts.clone();
        let device_count = contexts.len();
        let dims = dims(&config, &vision);
        let weights = prepare_weights::Weights::open(&options.weights_directory, dims)?;
        let kda_spec = Glm53Flash::kda_spec(&config);
        let dsa_spec = Glm53Flash::dsa_spec(&config);
        let mut kda = Vec::with_capacity(device_count);
        let mut caches = Vec::with_capacity(device_count);
        let mut dsa = Vec::with_capacity(device_count);
        let mut experts = Vec::with_capacity(device_count);
        let source = std::sync::Arc::new(prepare::Glm53FlashExpertSource::new(weights.clone(), config.layer_count + usize::from(mtp_enabled), config.expert_count));
        for (device, context) in contexts.iter().enumerate() {
            context.activate().map_err(|error| format!("activate d{device} DSA APE: {error:?}"))?;
            kda.push(KdaState::new(config.layer_count, kda_spec).map_err(|error| format!("KDA state: {error:?}"))?);
            caches.push(RocmKvCache::with_capacity(state_layers, options.max_sequence_length));
            let mut state = RocmDsaState::new(state_layers, options.max_sequence_length, dsa_spec.head_dim, dsa_spec.top_k).map_err(|error| format!("DSA state: {error}"))?;
            let owns_mtp = mtp_enabled && device + 1 == contexts.len();
            for layer in 0..state_layers {
                if layer == config.layer_count && owns_mtp || layer < config.layer_count && model.is_full_attention(layer) && chain.device_of_layer(layer).ok() == Some(device) {
                    state.set_kpool_ape(context, layer, weights.kpool_ape_f32(layer)?)?;
                }
            }
            dsa.push(state);
            experts.push(RocmPrefillExperts::fp8_source(source.clone()));
        }
        let mut layers = Vec::with_capacity(options.layer_end - options.layer_start);
        for layer in options.layer_start..options.layer_end {
            let device = chain.device_of_layer(layer).map_err(|error| format!("L{layer} 放置: {error:?}"))?;
            let context = &contexts[device];
            context.activate().map_err(|error| format!("activate d{device}: {error:?}"))?;
            let prepared = prepare::prepare_layer(context, &weights, &config, &model, layer).map_err(|error| format!("L{layer} prepare: {error:?}"))?;
            if let Glm53PreparedLayer::Moe(prepared) = &prepared {
                prepared.router_weight.prepare_router(context.device_id()).map_err(|error| format!("L{layer} router 常驻: {error:?}"))?;
            }
            layers.push((layer, device, prepared));
        }
        // paged KV/DSA 的 block ID 是模型运行常量。按最大上下文在加载期
        // 一次上传，避免首次 prefill 或后续跨 64-token block 时发生 H2D。
        let mut full_attention_devices = vec![false; device_count];
        for (layer, device, _) in &layers {
            full_attention_devices[*device] |= model.is_full_attention(*layer);
        }
        for (device, owns_full_attention) in full_attention_devices.into_iter().enumerate() {
            if owns_full_attention {
                let context = &contexts[device];
                context.activate().map_err(|error| format!("activate d{device} block table: {error:?}"))?;
                caches[device].prepare_block_table(context).map_err(|error| format!("d{device} KV block table 常驻: {error:?}"))?;
                dsa[device].prepare_block_table(context).map_err(|error| format!("d{device} DSA block table 常驻: {error:?}"))?;
            }
        }
        // 官方 FP8 experts 以 codes + block scale 原形常驻。按设备并行预载，
        // forward 只能消费 device route，不再允许按 top-k 临时读盘/上传。
        let mut expert_layers = vec![Vec::new(); device_count];
        for (layer, device, prepared) in &layers {
            if matches!(prepared, Glm53PreparedLayer::Moe(_)) {
                expert_layers[*device].push(*layer);
            }
        }
        experts.par_iter_mut().enumerate().try_for_each(|(device, state)| -> Result<(), String> {
            let context = contexts[device];
            context.activate().map_err(|error| format!("activate d{device} experts: {error}"))?;
            for &layer in &expert_layers[device] {
                state.preload_layer(&context, layer, config.expert_count).map_err(|error| format!("d{device} L{layer} FP8 experts 常驻: {error:?}"))?;
                eprintln!("[glm53-fp8-resident] device={} layer={layer} experts={}", context.device_id(), config.expert_count);
            }
            Ok(())
        })?;
        let output = if options.layer_end == config.layer_count {
            let context = contexts.last().expect("已校验设备非空");
            Some(prepare::load_prepare_output(context, &weights, &config).map_err(|error| format!("输出头 prepare: {error:?}"))?)
        } else {
            None
        };
        // MTP 头常驻末卡:eh_proj/norm/DSA-MLA + L{layer_count} 的 MoE FFN。
        // MLA/DSA 状态复用末卡 cache 的 layer_count 槽;采样复用主干 output head。
        let mtp = if mtp_enabled {
            let device = device_count - 1;
            let context = contexts[device];
            context.activate().map_err(|error| format!("activate d{device} MTP: {error}"))?;
            let mtp_weights = prepare::prepare_mtp(&context, &weights, &config).map_err(|error| format!("MTP prepare: {error:?}"))?;
            let router = weights.load_moe_router(config.layer_count).map_err(|error| format!("MTP router: {error}"))?;
            let shared = weights.load_shared_experts(config.layer_count).map_err(|error| format!("MTP shared: {error}"))?;
            let mut mtp_experts = RocmPrefillExperts::fp8_source(source.clone());
            mtp_experts.preload_layer(&context, config.layer_count, config.expert_count).map_err(|error| format!("MTP experts 常驻: {error:?}"))?;
            eprintln!("[glm53-fp8-resident] device={} layer={} experts={} (MTP)", contexts[device].device_id(), config.layer_count, config.expert_count);
            let router_weight = <RocmContext as crate::backend::BackendResources>::prepare_f32(&context, &router.router, config.expert_count, config.hidden_size).map_err(|error| format!("MTP router 常驻: {error:?}"))?;
            router_weight.prepare_router(context.device_id()).map_err(|error| format!("MTP router resident: {error:?}"))?;
            let router_bias = prepare::prepare_f32_vector(&context, &router.correction_bias, "MTP router bias").map_err(|error| format!("MTP router bias: {error:?}"))?;
            let shared = prepare::prepare_dense_mlp(&context, &shared).map_err(|error| format!("MTP shared FFN: {error:?}"))?;
            Some(Box::new(MtpRuntime { weights: mtp_weights, router_weight, router_bias, shared, experts: mtp_experts, layer: config.layer_count }))
        } else {
            None
        };
        let kda_snapshot: Vec<Option<KdaSnapshot>> = (0..device_count).map(|_| None).collect();
        Ok(Self { config, model, vision, contexts, weights, layers, kda, caches, dsa, experts, output, mtp, kda_snapshot, position: 0 })
    }

    /// head prefill：CPU embedding 行读取与首卡 H2D 留在 stage 入口；多个
    /// chunk 随后在本机各设备之间连续流动，末卡每完成一个 chunk 就回调。
    pub fn prefill_tokens_pipeline<F>(&mut self, chunks: Vec<Vec<u32>>, mut output: F) -> Result<(), BackendError>
    where
        F: FnMut(usize, usize, Vec<u16>) -> Result<(), BackendError>,
    {
        if self.layers.first().map(|(layer, _, _)| *layer) != Some(0) {
            return Err(BackendError::Compute { msg: "非 head 段不能从 token 建立 hidden".to_owned() });
        }
        let mut position = self.position;
        let mut chunks = chunks.into_iter();
        let boundary = *self.contexts.last().expect("配置已校验设备非空");
        self.forward_pipeline(
            move || {
                let Some(tokens) = chunks.next() else { return Ok(None) };
                let input = PipelineInput::Tokens { position, tokens };
                if let PipelineInput::Tokens { tokens, .. } = &input {
                    position += tokens.len();
                }
                Ok(Some(input))
            },
            |position, rows, hidden| {
                let values = boundary.tensor_to_bf16_bits(&hidden)?;
                output(position, rows, values)
            },
        )
    }

    /// head 多模态 prefill:图像在进入 pipeline 前已由 `vision_overlay` 编码为
    /// BF16 行;stage 0 在每个 chunk 的文本 embedding 上覆盖 image token 行。
    pub fn prefill_multimodal_pipeline<F>(&mut self, chunks: Vec<Vec<u32>>, overlay: Arc<VisionOverlay>, mut output: F) -> Result<(), BackendError>
    where
        F: FnMut(usize, usize, Vec<u16>) -> Result<(), BackendError>,
    {
        if self.layers.first().map(|(layer, _, _)| *layer) != Some(0) {
            return Err(BackendError::Compute { msg: "非 head 段不能从 token 建立 hidden".to_owned() });
        }
        let mut position = self.position;
        let mut chunks = chunks.into_iter();
        let boundary = *self.contexts.last().expect("配置已校验设备非空");
        self.forward_pipeline(
            move || {
                let Some(tokens) = chunks.next() else { return Ok(None) };
                let input = PipelineInput::VisionTokens { position, tokens, overlay: overlay.clone() };
                if let PipelineInput::VisionTokens { tokens, .. } = &input {
                    position += tokens.len();
                }
                Ok(Some(input))
            },
            |position, rows, hidden| {
                let values = boundary.tensor_to_bf16_bits(&hidden)?;
                output(position, rows, values)
            },
        )
    }

    /// 在首卡逐张编码图像并读回 BF16 行,构建 chunk 覆盖数据。
    /// 每张图的视觉输出行数必须与其 token 占位区间一致。
    pub fn vision_overlay(&self, images: &[ImageTensor], ranges: &[ImageTokenRange]) -> Result<VisionOverlay, BackendError> {
        let Some(context) = self.contexts.first() else {
            return Err(BackendError::Compute { msg: "GLM-5.3-Flash 视觉编码没有可用设备".to_owned() });
        };
        context.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let mut encoded = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            let hidden = glm53_vision_encode(context, &self.vision, &self.weights, image).map_err(|error| BackendError::Compute { msg: format!("图像 {index} 视觉编码: {error:?}") })?;
            encoded.push(context.tensor_to_bf16_bits(&hidden)?);
        }
        for range in ranges {
            let Some(bits) = encoded.get(range.image_index) else {
                return Err(BackendError::Compute { msg: format!("图像 {} 缺少视觉输出", range.image_index) });
            };
            let expected = range.tokens.len().checked_mul(self.config.hidden_size).ok_or_else(|| BackendError::Compute { msg: "视觉覆盖行数溢出".to_owned() })?;
            if bits.len() != expected {
                return Err(BackendError::Compute { msg: format!("图像 {} 视觉输出 rows={}，占位 token={}", range.image_index, bits.len() / self.config.hidden_size, range.tokens.len()) });
            }
        }
        Ok(VisionOverlay { images: encoded, ranges: ranges.to_vec() })
    }

    /// checkpoint 是否携带视觉塔张量;能力探测用,不触发装载。
    pub fn vision_available(&self) -> bool {
        self.weights.has_vision_tower()
    }

    /// tail prefill：输入闭包可以阻塞接收网络帧；它在独立 feeder 线程运行，
    /// 因而 tail stage 计算 chunk N 时仍可接收 chunk N+1。
    pub fn prefill_hidden_pipeline<S>(&mut self, source: S) -> Result<(RocmTensor, usize), BackendError>
    where
        S: FnMut() -> Result<Option<(usize, usize, usize, Vec<u16>)>, BackendError> + Send,
    {
        self.prefill_hidden_pipeline_with_mtp(source, None)
    }

    /// MTP 启用时带完整 prompt tokens:pipeline 各 chunk 的 collapse 输出先
    /// 深拷贝收集,结束后统一按移位 (emb(t_i), h_{i-1}) 重放,把 L45 的
    /// KV/DSA 链从位置 1 建到 prompt 末尾;否则 decode 侧首次 MTP append
    /// 会因 DSA 链缺 prefill 前缀被拒绝。
    pub fn prefill_hidden_pipeline_with_mtp<S>(&mut self, source: S, prompt_tokens: Option<Vec<u32>>) -> Result<(RocmTensor, usize), BackendError>
    where
        S: FnMut() -> Result<Option<(usize, usize, usize, Vec<u16>)>, BackendError> + Send,
    {
        let mut source = source;
        let mut last = None;
        let collect_mtp = prompt_tokens.is_some();
        let mut chunks: Vec<(usize, usize, RocmTensor)> = Vec::new();
        let tail_context = self.contexts.last().copied().expect("已校验设备非空");
        self.forward_pipeline(
            move || source().map(|input| input.map(|(position, rows, cols, values)| PipelineInput::Hidden { position, rows, cols, values })),
            |position, rows, hidden| {
                if collect_mtp {
                    let collapsed = glm53_collapse_hidden(&tail_context, &hidden, 4)?;
                    let saved = clone_resident_rows(&tail_context, &collapsed, 0, rows)?;
                    chunks.push((position, rows, saved));
                }
                last = Some((hidden, rows));
                Ok(())
            },
        )?;
        if let Some(tokens) = prompt_tokens {
            let mut prev_tail: Option<RocmTensor> = None;
            for (position, rows, collapsed) in chunks {
                self.mtp_prefill_chunk(&tokens, position, rows, &collapsed, &mut prev_tail)?;
            }
        }
        last.ok_or_else(|| BackendError::Compute { msg: "GLM-5.3-Flash tail prefill 没有输入 chunk".to_owned() })
    }

    /// prompt chunk 的 MTP 移位推进:位置 max(position,1)..position+rows 各喂
    /// (emb(t_i), h_{i-1});跨 chunk 边界用上一 chunk 尾行 hidden(prev_tail)。
    fn mtp_prefill_chunk(&mut self, tokens: &[u32], position: usize, rows: usize, collapsed: &RocmTensor, prev_tail: &mut Option<RocmTensor>) -> Result<(), BackendError> {
        if self.mtp.is_none() {
            return Ok(());
        }
        if rows == 0 || collapsed.rows != rows {
            return Err(BackendError::Compute { msg: format!("MTP prefill chunk shape position={position} rows={rows} collapsed=[{},{}] 非法", collapsed.rows, collapsed.cols) });
        }
        let context = self.contexts.last().copied().expect("已校验设备非空");
        context.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let next_tail = clone_resident_rows(&context, collapsed, rows - 1, 1)?;
        let Some((start, mtp_position, length)) = mtp_prefill_window(position, rows)? else {
            *prev_tail = Some(next_tail);
            return Ok(());
        };
        let end = start + length;
        if end > tokens.len() {
            return Err(BackendError::Compute { msg: format!("MTP prefill token 区间 [{start},{end}) 超过 prompt={}", tokens.len()) });
        }
        // 移位 hidden = 位置 start-1..end-1 的主干输出。
        let hidden = if start == 1 {
            // 首个 chunk 从位置 1 起:h_0.. 即 collapsed 前 length 行。
            clone_resident_rows(&context, collapsed, start - position, length)?
        } else {
            // 后续 chunk:h_{position-1}(上一 chunk 尾行)拼 collapsed 前 length-1 行。
            let tail = prev_tail.as_ref().expect("跨 chunk 移位需要上一 chunk 尾行");
            concat_two_rows(&context, tail, collapsed, length)?
        };
        let shift_tokens = &tokens[start..end];
        let config = &self.config;
        let device = self.contexts.len() - 1;
        let MtpRuntime { weights: mtp_weights, layer, .. } = &mut **self.mtp.as_mut().expect("已校验 MTP 装载");
        let raw = self.weights.embedding_rows_bf16(shift_tokens, config.vocab_size).map_err(|error| BackendError::Compute { msg: format!("MTP prefill embedding: {error}") })?;
        let bits = raw.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
        let embedding = context.tensor_from_bf16_bits(bits, length, config.hidden_size).map_err(|error| BackendError::Compute { msg: error.to_string() })?;
        let mla = Glm53Flash::mla_spec(config);
        let dsa_spec = Glm53Flash::dsa_spec(config);
        crate::runtime::glm53_flash::layer::glm53_mtp_cache(&context, &mut self.caches[device], &mut self.dsa[device], mtp_weights, &mla, &dsa_spec, *layer, mtp_position, &embedding, &hidden, config.hidden_size, config.rms_eps, &[], &[])?;
        *prev_tail = Some(next_tail);
        Ok(())
    }

    fn forward_pipeline<S, O>(&mut self, mut source: S, mut output: O) -> Result<(), BackendError>
    where
        S: FnMut() -> Result<Option<PipelineInput>, BackendError> + Send,
        O: FnMut(usize, usize, RocmTensor) -> Result<(), BackendError>,
    {
        if self.contexts.is_empty() {
            return Err(BackendError::Compute { msg: "GLM-5.3-Flash pipeline 没有设备".to_owned() });
        }
        let start_position = self.position;
        let config = &self.config;
        let model = &self.model;
        let weights = &self.weights;
        let layers = &self.layers;
        let contexts = &self.contexts;
        let device_count = contexts.len();
        let (input_tx, input_rx) = sync_channel::<Result<PipelineInput, BackendError>>(1);
        let mut stage_txs = Vec::with_capacity(device_count);
        let mut stage_rxs = Vec::with_capacity(device_count);
        for _ in 0..device_count {
            let (tx, rx) = sync_channel::<Result<PipelineValue, BackendError>>(1);
            stage_txs.push(Some(tx));
            stage_rxs.push(Some(rx));
        }
        let final_rx = stage_rxs[device_count - 1].take().expect("末 stage receiver 只取一次");
        let mut next_output = start_position;
        let mut completed = 0usize;
        let mut first_error = None;

        std::thread::scope(|scope| {
            scope.spawn(move || {
                loop {
                    match source() {
                        Ok(Some(input)) => {
                            if input_tx.send(Ok(input)).is_err() {
                                return;
                            }
                        }
                        Ok(None) => return,
                        Err(error) => {
                            let _ = input_tx.send(Err(error));
                            return;
                        }
                    }
                }
            });

            let mut kda = self.kda.iter_mut();
            let mut caches = self.caches.iter_mut();
            let mut dsa = self.dsa.iter_mut();
            let mut experts = self.experts.iter_mut();

            let context = contexts[0];
            let tx = stage_txs[0].take().expect("首 stage sender 只取一次");
            let stage_kda = kda.next().expect("KDA state 与设备一一对应");
            let stage_cache = caches.next().expect("KV state 与设备一一对应");
            let stage_dsa = dsa.next().expect("DSA state 与设备一一对应");
            let stage_experts = experts.next().expect("expert state 与设备一一对应");
            scope.spawn(move || {
                let mut next_position = start_position;
                for input in input_rx {
                    let input = match input {
                        Ok(input) => input,
                        Err(error) => {
                            let _ = tx.send(Err(error));
                            return;
                        }
                    };
                    let (position, rows, hidden) = match build_pipeline_input(context, config, weights, input) {
                        Ok(value) => value,
                        Err(error) => {
                            let _ = tx.send(Err(error));
                            return;
                        }
                    };
                    if position != next_position {
                        let _ = tx.send(Err(BackendError::Compute { msg: format!("GLM-5.3-Flash stage 0 position 不连续: next={next_position} input={position}") }));
                        return;
                    }
                    let transfers = crate::kernel::rocm::hip::host_transfer_bytes();
                    let hidden = match run_device_stage(context, 0, layers, config, model, stage_kda, stage_cache, stage_dsa, stage_experts, position, hidden, None) {
                        Ok(hidden) => hidden,
                        Err(error) => {
                            let _ = tx.send(Err(error));
                            return;
                        }
                    };
                    if let Err(error) = require_no_stage_host_transfer(transfers) {
                        let _ = tx.send(Err(error));
                        return;
                    }
                    next_position += rows;
                    if tx.send(Ok(PipelineValue { position, rows, hidden })).is_err() {
                        return;
                    }
                }
            });

            for device in 1..device_count {
                let rx = stage_rxs[device - 1].take().expect("stage receiver 只取一次");
                let tx = stage_txs[device].take().expect("stage sender 只取一次");
                let context = contexts[device];
                let stage_kda = kda.next().expect("KDA state 与设备一一对应");
                let stage_cache = caches.next().expect("KV state 与设备一一对应");
                let stage_dsa = dsa.next().expect("DSA state 与设备一一对应");
                let stage_experts = experts.next().expect("expert state 与设备一一对应");
                scope.spawn(move || run_pipeline_stage(rx, tx, context, device, layers, config, model, stage_kda, stage_cache, stage_dsa, stage_experts, start_position));
            }

            for item in final_rx {
                match item {
                    Ok(value) => {
                        if value.position != next_output && first_error.is_none() {
                            first_error = Some(BackendError::Compute { msg: format!("GLM-5.3-Flash pipeline 输出 position 不连续: next={next_output} output={}", value.position) });
                        }
                        next_output = value.position + value.rows;
                        completed += 1;
                        if first_error.is_none() {
                            if let Err(error) = output(value.position, value.rows, value.hidden) {
                                first_error = Some(error);
                            }
                        }
                    }
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
        });

        if let Some(error) = first_error {
            return Err(error);
        }
        if completed == 0 {
            return Err(BackendError::Compute { msg: "GLM-5.3-Flash pipeline 没有完成 chunk".to_owned() });
        }
        self.position = next_output;
        Ok(())
    }

    /// head 入口:embedding 后展开 mHC copies,返回本段末尾的展开态。
    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Result<RocmTensor, BackendError> {
        if self.layers.first().map(|(layer, _, _)| *layer) != Some(0) {
            return Err(BackendError::Compute { msg: "非 head 段不能从 token 建立 hidden".to_owned() });
        }
        if tokens.is_empty() {
            return Err(BackendError::Compute { msg: "prefill chunk 为空".to_owned() });
        }
        let position = self.position;
        let rows = tokens.len();
        let hyper = Glm53Flash::hyper_connection_spec(&self.config);
        let first = self.contexts[0];
        first.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let hidden = {
            let raw = self.weights.embedding_rows_bf16(tokens, self.config.vocab_size).map_err(|error| BackendError::Compute { msg: format!("embedding: {error}") })?;
            let bits = raw.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            let embedding = first.tensor_from_bf16_bits(bits, rows, self.config.hidden_size)?;
            glm53_expand_hidden(&first, &embedding, hyper.copies)?
        };
        self.forward_expanded(position, rows, hidden)
    }

    /// tail 入口:从网络收到的 BF16 mHC 展开态继续执行。
    pub fn forward_hidden_bits(&mut self, position: usize, rows: usize, cols: usize, values: Vec<u16>) -> Result<RocmTensor, BackendError> {
        let expected_cols = self.config.hidden_size * self.config.hyper_connection_copies;
        if rows == 0 || cols != expected_cols || values.len() != rows.saturating_mul(cols) {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash stage hidden shape=[{rows},{cols}] values={}，期望 cols={expected_cols}", values.len()) });
        }
        let first = self.contexts[0];
        first.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let hidden = first.tensor_from_bf16_bits(values, rows, cols)?;
        self.forward_expanded(position, rows, hidden)
    }

    fn forward_expanded(&mut self, position: usize, rows: usize, hidden: RocmTensor) -> Result<RocmTensor, BackendError> {
        self.forward_expanded_collect(position, rows, hidden, None)
    }

    fn forward_expanded_collect(&mut self, position: usize, rows: usize, mut hidden: RocmTensor, mut collect: Option<&mut Vec<Vec<(usize, RocmTensor)>>>) -> Result<RocmTensor, BackendError> {
        if position != self.position {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash stage position 不连续: next={} input={position}", self.position) });
        }
        let transfers_before = crate::kernel::rocm::hip::host_transfer_bytes();
        for device in 0..self.contexts.len() {
            hidden = run_device_stage(
                self.contexts[device],
                device,
                &self.layers,
                &self.config,
                &self.model,
                &mut self.kda[device],
                &mut self.caches[device],
                &mut self.dsa[device],
                &mut self.experts[device],
                position,
                hidden,
                collect.as_mut().map(|collect| &mut collect[device]),
            )?;
        }
        require_no_stage_host_transfer(transfers_before)?;
        self.position += rows;
        Ok(hidden)
    }

    /// 完整尾段输出:collapse mHC copies 后只对指定行执行 head。
    pub fn sample_hidden(&self, hidden: &RocmTensor, row: usize) -> Result<u32, crate::backend::BackendError> {
        let output = self.output.as_ref().ok_or_else(|| BackendError::Compute { msg: "GLM-5.3-Flash 输出头未装载".to_owned() })?;
        let context = self.contexts.last().expect("已校验设备非空");
        context.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let hidden = glm53_collapse_hidden(context, hidden, self.config.hyper_connection_copies)?;
        let plan = crate::runtime::output::OutputPlan { eps: self.config.rms_eps, norm: crate::runtime::output::OutputNorm::Rms, excluded_tokens: Vec::new() };
        let result = crate::runtime::output::last_token_output(context, &output.head, &hidden, row, &plan)?;
        Ok(result.token_id)
    }

    /// 完整尾段输出:collapse 后把 verify 各行合成一次 norm / LM head。
    pub fn sample_all_rows(&self, hidden: &RocmTensor, rows: usize) -> Result<Vec<u32>, BackendError> {
        let output = self.output.as_ref().ok_or_else(|| BackendError::Compute { msg: "GLM-5.3-Flash 输出头未装载".to_owned() })?;
        let context = self.contexts.last().expect("已校验设备非空");
        context.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let collapsed = glm53_collapse_hidden(context, hidden, self.config.hyper_connection_copies)?;
        if collapsed.rows != rows {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash verify 输出 rows={}，期望 {rows}", collapsed.rows) });
        }
        let plan = crate::runtime::output::OutputPlan { eps: self.config.rms_eps, norm: crate::runtime::output::OutputNorm::Rms, excluded_tokens: Vec::new() };
        crate::runtime::output::token_ids(context, &output.head, &collapsed, &plan)
    }

    /// tail speculative verify:推进前字节快照 KDA 状态,forward 中收集
    /// 各卡 KDA 层 anchor 行(行 0)的 attention 输入,供拒绝后重放。
    pub fn verify_bits_collected(&mut self, position: usize, rows: usize, cols: usize, values: Vec<u16>) -> Result<RocmTensor, BackendError> {
        self.snapshot_kda()?;
        let expected_cols = self.config.hidden_size * self.config.hyper_connection_copies;
        if rows == 0 || cols != expected_cols || values.len() != rows.saturating_mul(cols) {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash verify hidden shape=[{rows},{cols}] values={}，期望 cols={expected_cols}", values.len()) });
        }
        let first = self.contexts[0];
        first.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let hidden = first.tensor_from_bf16_bits(values, rows, cols)?;
        let mut collect: Vec<Vec<(usize, RocmTensor)>> = vec![Vec::new(); self.contexts.len()];
        let completed = self.forward_expanded_collect(position, rows, hidden, Some(&mut collect))?;
        for (device, inputs) in collect.into_iter().enumerate() {
            if let Some(snapshot) = self.kda_snapshot[device].as_mut() {
                snapshot.layer_inputs = inputs;
            }
        }
        Ok(completed)
    }

    /// head speculative verify:embedding/展开后带 KDA 快照与 anchor 行收集。
    pub fn verify_tokens_collected(&mut self, tokens: &[u32]) -> Result<RocmTensor, BackendError> {
        if self.layers.first().map(|(layer, _, _)| *layer) != Some(0) {
            return Err(BackendError::Compute { msg: "非 head 段不能从 token 建立 hidden".to_owned() });
        }
        if tokens.is_empty() {
            return Err(BackendError::Compute { msg: "verify chunk 为空".to_owned() });
        }
        self.snapshot_kda()?;
        let position = self.position;
        let rows = tokens.len();
        let hyper = Glm53Flash::hyper_connection_spec(&self.config);
        let first = self.contexts[0];
        first.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let hidden = {
            let raw = self.weights.embedding_rows_bf16(tokens, self.config.vocab_size).map_err(|error| BackendError::Compute { msg: format!("embedding: {error}") })?;
            let bits = raw.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            let embedding = first.tensor_from_bf16_bits(bits, rows, self.config.hidden_size)?;
            glm53_expand_hidden(&first, &embedding, hyper.copies)?
        };
        let mut collect: Vec<Vec<(usize, RocmTensor)>> = vec![Vec::new(); self.contexts.len()];
        let completed = self.forward_expanded_collect(position, rows, hidden, Some(&mut collect))?;
        for (device, inputs) in collect.into_iter().enumerate() {
            if let Some(snapshot) = self.kda_snapshot[device].as_mut() {
                snapshot.layer_inputs = inputs;
            }
        }
        Ok(completed)
    }

    /// KDA 字节快照:拷贝每卡已分配层的 conv/recurrent,并记录层序与 position。
    fn snapshot_kda(&mut self) -> Result<(), BackendError> {
        for device in 0..self.contexts.len() {
            let context = self.contexts[device];
            context.activate().map_err(|error| BackendError::Compute { msg: error })?;
            let mut conv = Vec::new();
            let mut recurrent = Vec::new();
            let mut layers = Vec::new();
            let mut positions = Vec::new();
            for layer in 0..self.kda[device].layer_count() {
                let Some(storage) = self.kda[device].layer_storage(layer) else { continue };
                conv.push(clone_resident_bytes(&context, &storage.conv)?);
                recurrent.push(clone_resident_bytes(&context, &storage.recurrent)?);
                layers.push(layer);
                positions.push(self.kda[device].layer_position(layer).unwrap_or(0));
            }
            self.kda_snapshot[device] = Some(KdaSnapshot { conv, recurrent, layers, positions, layer_inputs: Vec::new() });
        }
        Ok(())
    }

    /// verify 拒绝后回退:恢复 KDA 字节快照并按序重放 retained 前缀，
    /// 主干 MLA/DSA truncate 到 keep_position。MTP speculative cache 由调用方
    /// 从 verify 起点前一行单独重建，不能跟主干保留相同长度。
    /// 重放只跑 KDA attention(mHC/FFN 无状态),无需重跑 FFN。
    pub fn rollback_verify(&mut self, keep_position: usize) -> Result<(), BackendError> {
        if keep_position > self.position {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash rollback keep={keep_position} 超过当前 position={}", self.position) });
        }
        let kda_spec = Glm53Flash::kda_spec(&self.config);
        for device in 0..self.contexts.len() {
            let context = self.contexts[device];
            context.activate().map_err(|error| BackendError::Compute { msg: error })?;
            let Some(snapshot) = self.kda_snapshot[device].take() else { continue };
            for (index, &layer) in snapshot.layers.iter().enumerate() {
                let Some(storage) = self.kda[device].layer_storage(layer) else { continue };
                restore_resident_bytes(&context, &storage.conv, &snapshot.conv[index])?;
                restore_resident_bytes(&context, &storage.recurrent, &snapshot.recurrent[index])?;
                self.kda[device].rewind_layer(layer, snapshot.positions[index])?;
            }
            for (layer, input) in snapshot.layer_inputs {
                let Some((_, _, prepared)) = self.layers.iter().find(|(candidate, owner, _)| *candidate == layer && *owner == device) else { continue };
                let core = match prepared {
                    Glm53PreparedLayer::Dense(prepared) => &prepared.core,
                    Glm53PreparedLayer::Moe(prepared) => &prepared.core,
                };
                let Glm53AttentionWeights::Kda(weights) = &core.attention else { continue };
                let position = self.kda[device].layer_position(layer).unwrap_or(keep_position);
                let retained = keep_position.checked_sub(position).ok_or_else(|| BackendError::Compute { msg: format!("L{layer} KDA rollback keep={keep_position} 早于快照 position={position}") })?;
                if retained == 0 || retained > input.rows {
                    return Err(BackendError::Compute { msg: format!("L{layer} KDA rollback retained={retained} 超过 verify rows={}", input.rows) });
                }
                let input = clone_resident_rows(&context, &input, 0, retained)?;
                glm53_kda_attention(&context, &mut self.kda[device], layer, position, &input, weights, &kda_spec)?;
            }
            for (layer, owner, _) in &self.layers {
                if *owner != device {
                    continue;
                }
                self.caches[device].truncate_layer_rows(*layer, keep_position)?;
                self.dsa[device].truncate_layer_rows(*layer, keep_position)?;
            }
        }
        self.position = keep_position;
        Ok(())
    }

    /// verify 全部保留时无需恢复 KDA，只释放本轮快照。
    pub fn commit_verify(&mut self) {
        self.kda_snapshot.fill_with(|| None);
    }

    /// MTP cache 的逻辑行数恒等于下一次 target position - 1。
    fn truncate_mtp(&mut self, rows: usize) -> Result<(), BackendError> {
        let device = self.contexts.len() - 1;
        if self.mtp.is_none() {
            return Err(BackendError::Compute { msg: "GLM-5.3-Flash MTP 未装载".to_owned() });
        }
        self.caches[device].truncate_layer_rows(self.config.layer_count, rows)?;
        self.dsa[device].truncate_layer_rows(self.config.layer_count, rows)
    }

    /// MTP 单步推进:位置 `position` 吃 (token, 前一位置 hidden)，返回 draft
    /// 与 shared_head norm 后的 hidden；后者直接作为下一递归深度的输入。
    fn mtp_step(&mut self, token: u32, previous_hidden: &RocmTensor, position: usize) -> Result<(u32, RocmTensor), BackendError> {
        let output = self.output.as_ref().ok_or_else(|| BackendError::Compute { msg: "GLM-5.3-Flash MTP 需要输出头采样".to_owned() })?;
        let device = self.contexts.len() - 1;
        let context = self.contexts[device];
        context.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let hidden_size = self.config.hidden_size;
        if previous_hidden.cols != hidden_size || previous_hidden.rows != 1 {
            return Err(BackendError::Compute { msg: format!("MTP 输入 hidden shape=[{},{}] 期望 [1,{hidden_size}]", previous_hidden.rows, previous_hidden.cols) });
        }
        let mla = Glm53Flash::mla_spec(&self.config);
        let dsa_spec = Glm53Flash::dsa_spec(&self.config);
        let moe_spec = Glm53Flash::moe_spec(&self.config);
        let MtpRuntime { weights: mtp, router_weight, router_bias, shared, experts, layer } = &mut **self.mtp.as_mut().ok_or_else(|| BackendError::Compute { msg: "GLM-5.3-Flash MTP 未装载".to_owned() })?;
        let raw = self.weights.embedding_rows_bf16(std::slice::from_ref(&token), self.config.vocab_size).map_err(|error| BackendError::Compute { msg: format!("MTP embedding: {error}") })?;
        let bits = raw.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
        let embedding = context.tensor_from_bf16_bits(bits, 1, hidden_size)?;
        let projected = crate::runtime::glm53_flash::layer::glm53_mtp_project(&context, mtp, &embedding, previous_hidden, hidden_size, self.config.rms_eps)?;
        let hidden = crate::runtime::glm53_flash::layer::glm53_mtp_layer(&context, Some(&mut self.caches[device]), &mut self.dsa[device], &mtp.layer, &mla, &dsa_spec, *layer, position, projected, self.config.rms_eps, &[], &[], |input| {
            moe_prefill(&context, &moe_spec, *layer, (router_weight, router_bias, shared), experts, input)
        })?;
        let normed = context.rmsnorm(&hidden, &mtp.output_norm, self.config.rms_eps)?;
        let token = crate::runtime::output::normalized_token_id(&context, &output.head, &normed, &[])?;
        Ok((token, normed))
    }

    /// 从一个 target token/hidden 递归生成固定上限的 draft；遇到 EOS 提前停止。
    pub fn mtp_drafts(&mut self, token: u32, previous_hidden: &RocmTensor, position: usize, count: usize) -> Result<Vec<u32>, BackendError> {
        if count == 0 || count > 8 {
            return Err(BackendError::Compute { msg: format!("MTP draft count={count} 必须在 1..=8") });
        }
        let mut drafts = Vec::with_capacity(count);
        let mut token = token;
        let mut hidden = previous_hidden.clone();
        for depth in 0..count {
            let (draft, next_hidden) = self.mtp_step(token, &hidden, position + depth)?;
            drafts.push(draft);
            token = draft;
            hidden = next_hidden;
            if self.config.eos_token_ids.contains(&draft) {
                break;
            }
        }
        Ok(drafts)
    }

    /// 已确认行只追赶 MTP cache；其输出不会参与采样，不能再跑整层 MoE/LM head。
    fn mtp_catch_up_rows(&mut self, tokens: &[u32], shifted_hidden: &RocmTensor, position: usize) -> Result<(), BackendError> {
        let device = self.contexts.len() - 1;
        let context = self.contexts[device];
        context.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let hidden_size = self.config.hidden_size;
        if tokens.is_empty() || shifted_hidden.cols != hidden_size || shifted_hidden.rows != tokens.len() {
            return Err(BackendError::Compute { msg: format!("MTP catch-up tokens={} hidden=[{},{}] 期望 [tokens,{hidden_size}]", tokens.len(), shifted_hidden.rows, shifted_hidden.cols) });
        }
        let raw = self.weights.embedding_rows_bf16(tokens, self.config.vocab_size).map_err(|error| BackendError::Compute { msg: format!("MTP catch-up embedding: {error}") })?;
        let bits = raw.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
        let embedding = context.tensor_from_bf16_bits(bits, tokens.len(), hidden_size)?;
        let mla = Glm53Flash::mla_spec(&self.config);
        let dsa_spec = Glm53Flash::dsa_spec(&self.config);
        let MtpRuntime { weights: mtp, layer, .. } = &mut **self.mtp.as_mut().ok_or_else(|| BackendError::Compute { msg: "GLM-5.3-Flash MTP 未装载".to_owned() })?;
        crate::runtime::glm53_flash::layer::glm53_mtp_cache(&context, &mut self.caches[device], &mut self.dsa[device], mtp, &mla, &dsa_spec, *layer, position, &embedding, shifted_hidden, hidden_size, self.config.rms_eps, &[], &[])
    }

    /// verify 后丢弃递归 draft cache，用 target 的真实 hidden 重建 retained 前缀，
    /// 再从最新确认 token 递归产生下一轮 drafts。
    pub fn mtp_rebuild_drafts(
        &mut self,
        verify_inputs: &[u32],
        target_hidden: &RocmTensor,
        previous_hidden: &RocmTensor,
        position: usize,
        retained_rows: usize,
        anchor: u32,
        draft_count: usize,
    ) -> Result<(Vec<u32>, RocmTensor), BackendError> {
        if retained_rows == 0 || retained_rows > verify_inputs.len() {
            return Err(BackendError::Compute { msg: format!("MTP rebuild retained={retained_rows} verify_inputs={}", verify_inputs.len()) });
        }
        let context = self.contexts.last().copied().expect("已校验设备非空");
        context.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let collapsed = glm53_collapse_hidden(&context, target_hidden, self.config.hyper_connection_copies)?;
        if collapsed.rows != verify_inputs.len() || previous_hidden.rows != 1 || previous_hidden.cols != collapsed.cols {
            return Err(BackendError::Compute { msg: format!("MTP rebuild shape verify={} target=[{},{}] previous=[{},{}]", verify_inputs.len(), collapsed.rows, collapsed.cols, previous_hidden.rows, previous_hidden.cols) });
        }
        let terminal_hidden = clone_resident_rows(&context, &collapsed, retained_rows - 1, 1)?;
        let shifted_hidden = concat_two_rows(&context, previous_hidden, &collapsed, retained_rows)?;
        let mtp_base = position.saturating_sub(1);
        self.truncate_mtp(mtp_base)?;
        self.mtp_catch_up_rows(&verify_inputs[..retained_rows], &shifted_hidden, mtp_base)?;
        let drafts = self.mtp_drafts(anchor, &terminal_hidden, position + retained_rows - 1, draft_count)?;
        Ok((drafts, terminal_hidden))
    }

    /// 主干 hidden collapse 后取指定行深拷贝;MTP 移位输入跨轮存活。
    pub fn collapse_row(&self, hidden: &RocmTensor, row: usize) -> Result<RocmTensor, BackendError> {
        let context = self.contexts.last().expect("已校验设备非空");
        context.activate().map_err(|error| BackendError::Compute { msg: error })?;
        let collapsed = glm53_collapse_hidden(context, hidden, self.config.hyper_connection_copies)?;
        if row >= collapsed.rows {
            return Err(BackendError::Compute { msg: format!("collapse_row {row} 越界 {}", collapsed.rows) });
        }
        clone_resident_rows(context, &collapsed, row, 1)
    }

    pub fn boundary_bits(&self, hidden: &RocmTensor) -> Result<Vec<u16>, BackendError> {
        let context = self.contexts.last().expect("已校验设备非空");
        context.tensor_to_bf16_bits(hidden)
    }

    pub fn boundary_cols(&self) -> usize {
        self.config.hidden_size * self.config.hyper_connection_copies
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) -> Result<(), BackendError> {
        let spec = Glm53Flash::kda_spec(&self.config);
        for state in &mut self.kda {
            *state = KdaState::new(self.config.layer_count, spec)?;
        }
        for cache in &mut self.caches {
            cache.truncate_rows(0)?;
        }
        for state in &mut self.dsa {
            state.truncate_rows(0)?;
        }
        self.kda_snapshot.fill_with(|| None);
        self.position = 0;
        Ok(())
    }

    pub fn device_count(&self) -> usize {
        self.contexts.len()
    }

    pub fn device_memory(&self) -> Result<Vec<crate::server::stage_transport::StageDeviceMemory>, String> {
        self.contexts
            .iter()
            .enumerate()
            .map(|(device, context)| {
                let model_units = self.layers.iter().filter(|(_, owner, _)| *owner == device).count();
                Ok(crate::server::stage_transport::StageDeviceMemory {
                    device: context.device_id(),
                    model_units,
                    available_bytes: context.stage_available_bytes().map_err(|error| format!("device {} 可用显存: {error:?}", context.device_id()))? as u64,
                    total_bytes: context.stage_total_bytes().map_err(|error| format!("device {} 总显存: {error:?}", context.device_id()))? as u64,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod smoke_tests {
    use super::mtp_prefill_window;

    #[test]
    fn mtp_prefill_window_is_shifted_one_row_behind_target() {
        assert_eq!(mtp_prefill_window(0, 1).unwrap(), None);
        assert_eq!(mtp_prefill_window(0, 2048).unwrap(), Some((1, 0, 2047)));
        assert_eq!(mtp_prefill_window(2048, 100).unwrap(), Some((2048, 2047, 100)));
    }
}

fn build_pipeline_input(context: RocmContext, config: &Glm53FlashConfig, weights: &prepare_weights::Weights, input: PipelineInput) -> Result<(usize, usize, RocmTensor), BackendError> {
    context.activate().map_err(|error| BackendError::Compute { msg: error })?;
    match input {
        PipelineInput::Tokens { position, tokens } => {
            if tokens.is_empty() {
                return Err(BackendError::Compute { msg: "prefill chunk 为空".to_owned() });
            }
            let rows = tokens.len();
            let raw = weights.embedding_rows_bf16(&tokens, config.vocab_size).map_err(|error| BackendError::Compute { msg: format!("embedding: {error}") })?;
            let bits = raw.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            let embedding = context.tensor_from_bf16_bits(bits, rows, config.hidden_size)?;
            let hidden = glm53_expand_hidden(&context, &embedding, config.hyper_connection_copies)?;
            Ok((position, rows, hidden))
        }
        PipelineInput::VisionTokens { position, tokens, overlay } => {
            if tokens.is_empty() {
                return Err(BackendError::Compute { msg: "prefill chunk 为空".to_owned() });
            }
            let rows = tokens.len();
            let raw = weights.embedding_rows_bf16(&tokens, config.vocab_size).map_err(|error| BackendError::Compute { msg: format!("embedding: {error}") })?;
            let mut bits = raw.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            splice_image_embeddings_bf16(&mut bits, config.hidden_size, position, &overlay.ranges, &overlay.images).map_err(|error| BackendError::Compute { msg: format!("视觉行覆盖: {error}") })?;
            let embedding = context.tensor_from_bf16_bits(bits, rows, config.hidden_size)?;
            let hidden = glm53_expand_hidden(&context, &embedding, config.hyper_connection_copies)?;
            Ok((position, rows, hidden))
        }
        PipelineInput::Hidden { position, rows, cols, values } => {
            let expected_cols = config.hidden_size * config.hyper_connection_copies;
            if rows == 0 || cols != expected_cols || values.len() != rows.saturating_mul(cols) {
                return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash stage hidden shape=[{rows},{cols}] values={}，期望 cols={expected_cols}", values.len()) });
            }
            let hidden = context.tensor_from_bf16_bits(values, rows, cols)?;
            Ok((position, rows, hidden))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pipeline_stage(
    rx: Receiver<Result<PipelineValue, BackendError>>,
    tx: SyncSender<Result<PipelineValue, BackendError>>,
    context: RocmContext,
    device: usize,
    layers: &[(usize, usize, Glm53PreparedLayer<RocmWeight>)],
    config: &Glm53FlashConfig,
    model: &Glm53Flash,
    kda: &mut KdaState<crate::backend::rocm::RocmKdaStorage>,
    cache: &mut RocmKvCache,
    dsa: &mut RocmDsaState,
    experts: &mut RocmPrefillExperts,
    start_position: usize,
) {
    let mut next_position = start_position;
    for item in rx {
        let value = match item {
            Ok(value) => value,
            Err(error) => {
                let _ = tx.send(Err(error));
                return;
            }
        };
        if value.position != next_position {
            let _ = tx.send(Err(BackendError::Compute { msg: format!("GLM-5.3-Flash stage {device} position 不连续: next={next_position} input={}", value.position) }));
            return;
        }
        let transfers = crate::kernel::rocm::hip::host_transfer_bytes();
        let hidden = match run_device_stage(context, device, layers, config, model, kda, cache, dsa, experts, value.position, value.hidden, None) {
            Ok(hidden) => hidden,
            Err(error) => {
                let _ = tx.send(Err(error));
                return;
            }
        };
        if let Err(error) = require_no_stage_host_transfer(transfers) {
            let _ = tx.send(Err(error));
            return;
        }
        next_position += value.rows;
        if tx.send(Ok(PipelineValue { hidden, ..value })).is_err() {
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_device_stage(
    context: RocmContext,
    device: usize,
    layers: &[(usize, usize, Glm53PreparedLayer<RocmWeight>)],
    config: &Glm53FlashConfig,
    model: &Glm53Flash,
    kda: &mut KdaState<crate::backend::rocm::RocmKdaStorage>,
    cache: &mut RocmKvCache,
    dsa: &mut RocmDsaState,
    experts: &mut RocmPrefillExperts,
    position: usize,
    hidden: RocmTensor,
    collect: Option<&mut Vec<(usize, RocmTensor)>>,
) -> Result<RocmTensor, BackendError> {
    context.activate().map_err(|error| BackendError::Compute { msg: error })?;
    let mut hidden = context.move_tensor_to_stage_ordered(hidden)?;
    let hyper = Glm53Flash::hyper_connection_spec(config);
    let mla = Glm53Flash::mla_spec(config);
    let dsa_spec = Glm53Flash::dsa_spec(config);
    let kda_spec = Glm53Flash::kda_spec(config);
    let moe_spec = Glm53Flash::moe_spec(config);
    let mut collector = collect.map(|collector| (collector, context.clone()));
    for (layer, _, prepared) in layers.iter().filter(|(_, owner, _)| *owner == device) {
        let spec = model.layer_spec(*layer).map_err(|error| BackendError::Compute { msg: format!("L{layer} spec: {error}") })?;
        let core = match prepared {
            Glm53PreparedLayer::Dense(prepared) => &prepared.core,
            Glm53PreparedLayer::Moe(prepared) => &prepared.core,
        };
        hidden = match (&core.attention, &spec.attention, prepared) {
            (Glm53AttentionWeights::Kda(_), AttentionSpec::Kda(_), Glm53PreparedLayer::Dense(prepared)) => {
                let ffn = &prepared.feedforward;
                let mut observer = |input: &RocmTensor| {
                    if let Some((collector, context)) = collector.as_mut() {
                        if let Ok(block) = clone_resident_tensor(context, input) {
                            collector.push((*layer, block));
                        }
                    }
                };
                glm53_kda_layer_with_observer(
                    &context,
                    kda,
                    &hyper,
                    core,
                    &kda_spec,
                    *layer,
                    position,
                    hidden,
                    config.rms_eps,
                    |input| {
                        let activated = context.gated_linear(input, &ffn.gate, &ffn.up, &crate::moe::Activation::Silu)?;
                        context.linear(&activated, &ffn.down)
                    },
                    &mut observer,
                )?
            }
            (Glm53AttentionWeights::Kda(_), AttentionSpec::Kda(_), Glm53PreparedLayer::Moe(prepared)) => {
                let router = (&prepared.router_weight, &prepared.router_bias, &prepared.shared);
                let mut observer = |input: &RocmTensor| {
                    if let Some((collector, context)) = collector.as_mut() {
                        if let Ok(block) = clone_resident_tensor(context, input) {
                            collector.push((*layer, block));
                        }
                    }
                };
                glm53_kda_layer_with_observer(&context, kda, &hyper, core, &kda_spec, *layer, position, hidden, config.rms_eps, |input| moe_prefill(&context, &moe_spec, *layer, router, experts, input), &mut observer)?
            }
            (Glm53AttentionWeights::DsaMla(_), AttentionSpec::Mla(_), Glm53PreparedLayer::Dense(prepared)) => {
                let ffn = &prepared.feedforward;
                glm53_dsa_mla_layer(&context, Some(cache), dsa, &hyper, core, &mla, &dsa_spec, *layer, position, hidden, config.rms_eps, &[], &[], |input| {
                    let activated = context.gated_linear(input, &ffn.gate, &ffn.up, &crate::moe::Activation::Silu)?;
                    context.linear(&activated, &ffn.down)
                })?
            }
            (Glm53AttentionWeights::DsaMla(_), AttentionSpec::Mla(_), Glm53PreparedLayer::Moe(prepared)) => {
                let router = (&prepared.router_weight, &prepared.router_bias, &prepared.shared);
                glm53_dsa_mla_layer(&context, Some(cache), dsa, &hyper, core, &mla, &dsa_spec, *layer, position, hidden, config.rms_eps, &[], &[], |input| moe_prefill(&context, &moe_spec, *layer, router, experts, input))?
            }
            _ => return Err(BackendError::Compute { msg: format!("L{layer} prepared 与 spec 不一致") }),
        };
    }
    Ok(hidden)
}

fn require_no_stage_host_transfer(before: (u64, u64)) -> Result<(), BackendError> {
    let after = crate::kernel::rocm::hip::host_transfer_bytes();
    if after == before {
        return Ok(());
    }
    Err(BackendError::Compute { msg: format!("GLM-5.3-Flash GPU resident 层路径发生 host 传输: H2D={} D2H={} bytes", after.0.saturating_sub(before.0), after.1.saturating_sub(before.1)) })
}

fn moe_prefill(
    context: &RocmContext,
    spec: &crate::moe::topk_moe::TopkMoeSpec,
    layer: usize,
    router: (&RocmWeight, &RocmWeight, &crate::moe::DenseFfn<RocmWeight>),
    experts: &mut RocmPrefillExperts,
    input: &RocmTensor,
) -> Result<RocmTensor, BackendError> {
    let (router_weight, router_bias, shared) = router;
    let shared_ref = [crate::moe::topk_moe::SharedExpertRef { gate: &shared.gate, up: &shared.up, down: &shared.down, output_gate: None }];
    let weights = crate::moe::topk_moe::MoeFfnRef { router_weight, router_bias, shared_experts: &shared_ref, selected_experts: None };
    crate::moe::prefill::prefill_experts_untraced(context, spec, &weights, layer, experts, input, None)
}

fn dims(config: &Glm53FlashConfig, vision: &Glm53FlashVisionConfig) -> crate::weight::model::glm53_flash::Glm53FlashDims {
    use crate::weight::model::glm53_flash::Glm53FlashDims;
    Glm53FlashDims {
        hidden_size: config.hidden_size,
        q_lora_rank: config.q_lora_rank,
        kv_lora_rank: config.kv_lora_rank,
        q_projection_size: config.num_attention_heads * config.qk_nope_head_dim,
        kv_projection_size: config.num_attention_heads * (config.qk_nope_head_dim + config.value_head_dim),
        kda_projection_size: config.kda_num_heads * config.kda_head_dim,
        kda_num_heads: config.kda_num_heads,
        kda_head_dim: config.kda_head_dim,
        kda_short_conv_kernel: config.kda_short_conv_kernel_size,
        kda_decay_rank: config.kda_decay_rank,
        dense_intermediate_size: config.dense_intermediate_size,
        moe_intermediate_size: config.expert_intermediate_size,
        expert_count: config.expert_count,
        index_num_heads: config.index_num_heads,
        index_head_dim: config.index_head_dim,
        index_kpool: config.index_kpool,
        // 官方 vision_config(glm5_next_vision)经 Glm53FlashVisionConfig 单一来源下发。
        vision: vision.dims(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Amd-1 真机 head 冒烟:加载 0..23 层到 8 卡,两轮 chunk prefill,
    /// 校验 boundary hidden 有限且非零。需要 62 片权重与 8 卡 ROCm,
    /// 仅在 --ignored 时执行。
    #[test]
    #[ignore = "需要 Amd-1 全量权重与 8 卡 ROCm"]
    fn head_smoke_prefill() {
        let options = Options {
            weights_directory: "/workspace/models/GLM-5.3-Flash".into(),
            devices: vec![0, 1, 2, 3, 4, 5, 6, 7],
            layer_ends: vec![3, 6, 9, 12, 14, 17, 20, 23],
            layer_start: 0,
            layer_end: 23,
            max_sequence_length: 4096,
            prefill_chunk_size: 512,
            mtp: false,
        };
        let started = std::time::Instant::now();
        let mut engine = Engine::load(&options).expect("head 引擎装载");
        eprintln!("[head-smoke] 装载完成: {} 卡 {:.1}s", engine.device_count(), started.elapsed().as_secs_f32());
        for chunk in 0..2 {
            let tokens: Vec<u32> = (0..512).map(|index| (index % 1000 + 100) as u32).collect();
            let boundary = engine.forward_tokens(&tokens).expect("prefill chunk");
            let values = engine.contexts.last().unwrap().tensor_to_f32(&boundary).expect("读回 boundary");
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            assert!(values.iter().all(|value| value.is_finite()), "boundary 出现非有限值");
            assert!(maximum > 0.0, "boundary 全零");
            eprintln!("[head-smoke] chunk {chunk}: position={} boundary[{}] max={maximum:.4} mean={mean:.5}", engine.position(), values.len());
        }
        // head 的单 token 通路只推进状态,采样必须留在完整 tail。
        for step in 0..3 {
            let boundary = engine.forward_tokens(&[100 + step as u32]).expect("decode token");
            assert_eq!(engine.boundary_bits(&boundary).unwrap().len(), engine.boundary_cols());
            eprintln!("[head-smoke] decode {step}: position={}", engine.position());
        }
    }
}
