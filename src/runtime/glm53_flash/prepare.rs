//! GLM-5.3-Flash checkpoint 权重到 backend resident 权重的适配层。

use crate::{
    backend::{Backend, BackendError, LinearWeight},
    moe::DenseFfn,
    runtime::{
        glm53_flash::{
            Glm53FlashConfig,
            layer::{Glm53HyperConnection, Glm53KdaWeights, KdaConvPath, KdaDecayPath, KdaFactoredGatePath},
        },
        output::{self, OutputHead},
    },
    weight::{
        Fp8Matrix,
        container::safetensor::TensorData,
        expert_source::{ExpertSource, ExpertSourceProvider, Fp8ExpertSource},
        format::official_fp8::Fp8ExpertWeights,
        model::glm53_flash::{Glm53DenseMlpWeights, Glm53FlashWeights, Glm53KdaLayerWeights, Glm53MhcTrio},
    },
};

/// dense FFN 层(前 3 层):层骨架 + dense 前馈。
pub struct Glm53DenseLayer<W> {
    pub core: crate::runtime::glm53_flash::layer::Glm53LayerWeights<W>,
    pub feedforward: DenseFfn<W>,
}

/// MoE 层:层骨架 + 路由器 + 共享专家;路由专家经 expert source 由 backend 管理。
pub struct Glm53MoeLayer<W> {
    pub core: crate::runtime::glm53_flash::layer::Glm53LayerWeights<W>,
    pub router_weight: W,
    pub router_bias: W,
    pub shared: DenseFfn<W>,
}

pub enum Glm53PreparedLayer<W> {
    Dense(Glm53DenseLayer<W>),
    Moe(Glm53MoeLayer<W>),
}

pub struct Glm53PreparedOutput<W> {
    pub head: OutputHead<W>,
}

/// LM head 很大,调用方应只在确定其驻留策略后显式执行。
pub fn load_prepare_output<B: Backend>(backend: &B, source: &Glm53FlashWeights, config: &Glm53FlashConfig) -> Result<Glm53PreparedOutput<B::Weight>, BackendError> {
    let final_norm = source.final_norm().map_err(|msg| part_error("final norm", msg))?;
    let lm_head = source.lm_head(config.vocab_size).map_err(|msg| part_error("LM head", msg))?;
    let head = output::prepare_output_head_weight_quantized(backend, LinearWeight::Bf16Bytes(&final_norm.data), LinearWeight::Bf16Bytes(&lm_head.data), config.vocab_size, config.hidden_size, crate::weight::LmHeadQuantization::Native)?;
    Ok(Glm53PreparedOutput { head })
}

/// 装配一个层:dense/MoE 与 KDA/DSA-MLA 四种组合按 config 的层表分发。
pub fn prepare_layer<B: Backend>(backend: &B, source: &Glm53FlashWeights, config: &Glm53FlashConfig, model: &crate::runtime::glm53_flash::Glm53Flash, layer: usize) -> Result<Glm53PreparedLayer<B::Weight>, BackendError> {
    let common = source.load_layer_common(layer).map_err(|msg| part_error(format!("L{layer} common"), msg))?;
    let mhc = source.load_mhc(layer, config.hyper_connection_copies).map_err(|msg| part_error(format!("L{layer} mhc"), msg))?;
    let input_norm = prepare_bf16(backend, &common.input_norm)?;
    let post_attention_norm = prepare_bf16(backend, &common.post_attention_norm)?;
    let attention = if model.is_full_attention(layer) {
        let weights = source.load_dsa_mla(layer).map_err(|msg| part_error(format!("L{layer} DSA-MLA"), msg))?;
        crate::runtime::glm53_flash::layer::Glm53AttentionWeights::DsaMla(prepare_dsa_mla(backend, &weights)?)
    } else {
        let weights = source.load_kda(layer).map_err(|msg| part_error(format!("L{layer} KDA"), msg))?;
        crate::runtime::glm53_flash::layer::Glm53AttentionWeights::Kda(prepare_kda(backend, &weights)?)
    };
    let core = crate::runtime::glm53_flash::layer::Glm53LayerWeights {
        input_norm,
        post_attention_norm,
        attention_hyper_connection: prepare_mhc(backend, &mhc.attention, config)?,
        attention,
        feedforward_hyper_connection: prepare_mhc(backend, &mhc.feedforward, config)?,
    };
    if layer < config.dense_layer_count {
        let weights = source.load_dense_mlp(layer).map_err(|msg| part_error(format!("L{layer} dense FFN"), msg))?;
        Ok(Glm53PreparedLayer::Dense(Glm53DenseLayer { core, feedforward: prepare_dense_mlp(backend, &weights)? }))
    } else {
        let router = source.load_moe_router(layer).map_err(|msg| part_error(format!("L{layer} MoE router"), msg))?;
        let shared = source.load_shared_experts(layer).map_err(|msg| part_error(format!("L{layer} shared experts"), msg))?;
        Ok(Glm53PreparedLayer::Moe(Glm53MoeLayer {
            core,
            router_weight: backend.prepare_f32(&router.router, config.expert_count, config.hidden_size)?,
            router_bias: prepare_f32_vector(backend, &router.correction_bias, "MoE router bias")?,
            shared: prepare_dense_mlp(backend, &shared)?,
        }))
    }
}

/// MTP 头(layers.{layer_count});transformer 层部分为 DSA-MLA + MoE,无 mHC。
pub fn prepare_mtp<B: Backend>(backend: &B, source: &Glm53FlashWeights, config: &Glm53FlashConfig) -> Result<crate::runtime::glm53_flash::layer::Glm53Mtp<B::Weight>, BackendError> {
    if config.mtp_layer_count == 0 {
        return Err(BackendError::Compute { msg: "GLM-5.3-Flash MTP 未启用(mtp_layer_count=0)".to_owned() });
    }
    let layer = config.layer_count;
    let mtp = source.load_mtp(layer).map_err(|msg| part_error("MTP head", msg))?;
    let attention = source.load_dsa_mla(layer).map_err(|msg| part_error("MTP DSA-MLA", msg))?;
    let common = source.load_layer_common(layer).map_err(|msg| part_error("MTP common", msg))?;
    Ok(crate::runtime::glm53_flash::layer::Glm53Mtp {
        embedding_norm: prepare_bf16(backend, &mtp.embedding_norm)?,
        hidden_norm: prepare_bf16(backend, &mtp.hidden_norm)?,
        input_projection: prepare_bf16(backend, &mtp.input_projection)?,
        layer: crate::runtime::glm53_flash::layer::Glm53MtpLayerWeights {
            input_norm: prepare_bf16(backend, &common.input_norm)?,
            post_attention_norm: prepare_bf16(backend, &common.post_attention_norm)?,
            attention: prepare_dsa_mla(backend, &attention)?,
        },
        output_norm: prepare_bf16(backend, &mtp.shared_head_norm)?,
    })
}

pub fn prepare_kda<B: Backend>(backend: &B, source: &Glm53KdaLayerWeights) -> Result<Glm53KdaWeights<B::Weight>, BackendError> {
    let conv = |source: &crate::weight::model::glm53_flash::Glm53KdaConvWeights| -> Result<KdaConvPath<B::Weight>, BackendError> {
        // ROCm KDA fused kernel 消费 F32 conv/norm 权重;conv1d 是 [heads,1,kernel],展平成 [heads,kernel]。
        let kernel = source.convolution.shape.last().copied().unwrap_or(0);
        let heads = source.convolution.shape.first().copied().unwrap_or(0);
        let values = crate::weight::container::safetensor::decode_to_f32("kda conv1d", "BF16", &source.convolution.data).map_err(crate::runtime::compute_error)?;
        Ok(KdaConvPath { projection: prepare_bf16(backend, &source.projection)?, convolution: backend.prepare_f32(&values, heads, kernel)? })
    };
    Ok(Glm53KdaWeights {
        query: conv(&source.query)?,
        key: conv(&source.key)?,
        value: conv(&source.value)?,
        decay: KdaDecayPath {
            first_projection: prepare_bf16(backend, &source.f_a_projection)?,
            second_projection: prepare_bf16(backend, &source.f_b_projection)?,
            a_log: prepare_f32_vector(backend, &source.a_log, "KDA a_log")?,
            dt_bias: prepare_f32_vector(backend, &source.dt_bias, "KDA dt_bias")?,
        },
        beta_projection: prepare_bf16(backend, &source.beta_projection)?,
        gate: KdaFactoredGatePath { input_projection: prepare_bf16(backend, &source.gate_input_projection)?, output_projection: prepare_bf16(backend, &source.gate_output_projection)? },
        // KDA fused kernel 消费 F32 权重(见 conv 注释)。
        output_norm: {
            let values = crate::weight::container::safetensor::decode_to_f32("kda o_norm", "BF16", &source.output_norm.data).map_err(crate::runtime::compute_error)?;
            backend.prepare_f32(&values, 1, values.len())?
        },
        output_projection: prepare_bf16(backend, &source.output_projection)?,
    })
}

fn prepare_dsa_mla<B: Backend>(backend: &B, source: &crate::weight::model::glm53_flash::Glm53DsaMlaLayerWeights) -> Result<crate::runtime::glm53_flash::layer::Glm53DsaMlaWeights<B::Weight>, BackendError> {
    Ok(crate::runtime::glm53_flash::layer::Glm53DsaMlaWeights {
        query: crate::runtime::glm53_flash::layer::MlaQueryPath { first_projection: prepare_fp8(backend, &source.query_a)?, norm: prepare_bf16(backend, &source.query_norm)?, second_projection: prepare_fp8(backend, &source.query_b)? },
        key_value: crate::runtime::glm53_flash::layer::MlaKvPath { first_projection: prepare_fp8(backend, &source.kv_a)?, norm: prepare_bf16(backend, &source.kv_norm)?, second_projection: prepare_bf16(backend, &source.kv_b)? },
        indexer: crate::runtime::glm53_flash::layer::Glm53IndexerWeights {
            query_projection: prepare_bf16(backend, &source.indexer_query)?,
            key_projection: prepare_bf16(backend, &source.indexer_key)?,
            head_weights_projection: prepare_bf16(backend, &source.indexer_head_weights)?,
            key_norm_weight: prepare_bf16(backend, &source.indexer_key_norm_weight)?,
            key_norm_bias: prepare_bf16(backend, &source.indexer_key_norm_bias)?,
            kpool_ape: prepare_bf16(backend, &source.indexer_kpool_ape)?,
            kpool_gate: prepare_bf16(backend, &source.indexer_kpool_gate)?,
        },
        output_projection: prepare_fp8(backend, &source.output)?,
    })
}

/// mHC:fn 为 BF16 线性权重,base/scale 为 F32 小向量;input_norm 是
/// copies*hidden 宽的单位 RMSNorm(checkpoint 无此张量,与 DeepSeek-V4 mHC 同构)。
fn prepare_mhc<B: Backend>(backend: &B, source: &Glm53MhcTrio, config: &Glm53FlashConfig) -> Result<Glm53HyperConnection<B::Weight>, BackendError> {
    let hidden = config.hyper_connection_copies * config.hidden_size;
    let unit = vec![1.0f32; hidden];
    let input_norm = backend.prepare_f32(&unit, 1, hidden)?;
    let mixes = (2 + config.hyper_connection_copies) * config.hyper_connection_copies;
    let expected = |values: &[f32], len: usize, label: &str| -> Result<(), BackendError> {
        if values.len() != len {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash mhc {label} 长度 {}，期望 {len}", values.len()) });
        }
        Ok(())
    };
    expected(&source.base, mixes, "base")?;
    // mHC 结构常量:scale 恒为 3 项(pre/combination/post 的动态调整系数)。
    expected(&source.scale, 3, "scale")?;
    if source.function.shape.as_slice() != [mixes, hidden] {
        return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash mhc fn shape {:?}，期望 [{mixes},{hidden}]", source.function.shape) });
    }
    Ok(Glm53HyperConnection {
        input_norm,
        function: backend.prepare_f32(&crate::weight::container::safetensor::decode_to_f32("mhc fn", "BF16", &source.function.data).map_err(crate::runtime::compute_error)?, mixes, hidden)?,
        base: backend.prepare_f32(&source.base, mixes, 1)?,
        scale: backend.prepare_f32(&source.scale, 3, 1)?,
    })
}

pub fn prepare_dense_mlp<B: Backend>(backend: &B, source: &Glm53DenseMlpWeights) -> Result<DenseFfn<B::Weight>, BackendError> {
    Ok(DenseFfn { gate: prepare_fp8(backend, &source.gate)?, up: prepare_fp8(backend, &source.up)?, down: prepare_fp8(backend, &source.down)? })
}

/// 官方块级 FP8 直接交给 backend;CPU oracle 可展开,Metal/ROCm 保持 packed。
fn prepare_fp8<B: Backend>(backend: &B, matrix: &Fp8Matrix) -> Result<B::Weight, BackendError> {
    backend.prepare_weight(LinearWeight::fp8(matrix), matrix.rows, matrix.cols)
}

fn prepare_bf16<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let (&first, rest) = tensor.shape.split_first().ok_or_else(|| BackendError::Compute { msg: format!("{} 缺少 tensor shape", tensor.name) })?;
    let cols = rest.iter().try_fold(1usize, |size, &dim| size.checked_mul(dim).ok_or_else(|| BackendError::Compute { msg: format!("{} tensor shape {:?} 溢出", tensor.name, tensor.shape) }))?;
    let (rows, cols) = if rest.is_empty() { (1, first) } else { (first, cols) };
    if tensor.dtype != "BF16" {
        return Err(BackendError::Compute { msg: format!("{} dtype={}，GLM-5.3-Flash resident 权重期望 BF16", tensor.dtype, tensor.name) });
    }
    backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, cols)
}

pub(crate) fn prepare_f32_vector<B: Backend>(backend: &B, values: &[f32], label: &str) -> Result<B::Weight, BackendError> {
    backend.prepare_f32(values, values.len(), 1).map_err(|error| BackendError::Compute { msg: format!("GLM-5.3-Flash {label} prepare: {error}") })
}

fn part_error(part: impl std::fmt::Display, msg: String) -> BackendError {
    BackendError::Compute { msg: format!("GLM-5.3-Flash 加载 {part}: {msg}") }
}

/// 官方 FP8 expert 数据源:288 路由专家按需读取,backend 负责 resident 生命周期。
pub struct Glm53FlashExpertSource {
    weights: Glm53FlashWeights,
    layer_count: usize,
    expert_count: usize,
}

impl Glm53FlashExpertSource {
    pub fn new(weights: Glm53FlashWeights, layer_count: usize, expert_count: usize) -> Self {
        Self { weights, layer_count, expert_count }
    }

    fn check(&self, layer: usize, expert: usize) -> Result<(), String> {
        if layer >= self.layer_count {
            return Err(format!("GLM-5.3-Flash FP8 expert layer 越界: {layer}/{}", self.layer_count));
        }
        if expert >= self.expert_count {
            return Err(format!("GLM-5.3-Flash FP8 expert {expert} 越界于 {}", self.expert_count));
        }
        Ok(())
    }
}

impl Fp8ExpertSource for Glm53FlashExpertSource {
    fn intermediate(&self) -> usize {
        self.weights.dims().moe_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.weights.dims().hidden_size
    }

    fn load_expert_fp8(&self, layer: usize, expert: usize) -> Result<Fp8ExpertWeights, String> {
        self.check(layer, expert)?;
        let weights = self.weights.load_expert(layer, expert)?;
        Ok(Fp8ExpertWeights { gate: weights.gate, up: weights.up, down: weights.down })
    }
}

impl ExpertSourceProvider for Glm53FlashExpertSource {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String> {
        if layer >= self.layer_count {
            return Err(format!("GLM-5.3-Flash expert source layer 越界: {layer}/{}", self.layer_count));
        }
        Ok(ExpertSource::Fp8(self))
    }
}

/// kpool APE 平铺 f32([kpool * head_dim]),注入 DsaState 的 CPU 选择路径。
pub fn kpool_ape_f32(weights: &crate::weight::model::glm53_flash::Glm53FlashWeights, layer: usize) -> Result<Vec<f32>, String> {
    weights.kpool_ape_f32(layer)
}
