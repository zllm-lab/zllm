//! 进程配置文件边界。
//!
//! Scheduler、Node、Stage 与 standalone runtime 拥有不同生命周期，分别解析自己的
//! 顶层结构。这里仅保存跨入口共享的 YAML 加载、iroh 身份和 backend 配置契约。

use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::runtime::glm52::GLM52_LAYER_COUNT;
use crate::server::{
    ServerConfig,
    iroh::{IrohConfig, parse_secret},
    node::NodeConfig,
    scheduler::SchedulerConfig,
};
use crate::weight::{LmHeadQuantization, ResidentWeightQuantization};

pub const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("读取配置 {path}: {source}")]
    Read { path: PathBuf, source: std::io::Error },
    #[error("解析 YAML 配置 {path}: {source}")]
    Parse { path: PathBuf, source: serde_yaml::Error },
    #[error("配置错误: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigCommand {
    Help,
    Load { path: PathBuf, check_only: bool, print_effective: bool },
}

pub fn parse_config_command(arguments: impl IntoIterator<Item = String>) -> Result<ConfigCommand, String> {
    let mut arguments = arguments.into_iter();
    let _binary = arguments.next();
    let mut path = None;
    let mut check_only = false;
    let mut print_effective = false;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--config" => {
                if path.is_some() {
                    return Err("--config 只能指定一次".to_owned());
                }
                path = Some(PathBuf::from(arguments.next().ok_or("--config 需要 YAML 路径")?));
            }
            "--check-config" => check_only = true,
            "--print-effective-config" => print_effective = true,
            "-h" | "--help" => return Ok(ConfigCommand::Help),
            _ => return Err(format!("未知参数 {argument}；入口只接受 --config、--check-config 和 --print-effective-config")),
        }
    }
    let path = path.ok_or("缺少 --config YAML_PATH")?;
    Ok(ConfigCommand::Load { path, check_only, print_effective })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerKind {
    Scheduler,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerProcessConfig {
    pub version: u32,
    pub kind: SchedulerKind,
    pub http: SchedulerHttpConfig,
    pub scheduler: SchedulerServiceConfig,
    pub artifacts: ArtifactConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerHttpConfig {
    pub listen: SocketAddr,
    pub public_base_url: String,
    #[serde(default)]
    pub api_keys: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerServiceConfig {
    pub iroh: IrohListenerConfig,
    #[serde(default)]
    pub node_api_key: Option<String>,
    #[serde(default)]
    pub advertised_models: Vec<String>,
    /// 模型 -> Anthropic 家族 tier（opus/sonnet/haiku），随 /v1/models 下发给客户端做模型族映射。
    #[serde(default)]
    pub anthropic_family_tiers: std::collections::HashMap<String, String>,
    /// 对外别名 -> 后台真实模型（如 claude-opus-5-2 -> glm-5.2）。/v1/models 同时下发别名，
    /// 请求命中别名时重写为真实模型再调度，供按模型名过滤的客户端（Claude Desktop 等）使用。
    #[serde(default)]
    pub model_aliases: std::collections::HashMap<String, String>,
    #[serde(default = "default_dispatch_wait_seconds")]
    pub dispatch_wait_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactConfig {
    pub directory: PathBuf,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IrohListenerConfig {
    #[serde(default)]
    pub secret_key: Option<String>,
    #[serde(default)]
    pub bind_addr: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IrohPeerConfig {
    #[serde(default)]
    pub secret_key: Option<String>,
    #[serde(default)]
    pub bind_addr: Option<String>,
    #[serde(default)]
    pub expected_peer: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeProcessConfig {
    pub version: u32,
    pub scheduler: NodeSchedulerConfig,
    #[serde(default)]
    pub iroh: IrohPeerConfig,
    pub node: NodeServiceConfig,
    pub model: NodeModelConfig,
    pub backend: NodeBackendConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeSchedulerConfig {
    pub ticket: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeServiceConfig {
    pub cache_directory: PathBuf,
    /// KV cache 换出与优雅退出时是否写入 SSD。关闭后 resident cache
    /// 仍可在内存/显存中复用，但退出状态直接释放，也不从 SSD 恢复。
    #[serde(default = "default_true")]
    pub persist_kv_cache: bool,
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    /// 调度器中的模型注册名；用于让同一模型的生产节点和测试节点相互隔离。
    #[serde(default)]
    pub model_alias: Option<String>,
    /// 所有模型/backend 共用的不可变 block graph 全局 LRU 上限。
    #[serde(default = "default_terminal_cache_global_entries")]
    pub terminal_cache_global_entries: usize,
    /// 旧会话终点可进入全局前缀池的前缀轮数上限。
    #[serde(default = "default_terminal_cache_prefix_rounds", alias = "terminal_cache_lineage_rounds")]
    pub terminal_cache_prefix_rounds: usize,
}

/// 进程内引擎只保留真实消费的 session、模型与 backend 配置。
/// HTTP、artifact、iroh、并发调度都属于服务宿主，不进入该结构。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddedConfig {
    pub version: u32,
    pub session: EmbeddedSessionConfig,
    pub model: NodeModelConfig,
    pub backend: NodeBackendConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddedSessionConfig {
    pub cache_directory: PathBuf,
    #[serde(default = "default_true")]
    pub persist_kv_cache: bool,
    #[serde(default = "default_terminal_cache_entries")]
    pub resident_cache_entries: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "architecture", rename_all = "snake_case")]
pub enum NodeModelConfig {
    Ornith(OrnithNodeModelConfig),
    Gemma4(Gemma4NodeModelConfig),
    Qwen36(Qwen36NodeModelConfig),
    DeepseekV4(DeepSeekV4NodeModelConfig),
    Glm52(Glm52NodeModelConfig),
    Glm53Flash(Glm53FlashNodeModelConfig),
    MinimaxH3(H3NodeModelConfig),
    Mistral(MistralNodeModelConfig),
    MiniCpm5(MiniCpm5NodeModelConfig),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeepSeekV4NodeModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    /// 每张卡负责层区间的开区间末尾，与 backend.devices 一一对应。
    pub layer_ends: Vec<usize>,
    #[serde(default)]
    pub execution: DeepSeekV4NodeExecutionConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeepSeekV4NodeExecutionConfig {
    #[serde(default = "default_deepseek_v4_core_cache_gib")]
    pub core_cache_gib: usize,
    #[serde(default = "default_deepseek_v4_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
    #[serde(default = "default_decode_priority_prefill_chunk_size")]
    pub decode_priority_prefill_chunk_size: usize,
    #[serde(default = "default_decode_priority_prefill_chunk_ceiling")]
    pub decode_priority_prefill_chunk_ceiling: usize,
    #[serde(default = "default_deepseek_v4_pool_gib")]
    pub device_pool_gib: usize,
    /// KV 准入的 reservation 页大小（token 数）；节点上报每卡 KV token 容量后，
    /// 调度器按 min(卡容量)/page_tokens 对最大并发封顶，而不是只数请求数。
    #[serde(default = "default_kv_reservation_page_tokens")]
    pub kv_reservation_page_tokens: usize,
    /// KV 容量折算的安全水位：从空闲显存里扣除，覆盖 recent ring、batch 工作区
    /// 等每会话固定开销。
    #[serde(default = "default_memory_reserve_bytes")]
    pub memory_reserve_bytes: usize,
    #[serde(default)]
    pub long_prefill_threshold_tokens: Option<usize>,
    #[serde(default)]
    pub long_prefill_chunk_size: Option<usize>,
    #[serde(default)]
    pub dspark: bool,
    #[serde(default)]
    pub dspark_draft_tokens: Option<usize>,
    /// 初始会话数达到该值才使用 DSpark；默认 1 保持原行为。
    #[serde(default = "default_deepseek_v4_dspark_min_sessions")]
    pub dspark_min_sessions: usize,
    #[serde(default)]
    pub dspark_confidence_threshold: Option<f32>,
    #[serde(default = "default_deepseek_v4_decode_batch_limit")]
    pub decode_batch_limit: usize,
    #[serde(default)]
    pub score_expert_top_k: Option<usize>,
    #[serde(default)]
    pub profile: bool,
}

impl Default for DeepSeekV4NodeExecutionConfig {
    fn default() -> Self {
        Self {
            core_cache_gib: default_deepseek_v4_core_cache_gib(),
            prefill_chunk_size: default_deepseek_v4_prefill_chunk_size(),
            decode_priority_prefill_chunk_size: default_decode_priority_prefill_chunk_size(),
            decode_priority_prefill_chunk_ceiling: default_decode_priority_prefill_chunk_ceiling(),
            device_pool_gib: default_deepseek_v4_pool_gib(),
            kv_reservation_page_tokens: default_kv_reservation_page_tokens(),
            memory_reserve_bytes: default_memory_reserve_bytes(),
            long_prefill_threshold_tokens: None,
            long_prefill_chunk_size: None,
            dspark: false,
            dspark_draft_tokens: None,
            dspark_min_sessions: default_deepseek_v4_dspark_min_sessions(),
            dspark_confidence_threshold: None,
            decode_batch_limit: default_deepseek_v4_decode_batch_limit(),
            score_expert_top_k: None,
            profile: false,
        }
    }
}

fn default_deepseek_v4_core_cache_gib() -> usize {
    16
}
fn default_deepseek_v4_prefill_chunk_size() -> usize {
    4096
}
fn default_deepseek_v4_pool_gib() -> usize {
    4
}
fn default_deepseek_v4_decode_batch_limit() -> usize {
    4
}
const fn default_deepseek_v4_dspark_min_sessions() -> usize {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrnithNodeModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    /// ROCm 多卡分层的每卡层边界（含）；省略时按设备数均分。
    /// 必须与 backend.devices 一一对应、严格递增，最后一项 = 层数-1（引擎加载时校验）。
    #[serde(default)]
    pub layer_ends: Option<Vec<usize>>,
    #[serde(default)]
    pub execution: OrnithNodeExecutionConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MistralNodeModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default)]
    pub execution: MistralNodeExecutionConfig,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MistralNodeExecutionConfig {
    #[serde(default)]
    pub kv_cache_format: KvCacheFormat,
}

impl Default for MistralNodeExecutionConfig {
    fn default() -> Self {
        Self { kv_cache_format: KvCacheFormat::Q8g64 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniCpm5NodeModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default)]
    pub execution: MiniCpm5NodeExecutionConfig,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniCpm5NodeExecutionConfig {
    #[serde(default)]
    pub kv_cache_format: KvCacheFormat,
}

impl Default for MiniCpm5NodeExecutionConfig {
    fn default() -> Self {
        Self { kv_cache_format: KvCacheFormat::Q8g64 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrnithNodeExecutionConfig {
    #[serde(default)]
    pub precise_gqa_prefill: bool,
    #[serde(default)]
    pub expert_batch_size: Option<usize>,
    #[serde(default)]
    pub kv_cache_format: KvCacheFormat,
    #[serde(default)]
    pub lazy_experts: bool,
    #[serde(default = "default_expert_cache_gib")]
    pub expert_cache_gib: usize,
    #[serde(default)]
    pub expert_prefetch_count: Option<usize>,
    #[serde(default = "default_terminal_cache_entries")]
    pub terminal_cache_entries: usize,
}

impl Default for OrnithNodeExecutionConfig {
    fn default() -> Self {
        Self {
            precise_gqa_prefill: false,
            expert_batch_size: None,
            kv_cache_format: KvCacheFormat::Q8g64,
            lazy_experts: false,
            expert_cache_gib: default_expert_cache_gib(),
            expert_prefetch_count: None,
            terminal_cache_entries: default_terminal_cache_entries(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gemma4NodeModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default)]
    pub execution: Gemma4ExecutionConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen36NodeModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default)]
    pub variant: Qwen36Variant,
    #[serde(default)]
    pub execution: Qwen36NodeExecutionConfig,
}

/// Qwen3.8-27B 与 Qwen3.6-27B 的 GGUF 文本主干超参一致（同 qwen35 architecture），
/// variant 只决定节点上报与请求匹配的 model_key。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen36Variant {
    #[default]
    Qwen36,
    Qwen38,
}

impl Qwen36Variant {
    pub fn model_key(self) -> &'static str {
        match self {
            Self::Qwen36 => "qwen36",
            Self::Qwen38 => "qwen38",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen36NodeExecutionConfig {
    #[serde(default)]
    pub precise_gqa_prefill: bool,
    #[serde(default)]
    pub kv_cache_format: KvCacheFormat,
    #[serde(default = "default_metal_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
    /// 单张图像允许的最大视觉 token 数；None 使用 checkpoint 的像素上限。
    /// Qwen3.6/3.8 的 patch=16、merge=2，因此每 token 对应 32×32 像素。
    #[serde(default)]
    pub vision_max_tokens: Option<usize>,
    /// MTP 投机解码（GGUF 需含 nextn 块）。
    #[serde(default)]
    pub mtp: bool,
    /// MTP 生效的最大上下文 token 数；0 表示不限制。上下文超过该值后 decode 降级
    /// 为普通单行(verify 双行每步扫两份 KV,长上下文成本超过接受收益:
    /// Qwen3.8-27B Q3_K_XL 实测 2.2K +9%、8.7K -10%、15.6K -43%,建议 4096)。
    #[serde(default)]
    pub mtp_context_limit: usize,
    /// 每轮 MTP 链式 draft 的候选 token 数;1 即旧版 nextn=1 双行 verify。
    /// K>1 时 draft 链自回归推进,主干一次 K+1 行 verify。
    #[serde(default = "default_mtp_draft_tokens_qwen36")]
    pub mtp_draft_tokens: usize,
    /// 带模型与完整词表元数据的 FR-Spec draft vocabulary JSON。
    #[serde(default)]
    pub mtp_draft_vocabulary: Option<PathBuf>,
    /// DSpark 投机解码 drafter(llama.cpp `dflash` GGUF,如 DimInfer
    /// Qwen3.8-27B-Dspark)。embedding/lm_head 复用主模型。
    #[serde(default)]
    pub dspark_directory: Option<PathBuf>,
    /// 每轮 draft token 数(社区 Qwen3.8-27B checkpoint 吞吐最优 4)。
    #[serde(default = "default_dspark_draft_tokens")]
    pub dspark_draft_tokens: usize,
    /// CUDA 混合执行时驻留 GPU 的连续尾部层数；None 按空闲显存自动规划。
    #[serde(default)]
    pub cuda_gpu_layers: Option<usize>,
    /// CUDA 混合执行的 CPU decode 线程数；None 使用 rayon 默认线程池。
    #[serde(default)]
    pub cuda_decode_cpu_threads: Option<usize>,
    /// CPU 前缀层是否常驻 GGUF packed 权重，避免每 token 重新 mmap fault。
    #[serde(default = "default_true")]
    pub cuda_cpu_packed_resident: bool,
    /// CUDA 自动分层时保留的显存 GiB。
    #[serde(default = "default_cuda_vram_reserve_gib")]
    pub cuda_vram_reserve_gib: usize,
}

impl Default for Qwen36NodeExecutionConfig {
    fn default() -> Self {
        Self {
            precise_gqa_prefill: false,
            kv_cache_format: KvCacheFormat::Q8g64,
            prefill_chunk_size: default_metal_prefill_chunk_size(),
            vision_max_tokens: None,
            mtp: false,
            mtp_context_limit: 0,
            mtp_draft_tokens: default_mtp_draft_tokens_qwen36(),
            mtp_draft_vocabulary: None,
            dspark_directory: None,
            dspark_draft_tokens: default_dspark_draft_tokens(),
            cuda_gpu_layers: None,
            cuda_decode_cpu_threads: None,
            cuda_cpu_packed_resident: true,
            cuda_vram_reserve_gib: default_cuda_vram_reserve_gib(),
        }
    }
}

const fn default_cuda_vram_reserve_gib() -> usize {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm53FlashNodeModelConfig {
    pub weights_directory: PathBuf,
    pub tokenizer: PathBuf,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    /// head 本机层边界与下游 tail。
    pub head: Glm53FlashHeadConfig,
    #[serde(default = "default_glm53_flash_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
    /// 绝对 position 超过该值后 chunk 与 long_prefill_chunk_size 取 min;0 关闭分档。
    #[serde(default)]
    pub long_prefill_threshold_tokens: usize,
    /// 长上下文档 chunk;缺省沿用 prefill_chunk_size。
    #[serde(default)]
    pub long_prefill_chunk_size: Option<usize>,
    /// MTP speculative decode:head 按 Speculative/Verify 协议驱动,自身不装 MTP 权重。
    #[serde(default)]
    pub mtp: bool,
    /// 每轮 MTP 递归生成的 draft token 数。
    #[serde(default = "default_mtp_draft_tokens")]
    pub mtp_draft_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm53FlashHeadConfig {
    /// head 负责 `0..stage_end`,tail 负责 `stage_end..45`。
    pub stage_end: usize,
    /// 每张卡负责层区间的开区间末尾,末项必须等于 stage_end。
    pub layer_ends: Vec<usize>,
    pub downstream: StagePeerConfig,
}

fn default_glm53_flash_prefill_chunk_size() -> usize {
    2048
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52NodeModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    #[serde(default)]
    pub compressed_tensors_directory: Option<PathBuf>,
    #[serde(default)]
    pub nvfp4_directory: Option<PathBuf>,
    #[serde(default)]
    pub gguf_directory: Option<PathBuf>,
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    pub head: Glm52HeadConfig,
    #[serde(default)]
    pub execution: Glm52NodeExecutionConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52HeadConfig {
    pub stage_end: usize,
    pub layer_ends: Vec<usize>,
    pub downstream: StagePeerConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StagePeerConfig {
    pub ticket: String,
    #[serde(default)]
    pub iroh: IrohPeerConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52NodeExecutionConfig {
    #[serde(default = "default_glm52_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
    #[serde(default)]
    pub kv_cache_format: KvCacheFormat,
    #[serde(default = "default_unlimited_entries")]
    pub terminal_cache_entries: usize,
    #[serde(default = "default_kv_reservation_page_tokens")]
    pub kv_reservation_page_tokens: usize,
    #[serde(default = "default_true")]
    pub preload_experts: bool,
    /// 同机相邻卡共享 routed expert 计算；当前只用于单行 decode 实验。
    #[serde(default)]
    pub cooperative_expert_pairs: bool,
    #[serde(default)]
    pub preload_layers_per_device: Option<usize>,
    #[serde(default = "default_memory_reserve_bytes")]
    pub memory_reserve_bytes: usize,
    #[serde(default)]
    pub kv_admission_layers_per_device: usize,
    #[serde(default)]
    pub mtp: bool,
    /// DSpark checkpoint 目录；与 MTP 互斥。
    #[serde(default)]
    pub dspark_directory: Option<PathBuf>,
    /// DSpark proposal backbone 的执行设备。CPU 仍复用 ROCm capture/cache，
    /// 只把独立 proposal 计算移出 16 卡流水线。
    #[serde(default)]
    pub dspark_backend: Glm52DsparkExecutionBackend,
    /// CPU DSpark worker 的 Linux CPU list（如 `0-31`）；其固定 team 继承同一 mask。
    #[serde(default)]
    pub dspark_cpu_affinity: Option<String>,
    #[serde(default = "default_dspark_draft_tokens")]
    pub dspark_draft_tokens: usize,
    /// 仅显式配置时启用 confidence head，并剪掉首个低置信 token 及其后缀。
    #[serde(default)]
    pub dspark_confidence_threshold: Option<f32>,
    /// DSpark capture、backbone 与 Markov projection 的 resident 格式；不改变 target 权重。
    #[serde(default)]
    pub dspark_weight_quantization: ResidentWeightQuantization,
    /// verify 行按 N 行分组提交:组间保持流水重叠,组内共享一次权重读取。
    /// 1 为逐行流水(旧行为),与行数相等则整批。
    #[serde(default = "default_dspark_verify_group_rows")]
    pub dspark_verify_group_rows: usize,
    /// 把 final norm、LM head 与采样放到末段最后一张卡，decode 只回传 token。
    #[serde(default)]
    pub tail_sampling: bool,
    #[serde(default = "default_mtp_draft_tokens")]
    pub mtp_draft_tokens: usize,
    #[serde(default)]
    pub reasoning_effort: Glm52ReasoningEffort,
    #[serde(default)]
    pub thinking_token_budget: Option<usize>,
    #[serde(default)]
    pub scheduling: Glm52SchedulingConfig,
    #[serde(default)]
    pub diagnostics: Glm52DiagnosticsConfig,
}

impl Default for Glm52NodeExecutionConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_size: default_glm52_prefill_chunk_size(),
            kv_cache_format: KvCacheFormat::Q8g64,
            terminal_cache_entries: default_unlimited_entries(),
            kv_reservation_page_tokens: default_kv_reservation_page_tokens(),
            preload_experts: true,
            cooperative_expert_pairs: false,
            preload_layers_per_device: None,
            memory_reserve_bytes: default_memory_reserve_bytes(),
            kv_admission_layers_per_device: 0,
            mtp: false,
            dspark_directory: None,
            dspark_backend: Glm52DsparkExecutionBackend::default(),
            dspark_cpu_affinity: None,
            dspark_draft_tokens: default_dspark_draft_tokens(),
            dspark_confidence_threshold: None,
            dspark_weight_quantization: ResidentWeightQuantization::Native,
            dspark_verify_group_rows: default_dspark_verify_group_rows(),
            tail_sampling: false,
            mtp_draft_tokens: default_mtp_draft_tokens(),
            reasoning_effort: Glm52ReasoningEffort::default(),
            thinking_token_budget: None,
            scheduling: Glm52SchedulingConfig::default(),
            diagnostics: Glm52DiagnosticsConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Glm52DsparkExecutionBackend {
    #[default]
    Rocm,
    Cpu,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Glm52ReasoningEffort {
    High,
    #[default]
    Max,
}

impl Glm52ReasoningEffort {
    #[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
    pub(crate) const fn prompt_value(self) -> &'static str {
        match self {
            Self::High => "High",
            Self::Max => "Max",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52SchedulingConfig {
    #[serde(default = "default_stage_execution_slots")]
    pub execution_slots: usize,
    #[serde(default = "default_stage_execution_slots")]
    pub decode_execution_slots: usize,
    #[serde(default = "default_pipeline_work_window")]
    pub pipeline_work_window: usize,
    #[serde(default = "default_prefill_admission_burst")]
    pub prefill_admission_burst: usize,
    #[serde(default = "default_decode_batch_limit")]
    pub decode_batch_limit: usize,
    #[serde(default = "default_prefill_batch_limit")]
    pub prefill_batch_limit: usize,
    #[serde(default = "default_append_prefill_chunk_size")]
    pub append_prefill_chunk_size: usize,
    #[serde(default = "default_decode_priority_prefill_chunk_size")]
    pub decode_priority_prefill_chunk_size: usize,
    #[serde(default = "default_decode_priority_prefill_chunk_ceiling")]
    pub decode_priority_prefill_chunk_ceiling: usize,
    #[serde(default = "default_long_prefill_threshold_tokens")]
    pub long_prefill_threshold_tokens: usize,
    #[serde(default = "default_long_prefill_chunk_size")]
    pub long_prefill_chunk_size: usize,
    #[serde(default)]
    pub profile_completion: bool,
}

impl Default for Glm52SchedulingConfig {
    fn default() -> Self {
        Self {
            execution_slots: default_stage_execution_slots(),
            decode_execution_slots: default_stage_execution_slots(),
            pipeline_work_window: default_pipeline_work_window(),
            prefill_admission_burst: default_prefill_admission_burst(),
            decode_batch_limit: default_decode_batch_limit(),
            prefill_batch_limit: default_prefill_batch_limit(),
            append_prefill_chunk_size: default_append_prefill_chunk_size(),
            decode_priority_prefill_chunk_size: default_decode_priority_prefill_chunk_size(),
            decode_priority_prefill_chunk_ceiling: default_decode_priority_prefill_chunk_ceiling(),
            long_prefill_threshold_tokens: default_long_prefill_threshold_tokens(),
            long_prefill_chunk_size: default_long_prefill_chunk_size(),
            profile_completion: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52DiagnosticsConfig {
    #[serde(default)]
    pub profile_boundaries: bool,
    #[serde(default)]
    pub trace_stage_output: bool,
    #[serde(default)]
    pub trace_stage_events: bool,
    #[serde(default)]
    pub official_chat_template: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3NodeModelConfig {
    pub weights_directory: PathBuf,
    pub qwen_weights_directory: PathBuf,
    #[serde(default)]
    pub qwen_tokenizer_directory: Option<PathBuf>,
    #[serde(default)]
    pub execution: H3NodeExecutionConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3NodeExecutionConfig {
    #[serde(default = "default_h3_steps")]
    pub steps: usize,
    #[serde(default = "default_h3_stream_chunk_layers")]
    pub stream_chunk_layers: usize,
    #[serde(default)]
    pub save_latent: bool,
    #[serde(default = "default_ffmpeg")]
    pub ffmpeg: String,
    /// 默认关闭；启用后允许以第 0 层残差判据跳过相似时间步的其余 DiT block。
    #[serde(default)]
    pub block_cache: Option<H3BlockCacheConfig>,
}

impl Default for H3NodeExecutionConfig {
    fn default() -> Self {
        Self { steps: default_h3_steps(), stream_chunk_layers: default_h3_stream_chunk_layers(), save_latent: false, ffmpeg: default_ffmpeg(), block_cache: None }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3BlockCacheConfig {
    #[serde(default = "default_h3_block_cache_threshold")]
    pub threshold: f32,
    #[serde(default = "default_h3_block_cache_start_percent")]
    pub start_percent: f32,
    #[serde(default = "default_h3_block_cache_end_percent")]
    pub end_percent: f32,
    #[serde(default = "default_h3_block_cache_max_consecutive_hits")]
    pub max_consecutive_hits: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheFormat {
    F16,
    #[default]
    Q8g64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeProcessConfig {
    Stage(StageProcessConfig),
    Standalone(StandaloneProcessConfig),
    /// kind: node —— 平台 runtime 进程兼任调度器节点(zllm-rt-rocm/metal)。
    Node(NodeProcessConfig),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageProcessConfig {
    pub version: u32,
    pub model: StageModelConfig,
    pub transport: StageTransportConfig,
    pub backend: BackendConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "architecture", rename_all = "snake_case")]
pub enum StageModelConfig {
    Glm52(Glm52StageModelConfig),
    Glm53Flash(Glm53FlashStageModelConfig),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm53FlashStageModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    pub layers: Glm53FlashStageLayersConfig,
    /// 装载 layers.{layer_count} 的 MTP 头并启用 speculative 采样。
    #[serde(default)]
    pub mtp: bool,
    /// 每轮 MTP 递归生成的 draft token 数,必须与 head 一致。
    #[serde(default = "default_mtp_draft_tokens")]
    pub mtp_draft_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm53FlashStageLayersConfig {
    pub start: usize,
    pub end: usize,
    /// 绝对层号的开区间末尾,与 backend.devices 一一对应。
    pub device_layer_ends: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52StageModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    #[serde(default)]
    pub compressed_tensors_directory: Option<PathBuf>,
    #[serde(default)]
    pub nvfp4_directory: Option<PathBuf>,
    #[serde(default)]
    pub gguf_directory: Option<PathBuf>,
    #[serde(default)]
    pub cache_directory: Option<PathBuf>,
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    #[serde(default = "default_glm52_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default)]
    pub generation: Option<StageGenerationConfig>,
    pub layers: Glm52StageLayersConfig,
    #[serde(default)]
    pub execution: Glm52StageExecutionConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52StageLayersConfig {
    pub start: usize,
    pub end: usize,
    pub device_layer_ends: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52StageExecutionConfig {
    #[serde(default = "default_stage_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
    #[serde(default)]
    pub kv_cache_format: KvCacheFormat,
    #[serde(default = "default_true")]
    pub preload_experts: bool,
    /// 同机相邻卡共享 routed expert 计算；当前只用于单行 decode 实验。
    #[serde(default)]
    pub cooperative_expert_pairs: bool,
    #[serde(default)]
    pub preload_layers_per_device: Option<usize>,
    #[serde(default = "default_stage_max_concurrency")]
    pub max_concurrency: usize,
    #[serde(default)]
    pub mtp: bool,
    /// DSpark checkpoint 目录；每段只读取本段 capture 对应的 FC 列切片。
    #[serde(default)]
    pub dspark_directory: Option<PathBuf>,
    /// 本段 DSpark capture projection 的 resident 格式；不改变 target 权重。
    #[serde(default)]
    pub dspark_weight_quantization: ResidentWeightQuantization,
    #[serde(default = "default_mtp_draft_tokens")]
    pub mtp_draft_tokens: usize,
    /// 带模型与完整词表元数据的 FR-Spec draft vocabulary JSON。
    #[serde(default)]
    pub mtp_draft_vocabulary: Option<PathBuf>,
    #[serde(default)]
    pub scheduling: Glm52SchedulingConfig,
    #[serde(default)]
    pub diagnostics: Glm52StageDiagnosticsConfig,
}

impl Default for Glm52StageExecutionConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_size: default_stage_prefill_chunk_size(),
            kv_cache_format: KvCacheFormat::Q8g64,
            preload_experts: true,
            cooperative_expert_pairs: false,
            preload_layers_per_device: None,
            max_concurrency: default_stage_max_concurrency(),
            mtp: false,
            dspark_directory: None,
            dspark_weight_quantization: ResidentWeightQuantization::Native,
            mtp_draft_tokens: default_mtp_draft_tokens(),
            mtp_draft_vocabulary: None,
            scheduling: Glm52SchedulingConfig::default(),
            diagnostics: Glm52StageDiagnosticsConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52StageDiagnosticsConfig {
    #[serde(default)]
    pub profile_boundaries: bool,
    #[serde(default)]
    pub trace_stage_output: bool,
    #[serde(default)]
    pub trace_stage_events: bool,
    #[serde(default)]
    pub input_artifact: Option<PathBuf>,
    #[serde(default)]
    pub output_artifact: Option<PathBuf>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub allow_generated_request_id: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum StageTransportConfig {
    Listen {
        #[serde(default)]
        iroh: IrohPeerConfig,
    },
    Connect {
        ticket: String,
        #[serde(default)]
        iroh: IrohPeerConfig,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StandaloneProcessConfig {
    pub version: u32,
    pub http: StandaloneHttpConfig,
    pub artifacts: ArtifactConfig,
    pub node: NodeServiceConfig,
    #[serde(default)]
    pub iroh: IrohPeerConfig,
    pub model: NodeModelConfig,
    pub backend: NodeBackendConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StandaloneHttpConfig {
    pub listen: SocketAddr,
    #[serde(default)]
    pub public_base_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "architecture", rename_all = "snake_case")]
pub enum StandaloneModelConfig {
    Glm52(Glm52StandaloneModelConfig),
    Gemma4(Gemma4StandaloneModelConfig),
    Ornith(OrnithStandaloneModelConfig),
    Qwen3Vl(Qwen3VlStandaloneModelConfig),
    Qwen36(Qwen36StandaloneModelConfig),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextGenerationConfig {
    pub prompt: String,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default)]
    pub decode_steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageGenerationConfig {
    pub prompt: String,
    #[serde(default)]
    pub decode_steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm52StandaloneModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    #[serde(default)]
    pub compressed_tensors_directory: Option<PathBuf>,
    #[serde(default)]
    pub nvfp4_directory: Option<PathBuf>,
    #[serde(default)]
    pub gguf_directory: Option<PathBuf>,
    pub generation: TextGenerationConfig,
    #[serde(default)]
    pub prefill_layer_ends: Option<Vec<usize>>,
    #[serde(default)]
    pub cache_directory: Option<PathBuf>,
    #[serde(default)]
    pub kv_cache_dump: Option<PathBuf>,
    #[serde(default)]
    pub kv_cache_load: Option<PathBuf>,
    #[serde(default)]
    pub expert_cache_gib: Option<usize>,
    #[serde(default)]
    pub expert_prefetch_count: Option<usize>,
    #[serde(default)]
    pub execution: Glm52StageExecutionConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gemma4StandaloneModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    pub generation: TextGenerationConfig,
    #[serde(default)]
    pub execution: Gemma4ExecutionConfig,
    #[serde(default)]
    pub images: Vec<PathBuf>,
    #[serde(default)]
    pub video_frames: Vec<PathBuf>,
    #[serde(default)]
    pub audio_files: Vec<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gemma4ExecutionConfig {
    #[serde(default = "default_metal_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
    /// 官方 MTP 投机头(mtp-*.gguf,gemma4-assistant)。启用后 decode 走
    /// draft/verify 循环:4 层 draft 读主干 KV,主干一次 K+1 行 verify。
    #[serde(default)]
    pub mtp_weights: Option<PathBuf>,
    /// 每轮链式 draft 候选数(verify 行数 = 该值 + 1)。
    #[serde(default = "default_gemma4_mtp_draft_tokens")]
    pub mtp_draft_tokens: usize,
    /// 重放模式下执行算子消融计时分解(输出无效,仅测量)。
    #[serde(default)]
    pub replay_ablation: bool,
}

fn default_gemma4_mtp_draft_tokens() -> usize {
    // K 扫参实测(M5, 12B IQ4_NL + Q8_0 MTP,verify rows=2 bug 修复后,2026-08-28):
    // K=1 ~14.7 / K=2 ~15.6-16.5 / K=3 ~15-16 tok/s(按位置接受率 ~76/77/59%,
    // tokens/round 1.72/2.29/2.74);K=2 收益稳定、方差小。llama.cpp 同机参照:
    // 裸 decode 17.4、MTP K=3 17.1 tok/s。
    2
}

impl Default for Gemma4ExecutionConfig {
    fn default() -> Self {
        Self { prefill_chunk_size: default_metal_prefill_chunk_size(), mtp_weights: None, mtp_draft_tokens: default_gemma4_mtp_draft_tokens(), replay_ablation: false }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrnithStandaloneModelConfig {
    pub weights: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    pub generation: TextGenerationConfig,
    #[serde(default)]
    pub execution: OrnithStandaloneExecutionConfig,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrnithStandaloneExecutionConfig {
    #[serde(default)]
    pub precise_gqa_prefill: bool,
    #[serde(default)]
    pub expert_batch_size: Option<usize>,
    #[serde(default)]
    pub kv_cache_format: KvCacheFormat,
    #[serde(default)]
    pub lazy_experts: bool,
    #[serde(default = "default_expert_cache_gib")]
    pub expert_cache_gib: usize,
    #[serde(default)]
    pub expert_prefetch_count: Option<usize>,
    #[serde(default)]
    pub mtp: bool,
}

impl Default for OrnithStandaloneExecutionConfig {
    fn default() -> Self {
        Self { precise_gqa_prefill: false, expert_batch_size: None, kv_cache_format: KvCacheFormat::Q8g64, lazy_experts: false, expert_cache_gib: default_expert_cache_gib(), expert_prefetch_count: None, mtp: false }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3VlStandaloneModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    pub image: PathBuf,
    pub generation: TextGenerationConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen36StandaloneModelConfig {
    pub weights_directory: PathBuf,
    #[serde(default)]
    pub lm_head_quantization: LmHeadQuantization,
    pub generation: TextGenerationConfig,
    #[serde(default)]
    pub images: Vec<PathBuf>,
    #[serde(default)]
    pub videos: Vec<PathBuf>,
    #[serde(default)]
    pub execution: Qwen36ExecutionConfig,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen36ExecutionConfig {
    #[serde(default)]
    pub precise_gqa_prefill: bool,
    #[serde(default)]
    pub expert_batch_size: Option<usize>,
    #[serde(default)]
    pub kv_cache_format: KvCacheFormat,
}

impl Default for Qwen36ExecutionConfig {
    fn default() -> Self {
        Self { precise_gqa_prefill: false, expert_batch_size: None, kv_cache_format: KvCacheFormat::Q8g64 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendConfig {
    Cpu(CpuBackendConfig),
    Metal(MetalBackendConfig),
    Cuda(CudaBackendConfig),
    Rocm(RocmBackendConfig),
    Huawei(HuaweiBackendConfig),
}

/// Node 只暴露实际会读取的 backend 配置；standalone 的 profile/trace 开关不能混入。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeBackendConfig {
    Cpu(CpuBackendConfig),
    Metal(NodeMetalBackendConfig),
    Cuda(CudaBackendConfig),
    Rocm(RocmBackendConfig),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeMetalBackendConfig {
    #[serde(default = "default_metal_device")]
    pub device: String,
    /// 静态 decode/verify 平铺录制重放；Metal 后端默认开启。
    #[serde(default = "default_true")]
    pub replay: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuBackendConfig {
    #[serde(default)]
    pub threads: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetalBackendConfig {
    #[serde(default = "default_metal_device")]
    pub device: String,
    /// 静态 decode/verify 平铺录制重放；Metal 后端默认开启。
    #[serde(default = "default_true")]
    pub replay: bool,
    #[serde(default)]
    pub profile_prefill_gpu: bool,
    #[serde(default)]
    pub profile_decode: bool,
    #[serde(default)]
    pub trace_layers: bool,
    #[serde(default)]
    pub trace_layer_directory: Option<PathBuf>,
    #[serde(default)]
    pub trace_tokens: bool,
    #[serde(default)]
    pub trace_logits: bool,
    #[serde(default)]
    pub decode_prefetch_count: Option<usize>,
    #[serde(default)]
    pub decode_resident_layers: Option<usize>,
    #[serde(default = "default_true")]
    pub prefetch_decode_core: bool,
    #[serde(default)]
    pub lm_head_f32: bool,
    #[serde(default)]
    pub prefill_report: Option<PathBuf>,
    /// decode 延迟批次每 command buffer 的算子上限;None 用引擎默认(16)。
    /// 加大可减少 CB 边界,过大会把 CPU 编码与 GPU 执行串行化。
    #[serde(default)]
    pub decode_batch_operations: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaBackendConfig {
    #[serde(default)]
    pub device: i32,
    pub include_directory: PathBuf,
    #[serde(default)]
    pub architecture: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RocmBackendConfig {
    pub devices: Vec<i32>,
    #[serde(default = "default_rocm_root")]
    pub root: PathBuf,
    #[serde(default = "default_hiprtc_cache_directory")]
    pub hiprtc_cache_directory: PathBuf,
    #[serde(default)]
    pub kernel_sync: bool,
    #[serde(default)]
    pub kernel_profile: bool,
    /// 单行 integrated MoE 使用固定地址 HIP Graph；诊断模式下自动回退 eager。
    #[serde(default)]
    pub decode_graph: bool,
    /// MoE router：true=F32 权重精确打分；false=BF16 驻留快速路径（近并列路由可能翻转）。
    #[serde(default = "default_true")]
    pub precise_router: bool,
    /// 单 token MLA decode 从串行 kernel 切到 split/WMMA kernel 的上下文阈值。
    #[serde(default = "default_rocm_mla_decode_split_threshold")]
    pub mla_decode_split_threshold: usize,
    /// DSA Q/K 同时做 Hadamard+Q8，并以 i8 WMMA 评分；真机 oracle 前默认关闭。
    #[serde(default)]
    pub dsa_hadamard_i8: bool,
    /// 仅 kernel_profile 下对每个 DSA 层采样多少行 Hadamard 粗排；不改变 exact selection。
    #[serde(default)]
    pub dsa_hadamard_shadow_samples: usize,
    /// 仅 kernel_profile 下对每个 DSA 层采样多少行 HISA block 粗排；不改变 exact selection。
    #[serde(default)]
    pub dsa_hisa_shadow_samples: usize,
    /// 单行 decode 由 CPU 全历史 DSA 产生候选，GPU 精确重排；MLA cache 仍由 GPU 本地读取。
    #[serde(default)]
    pub dsa_cpu_select: bool,
    /// CPU 保存全量 MLA，GPU 每层只保留固定行数的精确 hot cache；0 表示关闭。
    #[serde(default)]
    pub mla_cpu_hot_rows: usize,
    /// prefill grouped-down 先写 route-major F32，再按固定路由顺序归约。
    #[serde(default)]
    pub grouped_down_route_buffer: bool,
    /// ROCm 尚无对应设备实现时，是否允许执行显式 CPU reference 路径。
    #[serde(default)]
    pub allow_cpu_reference_fallback: bool,
    #[serde(default = "default_true")]
    pub memory_pool: bool,
    #[serde(default)]
    pub accelerator_name: Option<String>,
    #[serde(default)]
    pub compute_units: Option<usize>,
    #[serde(default)]
    pub accelerator_memory_bytes: Option<u64>,
    #[serde(default)]
    pub recommended_working_set_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuaweiBackendConfig {
    #[serde(default)]
    pub device: i32,
}

impl SchedulerProcessConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let mut config: Self = read_yaml(path)?;
        config.validate()?;
        let base = config_base(path)?;
        resolve_path(&base, &mut config.artifacts.directory);
        Ok(config)
    }

    pub fn runtime(&self) -> Result<(SocketAddr, ServerConfig), ConfigError> {
        let mut api_keys = self.http.api_keys.iter().map(|value| value.trim()).filter(|value| !value.is_empty()).map(str::to_owned).collect::<Vec<_>>();
        api_keys.sort_unstable();
        api_keys.dedup();
        let mut models = self.scheduler.advertised_models.iter().map(|value| value.trim()).filter(|value| !value.is_empty()).map(str::to_owned).collect::<Vec<_>>();
        let mut model_aliases = self.scheduler.model_aliases.iter().map(|(alias, model)| (alias.trim().to_owned(), model.trim().to_owned())).collect::<std::collections::HashMap<String, String>>();
        model_aliases.retain(|alias, model| !alias.is_empty() && !model.is_empty() && alias != model);
        models.sort_unstable();
        models.dedup();
        let scheduler = SchedulerConfig {
            artifact_dir: self.artifacts.directory.clone(),
            public_base_url: self.http.public_base_url.trim_end_matches('/').to_owned(),
            dispatch_wait: std::time::Duration::from_secs(self.scheduler.dispatch_wait_seconds),
            iroh: self.scheduler.iroh.runtime()?,
        };
        Ok((
            self.http.listen,
            ServerConfig { api_keys, anonymous_cache_namespace: None, node_api_key: nonempty(self.scheduler.node_api_key.as_deref()), models, anthropic_family_tiers: self.scheduler.anthropic_family_tiers.clone(), model_aliases, scheduler },
        ))
    }

    fn validate(&self) -> Result<(), ConfigError> {
        validate_version(self.version)?;
        if self.http.public_base_url.trim().is_empty() {
            return Err(ConfigError::Invalid("http.public_base_url 不能为空".to_owned()));
        }
        Ok(())
    }
}

impl NodeProcessConfig {
    /// 路径解析与校验;由 `RuntimeProcessConfig::load` 的 `kind: node`
    /// 分支调用(zllm-rt-rocm/metal 的 node 角色)。
    fn resolve_and_validate(&mut self, base: &Path) -> Result<(), ConfigError> {
        resolve_path(base, &mut self.node.cache_directory);
        resolve_node_model_backend(base, &mut self.model, &mut self.backend);
        self.validate()
    }

    pub fn runtime(&self) -> Result<NodeConfig, ConfigError> {
        Ok(NodeConfig {
            upstream: crate::server::node::NodeUpstream::Ticket(self.scheduler.ticket.trim().to_owned()),
            api_key: nonempty(self.scheduler.api_key.as_deref()),
            cache_dir: self.node.cache_directory.clone(),
            persist_kv_cache: self.node.persist_kv_cache,
            iroh: self.iroh.runtime()?,
            max_concurrency: self.node.max_concurrency,
            model_alias: self.node.model_alias.as_deref().map(str::trim).filter(|value| !value.is_empty()).map(str::to_owned),
            terminal_cache_global_entries: self.node.terminal_cache_global_entries,
            terminal_cache_prefix_rounds: self.node.terminal_cache_prefix_rounds,
        })
    }

    fn validate(&self) -> Result<(), ConfigError> {
        validate_version(self.version)?;
        if self.scheduler.ticket.trim().is_empty() {
            return Err(ConfigError::Invalid("scheduler.ticket 不能为空".to_owned()));
        }
        if self.node.max_concurrency == Some(0) {
            return Err(ConfigError::Invalid("node.max_concurrency 必须大于 0".to_owned()));
        }
        if self.node.terminal_cache_global_entries == 0 {
            return Err(ConfigError::Invalid("node.terminal_cache_global_entries 必须大于 0".to_owned()));
        }
        if self.node.model_alias.as_deref().is_some_and(|value| value.trim().is_empty()) {
            return Err(ConfigError::Invalid("node.model_alias 不能为空字符串".to_owned()));
        }
        self.backend.validate()?;
        validate_node_model_backend(&self.model, &self.backend)
    }
}

fn resolve_node_model_backend(base: &Path, model: &mut NodeModelConfig, backend: &mut NodeBackendConfig) {
    match model {
        NodeModelConfig::Ornith(model) => resolve_path(base, &mut model.weights_directory),
        NodeModelConfig::Gemma4(model) => {
            resolve_path(base, &mut model.weights_directory);
            resolve_optional_path(base, &mut model.execution.mtp_weights);
        }
        NodeModelConfig::Qwen36(model) => {
            resolve_path(base, &mut model.weights_directory);
            resolve_optional_path(base, &mut model.execution.mtp_draft_vocabulary);
            resolve_optional_path(base, &mut model.execution.dspark_directory);
        }
        NodeModelConfig::DeepseekV4(model) => resolve_path(base, &mut model.weights_directory),
        NodeModelConfig::Glm53Flash(model) => {
            resolve_path(base, &mut model.weights_directory);
            resolve_path(base, &mut model.tokenizer);
        }
        NodeModelConfig::Glm52(model) => {
            resolve_path(base, &mut model.weights_directory);
            resolve_optional_path(base, &mut model.compressed_tensors_directory);
            resolve_optional_path(base, &mut model.nvfp4_directory);
            resolve_optional_path(base, &mut model.gguf_directory);
            resolve_optional_path(base, &mut model.execution.dspark_directory);
            if let Some(tokenizer) = &mut model.tokenizer {
                resolve_path(base, tokenizer);
            } else {
                model.tokenizer = Some(model.compressed_tensors_directory.as_ref().or(model.nvfp4_directory.as_ref()).unwrap_or(&model.weights_directory).join("tokenizer.json"));
            }
        }
        NodeModelConfig::MinimaxH3(model) => {
            resolve_path(base, &mut model.weights_directory);
            resolve_path(base, &mut model.qwen_weights_directory);
            resolve_optional_path(base, &mut model.qwen_tokenizer_directory);
        }
        NodeModelConfig::Mistral(model) => resolve_path(base, &mut model.weights_directory),
        NodeModelConfig::MiniCpm5(model) => resolve_path(base, &mut model.weights_directory),
    }
    match backend {
        NodeBackendConfig::Rocm(backend) => {
            resolve_path(base, &mut backend.root);
            resolve_path(base, &mut backend.hiprtc_cache_directory);
        }
        NodeBackendConfig::Cuda(backend) => resolve_path(base, &mut backend.include_directory),
        NodeBackendConfig::Cpu(_) | NodeBackendConfig::Metal(_) => {}
    }
}

fn validate_node_model_backend(model: &NodeModelConfig, backend: &NodeBackendConfig) -> Result<(), ConfigError> {
    match (model, backend) {
        (NodeModelConfig::Ornith(model), NodeBackendConfig::Metal(_)) => validate_ornith(model),
        (NodeModelConfig::Ornith(model), NodeBackendConfig::Cuda(_)) => validate_ornith(model),
        // Ornith 只使用 full-attention cached GQA，ROCm 已有设备 prefill/decode 实现。
        (NodeModelConfig::Ornith(model), NodeBackendConfig::Rocm(backend)) => validate_ornith_rocm(model, backend),
        (NodeModelConfig::Ornith(_), _) => Err(ConfigError::Invalid("Ornith Node 当前支持 Metal/CUDA/ROCm backend".to_owned())),
        (NodeModelConfig::Gemma4(model), NodeBackendConfig::Metal(backend)) => validate_gemma4_metal(model, backend),
        (NodeModelConfig::Gemma4(model), NodeBackendConfig::Cuda(_)) => validate_gemma4_cuda(model),
        (NodeModelConfig::Qwen36(model), NodeBackendConfig::Metal(_)) => validate_qwen36(model),
        // CUDA 走 safetensors MLX affine,无 nextn 权重;GGUF 在 CUDA 上显存不可行,故不支持 MTP。
        (NodeModelConfig::Qwen36(model), NodeBackendConfig::Cuda(_)) if !model.execution.mtp => validate_qwen36(model),
        (NodeModelConfig::Qwen36(_), NodeBackendConfig::Cuda(_)) => Err(ConfigError::Invalid("Qwen3.6/Qwen3.8 CUDA Node 当前不支持 MTP".to_owned())),
        (NodeModelConfig::DeepseekV4(model), NodeBackendConfig::Rocm(backend)) => validate_deepseek_v4(model, backend),
        (NodeModelConfig::Glm53Flash(model), NodeBackendConfig::Rocm(backend)) => validate_glm53_flash(model, backend),
        (NodeModelConfig::Glm52(model), NodeBackendConfig::Rocm(backend)) => validate_glm52(model, backend),
        (NodeModelConfig::MinimaxH3(model), NodeBackendConfig::Rocm(backend)) if matches!(backend.devices.len(), 1 | 2 | 4 | 8) => {
            validate_h3(model)?;
            if backend.devices.len() > 1 && model.execution.stream_chunk_layers != 50 {
                return Err(ConfigError::Invalid("MiniMax-H3 Ulysses 多卡要求 execution.stream_chunk_layers=50".to_owned()));
            }
            Ok(())
        }
        (NodeModelConfig::MinimaxH3(_), NodeBackendConfig::Rocm(_)) => Err(ConfigError::Invalid("MiniMax-H3 Node 的 backend.devices 数量必须是 1/2/4/8".to_owned())),
        (NodeModelConfig::Gemma4(_), _) => Err(ConfigError::Invalid("Gemma 4 Node 当前支持 Metal/CUDA backend".to_owned())),
        (NodeModelConfig::Qwen36(_), _) => Err(ConfigError::Invalid("Qwen3.6/Qwen3.8 Node 当前支持 Metal/CUDA backend".to_owned())),
        (NodeModelConfig::DeepseekV4(_), _) => Err(ConfigError::Invalid("DeepSeek-V4 Node 当前只支持 ROCm backend".to_owned())),
        (NodeModelConfig::Glm53Flash(_), _) => Err(ConfigError::Invalid("GLM-5.3-Flash Node 当前只支持 ROCm backend".to_owned())),
        (NodeModelConfig::Glm52(_), _) => Err(ConfigError::Invalid("GLM-5.2 Node 当前只支持 ROCm backend".to_owned())),
        (NodeModelConfig::MinimaxH3(_), _) => Err(ConfigError::Invalid("MiniMax-H3 Node 当前只支持 ROCm backend".to_owned())),
        (NodeModelConfig::Mistral(model), NodeBackendConfig::Metal(_) | NodeBackendConfig::Cuda(_)) => validate_mistral(model),
        (NodeModelConfig::Mistral(_), _) => Err(ConfigError::Invalid("Mistral Node 当前支持 Metal/CUDA backend".to_owned())),
        (NodeModelConfig::MiniCpm5(model), NodeBackendConfig::Metal(_) | NodeBackendConfig::Cpu(_)) => validate_minicpm5(model),
        (NodeModelConfig::MiniCpm5(_), _) => Err(ConfigError::Invalid("MiniCPM5 Node 当前支持 Metal/CPU backend".to_owned())),
    }
}

impl EmbeddedConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let mut config: Self = read_yaml(path)?;
        validate_version(config.version)?;
        let base = config_base(path)?;
        resolve_path(&base, &mut config.session.cache_directory);
        resolve_node_model_backend(&base, &mut config.model, &mut config.backend);
        config.backend.validate()?;
        validate_node_model_backend(&config.model, &config.backend)?;
        Ok(config)
    }
}

impl RuntimeProcessConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let mut config: Self = read_yaml(path)?;
        let base = config_base(path)?;
        match &mut config {
            Self::Stage(config) => config.resolve_and_validate(&base)?,
            Self::Standalone(config) => config.resolve_and_validate(&base)?,
            Self::Node(config) => config.resolve_and_validate(&base)?,
        }
        Ok(config)
    }

    pub fn backend(&self) -> &BackendConfig {
        match self {
            Self::Stage(config) => &config.backend,
            Self::Standalone(_) => unreachable!("standalone 服务使用 NodeBackendConfig"),
            Self::Node(_) => unreachable!("node 角色的 backend 是 NodeBackendConfig,不暴露 BackendConfig"),
        }
    }
}

/// tools 一次性验证工具的配置装载。执行数据(权重路径/prompt/步数)留在
/// argv 位置参数;可调配置项(缓存预算/trace 开关/采样等)来自可选的
/// `--config <tool.yaml>`,未提供时全部走默认。每个 tool 在自己的 bin
/// 文件内定义 serde schema(贴近使用点),不使用环境变量。
pub fn load_tool_config<T: Default + DeserializeOwned + Serialize>(entry_name: &str) -> Result<T, String> {
    let mut path = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--config" {
            path = Some(arguments.next().ok_or("--config 需要 YAML 路径".to_owned())?);
        }
    }
    let Some(path) = path else { return Ok(T::default()) };
    let config: T = read_yaml(std::path::Path::new(&path)).map_err(|error| error.to_string())?;
    if std::env::args().any(|argument| argument == "--print-effective-config") {
        print!("{}", serde_yaml::to_string(&config).map_err(|error| error.to_string())?);
    }
    let _ = entry_name;
    Ok(config)
}

impl StageProcessConfig {
    fn resolve_and_validate(&mut self, base: &Path) -> Result<(), ConfigError> {
        validate_version(self.version)?;
        self.backend.validate()?;
        let BackendConfig::Rocm(backend) = &mut self.backend else { return Err(ConfigError::Invalid("Stage 当前只支持 ROCm backend".to_owned())) };
        resolve_path(base, &mut backend.root);
        resolve_path(base, &mut backend.hiprtc_cache_directory);
        match &mut self.model {
            StageModelConfig::Glm52(model) => {
                resolve_path(base, &mut model.weights_directory);
                resolve_optional_path(base, &mut model.compressed_tensors_directory);
                resolve_optional_path(base, &mut model.nvfp4_directory);
                resolve_optional_path(base, &mut model.gguf_directory);
                resolve_optional_path(base, &mut model.cache_directory);
                if let Some(tokenizer) = &mut model.tokenizer {
                    resolve_path(base, tokenizer);
                } else {
                    model.tokenizer = Some(model.compressed_tensors_directory.as_ref().or(model.nvfp4_directory.as_ref()).unwrap_or(&model.weights_directory).join("tokenizer.json"));
                }
                if let Some(generation) = &model.generation {
                    validate_stage_generation(generation)?;
                }
                resolve_optional_path(base, &mut model.execution.diagnostics.input_artifact);
                resolve_optional_path(base, &mut model.execution.diagnostics.output_artifact);
                resolve_optional_path(base, &mut model.execution.mtp_draft_vocabulary);
                resolve_optional_path(base, &mut model.execution.dspark_directory);
                validate_stage_glm52(model, backend)?;
            }
            StageModelConfig::Glm53Flash(model) => {
                resolve_path(base, &mut model.weights_directory);
                validate_stage_glm53_flash(model, backend)?;
            }
        }
        validate_stage_transport(&self.model, &self.transport)?;
        Ok(())
    }
}

impl StandaloneProcessConfig {
    fn resolve_and_validate(&mut self, base: &Path) -> Result<(), ConfigError> {
        resolve_path(base, &mut self.artifacts.directory);
        let mut node = NodeProcessConfig {
            version: self.version,
            scheduler: NodeSchedulerConfig { ticket: "embedded".to_owned(), api_key: None },
            iroh: self.iroh.clone(),
            node: self.node.clone(),
            model: self.model.clone(),
            backend: self.backend.clone(),
        };
        node.resolve_and_validate(base)?;
        self.node = node.node;
        self.model = node.model;
        self.backend = node.backend;
        Ok(())
    }

    pub fn server(&self) -> Result<(SocketAddr, ServerConfig), ConfigError> {
        let public_base_url = self.http.public_base_url.clone().unwrap_or_else(|| format!("http://{}", self.http.listen));
        let scheduler = SchedulerConfig { artifact_dir: self.artifacts.directory.clone(), public_base_url, dispatch_wait: std::time::Duration::from_secs(30), iroh: IrohConfig::default() };
        Ok((
            self.http.listen,
            ServerConfig { api_keys: Vec::new(), anonymous_cache_namespace: Some("standalone".to_owned()), node_api_key: None, models: Vec::new(), anthropic_family_tiers: Default::default(), model_aliases: Default::default(), scheduler },
        ))
    }
}

impl IrohListenerConfig {
    pub fn runtime(&self) -> Result<IrohConfig, ConfigError> {
        Ok(IrohConfig { secret_key: parse_optional_secret(self.secret_key.as_deref())?, bind_addr: nonempty(self.bind_addr.as_deref()), expected_peer: None })
    }
}

impl IrohPeerConfig {
    pub fn runtime(&self) -> Result<IrohConfig, ConfigError> {
        Ok(IrohConfig { secret_key: parse_optional_secret(self.secret_key.as_deref())?, bind_addr: nonempty(self.bind_addr.as_deref()), expected_peer: nonempty(self.expected_peer.as_deref()) })
    }
}

impl BackendConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::Cpu(config) if config.threads == Some(0) => Err(ConfigError::Invalid("backend.threads 必须大于 0".to_owned())),
            Self::Metal(config) if config.device != "default" => Err(ConfigError::Invalid("当前 Metal backend 只支持 device: default".to_owned())),
            Self::Cuda(config) if config.device < 0 => Err(ConfigError::Invalid("CUDA backend.device 不能为负数".to_owned())),
            Self::Rocm(config) if config.devices.is_empty() => Err(ConfigError::Invalid("ROCm backend 至少需要一个 device".to_owned())),
            Self::Rocm(config) if has_duplicates(&config.devices) => Err(ConfigError::Invalid("ROCm backend devices 不能重复".to_owned())),
            _ => Ok(()),
        }
    }
}

impl NodeBackendConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::Cpu(config) if config.threads == Some(0) => Err(ConfigError::Invalid("backend.threads 必须大于 0".to_owned())),
            Self::Metal(config) if config.device != "default" => Err(ConfigError::Invalid("当前 Metal backend 只支持 device: default".to_owned())),
            Self::Cuda(config) if config.device < 0 => Err(ConfigError::Invalid("CUDA backend.device 不能为负数".to_owned())),
            Self::Rocm(config) if config.devices.is_empty() => Err(ConfigError::Invalid("ROCm backend 至少需要一个 device".to_owned())),
            Self::Rocm(config) if has_duplicates(&config.devices) => Err(ConfigError::Invalid("ROCm backend devices 不能重复".to_owned())),
            _ => Ok(()),
        }
    }
}

#[allow(dead_code)]
fn validate_standalone_backend(model: &StandaloneModelConfig, backend: &BackendConfig) -> Result<(), ConfigError> {
    if matches!(model, StandaloneModelConfig::Qwen36(_)) && matches!(backend, BackendConfig::Rocm(config) if !config.allow_cpu_reference_fallback) {
        return Err(ConfigError::Invalid("Qwen3.6 ROCm 的部分 cached GQA 路径尚需 CPU reference，必须显式设置 backend.allow_cpu_reference_fallback: true".to_owned()));
    }
    match (model, backend) {
        (StandaloneModelConfig::Gemma4(model), BackendConfig::Metal(backend)) if model.execution.replay_ablation && !backend.replay => Err(ConfigError::Invalid("Gemma 4 replay_ablation 必须与 backend.replay 一起启用".to_owned())),
        (StandaloneModelConfig::Gemma4(_), BackendConfig::Metal(_)) => Ok(()),
        (StandaloneModelConfig::Gemma4(model), BackendConfig::Cuda(_)) if model.execution.mtp_weights.is_some() || model.execution.replay_ablation => Err(ConfigError::Invalid("Gemma 4 CUDA 当前不支持 Metal MTP/replay 选项".to_owned())),
        (StandaloneModelConfig::Gemma4(_), BackendConfig::Cuda(_)) => Ok(()),
        (StandaloneModelConfig::Gemma4(_), _) => Err(ConfigError::Invalid("Gemma 4 standalone 当前支持 Metal/CUDA backend".to_owned())),
        (StandaloneModelConfig::Glm52(model), BackendConfig::Cpu(_) | BackendConfig::Metal(_)) if model.prefill_layer_ends.is_none() => Ok(()),
        (StandaloneModelConfig::Glm52(model), BackendConfig::Rocm(backend)) => {
            if let Some(ends) = &model.prefill_layer_ends
                && (ends.len() != backend.devices.len() || ends.last().copied() != Some(GLM52_LAYER_COUNT - 1) || ends.windows(2).any(|pair| pair[0] >= pair[1]))
            {
                return Err(ConfigError::Invalid("GLM-5.2 standalone prefill_layer_ends 必须与 ROCm devices 一一对应、严格递增并以 77 结尾".to_owned()));
            }
            Ok(())
        }
        (StandaloneModelConfig::Glm52(_), BackendConfig::Cuda(_) | BackendConfig::Huawei(_)) => Err(ConfigError::Invalid("GLM-5.2 standalone 当前不支持所选 backend".to_owned())),
        (StandaloneModelConfig::Glm52(_), _) => Err(ConfigError::Invalid("prefill_layer_ends 只适用于 ROCm 多设备 standalone".to_owned())),
        (StandaloneModelConfig::Ornith(_), BackendConfig::Cpu(_) | BackendConfig::Metal(_) | BackendConfig::Cuda(_) | BackendConfig::Rocm(_)) => Ok(()),
        (StandaloneModelConfig::Ornith(_), BackendConfig::Huawei(_)) => Err(ConfigError::Invalid("Ornith standalone 当前不支持 Huawei backend".to_owned())),
        (StandaloneModelConfig::Qwen3Vl(_), BackendConfig::Cpu(_)) => Ok(()),
        (StandaloneModelConfig::Qwen3Vl(_), _) => Err(ConfigError::Invalid("Qwen3-VL standalone 当前只支持 CPU backend".to_owned())),
        (StandaloneModelConfig::Qwen36(_), BackendConfig::Metal(_)) => Ok(()),
        (StandaloneModelConfig::Qwen36(model), BackendConfig::Cpu(_) | BackendConfig::Cuda(_) | BackendConfig::Rocm(_)) if model.images.is_empty() && model.videos.is_empty() => Ok(()),
        (StandaloneModelConfig::Qwen36(_), BackendConfig::Cpu(_) | BackendConfig::Cuda(_) | BackendConfig::Rocm(_)) => Err(ConfigError::Invalid("Qwen3.6 图像/视频输入当前只支持 Metal backend".to_owned())),
        (StandaloneModelConfig::Qwen36(_), BackendConfig::Huawei(_)) => Err(ConfigError::Invalid("Qwen3.6 standalone 尚未接入 Huawei backend".to_owned())),
    }
}

fn validate_ornith(model: &OrnithNodeModelConfig) -> Result<(), ConfigError> {
    if model.max_sequence_length == 0 {
        return Err(ConfigError::Invalid("model.max_sequence_length 必须大于 0".to_owned()));
    }
    validate_ornith_execution(&model.execution)
}

fn validate_mistral(model: &MistralNodeModelConfig) -> Result<(), ConfigError> {
    if model.max_sequence_length == 0 {
        return Err(ConfigError::Invalid("model.max_sequence_length 必须大于 0".to_owned()));
    }
    Ok(())
}

fn validate_minicpm5(model: &MiniCpm5NodeModelConfig) -> Result<(), ConfigError> {
    if model.max_sequence_length == 0 {
        return Err(ConfigError::Invalid("model.max_sequence_length 必须大于 0".to_owned()));
    }
    Ok(())
}

/// Ornith + ROCm 分层部署：layer_ends 与 devices 一一对应、严格递增；
/// 最后一项等于模型层数-1 由引擎在打开 GGUF 后校验（层数来自权重元数据）。
fn validate_ornith_rocm(model: &OrnithNodeModelConfig, backend: &RocmBackendConfig) -> Result<(), ConfigError> {
    validate_ornith(model)?;
    if let Some(ends) = &model.layer_ends {
        if ends.len() != backend.devices.len() {
            return Err(ConfigError::Invalid("Ornith ROCm layer_ends 必须与 backend.devices 一一对应".to_owned()));
        }
        if ends.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ConfigError::Invalid("Ornith ROCm layer_ends 必须严格递增".to_owned()));
        }
    }
    Ok(())
}

fn validate_gemma4(model: &Gemma4NodeModelConfig) -> Result<(), ConfigError> {
    if model.max_sequence_length == 0 {
        return Err(ConfigError::Invalid("model.max_sequence_length 必须大于 0".to_owned()));
    }
    if model.execution.prefill_chunk_size == 0 {
        return Err(ConfigError::Invalid("Gemma 4 execution.prefill_chunk_size 必须大于 0".to_owned()));
    }
    if !(1..=4).contains(&model.execution.mtp_draft_tokens) {
        return Err(ConfigError::Invalid("Gemma 4 execution.mtp_draft_tokens 必须在 1..=4".to_owned()));
    }
    Ok(())
}

fn validate_gemma4_metal(model: &Gemma4NodeModelConfig, backend: &NodeMetalBackendConfig) -> Result<(), ConfigError> {
    validate_gemma4(model)?;
    if model.execution.replay_ablation && !backend.replay {
        return Err(ConfigError::Invalid("Gemma 4 replay_ablation 必须与 backend.replay 一起启用".to_owned()));
    }
    Ok(())
}

fn validate_gemma4_cuda(model: &Gemma4NodeModelConfig) -> Result<(), ConfigError> {
    validate_gemma4(model)?;
    if model.execution.mtp_weights.is_some() || model.execution.replay_ablation {
        return Err(ConfigError::Invalid("Gemma 4 CUDA 当前不支持 Metal MTP/replay 选项".to_owned()));
    }
    Ok(())
}

fn validate_qwen36(model: &Qwen36NodeModelConfig) -> Result<(), ConfigError> {
    if model.max_sequence_length == 0 {
        return Err(ConfigError::Invalid("model.max_sequence_length 必须大于 0".to_owned()));
    }
    if model.execution.prefill_chunk_size == 0 {
        return Err(ConfigError::Invalid("Qwen3.6/Qwen3.8 execution.prefill_chunk_size 必须大于 0".to_owned()));
    }
    if model.execution.vision_max_tokens == Some(0) {
        return Err(ConfigError::Invalid("Qwen3.6/Qwen3.8 execution.vision_max_tokens 必须大于 0".to_owned()));
    }
    if model.execution.mtp_draft_vocabulary.is_some() && !model.execution.mtp {
        return Err(ConfigError::Invalid("Qwen3.6/Qwen3.8 mtp_draft_vocabulary 必须与 mtp 一起启用".to_owned()));
    }
    if !model.execution.mtp && model.execution.mtp_draft_tokens != default_mtp_draft_tokens_qwen36() {
        return Err(ConfigError::Invalid("Qwen3.6/Qwen3.8 mtp_draft_tokens 必须与 mtp 一起启用".to_owned()));
    }
    if !(1..=4).contains(&model.execution.mtp_draft_tokens) {
        return Err(ConfigError::Invalid("Qwen3.6/Qwen3.8 mtp_draft_tokens 必须在 1..=4(链深越大 verify 行数越多,27B 量化权重建议不超过 3)".to_owned()));
    }
    Ok(())
}

/// node 与 standalone 两个入口共用的 Ornith 专家执行参数校验。
fn validate_ornith_expert_options(expert_batch_size: Option<usize>, expert_prefetch_count: Option<usize>, expert_cache_gib: usize) -> Result<(), ConfigError> {
    if expert_batch_size == Some(0) || expert_prefetch_count == Some(0) || expert_cache_gib == 0 {
        return Err(ConfigError::Invalid("Ornith expert_batch_size、expert_prefetch_count 与 expert_cache_gib 必须大于 0".to_owned()));
    }
    Ok(())
}

fn validate_ornith_execution(execution: &OrnithNodeExecutionConfig) -> Result<(), ConfigError> {
    validate_ornith_expert_options(execution.expert_batch_size, execution.expert_prefetch_count, execution.expert_cache_gib)?;
    if execution.terminal_cache_entries == 0 {
        return Err(ConfigError::Invalid("Ornith terminal_cache_entries 必须大于 0".to_owned()));
    }
    Ok(())
}

#[allow(dead_code)]
fn validate_ornith_standalone_execution(execution: &OrnithStandaloneExecutionConfig) -> Result<(), ConfigError> {
    validate_ornith_expert_options(execution.expert_batch_size, execution.expert_prefetch_count, execution.expert_cache_gib)
}

fn validate_scheduling(scheduling: &Glm52SchedulingConfig) -> Result<(), ConfigError> {
    if scheduling.execution_slots == 0
        || scheduling.decode_execution_slots == 0
        || scheduling.pipeline_work_window == 0
        || scheduling.prefill_admission_burst == 0
        || scheduling.decode_batch_limit == 0
        || scheduling.prefill_batch_limit == 0
        || scheduling.append_prefill_chunk_size == 0
        || scheduling.decode_priority_prefill_chunk_size == 0
        || scheduling.decode_priority_prefill_chunk_ceiling < scheduling.decode_priority_prefill_chunk_size
        || scheduling.long_prefill_threshold_tokens == 0
        || scheduling.long_prefill_chunk_size == 0
    {
        return Err(ConfigError::Invalid("GLM-5.2 scheduling 参数必须大于 0".to_owned()));
    }
    Ok(())
}

fn validate_glm52_stage_execution(execution: &Glm52StageExecutionConfig) -> Result<(), ConfigError> {
    if execution.prefill_chunk_size == 0 || execution.max_concurrency == 0 || execution.preload_layers_per_device == Some(0) || !(1..=8).contains(&execution.mtp_draft_tokens) {
        return Err(ConfigError::Invalid("GLM-5.2 execution chunk、并发和 preload 参数必须大于 0".to_owned()));
    }
    if execution.mtp_draft_vocabulary.is_some() && !execution.mtp {
        return Err(ConfigError::Invalid("GLM-5.2 mtp_draft_vocabulary 必须与 mtp 一起启用".to_owned()));
    }
    if execution.mtp && execution.dspark_directory.is_some() {
        return Err(ConfigError::Invalid("GLM-5.2 MTP 与 DSpark 不能同时启用".to_owned()));
    }
    validate_scheduling(&execution.scheduling)
}

fn validate_glm53_flash(model: &Glm53FlashNodeModelConfig, backend: &RocmBackendConfig) -> Result<(), ConfigError> {
    const GLM53_FLASH_LAYER_COUNT: usize = 45;
    if model.max_sequence_length == 0 || model.prefill_chunk_size == 0 || !(1..=8).contains(&model.mtp_draft_tokens) {
        return Err(ConfigError::Invalid("GLM-5.3-Flash 序列长度与 prefill chunk 必须大于 0".to_owned()));
    }
    let head = &model.head;
    if head.stage_end == 0 || head.stage_end >= GLM53_FLASH_LAYER_COUNT {
        return Err(ConfigError::Invalid(format!("GLM-5.3-Flash head.stage_end 必须在 1..{GLM53_FLASH_LAYER_COUNT}")));
    }
    if head.layer_ends.len() != backend.devices.len() || head.layer_ends.is_empty() {
        return Err(ConfigError::Invalid(format!("GLM-5.3-Flash head.layer_ends 数量 {} 必须与 backend.devices 数量 {} 一致且非空", head.layer_ends.len(), backend.devices.len())));
    }
    if head.layer_ends.last().copied() != Some(head.stage_end) || head.layer_ends.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ConfigError::Invalid("GLM-5.3-Flash head.layer_ends 必须严格递增且末项等于 stage_end".to_owned()));
    }
    if head.downstream.ticket.trim().is_empty() {
        return Err(ConfigError::Invalid("GLM-5.3-Flash head.downstream.ticket 不能为空".to_owned()));
    }
    Ok(())
}

fn validate_glm52(model: &Glm52NodeModelConfig, backend: &RocmBackendConfig) -> Result<(), ConfigError> {
    if model.max_sequence_length == 0
        || model.execution.prefill_chunk_size == 0
        || model.execution.terminal_cache_entries == 0
        || model.execution.kv_reservation_page_tokens == 0
        || model.execution.memory_reserve_bytes == 0
        || model.execution.preload_layers_per_device == Some(0)
        || !(1..=8).contains(&model.execution.mtp_draft_tokens)
        || !(1..=8).contains(&model.execution.dspark_draft_tokens)
    {
        return Err(ConfigError::Invalid("GLM-5.2 序列、chunk 与 cache 大小必须大于 0".to_owned()));
    }
    if model.execution.dspark_confidence_threshold.is_some_and(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(&threshold)) {
        return Err(ConfigError::Invalid("GLM-5.2 dspark_confidence_threshold 必须在 [0,1]".to_owned()));
    }
    if model.execution.dspark_cpu_affinity.as_ref().is_some_and(|affinity| affinity.trim().is_empty()) {
        return Err(ConfigError::Invalid("GLM-5.2 dspark_cpu_affinity 不能为空".to_owned()));
    }
    validate_scheduling(&model.execution.scheduling)?;
    let weight_overrides = [model.compressed_tensors_directory.is_some(), model.nvfp4_directory.is_some(), model.gguf_directory.is_some()].into_iter().filter(|configured| *configured).count();
    if weight_overrides > 1 {
        return Err(ConfigError::Invalid("GLM-5.2 compressed_tensors_directory、nvfp4_directory 与 gguf_directory 最多配置一个".to_owned()));
    }
    if model.execution.mtp && !model.execution.tail_sampling {
        return Err(ConfigError::Invalid("GLM-5.2 MTP 必须与 tail_sampling 一起启用，LM head 与 L78 固定驻留尾卡".to_owned()));
    }
    if model.execution.mtp && model.execution.dspark_directory.is_some() {
        return Err(ConfigError::Invalid("GLM-5.2 MTP 与 DSpark 不能同时启用".to_owned()));
    }
    if backend.mla_decode_split_threshold == 0 {
        return Err(ConfigError::Invalid("ROCm MLA decode split 阈值必须大于 0".to_owned()));
    }
    if model.head.stage_end == 0 || model.head.stage_end >= GLM52_LAYER_COUNT {
        return Err(ConfigError::Invalid(format!("GLM-5.2 Node head.stage_end 必须在 1..{GLM52_LAYER_COUNT}")));
    }
    if model.head.downstream.ticket.trim().is_empty() {
        return Err(ConfigError::Invalid("model.head.downstream.ticket 不能为空".to_owned()));
    }
    let logical_stage_count = if model.execution.cooperative_expert_pairs {
        if backend.devices.len() % 2 != 0 {
            return Err(ConfigError::Invalid("GLM-5.2 cooperative_expert_pairs 要求偶数张设备".to_owned()));
        }
        backend.devices.len() / 2
    } else {
        backend.devices.len()
    };
    if model.head.layer_ends.len() != logical_stage_count || model.head.layer_ends.last().copied() != Some(model.head.stage_end - 1) || model.head.layer_ends.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ConfigError::Invalid("GLM-5.2 head.layer_ends 必须与逻辑 stage 一一对应、严格递增，最后等于 stage_end-1；cooperative 模式每两张设备为一个 stage".to_owned()));
    }
    Ok(())
}

fn validate_deepseek_v4(model: &DeepSeekV4NodeModelConfig, backend: &RocmBackendConfig) -> Result<(), ConfigError> {
    const LAYERS: usize = 43;
    let execution = &model.execution;
    if model.max_sequence_length == 0
        || execution.core_cache_gib == 0
        || execution.prefill_chunk_size == 0
        || execution.decode_priority_prefill_chunk_size == 0
        || execution.decode_priority_prefill_chunk_ceiling < execution.decode_priority_prefill_chunk_size
        || execution.device_pool_gib == 0
        || execution.decode_batch_limit == 0
        || execution.kv_reservation_page_tokens == 0
        || execution.memory_reserve_bytes == 0
    {
        return Err(ConfigError::Invalid("DeepSeek-V4 序列、cache、chunk 与 pool 大小必须大于 0".to_owned()));
    }
    if execution.dspark_draft_tokens == Some(0) || execution.dspark_min_sessions == 0 {
        return Err(ConfigError::Invalid("DeepSeek-V4 dspark_draft_tokens 与 dspark_min_sessions 必须大于 0".to_owned()));
    }
    if execution.dspark_confidence_threshold.is_some_and(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(&threshold)) {
        return Err(ConfigError::Invalid("DeepSeek-V4 dspark_confidence_threshold 必须在 [0,1]".to_owned()));
    }
    if execution.score_expert_top_k.is_some_and(|top_k| !(1..=6).contains(&top_k)) {
        return Err(ConfigError::Invalid("DeepSeek-V4 score_expert_top_k 必须在 1..=6".to_owned()));
    }
    if !matches!((execution.long_prefill_threshold_tokens, execution.long_prefill_chunk_size), (None, None) | (Some(1..), Some(1..))) {
        return Err(ConfigError::Invalid("DeepSeek-V4 long prefill threshold/chunk 必须同时省略或同时大于 0".to_owned()));
    }
    if model.layer_ends.len() != backend.devices.len() || model.layer_ends.last().copied() != Some(LAYERS) || model.layer_ends.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ConfigError::Invalid(format!("DeepSeek-V4 layer_ends 必须与 backend.devices 一一对应、严格递增，最后等于 {LAYERS}")));
    }
    Ok(())
}

fn validate_h3(model: &H3NodeModelConfig) -> Result<(), ConfigError> {
    if model.execution.steps < 2 || model.execution.stream_chunk_layers == 0 || model.execution.stream_chunk_layers > 50 {
        return Err(ConfigError::Invalid("MiniMax-H3 execution.steps 必须 >=2，stream_chunk_layers 必须在 1..=50".to_owned()));
    }
    if model.execution.ffmpeg.trim().is_empty() {
        return Err(ConfigError::Invalid("MiniMax-H3 execution.ffmpeg 不能为空".to_owned()));
    }
    Ok(())
}

fn validate_stage_glm52(model: &Glm52StageModelConfig, backend: &RocmBackendConfig) -> Result<(), ConfigError> {
    let layers = &model.layers;
    if layers.start >= layers.end || layers.end > GLM52_LAYER_COUNT {
        return Err(ConfigError::Invalid(format!("Stage layers 必须满足 0 <= start < end <= {GLM52_LAYER_COUNT}")));
    }
    let logical_stage_count = if model.execution.cooperative_expert_pairs {
        if backend.devices.len() % 2 != 0 {
            return Err(ConfigError::Invalid("GLM-5.2 cooperative_expert_pairs 要求偶数张设备".to_owned()));
        }
        backend.devices.len() / 2
    } else {
        backend.devices.len()
    };
    if layers.device_layer_ends.len() != logical_stage_count
        || layers.device_layer_ends.last().copied() != Some(layers.end - 1)
        || layers.device_layer_ends.windows(2).any(|pair| pair[0] >= pair[1])
        || layers.device_layer_ends.first().is_some_and(|end| *end < layers.start)
    {
        return Err(ConfigError::Invalid("Stage device_layer_ends 必须与逻辑 stage 一一对应并覆盖完整层区间；cooperative 模式每两张设备为一个 stage".to_owned()));
    }
    if model.max_sequence_length == 0 {
        return Err(ConfigError::Invalid("Stage max_sequence_length 必须大于 0".to_owned()));
    }
    if backend.mla_decode_split_threshold == 0 {
        return Err(ConfigError::Invalid("ROCm MLA decode split 阈值必须大于 0".to_owned()));
    }
    validate_glm52_stage_execution(&model.execution)?;
    if model.compressed_tensors_directory.is_some() && model.nvfp4_directory.is_some() {
        return Err(ConfigError::Invalid("Stage compressed_tensors_directory 与 nvfp4_directory 不能同时配置".to_owned()));
    }
    if model.layers.start > 0 && model.layers.end == GLM52_LAYER_COUNT && model.cache_directory.is_none() {
        return Err(ConfigError::Invalid("GLM-5.2 末段 Stage 必须配置 model.cache_directory".to_owned()));
    }
    Ok(())
}

fn validate_stage_glm53_flash(model: &Glm53FlashStageModelConfig, backend: &RocmBackendConfig) -> Result<(), ConfigError> {
    const LAYER_COUNT: usize = 45;
    let layers = &model.layers;
    if layers.start == 0 || layers.start >= layers.end || layers.end != LAYER_COUNT {
        return Err(ConfigError::Invalid(format!("GLM-5.3-Flash tail 必须满足 0 < start < end = {LAYER_COUNT}")));
    }
    if model.max_sequence_length == 0 || !(1..=8).contains(&model.mtp_draft_tokens) {
        return Err(ConfigError::Invalid("GLM-5.3-Flash tail max_sequence_length 必须大于 0".to_owned()));
    }
    if layers.device_layer_ends.len() != backend.devices.len()
        || layers.device_layer_ends.last().copied() != Some(layers.end)
        || layers.device_layer_ends.windows(2).any(|pair| pair[0] >= pair[1])
        || layers.device_layer_ends.first().is_some_and(|end| *end <= layers.start)
    {
        return Err(ConfigError::Invalid("GLM-5.3-Flash tail device_layer_ends 必须与 devices 一一对应并覆盖完整层区间".to_owned()));
    }
    Ok(())
}

fn validate_stage_transport(model: &StageModelConfig, transport: &StageTransportConfig) -> Result<(), ConfigError> {
    match (model, transport) {
        (StageModelConfig::Glm52(model), StageTransportConfig::Connect { ticket, .. }) if model.layers.start == 0 && model.layers.end < 78 && model.generation.is_some() && !ticket.trim().is_empty() => Ok(()),
        (StageModelConfig::Glm52(model), StageTransportConfig::Listen { .. }) if model.layers.start > 0 && model.layers.end == 78 && model.generation.is_none() => Ok(()),
        (StageModelConfig::Glm52(model), _) if model.layers.start == 0 => Err(ConfigError::Invalid("首段 Stage 必须配置 generation，并以 connect transport 连接后继段".to_owned())),
        (StageModelConfig::Glm52(_), _) => Err(ConfigError::Invalid("末段 Stage 不配置 generation，并以 listen transport 接收前驱段".to_owned())),
        (StageModelConfig::Glm53Flash(_), StageTransportConfig::Listen { .. }) => Ok(()),
        (StageModelConfig::Glm53Flash(_), _) => Err(ConfigError::Invalid("GLM-5.3-Flash tail 必须以 listen transport 接收 head".to_owned())),
    }
}

#[allow(dead_code)]
fn validate_generation(generation: &TextGenerationConfig) -> Result<(), ConfigError> {
    if generation.prompt.is_empty() || generation.max_sequence_length == 0 { Err(ConfigError::Invalid("standalone generation.prompt 不能为空且 max_sequence_length 必须大于 0".to_owned())) } else { Ok(()) }
}

fn validate_stage_generation(generation: &StageGenerationConfig) -> Result<(), ConfigError> {
    if generation.prompt.is_empty() { Err(ConfigError::Invalid("stage generation.prompt 不能为空".to_owned())) } else { Ok(()) }
}

fn read_yaml<T: DeserializeOwned>(path: &Path) -> Result<T, ConfigError> {
    let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::Read { path: path.to_owned(), source })?;
    serde_yaml::from_str(&contents).map_err(|source| ConfigError::Parse { path: path.to_owned(), source })
}

fn config_base(path: &Path) -> Result<PathBuf, ConfigError> {
    let absolute = if path.is_absolute() { path.to_owned() } else { std::env::current_dir().map_err(|source| ConfigError::Read { path: path.to_owned(), source })?.join(path) };
    Ok(absolute.parent().unwrap_or_else(|| Path::new(".")).to_owned())
}

fn resolve_path(base: &Path, path: &mut PathBuf) {
    if path.is_relative() {
        *path = base.join(&*path);
    }
}

fn resolve_optional_path(base: &Path, path: &mut Option<PathBuf>) {
    if let Some(path) = path {
        resolve_path(base, path);
    }
}

fn validate_version(version: u32) -> Result<(), ConfigError> {
    if version == CONFIG_VERSION { Ok(()) } else { Err(ConfigError::Invalid(format!("不支持 version={version}，当前只支持 {CONFIG_VERSION}"))) }
}

fn parse_optional_secret(value: Option<&str>) -> Result<Option<iroh::SecretKey>, ConfigError> {
    nonempty(value).map(|value| parse_secret(&value).map_err(ConfigError::Invalid)).transpose()
}

fn nonempty(value: Option<&str>) -> Option<String> {
    value.map(str::trim).filter(|value| !value.is_empty()).map(str::to_owned)
}

fn has_duplicates<T: Ord + Copy>(values: &[T]) -> bool {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).any(|pair| pair[0] == pair[1])
}

const fn default_dispatch_wait_seconds() -> u64 {
    30
}
fn default_metal_device() -> String {
    "default".to_owned()
}
fn default_rocm_root() -> PathBuf {
    PathBuf::from("/opt/rocm")
}
fn default_hiprtc_cache_directory() -> PathBuf {
    PathBuf::from("/tmp/zllm-hiprtc")
}
const fn default_rocm_mla_decode_split_threshold() -> usize {
    1024
}
const fn default_true() -> bool {
    true
}
const fn default_max_sequence_length() -> usize {
    4096
}
const fn default_expert_cache_gib() -> usize {
    12
}
const fn default_terminal_cache_entries() -> usize {
    1
}
const fn default_terminal_cache_global_entries() -> usize {
    64
}
const fn default_terminal_cache_prefix_rounds() -> usize {
    4
}
const fn default_glm52_prefill_chunk_size() -> usize {
    2048
}
const fn default_unlimited_entries() -> usize {
    usize::MAX
}
const fn default_kv_reservation_page_tokens() -> usize {
    1024
}
const fn default_memory_reserve_bytes() -> usize {
    2usize << 30
}
const fn default_stage_execution_slots() -> usize {
    1
}
const fn default_pipeline_work_window() -> usize {
    8
}
const fn default_prefill_admission_burst() -> usize {
    1
}
const fn default_decode_batch_limit() -> usize {
    4
}
const fn default_prefill_batch_limit() -> usize {
    4
}
const fn default_append_prefill_chunk_size() -> usize {
    2048
}
const fn default_decode_priority_prefill_chunk_size() -> usize {
    32
}
const fn default_decode_priority_prefill_chunk_ceiling() -> usize {
    256
}
const fn default_long_prefill_threshold_tokens() -> usize {
    128 * 1024
}
const fn default_long_prefill_chunk_size() -> usize {
    2048
}
const fn default_h3_steps() -> usize {
    20
}
const fn default_h3_stream_chunk_layers() -> usize {
    1
}
const fn default_h3_block_cache_threshold() -> f32 {
    0.08
}
const fn default_h3_block_cache_start_percent() -> f32 {
    0.10
}
const fn default_h3_block_cache_end_percent() -> f32 {
    0.95
}
const fn default_h3_block_cache_max_consecutive_hits() -> usize {
    2
}
const fn default_glm52_max_sequence_length() -> usize {
    1_048_576
}
const fn default_stage_prefill_chunk_size() -> usize {
    2048
}
const fn default_stage_max_concurrency() -> usize {
    64
}

const fn default_mtp_draft_tokens() -> usize {
    3
}

/// qwen36 MTP 链式 draft 默认回到 nextn=1 旧行为;显式配置 K>1 才启用多候选。
const fn default_mtp_draft_tokens_qwen36() -> usize {
    1
}

const fn default_dspark_draft_tokens() -> usize {
    7
}

const fn default_dspark_verify_group_rows() -> usize {
    1
}
const fn default_metal_prefill_chunk_size() -> usize {
    2048
}
fn default_ffmpeg() -> String {
    "ffmpeg".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_rejects_backend_fields() {
        let error = serde_yaml::from_str::<SchedulerProcessConfig>(
            "version: 1\nkind: scheduler\nhttp:\n  listen: 127.0.0.1:8000\n  public_base_url: http://127.0.0.1:8000\nscheduler:\n  iroh: {}\nartifacts:\n  directory: ./artifacts\nbackend:\n  kind: cpu\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field `backend`"));
    }

    #[test]
    fn backend_uses_tagged_specific_fields() {
        let backend: BackendConfig = serde_yaml::from_str("kind: rocm\ndevices: [0, 1]\nroot: /opt/rocm\nhiprtc_cache_directory: /tmp/zllm\n").unwrap();
        backend.validate().unwrap();
        let error = serde_yaml::from_str::<BackendConfig>("kind: metal\ndevices: [0]\n").unwrap_err();
        assert!(error.to_string().contains("unknown field `devices`"));

        let error = serde_yaml::from_str::<NodeBackendConfig>("kind: metal\nprofile_decode: true\n").unwrap_err();
        assert!(error.to_string().contains("unknown field `profile_decode`"));
    }

    #[test]
    fn metal_replay_is_a_backend_capability_and_defaults_on() {
        let backend: NodeMetalBackendConfig = serde_yaml::from_str("{}").unwrap();
        assert!(backend.replay);
        let disabled: NodeMetalBackendConfig = serde_yaml::from_str("replay: false\n").unwrap();
        assert!(!disabled.replay);

        let mut model = Gemma4NodeModelConfig { weights_directory: PathBuf::from("./model.gguf"), lm_head_quantization: LmHeadQuantization::default(), max_sequence_length: 8192, execution: Gemma4ExecutionConfig::default() };
        validate_gemma4_metal(&model, &backend).unwrap();
        validate_gemma4_cuda(&model).unwrap();

        model.execution.replay_ablation = true;
        assert!(validate_gemma4_metal(&model, &disabled).is_err());
    }

    #[test]
    fn terminal_cache_prefix_rounds_defaults_to_four_and_accepts_legacy_name() {
        let defaults: NodeServiceConfig = serde_yaml::from_str("cache_directory: /tmp/zllm\n").unwrap();
        assert_eq!(defaults.terminal_cache_prefix_rounds, 4);
        assert!(defaults.persist_kv_cache);
        let legacy: NodeServiceConfig = serde_yaml::from_str("cache_directory: /tmp/zllm\nterminal_cache_lineage_rounds: 3\n").unwrap();
        assert_eq!(legacy.terminal_cache_prefix_rounds, 3);
        let benchmark: NodeServiceConfig = serde_yaml::from_str("cache_directory: /tmp/zllm\npersist_kv_cache: false\n").unwrap();
        assert!(!benchmark.persist_kv_cache);
    }

    #[test]
    fn example_configs_parse_with_separate_process_shapes() {
        for path in ["config/scheduler.yaml", "config/node-ornith.yaml", "config/node-glm52.yaml", "config/node-deepseek-v4.yaml", "config/node-h3.yaml"] {
            if path.contains("scheduler") {
                SchedulerProcessConfig::load(Path::new(path)).unwrap();
            } else {
                // node yaml 统一走 RuntimeProcessConfig 的 kind: node 变体。
                let config = RuntimeProcessConfig::load(Path::new(path)).unwrap();
                assert!(matches!(config, RuntimeProcessConfig::Node(_)), "{path} 应为 kind: node");
            }
        }
        for path in ["config/stage-glm52-head.yaml", "config/stage-glm52-tail.yaml", "config/standalone-ornith-metal.yaml", "config/standalone-glm52-rocm.yaml"] {
            RuntimeProcessConfig::load(Path::new(path)).unwrap();
        }
        for path in ["config/embedded-gemma4-metal.yaml", "config/embedded-qwen36-metal.yaml"] {
            EmbeddedConfig::load(Path::new(path)).unwrap();
        }
    }

    #[test]
    fn embedded_config_rejects_service_fields() {
        let error = serde_yaml::from_str::<EmbeddedConfig>("version: 1\nhttp: {}\nsession:\n  cache_directory: ./cache\nmodel:\n  architecture: gemma4\n  weights_directory: ./model.gguf\nbackend:\n  kind: metal\n").unwrap_err();
        assert!(error.to_string().contains("unknown field `http`"));
    }

    #[test]
    fn glm52_examples_keep_the_8x8_topology() {
        let RuntimeProcessConfig::Node(head) = RuntimeProcessConfig::load(Path::new("config/node-glm52.yaml")).unwrap() else { unreachable!() };
        let NodeModelConfig::Glm52(head_model) = head.model else { unreachable!() };
        let NodeBackendConfig::Rocm(head_backend) = head.backend else { unreachable!() };
        assert_eq!(head.node.max_concurrency, Some(8));
        assert_eq!(head_model.head.stage_end, 38);
        assert_eq!(head_model.head.layer_ends, [2, 7, 12, 17, 22, 27, 32, 37]);
        assert_eq!(head_backend.devices, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(head_model.execution.mtp);
        assert!(head_model.execution.tail_sampling);
        assert_eq!(head_model.execution.mtp_draft_tokens, 3);
        assert_eq!(head_model.execution.reasoning_effort, Glm52ReasoningEffort::Max);
        assert_eq!(head_model.execution.thinking_token_budget, Some(16_384));

        let RuntimeProcessConfig::Stage(tail) = RuntimeProcessConfig::load(Path::new("config/stage-glm52-tail.yaml")).unwrap() else { unreachable!() };
        let StageModelConfig::Glm52(tail_model) = tail.model else { unreachable!() };
        let BackendConfig::Rocm(tail_backend) = tail.backend else { unreachable!() };
        assert_eq!((tail_model.layers.start, tail_model.layers.end), (38, 78));
        assert_eq!(tail_model.layers.device_layer_ends, [42, 47, 52, 57, 62, 67, 73, 77]);
        assert_eq!(tail_backend.devices, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(tail_model.execution.max_concurrency, 8);
        assert!(tail_model.execution.mtp);
    }

    #[test]
    fn scheduling_rejects_zero_values() {
        let mut scheduling = Glm52SchedulingConfig { execution_slots: 0, ..Glm52SchedulingConfig::default() };
        assert!(validate_scheduling(&scheduling).is_err());
        scheduling = Glm52SchedulingConfig::default();
        scheduling.decode_execution_slots = 0;
        assert!(validate_scheduling(&scheduling).is_err());
        scheduling = Glm52SchedulingConfig::default();
        scheduling.pipeline_work_window = 0;
        assert!(validate_scheduling(&scheduling).is_err());
        scheduling = Glm52SchedulingConfig::default();
        scheduling.append_prefill_chunk_size = 0;
        assert!(validate_scheduling(&scheduling).is_err());
        scheduling = Glm52SchedulingConfig::default();
        scheduling.decode_priority_prefill_chunk_size = 0;
        assert!(validate_scheduling(&scheduling).is_err());
        scheduling = Glm52SchedulingConfig::default();
        scheduling.long_prefill_threshold_tokens = 0;
        assert!(validate_scheduling(&scheduling).is_err());
        scheduling = Glm52SchedulingConfig::default();
        scheduling.long_prefill_chunk_size = 0;
        assert!(validate_scheduling(&scheduling).is_err());
    }

    #[test]
    fn deepseek_dspark_min_sessions_defaults_to_one_and_rejects_zero() {
        let defaults: DeepSeekV4NodeExecutionConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(defaults.dspark_min_sessions, 1);
        let tuned: DeepSeekV4NodeExecutionConfig = serde_yaml::from_str("dspark_min_sessions: 2\n").unwrap();
        assert_eq!(tuned.dspark_min_sessions, 2);

        let RuntimeProcessConfig::Node(config) = RuntimeProcessConfig::load(Path::new("config/node-deepseek-v4.yaml")).unwrap() else { unreachable!() };
        let NodeModelConfig::DeepseekV4(mut model) = config.model else { unreachable!() };
        let NodeBackendConfig::Rocm(backend) = config.backend else { unreachable!() };
        model.execution.dspark_min_sessions = 0;
        assert!(validate_deepseek_v4(&model, &backend).is_err());
    }

    #[test]
    fn glm52_reasoning_defaults_remain_backward_compatible() {
        let execution: Glm52NodeExecutionConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(execution.prefill_chunk_size, 2048);
        assert_eq!(execution.scheduling.append_prefill_chunk_size, 2048);
        assert_eq!(execution.reasoning_effort, Glm52ReasoningEffort::Max);
        assert_eq!(execution.thinking_token_budget, None);
        assert!(!execution.tail_sampling);
        assert_eq!(execution.dspark_backend, Glm52DsparkExecutionBackend::Rocm);
        assert_eq!(execution.dspark_cpu_affinity, None);
        assert_eq!(execution.dspark_weight_quantization, ResidentWeightQuantization::Native);
        assert_eq!(execution.dspark_confidence_threshold, None);
        let execution: Glm52NodeExecutionConfig = serde_yaml::from_str("dspark_weight_quantization: q8g128\n").unwrap();
        assert_eq!(execution.dspark_weight_quantization, ResidentWeightQuantization::Q8g128);
        let execution: Glm52NodeExecutionConfig = serde_yaml::from_str("dspark_backend: cpu\n").unwrap();
        assert_eq!(execution.dspark_backend, Glm52DsparkExecutionBackend::Cpu);
        let execution: Glm52NodeExecutionConfig = serde_yaml::from_str("dspark_cpu_affinity: 0-31\n").unwrap();
        assert_eq!(execution.dspark_cpu_affinity.as_deref(), Some("0-31"));
        let execution: Glm52NodeExecutionConfig = serde_yaml::from_str("dspark_confidence_threshold: 0.7\n").unwrap();
        assert_eq!(execution.dspark_confidence_threshold, Some(0.7));
        let stage = Glm52StageExecutionConfig::default();
        assert_eq!(stage.prefill_chunk_size, 2048);
        assert_eq!(stage.scheduling.append_prefill_chunk_size, 2048);
        assert_eq!(stage.dspark_weight_quantization, ResidentWeightQuantization::Native);
        let stage: Glm52StageExecutionConfig = serde_yaml::from_str("dspark_weight_quantization: q8g128\n").unwrap();
        assert_eq!(stage.dspark_weight_quantization, ResidentWeightQuantization::Q8g128);
        assert!(serde_yaml::from_str::<Glm52NodeExecutionConfig>("reasoning_effort: medium\n").is_err());
    }

    #[test]
    fn node只接受已有真实engine的ornith与qwen后端() {
        let base = "version: 1\nscheduler:\n  ticket: embedded\nnode:\n  cache_directory: ./cache\n";
        let parse = |model: &str, backend: &str| serde_yaml::from_str::<NodeProcessConfig>(&format!("{base}model:\n{model}backend:\n{backend}")).unwrap();

        let ornith = parse("  architecture: ornith\n  weights_directory: ./w\n", "  kind: cpu\n");
        assert!(ornith.validate().is_err());
        let ornith = parse("  architecture: ornith\n  weights_directory: ./w\n", "  kind: cuda\n  include_directory: ./include\n");
        ornith.validate().unwrap();
        let ornith = parse("  architecture: ornith\n  weights_directory: ./w\n", "  kind: rocm\n  devices: [0]\n  allow_cpu_reference_fallback: false\n");
        ornith.validate().unwrap();

        let qwen36 = parse("  architecture: qwen36\n  weights_directory: ./w\n", "  kind: cpu\n");
        assert!(qwen36.validate().is_err());
        let qwen36 = parse("  architecture: qwen36\n  weights_directory: ./w\n", "  kind: cuda\n  include_directory: ./include\n");
        qwen36.validate().unwrap();

        let qwen36_mtp = parse("  architecture: qwen36\n  weights_directory: ./w\n  execution:\n    mtp: true\n", "  kind: cuda\n  include_directory: ./include\n");
        let error = qwen36_mtp.validate().unwrap_err();
        assert!(error.to_string().contains("不支持 MTP"));

        let mistral = parse("  architecture: mistral\n  weights_directory: ./w\n", "  kind: cuda\n  include_directory: ./include\n");
        mistral.validate().unwrap();

        let gemma4 = parse("  architecture: gemma4\n  weights_directory: ./w\n", "  kind: cpu\n");
        assert!(gemma4.validate().is_err());
    }

    #[test]
    fn rocm_root_resolves_from_yaml_directory() {
        let base = Path::new("/tmp/zllm-config");
        let mut root = PathBuf::from("./rocm");
        resolve_path(base, &mut root);
        assert_eq!(root, base.join("rocm"));
    }

    #[test]
    fn glm52_tail_requires_cache_directory() {
        let RuntimeProcessConfig::Stage(config) = RuntimeProcessConfig::load(Path::new("config/stage-glm52-tail.yaml")).unwrap() else { unreachable!() };
        let StageModelConfig::Glm52(mut model) = config.model else { unreachable!() };
        let BackendConfig::Rocm(backend) = config.backend else { unreachable!() };
        model.cache_directory = None;
        let error = validate_stage_glm52(&model, &backend).unwrap_err();
        assert!(error.to_string().contains("model.cache_directory"));
    }

    /// 构造合法 GLM-5.2 head NodeModelConfig 的最小骨架,只动被测字段。
    /// `head` 整段由调用方控制(stage_end / layer_ends / downstream 都可改);
    /// `devices` 决定 backend 的卡数,用于与 head.layer_ends 数量对齐的校验。
    fn glm52_node_fixture(head: Glm52HeadConfig, devices: Vec<i32>) -> (Glm52NodeModelConfig, RocmBackendConfig) {
        let model = Glm52NodeModelConfig {
            weights_directory: PathBuf::from("/tmp/glm52"),
            lm_head_quantization: LmHeadQuantization::default(),
            compressed_tensors_directory: Some(PathBuf::from("/tmp/glm52-ct")),
            nvfp4_directory: None,
            gguf_directory: None,
            tokenizer: Some(PathBuf::from("/tmp/glm52-ct/tokenizer.json")),
            max_sequence_length: 1024,
            head,
            execution: Glm52NodeExecutionConfig::default(),
        };
        let backend = RocmBackendConfig { devices, ..test_rocm_backend() };
        (model, backend)
    }

    fn test_rocm_backend() -> RocmBackendConfig {
        RocmBackendConfig {
            devices: Vec::new(),
            root: PathBuf::from("/opt/rocm"),
            hiprtc_cache_directory: PathBuf::from("./hiprtc"),
            kernel_sync: false,
            kernel_profile: false,
            decode_graph: false,
            precise_router: true,
            allow_cpu_reference_fallback: false,
            memory_pool: true,
            accelerator_name: None,
            compute_units: None,
            accelerator_memory_bytes: None,
            recommended_working_set_bytes: None,
            mla_decode_split_threshold: 1,
            dsa_hadamard_i8: false,
            dsa_hadamard_shadow_samples: 0,
            dsa_hisa_shadow_samples: 0,
            dsa_cpu_select: false,
            mla_cpu_hot_rows: 0,
            grouped_down_route_buffer: false,
        }
    }

    #[test]
    fn glm52_node_head_stage_end_must_be_in_range() {
        // stage_end == 0 或 >= GLM52_LAYER_COUNT 都要拒。校验是分布式 head NodeEngine 拓扑守门,
        // 否则 rocm_node.rs 会按 head 跑 0..stage_end 触发空层 / 越界 panic。
        for invalid in [0usize, GLM52_LAYER_COUNT, GLM52_LAYER_COUNT + 1] {
            let head = Glm52HeadConfig { stage_end: invalid, layer_ends: vec![invalid.saturating_sub(1).clamp(1, GLM52_LAYER_COUNT - 1)], downstream: StagePeerConfig { ticket: "ticket".into(), iroh: IrohPeerConfig::default() } };
            let (model, backend) = glm52_node_fixture(head, vec![0]);
            let error = validate_glm52(&model, &backend).unwrap_err();
            assert!(error.to_string().contains("stage_end"), "stage_end={invalid} 应拒: {error}");
        }
    }

    #[test]
    fn glm52_node_head_requires_downstream_ticket() {
        let head = Glm52HeadConfig { stage_end: 38, layer_ends: vec![37], downstream: StagePeerConfig { ticket: "   ".into(), iroh: IrohPeerConfig::default() } };
        let (mut model, backend) = glm52_node_fixture(head, vec![0]);
        let error = validate_glm52(&model, &backend).unwrap_err();
        assert!(error.to_string().contains("downstream.ticket"), "空 ticket 应拒: {error}");
        // 改成真 ticket 后必须通过(其它字段也合法)。
        model.head.downstream.ticket = "tail-ticket".into();
        validate_glm52(&model, &backend).expect("合法 head 配置应通过");
    }

    #[test]
    fn glm52_node_head_layer_ends_must_match_devices_and_stage_end() {
        // 8 卡必须 8 项 layer_ends,严格递增,末项 = stage_end-1。
        let head = Glm52HeadConfig { stage_end: 38, layer_ends: vec![7, 15, 23, 31, 37], downstream: StagePeerConfig { ticket: "t".into(), iroh: IrohPeerConfig::default() } };
        let (mut model, backend) = glm52_node_fixture(head, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // 5 项 layer_ends 对 8 卡 → 拒
        let error = validate_glm52(&model, &backend).unwrap_err();
        assert!(error.to_string().contains("layer_ends"), "数量不匹配应拒: {error}");
        // 末项 != stage_end-1(37) → 拒
        model.head.layer_ends = vec![2, 7, 12, 17, 22, 27, 32, 36];
        let error = validate_glm52(&model, &backend).unwrap_err();
        assert!(error.to_string().contains("layer_ends"), "末项错误应拒: {error}");
        // 非严格递增 → 拒
        model.head.layer_ends = vec![2, 7, 12, 17, 22, 27, 37, 32];
        let error = validate_glm52(&model, &backend).unwrap_err();
        assert!(error.to_string().contains("layer_ends"), "非严格递增应拒: {error}");
        // 合法 8 卡拓扑
        model.head.layer_ends = vec![2, 7, 12, 17, 22, 27, 32, 37];
        validate_glm52(&model, &backend).expect("合法 8 卡拓扑应通过");
    }

    #[test]
    fn glm52_node_head_mtp_requires_tail_sampling() {
        // mtp=true 必须 tail_sampling=true,反之不行(已存在的校验)。本测试固化该约束,
        // 避免有人把 GLM-5.2 head 部署成"MTP 但 LM head 留在 head"的非法拓扑。
        let head = Glm52HeadConfig { stage_end: 38, layer_ends: vec![37], downstream: StagePeerConfig { ticket: "t".into(), iroh: IrohPeerConfig::default() } };
        let (mut model, backend) = glm52_node_fixture(head, vec![0]);
        model.execution.mtp = true;
        model.execution.tail_sampling = false;
        let error = validate_glm52(&model, &backend).unwrap_err();
        assert!(error.to_string().contains("tail_sampling"), "MTP 必带 tail_sampling: {error}");
        model.execution.tail_sampling = true;
        validate_glm52(&model, &backend).expect("MTP + tail_sampling 合法");
    }
}
