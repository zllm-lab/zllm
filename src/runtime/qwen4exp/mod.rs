//! Qwen4-Exp(Qwen3.8-Flash-Next)模型算法与执行组合。
//!
//! 结构:GDN / QSA 稀疏注意力 3:1 混合 + 全层 softmax 路由 MoE(512 选 10 +
//! sigmoid 门共享专家)+ Hyper-Connection 四流残差(取代全部层 norm)+
//! 单层 PLE n-gram 哈希嵌入。平台无关规格与 GGUF 装配在本文件;
//! CPU reference 组合见同目录 `cpu.rs`。
//!
//! 语义来源:llama.cpp `src/models/qwen4exp.cpp`(GGUF `qwen4exp` 架构)。

pub mod cpu;
#[cfg(feature = "with-cuda")]
pub mod cuda;
#[cfg(feature = "with-cuda")]
mod cuda_mtp;
#[cfg(feature = "with-cuda")]
pub mod cuda_node;

use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetSpec, GdnOutputGate},
        gqa::{CausalWindow, GqaSpec},
        hybrid::{DeltaNetWeights, FullAttentionWeights},
    },
    backend::{Backend, BackendError},
    moe::{
        Activation,
        topk_moe::{ScoringFunc, TopkMoeSpec},
    },
    weight::container::gguf::{GgufMatrix, GgufReader, GgufValue},
};

pub use crate::model_spec::qwen4exp::Qwen4ExpConfig;

impl Qwen4ExpConfig {
    /// 从 GGUF `qwen4exp.*` 元数据构造配置(UD-Q4_K_XL 等量化通用)。
    pub fn from_gguf(reader: &GgufReader) -> Result<Self, String> {
        let value = |key: &str| reader.metadata_u64(key).map_err(|error| format!("Qwen4-Exp GGUF {error}"));
        let float = |key: &str| match reader.metadata(key) {
            Some(GgufValue::Float(number)) => Ok(*number as f32),
            Some(GgufValue::Unsigned(number)) => Ok(*number as f32),
            _ => Err(format!("Qwen4-Exp GGUF metadata {key} 缺失或不是数值")),
        };
        let array = |key: &str| reader.metadata(key).and_then(GgufValue::as_i64_array).ok_or_else(|| format!("Qwen4-Exp GGUF metadata {key} 缺失或不是数组"));

        reader.expect_metadata_str("general.architecture", "qwen4exp")?;
        let num_layers = value("qwen4exp.block_count")? as usize;
        let full_attention_interval = value("qwen4exp.full_attention_interval")? as usize;

        let head_dim = value("qwen4exp.attention.key_length")? as usize;
        if head_dim != value("qwen4exp.attention.value_length")? as usize {
            return Err("Qwen4-Exp attention.key_length != value_length".to_owned());
        }
        let rope_dim = value("qwen4exp.rope.dimension_count")? as usize;
        let sections = array("qwen4exp.rope.dimension_sections")?;
        if sections.len() < 3 {
            return Err(format!("Qwen4-Exp rope.dimension_sections 数组长度 {} 不足", sections.len()));
        }
        let mrope_section = [sections[0] as usize, sections[1] as usize, sections[2] as usize];

        // 压缩率逐层数组:全注意力层统一非 0,其余层 0
        let ratios = array("qwen4exp.attention.compress_ratios")?;
        if ratios.len() != num_layers {
            return Err(format!("Qwen4-Exp compress_ratios 长度 {} 与层数 {} 不一致", ratios.len(), num_layers));
        }
        let mut compress_ratio = 0;
        for (layer, ratio) in ratios.iter().enumerate() {
            let expect_full = (layer + 1).is_multiple_of(full_attention_interval);
            if expect_full != (*ratio > 0) {
                return Err(format!("Qwen4-Exp compress_ratios[{layer}]={ratio} 与层型不一致"));
            }
            if *ratio > 0 && compress_ratio == 0 {
                compress_ratio = *ratio as usize;
            }
        }

        // PLE 常量(u64 精确,经 i64 数组读取)
        let ple = if array("qwen4exp.ple.layers").is_ok() {
            let layers = array("qwen4exp.ple.layers")?;
            if layers.len() != 1 {
                return Err(format!("Qwen4-Exp 只支持单 PLE 层,GGUF 声明 {} 层", layers.len()));
            }
            let layer_multipliers = array("qwen4exp.ple.layer_multipliers")?.iter().map(|v| *v as u64).collect::<Vec<_>>();
            let head_offsets = array("qwen4exp.ple.head_offsets")?;
            let head_vocab_sizes = array("qwen4exp.ple.head_vocab_sizes")?;
            let narrow = |values: &[i64], name: &str| -> Result<Vec<u32>, String> { values.iter().map(|v| u32::try_from(*v).map_err(|_| format!("Qwen4-Exp {name} 元素 {v} 超出 u32"))).collect() };
            Some(crate::model_spec::qwen4exp::Qwen4ExpPleConfig {
                layer: layers[0] as usize,
                ngram_size: value("qwen4exp.ple.ngram_size")? as usize,
                heads_per_ngram: value("qwen4exp.ple.heads_per_ngram")? as usize,
                head_dim: value("qwen4exp.embedding_length_per_layer_input")? as usize,
                conv_kernel: value("qwen4exp.ple.conv_kernel")? as usize,
                layer_multipliers,
                head_offsets: narrow(&head_offsets, "ple.head_offsets")?,
                head_vocab_sizes: narrow(&head_vocab_sizes, "ple.head_vocab_sizes")?,
                eos_token_id: value("qwen4exp.ple.eos_token_id")? as u32,
            })
        } else {
            None
        };

        let eos_from_gguf = value("tokenizer.ggml.eos_token_id")? as u32;
        let bos = reader.metadata("tokenizer.ggml.bos_token_id").and_then(GgufValue::as_u64).unwrap_or(248_044) as u32;
        let cfg = Self {
            vocab_size: reader.tensor("token_embd.weight").and_then(|tensor| tensor.dims.get(1)).copied().ok_or("Qwen4-Exp token_embd.weight 缺少词表维度")?,
            hidden_size: value("qwen4exp.embedding_length")? as usize,
            num_layers,
            full_attention_interval,
            num_attention_heads: value("qwen4exp.attention.head_count")? as usize,
            num_kv_heads: value("qwen4exp.attention.head_count_kv")? as usize,
            head_dim,
            rope_dim,
            rope_theta: float("qwen4exp.rope.freq_base")?,
            mrope_section,
            indexer: crate::model_spec::qwen4exp::Qwen4ExpIndexerConfig {
                head_count: value("qwen4exp.attention.indexer.head_count")? as usize,
                head_dim: value("qwen4exp.attention.indexer.key_length")? as usize,
                top_k: value("qwen4exp.attention.indexer.top_k")? as usize,
                compress_ratio,
            },
            linear_key_heads: value("qwen4exp.ssm.group_count")? as usize,
            linear_value_heads: value("qwen4exp.ssm.time_step_rank")? as usize,
            linear_head_dim: value("qwen4exp.ssm.state_size")? as usize,
            linear_conv_kernel_size: value("qwen4exp.ssm.conv_kernel")? as usize,
            num_experts: value("qwen4exp.expert_count")? as usize,
            num_experts_per_tok: value("qwen4exp.expert_used_count")? as usize,
            expert_intermediate_size: value("qwen4exp.expert_feed_forward_length")? as usize,
            shared_expert_intermediate_size: value("qwen4exp.expert_shared_feed_forward_length")? as usize,
            hyper_connection: crate::model_spec::qwen4exp::Qwen4ExpHyperConnectionConfig { streams: value("qwen4exp.hyper_connection.count")? as usize, low_rank: value("qwen4exp.hyper_connection.low_rank")? as usize },
            ple,
            rms_norm_eps: float("qwen4exp.attention.layer_norm_rms_epsilon")?,
            max_position_embeddings: value("qwen4exp.context_length")? as usize,
            bos_token_id: bos,
            eos_token_ids: vec![eos_from_gguf],
        };
        // ssm.inner_size = value 头 × state;校验 key/value 头维一致。
        let inner = value("qwen4exp.ssm.inner_size")? as usize;
        if cfg.linear_value_heads * cfg.linear_head_dim != inner || value("qwen4exp.ssm.state_size")? != cfg.linear_head_dim as u64 {
            return Err("Qwen4-Exp ssm 元数据与头维展开不一致".to_owned());
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// QSA(全注意力层)GQA 规格:输出门 = sigmoid,与 hybrid 底座一致。
    pub fn qsa_spec(&self) -> GqaSpec {
        GqaSpec {
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            rope_theta: self.rope_theta,
            use_qk_norm: true,
            window: CausalWindow::Full,
            score_scale: 1.0 / (self.head_dim as f32).sqrt(),
            output_gate: true,
        }
    }

    /// GDN(线性注意力层)规格:Qwen4-Exp 输出门是 sigmoid(llama.cpp 唯一差异点)。
    pub fn gated_delta_net_spec(&self) -> GatedDeltaNetSpec {
        GatedDeltaNetSpec {
            key_heads: self.linear_key_heads,
            value_heads: self.linear_value_heads,
            key_head_dim: self.linear_head_dim,
            value_head_dim: self.linear_head_dim,
            conv_kernel: self.linear_conv_kernel_size,
            rms_eps: self.rms_norm_eps,
            output_gate: GdnOutputGate::Sigmoid,
        }
    }

    /// softmax 路由 512 选 10 + 1 sigmoid 门共享专家(llama.cpp norm=true,无缩放)。
    pub fn moe_spec(&self) -> TopkMoeSpec {
        TopkMoeSpec {
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            num_shared_experts: 1,
            scoring_func: ScoringFunc::Softmax,
            normalize_selected: true,
            routed_scaling_factor: 1.0,
            intermediate_size: self.expert_intermediate_size,
            shared_intermediate_size: self.shared_expert_intermediate_size,
            activation: Activation::Silu,
        }
    }
}

// ============================================================================
// GGUF 张量装配。HC/PLE/indexer 的 norm 权重转换器已做 +1 折叠,
// 一律按普通 RMSNorm 语义直读,不做 Gemma 偏移。
// ============================================================================

/// 一个 Hyper-Connection 模块(attn 前 / FFN 前 / 输出头)。
#[derive(Debug)]
pub struct Qwen4ExpHyperConnection<W> {
    pub norm: W,
    pub down: W,
    pub up: W,
    /// `[hc_dim, streams]`;输出头模块无 inject。
    pub inject: Option<W>,
}

/// QSA 索引器投影(MQA:4 查询头 + 1 共享键头,头维 128)。
#[derive(Debug)]
pub struct Qwen4ExpIndexer<W> {
    pub q_proj: W,
    pub k_proj: W,
    pub q_norm: W,
    pub k_norm: W,
}

#[derive(Debug)]
pub enum Qwen4ExpMixer<W> {
    Delta(DeltaNetWeights<W>),
    SparseAttention { attention: FullAttentionWeights<W>, indexer: Qwen4ExpIndexer<W> },
}

#[derive(Debug)]
pub struct Qwen4ExpPle<W> {
    pub key: W,
    pub value: W,
    pub norm_key: W,
    pub norm_query: W,
    pub norm_conv: W,
    pub conv1d: W,
}

#[derive(Debug)]
pub struct Qwen4ExpMoe<W> {
    pub router: W,
    /// softmax 路由无 bias;MoeFfnRef 需要占位零向量。
    pub router_bias: W,
    pub shared_gate: W,
    pub shared_up: W,
    pub shared_down: W,
    /// 共享专家 sigmoid 输出门(`ffn_gate_inp_shexp`,逐 token 标量)。
    pub shared_output_gate: W,
}

#[derive(Debug)]
pub struct Qwen4ExpLayer<W> {
    pub hc_attn: Qwen4ExpHyperConnection<W>,
    pub mixer: Qwen4ExpMixer<W>,
    pub ple: Option<Qwen4ExpPle<W>>,
    pub hc_ffn: Qwen4ExpHyperConnection<W>,
    pub moe: Qwen4ExpMoe<W>,
}

fn prepare_hc<B: Backend>(backend: &B, reader: &GgufReader, prefix: &str, with_inject: bool) -> Result<Qwen4ExpHyperConnection<B::Weight>, BackendError> {
    let inject = if with_inject { Some(super::prepare_gguf_matrix(backend, reader, &format!("{prefix}_inject.weight"))?) } else { None };
    Ok(Qwen4ExpHyperConnection {
        norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}_norm.weight"))?,
        down: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}_down.weight"))?,
        up: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}_up.weight"))?,
        inject,
    })
}

pub fn prepare_qwen4exp_layers<B: Backend>(backend: &B, source: &Qwen4ExpGguf) -> Result<Vec<Qwen4ExpLayer<B::Weight>>, BackendError> {
    let cfg = source.config();
    (0..cfg.num_layers).map(|layer| prepare_qwen4exp_layer(backend, source, layer)).collect()
}

pub fn prepare_qwen4exp_layer<B: Backend>(backend: &B, source: &Qwen4ExpGguf, layer: usize) -> Result<Qwen4ExpLayer<B::Weight>, BackendError> {
    let cfg = source.config();
    prepare_layer(backend, source.reader(), cfg, layer, cfg.is_full_attention(layer))
}

fn prepare_layer<B: Backend>(backend: &B, reader: &GgufReader, cfg: &Qwen4ExpConfig, layer: usize, full_attention: bool) -> Result<Qwen4ExpLayer<B::Weight>, BackendError> {
    let prefix = format!("blk.{layer}");
    let mixer = if full_attention {
        let indexer = Qwen4ExpIndexer {
            // indexer 投影是 BF16 GGUF 类型,reference 直读 F32。
            q_proj: super::prepare_gguf_f32_matrix(backend, reader, &format!("{prefix}.indexer.q_proj.weight"))?,
            k_proj: super::prepare_gguf_f32_matrix(backend, reader, &format!("{prefix}.indexer.k_proj.weight"))?,
            q_norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.indexer.q_norm.weight"))?,
            k_norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.indexer.k_norm.weight"))?,
        };
        Qwen4ExpMixer::SparseAttention {
            attention: FullAttentionWeights {
                query_gate: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_q.weight"))?,
                query_norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.attn_q_norm.weight"))?,
                key: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_k.weight"))?,
                key_norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.attn_k_norm.weight"))?,
                value: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_v.weight"))?,
                output: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_output.weight"))?,
            },
            indexer,
        }
    } else {
        Qwen4ExpMixer::Delta(DeltaNetWeights {
            qkv: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_qkv.weight"))?,
            z: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_gate.weight"))?,
            alpha: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ssm_alpha.weight"))?,
            beta: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ssm_beta.weight"))?,
            conv: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ssm_conv1d.weight"))?,
            a_log: super::prepare_gguf_a_log_vector(backend, reader, &format!("{prefix}.ssm_a"))?,
            dt_bias: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.ssm_dt.bias"))?,
            norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.ssm_norm.weight"))?,
            output: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ssm_out.weight"))?,
        })
    };
    let ple = if cfg.ple.as_ref().is_some_and(|ple| ple.layer == layer) {
        Some(Qwen4ExpPle {
            key: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ple_key.weight"))?,
            value: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ple_value.weight"))?,
            norm_key: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.ple_norm_key.weight"))?,
            norm_query: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.ple_norm_query.weight"))?,
            norm_conv: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.ple_norm_conv.weight"))?,
            conv1d: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ple_conv1d.weight"))?,
        })
    } else {
        None
    };
    let moe = Qwen4ExpMoe {
        router: super::prepare_gguf_f32_matrix(backend, reader, &format!("{prefix}.ffn_gate_inp.weight"))?,
        router_bias: backend.prepare_f32(&vec![0.0; cfg.num_experts], 1, cfg.num_experts)?,
        shared_gate: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_gate_shexp.weight"))?,
        shared_up: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_up_shexp.weight"))?,
        shared_down: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_down_shexp.weight"))?,
        shared_output_gate: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.ffn_gate_inp_shexp.weight"))?,
    };
    Ok(Qwen4ExpLayer { hc_attn: prepare_hc(backend, reader, &format!("{prefix}.hc_attn"), true)?, mixer, ple, hc_ffn: prepare_hc(backend, reader, &format!("{prefix}.hc_ffn"), true)?, moe })
}

// ============================================================================
// Qwen4ExpGguf —— GGUF 权重 wrapper + MoE expert 数据源。
// ============================================================================

use crate::tokenizer::{Detokenizer, Tokenizer};
use crate::weight::expert_source::{ExpertSource, ExpertSourceProvider, GgufExpertSource, GgufExpertWeights};
use std::path::Path;

pub struct Qwen4ExpGguf {
    reader: GgufReader,
    cfg: Qwen4ExpConfig,
    resident_experts: Option<Vec<GgufExpertWeights>>,
}

impl GgufExpertSource for Qwen4ExpGguf {
    fn intermediate(&self) -> usize {
        self.cfg.expert_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }

    fn load_expert_gguf(&self, layer: usize, expert: usize) -> Result<GgufExpertWeights, String> {
        if layer >= self.cfg.num_layers || expert >= self.cfg.num_experts {
            return Err(format!("Qwen4-Exp expert 索引越界: layer={layer}/{}, expert={expert}/{}", self.cfg.num_layers, self.cfg.num_experts));
        }
        if let Some(resident) = &self.resident_experts {
            return Ok(resident[layer * self.cfg.num_experts + expert].clone());
        }
        Ok(GgufExpertWeights {
            gate: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_gate_exps.weight"), expert)?,
            up: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_up_exps.weight"), expert)?,
            down: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_down_exps.weight"), expert)?,
        })
    }
}

impl ExpertSourceProvider for Qwen4ExpGguf {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String> {
        if layer >= self.cfg.num_layers {
            return Err(format!("Qwen4-Exp expert source layer 越界: {layer} >= {}", self.cfg.num_layers));
        }
        Ok(ExpertSource::Gguf(self))
    }
}

impl Qwen4ExpGguf {
    pub fn open(path: &Path) -> Result<Self, String> {
        let reader = GgufReader::open(&GgufReader::locate(path)?)?;
        let cfg = Qwen4ExpConfig::from_gguf(&reader)?;
        Ok(Self { reader, cfg, resident_experts: None })
    }

    /// 由执行组合确认主存预算后调用。clone 共享 packed 字节,请求阶段不再读文件。
    pub fn make_experts_resident(&mut self) -> Result<usize, String> {
        let mut experts = Vec::with_capacity(self.cfg.num_layers * self.cfg.num_experts);
        let mut bytes = 0;
        for layer in 0..self.cfg.num_layers {
            for expert in 0..self.cfg.num_experts {
                let weights = self.load_expert_gguf(layer, expert)?;
                bytes += weights.gate.bytes()?.len() + weights.up.bytes()?.len() + weights.down.bytes()?.len();
                experts.push(weights);
            }
            eprintln!("[qwen4exp-host-resident] layer={layer} bytes={bytes}");
        }
        self.resident_experts = Some(experts);
        Ok(bytes)
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    pub fn embedding_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        self.reader.embedding_rows("token_embd.weight", token_ids, self.cfg.hidden_size, self.cfg.vocab_size)
    }

    pub fn output_head(&self) -> Result<GgufMatrix, String> {
        self.reader.read_matrix("output.weight")
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        self.reader.bpe_tokenizer().map_err(|error| format!("Qwen4-Exp tokenizer: {error}"))
    }

    pub fn detokenizer(&self) -> Result<Detokenizer, String> {
        self.reader.bpe_detokenizer().map_err(|error| format!("Qwen4-Exp detokenizer: {error}"))
    }

    /// PLE n-gram 行 gather:每 token `head_count` 行 × head_dim。
    /// 51B 参数表永远不全量驻留,只解码被哈希命中的行。
    pub fn ple_rows_f32(&self, ple: &crate::model_spec::qwen4exp::Qwen4ExpPleConfig, rows: &[u32]) -> Result<Vec<f32>, String> {
        let mut output = Vec::with_capacity(rows.len() * ple.head_dim);
        for &row in rows {
            output.extend(self.reader.read_matrix_row_f32("per_layer_token_embd.weight", row as usize)?);
        }
        Ok(output)
    }
}

// ============================================================================
// PLE n-gram 哈希(llama.cpp host-side 语义):
//   mixed = t[p]*m[0] ^ t[p-1]*m[1] (^ t[p-2]*m[2]);
//   row = mixed % vocab[h] + offset[h]
// EOS(或缺前驱)重置窗口;token 自身的 EOS 不截断自身上下文。
// ============================================================================

/// `tokens` 为当前批,`prev` 为其前的 ngram_size-1 个前驱(最旧在前)。
/// 前驱窗口 = prev ++ 当前批前缀(llama.cpp 的 prev tokens 含本批)。
pub fn ple_row_indices(ple: &crate::model_spec::qwen4exp::Qwen4ExpPleConfig, tokens: &[u32], prev: &[u32]) -> Vec<u32> {
    let ngram = ple.ngram_size;
    let heads = ple.head_count();
    let window: Vec<u32> = prev.iter().chain(tokens).copied().collect();
    let mut indices = Vec::with_capacity(tokens.len() * heads);
    for (index, &token) in tokens.iter().enumerate() {
        let mut ctx = vec![ple.eos_token_id; ngram];
        ctx[0] = token;
        // EOS 只重置更老的前驱(读序从最新到最旧,cut 向老传播),与 llama.cpp 一致。
        let mut cut = false;
        for step in 1..ngram {
            let predecessor = (prev.len() + index).checked_sub(step).and_then(|position| window.get(position)).copied().unwrap_or(ple.eos_token_id);
            let value = if cut { ple.eos_token_id } else { predecessor };
            cut = cut || value == ple.eos_token_id;
            ctx[step] = value;
        }
        for size in 2..=ngram {
            let mut mixed = (ctx[0] as u64).wrapping_mul(ple.layer_multipliers[0]);
            for position in 1..size {
                mixed ^= (ctx[position] as u64).wrapping_mul(ple.layer_multipliers[position]);
            }
            for head in (size - 2) * ple.heads_per_ngram..(size - 1) * ple.heads_per_ngram {
                indices.push((mixed % ple.head_vocab_sizes[head] as u64 + ple.head_offsets[head] as u64) as u32);
            }
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ple_config() -> crate::model_spec::qwen4exp::Qwen4ExpPleConfig {
        crate::model_spec::qwen4exp::Qwen4ExpPleConfig {
            layer: 1,
            ngram_size: 3,
            heads_per_ngram: 2,
            head_dim: 4,
            conv_kernel: 4,
            layer_multipliers: vec![1_000_003, 1_000_033, 1_000_037],
            head_offsets: vec![0, 0, 100, 100],
            head_vocab_sizes: vec![100, 101, 102, 103],
            eos_token_id: 7,
        }
    }

    /// 手算对照:bigram 头哈希与 trigram 头哈希各自取模落位。
    #[test]
    fn ple_hash_matches_manual_computation() {
        let ple = ple_config();
        // 前驱 [5, 6],当前批 [8, 9]:token 8 看 (8,6),token 9 看 (9,8)
        let rows = ple_row_indices(&ple, &[8, 9], &[5, 6]);
        assert_eq!(rows.len(), 2 * 4);
        let mixed8_bi = 8u64 * 1_000_003 ^ 6u64 * 1_000_033;
        let mixed8_tri = 8u64 * 1_000_003 ^ 6u64 * 1_000_033 ^ 5u64 * 1_000_037;
        assert_eq!(rows[0], (mixed8_bi % 100) as u32);
        assert_eq!(rows[1], (mixed8_bi % 101) as u32);
        assert_eq!(rows[2], (mixed8_tri % 102 + 100) as u32);
        assert_eq!(rows[3], (mixed8_tri % 103 + 100) as u32);
        let mixed9_tri = 9u64 * 1_000_003 ^ 8u64 * 1_000_033 ^ 6u64 * 1_000_037;
        // token 9 的头组从 rows[4] 开始(bi0, bi1, tri0, tri1)
        assert_eq!(rows[4 + 0], ((9u64 * 1_000_003 ^ 8u64 * 1_000_033) % 100) as u32);
        assert_eq!(rows[4 + 1], ((9u64 * 1_000_003 ^ 8u64 * 1_000_033) % 101) as u32);
        assert_eq!(rows[4 + 3], (mixed9_tri % 103 + 100) as u32);
    }

    /// EOS 只重置比它更老的前驱;最新前驱非 EOS 则保留。序列起点缺前驱读 EOS。
    #[test]
    fn ple_hash_eos_resets_only_older_context() {
        let ple = ple_config();
        let eos = 7u64;
        // prev=[eos, 6]:EOS 在最老位,trigram 的第三位归 EOS,bigram 仍是 (8,6)
        let rows = ple_row_indices(&ple, &[8], &[ple.eos_token_id, 6]);
        assert_eq!(rows[0], ((8u64 * 1_000_003 ^ 6u64 * 1_000_033) % 100) as u32);
        assert_eq!(rows[2], ((8u64 * 1_000_003 ^ 6u64 * 1_000_033 ^ eos * 1_000_037) % 102 + 100) as u32);
        // prev=[5, eos]:EOS 紧邻,更老位置一并重置;token 自身仍是 8
        let rows = ple_row_indices(&ple, &[8], &[5, ple.eos_token_id]);
        assert_eq!(rows[0], ((8u64 * 1_000_003 ^ eos * 1_000_033) % 100) as u32);
        // 序列起点:无前驱,全部读 EOS
        let rows = ple_row_indices(&ple, &[8], &[]);
        assert_eq!(rows[0], ((8u64 * 1_000_003 ^ eos * 1_000_033) % 100) as u32);
    }

    #[test]
    fn ple_hash_is_independent_of_chunk_boundaries() {
        let ple = ple_config();
        let tokens = [8, 9, 10, 11, 12, 7, 13, 14];
        let expected = ple_row_indices(&ple, &tokens, &[]);
        for split in 1..tokens.len() {
            let mut actual = ple_row_indices(&ple, &tokens[..split], &[]);
            actual.extend(ple_row_indices(&ple, &tokens[split..], &tokens[..split]));
            assert_eq!(actual, expected, "split={split}");
        }
    }

    /// 标准配置校验:Flash-Next 常量必须自洽。
    #[test]
    fn flash_next_config_is_valid() {
        let cfg = Qwen4ExpConfig::standard_flash_next();
        cfg.validate().expect("Flash-Next 常量必须有效");
        assert_eq!(cfg.num_layers, 48);
        assert!(cfg.is_full_attention(3) && cfg.is_full_attention(47) && !cfg.is_full_attention(0));
        assert_eq!(cfg.hc_dim(), 10_240);
        assert_eq!(cfg.gated_delta_net_spec().conv_dim(), 2_048 * 2 + 6_144);
        assert!(matches!(cfg.gated_delta_net_spec().output_gate, GdnOutputGate::Sigmoid));
        assert_eq!(cfg.ple.as_ref().expect("Flash-Next 有 PLE").head_count(), 16);
        assert_eq!(cfg.ple.as_ref().unwrap().conv_history(), 9);
    }

    /// MoE 规格:softmax、选中归一化、无缩放。
    #[test]
    fn moe_spec_matches_llama_cpp_routing() {
        let spec = Qwen4ExpConfig::standard_flash_next().moe_spec();
        assert!(matches!(spec.scoring_func, ScoringFunc::Softmax));
        assert_eq!(spec.num_experts, 512);
        assert_eq!(spec.top_k, 10);
        assert_eq!(spec.num_shared_experts, 1);
        assert!(spec.normalize_selected);
        assert_eq!(spec.routed_scaling_factor, 1.0);
    }
}
