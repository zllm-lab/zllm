//! Mistral 风格 dense LLM GGUF loader。
//!
//! 当前已验证：Mistral-Small-3.2-24B-Instruct-2506。
//! 兼容任何 llama.cpp 风格 GGUF dense decoder（hidden_size / num_heads / num_kv_heads /
//! head_dim / vocab_size / rope_theta 任意数值，仅硬约束：head_dim==value_dim、
//! hidden_size % num_heads == 0、num_heads % num_kv_heads == 0、tensor 命名对齐）。
//!
//! 架构（典型 Mistral-Small-3.2-24B）：
//! - hidden_size=5120, intermediate_size=32768 (SwiGLU 合并宽度), num_hidden_layers=40
//! - num_attention_heads=32, num_key_value_heads=8 (GQA 4:1), head_dim=128
//! - query_cols = num_heads * head_dim = 4096, kv_cols = num_kv_heads * head_dim = 1024
//! - vocab_size=131072, max_position_embeddings=131072, rope_theta=1e9
//! - SwiGLU MLP（Llama 风格 down(silu(gate)*up)，对应 `Activation::Silu`）
//! - RMSNorm eps=1e-5, 无 QK-norm, 无 bias
//!
//! **GGUF metadata `llama.feed_forward_length` 存的是单个矩阵（gate/up 任一）的输出维度**，
//! 即 `intermediate_size / 2 = 16384`（HF transformers 里的 `intermediate_size=32768` 是
//! SwiGLU 合并宽度，本字段取 half）。
//!
//! GGUF tensor 命名（llama.cpp convention，**不是 HF Safetensors 路径**）：
//! - `token_embd.weight`: [hidden, vocab]（GGUF dims 顺序 = [cols, rows]）
//! - `output.weight`: [hidden, vocab]（与 embed_tokens 不 tie，Mistral 标准）
//! - `output_norm.weight`: [hidden] F32
//! - `blk.{i}.attn_norm.weight`: [hidden] F32（attention 之前的 RMSNorm）
//! - `blk.{i}.ffn_norm.weight`: [hidden] F32（MLP 之前的 RMSNorm；llama.cpp GGUF 用 ffn_norm 而非 post_attention_norm）
//! - `blk.{i}.attn_q.weight`: [hidden, query_cols]
//! - `blk.{i}.attn_k.weight` / `attn_v.weight`: [hidden, kv_cols]
//! - `blk.{i}.attn_output.weight`: [query_cols, hidden]
//! - `blk.{i}.ffn_gate.weight` / `ffn_up.weight`: [hidden, intermediate]
//! - `blk.{i}.ffn_down.weight`: [intermediate, hidden]
//!
//! 每层 9 tensor × N 层 + 3 head + 1 embed = **9N + 3 tensor**。

use std::path::Path;

use crate::tokenizer::{Detokenizer, Tokenizer};
use crate::weight::container::gguf::{GgufMatrix, GgufReader, GgufValue};

/// Mistral 风格 dense decoder 配置（从 GGUF metadata 自动解析）。
///
/// 当前已验证：Mistral-Small-3.2-24B-Instruct-2506。其他 Mistral 变体（3.1、Next）或
/// 同架构 Llama 风格 decoder 可直接复用本 config 字段，仅数值变化。
#[derive(Debug, Clone, Copy)]
pub struct MistralConfig {
    pub layer_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl MistralConfig {
    /// GGUF metadata 中 `llama.*` 系列键填充 config；具体值由 tensor shape 二次校验。
    ///
    /// **不**对 metadata 做固定值校验（`expect_metadata_u64(14336)` 等）——数值因
    /// GGUF converter 而异。真正的硬约束是 tensor shape，由 `validate_tensors()`
    /// 在 `open()` 末尾用 `expect_tensor` 执行。
    pub fn from_gguf_metadata(reader: &GgufReader) -> Result<Self, String> {
        reader.expect_metadata_str("general.architecture", "llama")?;
        let layer_count = reader.metadata_u64("llama.block_count")? as usize;
        let hidden_size = reader.metadata_u64("llama.embedding_length")? as usize;
        // llama.cpp `feed_forward_length` = 单矩阵宽度 = HF intermediate_size / 2 (SwiGLU)
        let intermediate_size = reader.metadata_u64("llama.feed_forward_length")? as usize;
        let num_heads = reader.metadata_u64("llama.attention.head_count")? as usize;
        let num_kv_heads = reader.metadata_u64("llama.attention.head_count_kv")? as usize;
        let head_dim = reader.metadata_u64("llama.attention.key_length")? as usize;
        let value_dim = reader.metadata_u64("llama.attention.value_length")? as usize;
        if value_dim != head_dim {
            return Err(format!("Mistral 假设 head_dim==value_dim，实际 head_dim={head_dim}, value_dim={value_dim}"));
        }
        let vocab_size = reader.metadata_u64("llama.vocab_size")? as usize;
        let max_position_embeddings = reader.metadata_u64("llama.context_length")? as usize;
        // llama.cpp 把 rope.freq_base 存为 f32（与 u32/u64 同样合法），读时按 f64 拿、归一为 f32。
        let rope_theta = reader.metadata("llama.rope.freq_base").and_then(GgufValue::as_f64).ok_or_else(|| "GGUF metadata llama.rope.freq_base 缺失或类型错误".to_owned())? as f32;
        let rms_eps = reader.metadata("llama.attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(1e-5);
        Ok(Self { layer_count, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, vocab_size, max_position_embeddings, rope_theta, rms_eps })
    }

    /// query 投影列数（attention output dim）。
    pub fn query_cols(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// k/v 投影列数。
    pub fn kv_cols(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.layer_count == 0 || self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err("Mistral 配置中 hidden/intermediate/layer_count 必须非零".to_owned());
        }
        if self.num_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err("Mistral attention 维度必须非零".to_owned());
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(format!("Mistral hidden_size={} 不是 num_heads={} 的整数倍", self.hidden_size, self.num_heads));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(format!("Mistral num_heads={} 不是 num_kv_heads={} 的整数倍", self.num_heads, self.num_kv_heads));
        }
        Ok(())
    }
}

/// Mistral 单层权重（host-side，按层 lazy 加载）。
///
/// 所有矩阵保留为 `GgufMatrix`（持有 file/offset），首次喂给 backend 时通过
/// `LinearWeight::gguf(&matrix)` 走在线 dequant + matmul，不在 host 展开。
pub struct MistralLayerWeights {
    pub input_norm: Vec<f32>,
    pub query: GgufMatrix,
    pub key: GgufMatrix,
    pub value: GgufMatrix,
    pub output: GgufMatrix,
    pub post_attention_norm: Vec<f32>,
    pub gate: GgufMatrix,
    pub up: GgufMatrix,
    pub down: GgufMatrix,
}

/// 整个 Mistral 模型权重封装（GGUF reader + config）。
pub struct MistralWeights {
    reader: GgufReader,
    config: MistralConfig,
}

impl MistralWeights {
    pub fn open(path: &Path) -> Result<Self, String> {
        let located = GgufReader::locate(path)?;
        let reader = GgufReader::open(&located)?;
        let config = MistralConfig::from_gguf_metadata(&reader)?;
        config.validate()?;
        let model = Self { reader, config };
        model.validate_tensors()?;
        Ok(model)
    }

    pub fn config(&self) -> &MistralConfig {
        &self.config
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    /// 预取一组 token 的 embedding 行（返回 f32 向量）。
    ///
    /// GGUF `token_embd.weight` 是 Q6_K/Q8_0 量化矩阵；每次只 dequant 需要的行，
    /// 避免一次性展开 vocab×hidden×4 bytes 的 F32 常驻内存（24B 模型约 256 MB）。
    pub fn embedding_rows_f32(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.reader.embedding_rows("token_embd.weight", tokens, self.config.hidden_size, self.config.vocab_size)
    }

    /// LM head 之前的 RMSNorm（output_norm.weight，F32 存储）。
    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.reader.read_tensor_f32("output_norm.weight")
    }

    /// LM head 矩阵（`output.weight`，GGUF 量化）。
    pub fn lm_head(&self) -> Result<GgufMatrix, String> {
        self.reader.read_matrix("output.weight")
    }

    /// 单层权重（host lazy）。
    pub fn load_layer(&self, layer: usize) -> Result<MistralLayerWeights, String> {
        if layer >= self.config.layer_count {
            return Err(format!("Mistral layer {layer} 越界，共 {} 层", self.config.layer_count));
        }
        let prefix = format!("blk.{layer}");
        let attention = format!("{prefix}.attn");
        let mlp = format!("{prefix}.ffn");
        Ok(MistralLayerWeights {
            input_norm: self.reader.read_tensor_f32(&format!("{prefix}.attn_norm.weight"))?,
            query: self.reader.read_matrix(&format!("{attention}_q.weight"))?,
            key: self.reader.read_matrix(&format!("{attention}_k.weight"))?,
            value: self.reader.read_matrix(&format!("{attention}_v.weight"))?,
            output: self.reader.read_matrix(&format!("{attention}_output.weight"))?,
            post_attention_norm: self.reader.read_tensor_f32(&format!("{prefix}.ffn_norm.weight"))?,
            gate: self.reader.read_matrix(&format!("{mlp}_gate.weight"))?,
            up: self.reader.read_matrix(&format!("{mlp}_up.weight"))?,
            down: self.reader.read_matrix(&format!("{mlp}_down.weight"))?,
        })
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        self.reader.bpe_tokenizer().map_err(|error| format!("构造 Mistral tokenizer: {error}"))
    }

    pub fn detokenizer(&self) -> Result<Detokenizer, String> {
        self.reader.bpe_detokenizer().map_err(|error| format!("构造 Mistral detokenizer: {error}"))
    }

    /// 校验 9N + 3 个 tensor 全部存在且 shape 与 config 匹配。
    fn validate_tensors(&self) -> Result<(), String> {
        let cfg = &self.config;
        // GGUF dims = [cols, rows]
        self.reader.expect_tensor("token_embd.weight", &[cfg.hidden_size, cfg.vocab_size])?;
        self.reader.expect_tensor("output.weight", &[cfg.hidden_size, cfg.vocab_size])?;
        self.reader.expect_tensor("output_norm.weight", &[cfg.hidden_size])?;
        let query_cols = cfg.query_cols();
        let kv_cols = cfg.kv_cols();
        for layer in 0..cfg.layer_count {
            let prefix = format!("blk.{layer}");
            let attention = format!("{prefix}.attn");
            let mlp = format!("{prefix}.ffn");
            self.reader.expect_tensor(&format!("{prefix}.attn_norm.weight"), &[cfg.hidden_size])?;
            self.reader.expect_tensor(&format!("{prefix}.ffn_norm.weight"), &[cfg.hidden_size])?;
            self.reader.expect_tensor(&format!("{attention}_q.weight"), &[cfg.hidden_size, query_cols])?;
            self.reader.expect_tensor(&format!("{attention}_k.weight"), &[cfg.hidden_size, kv_cols])?;
            self.reader.expect_tensor(&format!("{attention}_v.weight"), &[cfg.hidden_size, kv_cols])?;
            self.reader.expect_tensor(&format!("{attention}_output.weight"), &[query_cols, cfg.hidden_size])?;
            self.reader.expect_tensor(&format!("{mlp}_gate.weight"), &[cfg.hidden_size, cfg.intermediate_size])?;
            self.reader.expect_tensor(&format!("{mlp}_up.weight"), &[cfg.hidden_size, cfg.intermediate_size])?;
            self.reader.expect_tensor(&format!("{mlp}_down.weight"), &[cfg.intermediate_size, cfg.hidden_size])?;
        }
        Ok(())
    }
}

// ── 向后兼容 alias ───────────────────────────────────────────────────────
//
// 这些 alias 保留给已存在的调用方（tests/mistral_small32_*.rs + bin/tools/mistral.rs +
// 任何使用旧符号的文档）。新代码请直接使用 `MistralConfig` / `MistralWeights` /
// `MistralLayerWeights`。
pub type MistralSmall32Config = MistralConfig;
pub type MistralSmall32Weights = MistralWeights;
pub type MistralSmall32LayerWeights = MistralLayerWeights;
