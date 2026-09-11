//! MiniCPM5 风格 dense LLM GGUF loader。
//!
//! 当前已验证：openbmb/MiniCPM5-1B-Instruct Q4_K_M。
//! 兼容任何 llama.cpp 风格 GGUF dense decoder（hidden_size / num_heads / num_kv_heads /
//! head_dim / vocab_size / rope_theta 任意数值，仅硬约束：head_dim==value_dim、
//! hidden_size % num_heads == 0、num_heads % num_kv_heads == 0、tensor 命名对齐）。
//!
//! 架构（典型 MiniCPM5-1B）：
//! - hidden_size=1536, intermediate_size=4608（gate/up 各自的输出宽度）, num_hidden_layers=24
//! - num_attention_heads=16, num_key_value_heads=2 (GQA 8:1), head_dim=128
//! - query_cols = 16*128 = 2048, kv_cols = 2*128 = 256
//! - vocab_size≈73728, max_position_embeddings=131072, rope_theta=1e6
//! - SwiGLU MLP（Llama 风格 down(silu(gate)*up)，对应 `Activation::Silu`）
//! - RMSNorm eps=1e-5, 无 QK-norm, 无 bias
//! - **1B 标准 tied embedding**：`output.weight` 与 `token_embd.weight` 共享，zllm 自动
//!   检测并切换 `lm_head()` 返回 `token_embd.weight` 矩阵。
//!
//! **GGUF metadata `llama.feed_forward_length` 存的是单个矩阵（gate/up 任一）的输出维度**，
//! 当前 1B 权重为 4608，对应 gate/up shape 均为 `[1536, 4608]`。
//!
//! GGUF tensor 命名（llama.cpp convention，**不是 HF Safetensors 路径**）：
//! - `token_embd.weight`: [hidden, vocab]（GGUF dims 顺序 = [cols, rows]）
//! - `output.weight`: [hidden, vocab]（仅 untied checkpoint 存在；tied 时缺失，lm_head 复用 token_embd）
//! - `output_norm.weight`: [hidden] F32
//! - `blk.{i}.attn_norm.weight`: [hidden] F32（attention 之前的 RMSNorm）
//! - `blk.{i}.ffn_norm.weight`: [hidden] F32（MLP 之前的 RMSNorm；llama.cpp GGUF 用 ffn_norm 而非 post_attention_norm）
//! - `blk.{i}.attn_q.weight`: [hidden, query_cols]
//! - `blk.{i}.attn_k.weight` / `attn_v.weight`: [hidden, kv_cols]
//! - `blk.{i}.attn_output.weight`: [query_cols, hidden]
//! - `blk.{i}.ffn_gate.weight` / `ffn_up.weight`: [hidden, intermediate]
//! - `blk.{i}.ffn_down.weight`: [intermediate, hidden]
//!
//! 每层 9 tensor × N 层 + 2~3 head + 1 embed = **9N + 2~3 tensor**。

use std::path::Path;

use crate::tokenizer::{Detokenizer, Tokenizer};
use crate::weight::container::gguf::{GgufMatrix, GgufReader, GgufValue};

/// MiniCPM5 风格 dense decoder 配置（从 GGUF metadata 自动解析）。
///
/// 当前已验证：MiniCPM5-1B-Instruct Q4_K M。其他 MiniCPM 变体（MiniCPM-V、4B）或
/// 同架构 Llama 风格 decoder 可直接复用本 config 字段，仅数值变化。
#[derive(Debug, Clone, Copy)]
pub struct MiniCpm5Config {
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
    /// `output.weight` 与 `token_embd.weight` 共享同一份量化（llama.cpp tied 模式）。
    /// 1B 模型标准做法；24B 等较大模型通常 untied。
    pub tied_embedding: bool,
}

impl MiniCpm5Config {
    /// GGUF metadata 中 `llama.*` 系列键填充 config；具体值由 tensor shape 二次校验。
    ///
    /// **不**对 metadata 做固定值校验（`expect_metadata_u64(14336)` 等）——数值因
    /// GGUF converter 而异。真正的硬约束是 tensor shape，由 `validate_tensors()`
    /// 在 `open()` 末尾用 `expect_tensor` 执行。
    pub fn from_gguf_metadata(reader: &GgufReader) -> Result<Self, String> {
        reader.expect_metadata_str("general.architecture", "llama")?;
        let layer_count = reader.metadata_u64("llama.block_count")? as usize;
        let hidden_size = reader.metadata_u64("llama.embedding_length")? as usize;
        // llama.cpp `feed_forward_length` = gate/up 任一矩阵的输出宽度。
        let intermediate_size = reader.metadata_u64("llama.feed_forward_length")? as usize;
        let num_heads = reader.metadata_u64("llama.attention.head_count")? as usize;
        let num_kv_heads = reader.metadata_u64("llama.attention.head_count_kv")? as usize;
        let head_dim = reader.metadata_u64("llama.attention.key_length")? as usize;
        let value_dim = reader.metadata_u64("llama.attention.value_length")? as usize;
        if value_dim != head_dim {
            return Err(format!("MiniCPM5 假设 head_dim==value_dim，实际 head_dim={head_dim}, value_dim={value_dim}"));
        }
        let vocab_size = reader.metadata_u64("llama.vocab_size")? as usize;
        let max_position_embeddings = reader.metadata_u64("llama.context_length")? as usize;
        // llama.cpp 把 rope.freq_base 存为 f32（与 u32/u64 同样合法），读时按 f64 拿、归一为 f32。
        let rope_theta = reader.metadata("llama.rope.freq_base").and_then(GgufValue::as_f64).ok_or_else(|| "GGUF metadata llama.rope.freq_base 缺失或类型错误".to_owned())? as f32;
        let rms_eps = reader.metadata("llama.attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(1e-5);
        // tied embedding 自动检测：llama.cpp converter 在 tied 模式下不会写 `output.weight`。
        // 1B 标准 tied，24B 通常 untied；让 GGUF 自身决定最稳。
        let tied_embedding = reader.tensor("output.weight").is_none();
        Ok(Self { layer_count, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, vocab_size, max_position_embeddings, rope_theta, rms_eps, tied_embedding })
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
            return Err("MiniCPM5 配置中 hidden/intermediate/layer_count 必须非零".to_owned());
        }
        if self.num_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err("MiniCPM5 attention 维度必须非零".to_owned());
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) || (self.num_heads * self.head_dim) != self.query_cols() {
            return Err(format!("MiniCPM5 hidden_size={} 不是 num_heads={} 的整数倍", self.hidden_size, self.num_heads));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(format!("MiniCPM5 num_heads={} 不是 num_kv_heads={} 的整数倍", self.num_heads, self.num_kv_heads));
        }
        Ok(())
    }
}

/// MiniCPM5 单层权重（host-side，按层 lazy 加载）。
///
/// 所有矩阵保留为 `GgufMatrix`（持有 file/offset），首次喂给 backend 时通过
/// `LinearWeight::gguf(&matrix)` 走在线 dequant + matmul，不在 host 展开。
pub struct MiniCpm5LayerWeights {
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

/// 整个 MiniCPM5 模型权重封装（GGUF reader + config）。
pub struct MiniCpm5Weights {
    reader: GgufReader,
    config: MiniCpm5Config,
}

impl MiniCpm5Weights {
    pub fn open(path: &Path) -> Result<Self, String> {
        let located = GgufReader::locate(path)?;
        let reader = GgufReader::open(&located)?;
        let config = MiniCpm5Config::from_gguf_metadata(&reader)?;
        config.validate()?;
        let model = Self { reader, config };
        model.validate_tensors()?;
        Ok(model)
    }

    pub fn config(&self) -> &MiniCpm5Config {
        &self.config
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    pub fn tied_embedding(&self) -> bool {
        self.config.tied_embedding
    }

    /// 预取一组 token 的 embedding 行（返回 f32 向量）。
    ///
    /// GGUF `token_embd.weight` 是 Q6_K/Q8_0/Q4_K 量化矩阵；每次只 dequant 需要的行，
    /// 避免一次性展开 vocab×hidden×4 bytes 的 F32 常驻内存。
    pub fn embedding_rows_f32(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.reader.embedding_rows("token_embd.weight", tokens, self.config.hidden_size, self.config.vocab_size)
    }

    /// embedding 矩阵本体(tied 时与 lm_head 同源;untied 时独立),供 backend 常驻上传。
    pub fn embedding_matrix(&self) -> Result<GgufMatrix, String> {
        self.reader.read_matrix("token_embd.weight")
    }

    /// LM head 之前的 RMSNorm（output_norm.weight，F32 存储）。
    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.reader.read_tensor_f32("output_norm.weight")
    }

    /// LM head 矩阵。
    ///
    /// **tied embedding** 模式下返回 `token_embd.weight`（同一份量化复用为 output head）；
    /// untied 模式返回 `output.weight`。runtime 必须接受两种情况。
    pub fn lm_head(&self) -> Result<GgufMatrix, String> {
        if self.config.tied_embedding { self.reader.read_matrix("token_embd.weight") } else { self.reader.read_matrix("output.weight") }
    }

    /// 单层权重（host lazy）。
    pub fn load_layer(&self, layer: usize) -> Result<MiniCpm5LayerWeights, String> {
        if layer >= self.config.layer_count {
            return Err(format!("MiniCPM5 layer {layer} 越界，共 {} 层", self.config.layer_count));
        }
        let prefix = format!("blk.{layer}");
        let attention = format!("{prefix}.attn");
        let mlp = format!("{prefix}.ffn");
        Ok(MiniCpm5LayerWeights {
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
        self.reader.bpe_tokenizer().map_err(|error| format!("构造 MiniCPM5 tokenizer: {error}"))
    }

    pub fn detokenizer(&self) -> Result<Detokenizer, String> {
        self.reader.bpe_detokenizer().map_err(|error| format!("构造 MiniCPM5 detokenizer: {error}"))
    }

    /// 校验 9N + 2~3 个 tensor 全部存在且 shape 与 config 匹配。tied embedding 时跳过 `output.weight`。
    fn validate_tensors(&self) -> Result<(), String> {
        let cfg = &self.config;
        // GGUF dims = [cols, rows]
        self.reader.expect_tensor("token_embd.weight", &[cfg.hidden_size, cfg.vocab_size])?;
        if !cfg.tied_embedding {
            self.reader.expect_tensor("output.weight", &[cfg.hidden_size, cfg.vocab_size])?;
        }
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

#[cfg(test)]
mod tests {
    const WEIGHTS: &str = "/path/to/MiniCPM5-1B-Q4_K_M.gguf";

    /// ChatML prompt 编码与 llama.cpp /tokenize 逐 id 对拍(2026-08-27 重录:思考模板)。
    /// 另外:本 GGUF 的 Q4_K/Q6_K dequant 已与 gguf-py(llama.cpp 移植)逐位核对一致。
    #[test]
    fn minicpm5_tokenize_matches_llama() {
        if !std::path::Path::new(WEIGHTS).exists() {
            eprintln!("skip: {WEIGHTS} 不存在");
            return;
        }
        let weights = super::MiniCpm5Weights::open(std::path::Path::new(WEIGHTS)).unwrap();
        let tokenizer = weights.tokenizer().unwrap();
        let prompt = crate::runtime::minicpm5::minicpm5_instruct_prompt("用一句话解释什么是注意力机制", None, true);
        let ids = tokenizer.tokenize_with_special(prompt.as_bytes(), true);
        assert_eq!(ids, [0, 130072, 8448, 220, 1066, 49667, 12778, 42013, 43211, 15963, 130073, 220, 130072, 130071, 220, 8, 220]);
    }
}
