//! Qwen4-Exp(Qwen3.8-Flash-Next)文本架构配置。
//!
//! 混合 GDN / QSA 稀疏注意力 + 全层 MoE + Hyper-Connection 四流残差 +
//! 单层 PLE n-gram 哈希嵌入。视觉塔复用 Qwen3-VL ViT(qwen36 已覆盖)。

/// QSA 索引器规格:MQA 轻量打分头,按 micro-block 选择可见 cell。
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpIndexerConfig {
    pub head_count: usize,
    pub head_dim: usize,
    /// 每 query 可见的 cell 预算(llama.cpp 语义:top_k + compress_ratio - 1)。
    pub top_k: usize,
    /// micro-block 大小(压缩率),仅全注意力层非 0。
    pub compress_ratio: usize,
}

/// PLE n-gram 哈希嵌入规格。
#[derive(Clone, Debug)]
pub struct Qwen4ExpPleConfig {
    /// 唯一的 PLE 层(0-indexed)。
    pub layer: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    /// 每头嵌入维度(总 = 头数 × 该值 = hidden_size)。
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// 哈希乘数(每 ngram 位置一个,u64 精确值)。
    pub layer_multipliers: Vec<u64>,
    pub head_offsets: Vec<u32>,
    pub head_vocab_sizes: Vec<u32>,
    pub eos_token_id: u32,
}

impl Qwen4ExpPleConfig {
    pub fn head_count(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// PLE 卷积的因果历史长度:(kernel-1) × dilation,ngram_size 即膨胀率。
    pub fn conv_history(&self) -> usize {
        (self.conv_kernel - 1) * self.ngram_size
    }
}

/// Hyper-Connection 四流残差规格。
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpHyperConnectionConfig {
    pub streams: usize,
    pub low_rank: usize,
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub full_attention_interval: usize,
    // QSA(全注意力层)
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub indexer: Qwen4ExpIndexerConfig,
    // GDN(线性注意力层)
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_head_dim: usize,
    pub linear_conv_kernel_size: usize,
    // MoE
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub expert_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    // Hyper-Connection 残差
    pub hyper_connection: Qwen4ExpHyperConnectionConfig,
    // PLE
    pub ple: Option<Qwen4ExpPleConfig>,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
}

impl Qwen4ExpConfig {
    /// Qwen3.8-Flash-Next(125B-A6B + 51B n-gram + 4B MTP)。
    pub fn standard_flash_next() -> Self {
        Self {
            vocab_size: 248_320,
            hidden_size: 2_560,
            num_layers: 48,
            full_attention_interval: 4,
            num_attention_heads: 24,
            num_kv_heads: 2,
            head_dim: 256,
            rope_dim: 64,
            rope_theta: 10_000_000.0,
            mrope_section: [11, 11, 10],
            indexer: Qwen4ExpIndexerConfig { head_count: 4, head_dim: 128, top_k: 2048, compress_ratio: 4 },
            linear_key_heads: 16,
            linear_value_heads: 48,
            linear_head_dim: 128,
            linear_conv_kernel_size: 4,
            num_experts: 512,
            num_experts_per_tok: 10,
            expert_intermediate_size: 640,
            shared_expert_intermediate_size: 640,
            hyper_connection: Qwen4ExpHyperConnectionConfig { streams: 4, low_rank: 320 },
            ple: Some(Qwen4ExpPleConfig {
                layer: 1,
                ngram_size: 3,
                heads_per_ngram: 8,
                head_dim: 160,
                conv_kernel: 4,
                layer_multipliers: vec![23703573157769, 20109073645365, 8052911324071],
                head_offsets: vec![0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275],
                head_vocab_sizes: vec![20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171],
                eos_token_id: 248_044,
            }),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262_144,
            bos_token_id: 248_044,
            eos_token_ids: vec![248_046, 248_044],
        }
    }

    /// 全注意力层判定与 llama.cpp 一致:(layer+1) % interval == 0。
    pub fn is_full_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    pub fn hc_dim(&self) -> usize {
        self.hyper_connection.streams * self.hidden_size
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.num_layers == 0 || self.full_attention_interval == 0 || self.hidden_size == 0 {
            return Err("Qwen4-Exp 层数/间隔/hidden 必须大于 0".into());
        }
        if self.num_attention_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 || self.num_attention_heads % self.num_kv_heads != 0 {
            return Err(format!("Qwen4-Exp GQA 头配置非法: heads={} kv={} head_dim={}", self.num_attention_heads, self.num_kv_heads, self.head_dim));
        }
        if self.rope_dim > self.head_dim || self.mrope_section.iter().sum::<usize>() * 2 != self.rope_dim {
            return Err(format!("Qwen4-Exp M-RoPE section {:?} 与 rope_dim {} 不一致", self.mrope_section, self.rope_dim));
        }
        if self.linear_key_heads == 0 || self.linear_value_heads == 0 || self.linear_head_dim == 0 || self.linear_conv_kernel_size == 0 {
            return Err("Qwen4-Exp GDN 头配置非法".into());
        }
        if self.num_experts == 0 || self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts || self.expert_intermediate_size == 0 || self.shared_expert_intermediate_size == 0 {
            return Err(format!("Qwen4-Exp MoE 配置非法: experts={} top-k={} inter={} shared={}", self.num_experts, self.num_experts_per_tok, self.expert_intermediate_size, self.shared_expert_intermediate_size));
        }
        if self.hyper_connection.streams <= 1 || self.hyper_connection.low_rank == 0 {
            return Err(format!("Qwen4-Exp HC 配置非法: streams={} low_rank={}", self.hyper_connection.streams, self.hyper_connection.low_rank));
        }
        if let Some(ple) = &self.ple {
            if ple.layer >= self.num_layers || ple.ngram_size < 2 || ple.heads_per_ngram == 0 {
                return Err(format!("Qwen4-Exp PLE 配置非法: layer={} ngram={} heads={}", ple.layer, ple.ngram_size, ple.heads_per_ngram));
            }
            if ple.head_count() * ple.head_dim != self.hidden_size {
                return Err(format!("Qwen4-Exp PLE 头数×维度 {} 与 hidden {} 不一致", ple.head_count() * ple.head_dim, self.hidden_size));
            }
            if ple.layer_multipliers.len() != ple.ngram_size || ple.head_offsets.len() != ple.head_count() || ple.head_vocab_sizes.len() != ple.head_count() {
                return Err(format!(
                    "Qwen4-Exp PLE 常量数组长度不匹配: multipliers={} offsets={} vocabs={} 期望 {}/{}/{}",
                    ple.layer_multipliers.len(),
                    ple.head_offsets.len(),
                    ple.head_vocab_sizes.len(),
                    ple.ngram_size,
                    ple.head_count(),
                    ple.head_count()
                ));
            }
            if self.is_full_attention(ple.layer) {
                return Err(format!("Qwen4-Exp PLE 层 {} 必须是线性注意力层", ple.layer));
            }
        }
        if self.indexer.head_count == 0 || self.indexer.head_dim == 0 || self.indexer.top_k == 0 || self.indexer.compress_ratio == 0 {
            return Err("Qwen4-Exp indexer 配置非法".into());
        }
        if self.vocab_size == 0 || !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err("Qwen4-Exp vocab/rms_eps 非法".into());
        }
        Ok(())
    }
}
