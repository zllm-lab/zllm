//! Qwen3.6/3.8 DSpark drafter(llama.cpp `dflash` GGUF)的规格与张量定位。
//!
//! drafter 只含 backbone 与头,Q8_0 量化;embedding 与 lm_head 复用目标模型。
//! 执行协议由 `speculative` 规格与 `runtime::dspark` 算法提供。

use crate::{
    speculative::{BlockDraftSpec, HiddenStateCapturePlan},
    weight::container::gguf::GgufReader,
};

/// GGUF 元数据前缀与张量名。fc 是 [hidden, 5×hidden]:5 份 capture 拼接后整体投影。
pub const FC_WEIGHT: &str = "fc.weight";
pub const ENC_OUTPUT_NORM: &str = "enc.output_norm.weight";
pub const OUTPUT_NORM: &str = "output_norm.weight";
pub const MARKOV_W1: &str = "markov_w1.weight";
pub const MARKOV_W2: &str = "markov_w2.weight";

pub fn layer_input_norm(layer: usize) -> String {
    format!("blk.{layer}.attn_norm.weight")
}

pub fn layer_query_norm(layer: usize) -> String {
    format!("blk.{layer}.attn_q_norm.weight")
}

pub fn layer_key_norm(layer: usize) -> String {
    format!("blk.{layer}.attn_k_norm.weight")
}

pub fn layer_query(layer: usize) -> String {
    format!("blk.{layer}.attn_q.weight")
}

pub fn layer_key(layer: usize) -> String {
    format!("blk.{layer}.attn_k.weight")
}

pub fn layer_value(layer: usize) -> String {
    format!("blk.{layer}.attn_v.weight")
}

pub fn layer_output(layer: usize) -> String {
    format!("blk.{layer}.attn_output.weight")
}

pub fn layer_post_attention_norm(layer: usize) -> String {
    format!("blk.{layer}.ffn_norm.weight")
}

pub fn layer_gate(layer: usize) -> String {
    format!("blk.{layer}.ffn_gate.weight")
}

pub fn layer_up(layer: usize) -> String {
    format!("blk.{layer}.ffn_up.weight")
}

pub fn layer_down(layer: usize) -> String {
    format!("blk.{layer}.ffn_down.weight")
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36DsparkSpec {
    pub layer_count: usize,
    pub block_size: usize,
    /// tap 的 hidden 捕获边界(HiddenStateCapturePlan 约定:N = 第 N-1 层输出)。
    /// llama.cpp `dflash.target_layers` 直接使用同一约定。
    pub capture_boundaries: Vec<usize>,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub feed_forward_length: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub mask_token_id: u32,
}

impl Qwen36DsparkSpec {
    pub fn from_gguf(source: &GgufReader) -> Result<Self, String> {
        source.expect_metadata_str("general.architecture", "dflash")?;
        let spec = Self {
            layer_count: source.metadata_u64("dflash.block_count")? as usize,
            block_size: source.metadata_u64("dflash.block_size")? as usize,
            capture_boundaries: source
                .metadata("dflash.target_layers")
                .and_then(|value| value.as_i64_array())
                .ok_or("GGUF metadata dflash.target_layers 缺失或类型错误")?
                .into_iter()
                .map(|value| usize::try_from(value).map_err(|_| "dflash.target_layers 元素为负".to_owned()))
                .collect::<Result<_, _>>()?,
            head_count: source.metadata_u64("dflash.attention.head_count")? as usize,
            kv_head_count: source.metadata_u64("dflash.attention.head_count_kv")? as usize,
            head_dim: source.metadata_u64("dflash.attention.key_length")? as usize,
            hidden_size: source.metadata_u64("dflash.embedding_length")? as usize,
            feed_forward_length: source.metadata_u64("dflash.feed_forward_length")? as usize,
            rms_eps: source.metadata("dflash.attention.layer_norm_rms_epsilon").and_then(|value| value.as_f64()).ok_or("GGUF metadata dflash.attention.layer_norm_rms_epsilon 缺失")? as f32,
            rope_theta: source.metadata("dflash.rope.freq_base").and_then(|value| value.as_f64()).ok_or("GGUF metadata dflash.rope.freq_base 缺失")? as f32,
            mask_token_id: u32::try_from(source.metadata_u64("tokenizer.ggml.mask_token_id")?).map_err(|_| "mask token id 超出 u32")?,
        };
        spec.validate()?;
        Ok(spec)
    }

    fn validate(&self) -> Result<(), String> {
        if self.layer_count == 0 || self.block_size < 2 || self.head_count == 0 || self.head_dim == 0 {
            return Err(format!("DSpark drafter 规格非法: {self:?}"));
        }
        if self.head_count % self.kv_head_count != 0 {
            return Err(format!("DSpark drafter heads={}/kv={} 不整除", self.head_count, self.kv_head_count));
        }
        if self.capture_boundaries.is_empty() || self.capture_boundaries.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!("DSpark drafter capture 边界非法: {:?}", self.capture_boundaries));
        }
        if !self.rms_eps.is_finite() || self.rms_eps <= 0.0 || !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!("DSpark drafter eps/rope_theta 非法: {}/{}", self.rms_eps, self.rope_theta));
        }
        Ok(())
    }

    pub fn block_spec(&self, draft_tokens: usize) -> Result<BlockDraftSpec, String> {
        BlockDraftSpec::new(self.block_size, draft_tokens, 1)
    }

    /// 目标层数必须已知(capture 边界不得超出)。
    pub fn capture_plan(&self, target_layer_count: usize) -> Result<HiddenStateCapturePlan, String> {
        HiddenStateCapturePlan::new(self.capture_boundaries.clone(), target_layer_count)
    }

    /// 校验 GGUF 的关键张量 shape(矩阵 dim 顺序为 GGUF 的 [ne0=cols, ne1=rows])。
    pub fn validate_tensors(&self, source: &GgufReader) -> Result<(), String> {
        let hidden = self.hidden_size;
        let query_dim = self.head_count * self.head_dim;
        let kv_dim = self.kv_head_count * self.head_dim;
        let ffn = self.feed_forward_length;
        let captures = self.capture_boundaries.len();
        let vocab = source.tensor(MARKOV_W1).and_then(|tensor| tensor.dims.last().copied()).ok_or("DSpark markov_w1 缺失")?;
        source.expect_tensor(FC_WEIGHT, &[captures * hidden, hidden])?;
        source.expect_tensor(ENC_OUTPUT_NORM, &[hidden])?;
        source.expect_tensor(OUTPUT_NORM, &[hidden])?;
        source.expect_tensor(MARKOV_W1, &[self.markov_rank(), vocab])?;
        source.expect_tensor(MARKOV_W2, &[self.markov_rank(), vocab])?;
        for layer in 0..self.layer_count {
            source.expect_tensor(&layer_input_norm(layer), &[hidden])?;
            source.expect_tensor(&layer_query_norm(layer), &[self.head_dim])?;
            source.expect_tensor(&layer_key_norm(layer), &[self.head_dim])?;
            source.expect_tensor(&layer_query(layer), &[hidden, query_dim])?;
            source.expect_tensor(&layer_key(layer), &[hidden, kv_dim])?;
            source.expect_tensor(&layer_value(layer), &[hidden, kv_dim])?;
            source.expect_tensor(&layer_output(layer), &[query_dim, hidden])?;
            source.expect_tensor(&layer_post_attention_norm(layer), &[hidden])?;
            source.expect_tensor(&layer_gate(layer), &[hidden, ffn])?;
            source.expect_tensor(&layer_up(layer), &[hidden, ffn])?;
            source.expect_tensor(&layer_down(layer), &[ffn, hidden])?;
        }
        Ok(())
    }

    /// markov_w1/w2 的公共秩;两 tensor 的 ne0。
    pub fn markov_rank(&self) -> usize {
        256
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_boundaries_follow_dflash_convention() {
        let spec = Qwen36DsparkSpec {
            layer_count: 5,
            block_size: 15,
            capture_boundaries: vec![2, 17, 32, 47, 62],
            head_count: 32,
            kv_head_count: 8,
            head_dim: 128,
            hidden_size: 5120,
            feed_forward_length: 17408,
            rms_eps: 1.0e-6,
            rope_theta: 1.0e7,
            mask_token_id: 248200,
        };
        spec.validate().unwrap();
        let plan = spec.capture_plan(64).unwrap();
        // 边界 2/17/32/47/62 = 第 1/16/31/46/61 层输出(社区 checkpoint 的 tap 层)
        assert!(plan.captures_layer_output(1));
        assert!(plan.captures_layer_output(61));
        assert!(!plan.captures_layer_output(0));
        assert!(!plan.captures_layer_output(62));
        assert_eq!(spec.block_spec(4).unwrap().block_size, 15);
    }
}
