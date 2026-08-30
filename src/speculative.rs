//! 推测解码的模型无关规格；具体 drafter、验证算法与 cache 提交属于 runtime。

/// 固定块 drafter 的执行规格。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDraftSpec {
    pub block_size: usize,
    pub speculative_tokens: usize,
    pub verifier_accept_k: usize,
}

impl BlockDraftSpec {
    pub fn new(block_size: usize, speculative_tokens: usize, verifier_accept_k: usize) -> Result<Self, String> {
        if block_size == 0 || speculative_tokens == 0 || speculative_tokens > block_size {
            return Err(format!("推测块规格非法: block_size={block_size} speculative_tokens={speculative_tokens}"));
        }
        if verifier_accept_k == 0 {
            return Err("verifier_accept_k 必须大于 0".to_owned());
        }
        Ok(Self { block_size, speculative_tokens, verifier_accept_k })
    }
}

/// drafter 请求的 verifier hidden-state 边界。0 是 embedding，N 是第 N-1 层输出。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiddenStateCapturePlan {
    boundaries: Vec<usize>,
}

impl HiddenStateCapturePlan {
    pub fn new(mut boundaries: Vec<usize>, layer_count: usize) -> Result<Self, String> {
        boundaries.sort_unstable();
        if boundaries.is_empty() || boundaries.windows(2).any(|pair| pair[0] == pair[1]) || boundaries.iter().any(|&boundary| boundary > layer_count) {
            return Err(format!("hidden capture 边界非法: {boundaries:?}, layer_count={layer_count}"));
        }
        Ok(Self { boundaries })
    }

    pub fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    pub fn captures_embedding(&self) -> bool {
        self.boundaries.first() == Some(&0)
    }

    pub fn captures_layer_output(&self, layer: usize) -> bool {
        self.boundaries.binary_search(&(layer + 1)).is_ok()
    }
}
