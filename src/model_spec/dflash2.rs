//! DFlash2 草稿网络的设备无关规格；不包含 target 模型或权重容器依赖。

#[derive(Clone, Debug, PartialEq)]
pub struct Dflash2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub target_layer_count: usize,
    pub target_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub sliding_window: usize,
    pub mask_token_id: u32,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Dflash2Config {
    pub fn glm53() -> Self {
        Self {
            hidden_size: 6144,
            intermediate_size: 12288,
            layer_count: 6,
            head_count: 64,
            kv_head_count: 8,
            head_dim: 128,
            vocab_size: 154880,
            target_layer_count: 78,
            target_layer_ids: vec![5, 19, 33, 47, 61, 75],
            block_size: 8,
            sliding_window: 2048,
            mask_token_id: 154856,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 256,
            selector_top_k: 16,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 1_048_576,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if [
            self.hidden_size,
            self.intermediate_size,
            self.layer_count,
            self.head_count,
            self.kv_head_count,
            self.head_dim,
            self.vocab_size,
            self.target_layer_count,
            self.sliding_window,
            self.conv_group_size,
            self.conv_kernel_size,
            self.selector_rank,
            self.selector_top_k,
        ]
        .contains(&0)
            || self.block_size < 2
            || self.max_position_embeddings < self.block_size
        {
            return Err(format!("DFlash2 维度或 block/context 非法: {self:?}"));
        }
        if !self.hidden_size.is_multiple_of(self.conv_group_size)
            || !self.head_count.is_multiple_of(self.kv_head_count)
            || !self.head_dim.is_multiple_of(2)
            || self.conv_kernel_size > self.block_size
            || self.selector_top_k > self.vocab_size
            || self.vocab_size > u32::MAX as usize
            || self.mask_token_id as usize >= self.vocab_size
        {
            return Err(format!("DFlash2 分组/head/词表维度不兼容: {self:?}"));
        }
        if self.target_layer_ids.is_empty() || self.target_layer_ids.windows(2).any(|p| p[0] >= p[1]) || self.target_layer_ids.iter().any(|&id| id >= self.target_layer_count) {
            return Err(format!("DFlash2 target_layer_ids={:?}，期望严格递增且小于 {}", self.target_layer_ids, self.target_layer_count));
        }
        if !self.rms_eps.is_finite() || self.rms_eps <= 0.0 || !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!("DFlash2 eps={} rope_theta={} 非法", self.rms_eps, self.rope_theta));
        }
        // 同时检查元素与字节上限，避免后续 shape 计算先溢出再进入容器校验。
        for dimensions in [
            vec![self.hidden_size, self.hidden_size, self.target_layer_ids.len()],
            vec![self.hidden_size, self.intermediate_size],
            vec![self.hidden_size, self.head_count, self.head_dim],
            vec![self.vocab_size, self.hidden_size],
            vec![self.vocab_size, self.selector_rank],
            vec![2, self.conv_kernel_size, self.hidden_size, self.hidden_size / self.conv_group_size],
            vec![self.max_position_embeddings, self.head_dim],
        ] {
            dimensions.iter().try_fold(4usize, |n, d| n.checked_mul(*d)).filter(|n| *n <= isize::MAX as usize).ok_or_else(|| format!("DFlash2 shape 字节数溢出: {dimensions:?}"))?;
        }
        Ok(())
    }
}
