//! SeedVR2 标准 7B NaDiT 与视频 VAE 的平台无关规格。

#[derive(Clone, Debug)]
pub struct SeedVr2Config {
    pub input_channels: usize,
    pub output_channels: usize,
    pub text_dim: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_hidden_size: usize,
    pub timestep_dim: usize,
    pub embedding_dim: usize,
    pub patch: [usize; 3],
    pub window: [usize; 3],
    pub norm_eps: f32,
    pub positive_text_tokens: usize,
    pub negative_text_tokens: usize,
}

impl SeedVr2Config {
    pub fn standard_7b() -> Self {
        Self {
            input_channels: 33,
            output_channels: 16,
            text_dim: 5_120,
            hidden_size: 3_072,
            num_layers: 36,
            num_heads: 24,
            head_dim: 128,
            mlp_hidden_size: 12_288,
            timestep_dim: 256,
            embedding_dim: 18_432,
            patch: [1, 2, 2],
            window: [4, 3, 3],
            norm_eps: 1.0e-5,
            positive_text_tokens: 58,
            negative_text_tokens: 64,
        }
    }

    pub fn attention_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
    pub fn patch_volume(&self) -> usize {
        self.patch.iter().product()
    }

    /// 源实现先 head_dim/2，再除以三轴，最后成对旋转；128 维头只旋转前 60 维。
    /// 频率来自权重中的 pixel RoPE buffer，不能替换成文本模型的 theta 公式。
    pub fn rope_frequency_count(&self) -> usize {
        self.head_dim / 2 / 3 / 2
    }
    pub fn rotary_dim(&self) -> usize {
        self.rope_frequency_count() * 2 * 3
    }

    pub fn validate(&self) -> Result<(), String> {
        if [
            self.input_channels,
            self.output_channels,
            self.text_dim,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.mlp_hidden_size,
            self.timestep_dim,
            self.embedding_dim,
            self.positive_text_tokens,
            self.negative_text_tokens,
        ]
        .contains(&0)
            || self.patch.contains(&0)
            || self.window.contains(&0)
        {
            return Err("SeedVR2 架构维度必须非零".to_owned());
        }
        if self.num_heads.checked_mul(self.head_dim) != Some(self.hidden_size)
            || self.hidden_size.checked_mul(4) != Some(self.mlp_hidden_size)
            || self.hidden_size.checked_mul(6) != Some(self.embedding_dim)
            || self.output_channels.checked_mul(2).and_then(|v| v.checked_add(1)) != Some(self.input_channels)
            || self.rope_frequency_count() == 0
            || !self.timestep_dim.is_multiple_of(2)
            || self.patch.iter().try_fold(1usize, |a, b| a.checked_mul(*b)).and_then(|v| v.checked_mul(self.input_channels)).is_none()
        {
            return Err(format!("SeedVR2 QKV、AdaSingle、MLP 或 patch 规格不一致: {self:?}"));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err("SeedVR2 norm eps 必须为正有限数".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SeedVr2VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_channels: [usize; 4],
    pub layers_per_block: usize,
    pub norm_groups: usize,
    pub norm_eps: f32,
    pub spatial_downsample_factor: usize,
    pub temporal_downsample_factor: usize,
    pub scaling_factor: f32,
    pub shifting_factor: f32,
}

impl SeedVr2VaeConfig {
    pub fn standard() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            norm_groups: 32,
            norm_eps: 1.0e-6,
            spatial_downsample_factor: 8,
            temporal_downsample_factor: 4,
            scaling_factor: 0.9152,
            shifting_factor: 0.0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if [self.in_channels, self.out_channels, self.latent_channels, self.layers_per_block, self.norm_groups].contains(&0)
            || self.block_channels.contains(&0)
            || self.block_channels.iter().any(|c| !c.is_multiple_of(self.norm_groups))
            || self.spatial_downsample_factor != 8
            || self.temporal_downsample_factor != 4
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
            || !self.scaling_factor.is_finite()
            || self.scaling_factor <= 0.0
            || !self.shifting_factor.is_finite()
        {
            return Err(format!("SeedVR2 VAE 配置不一致: {self:?}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standard_7b_dimensions_and_partial_pixel_rope() {
        let c = SeedVr2Config::standard_7b();
        c.validate().unwrap();
        assert_eq!(c.input_channels * c.patch_volume(), 132);
        assert_eq!(c.output_channels * c.patch_volume(), 64);
        assert_eq!(c.attention_dim(), 3072);
        assert_eq!((c.rope_frequency_count(), c.rotary_dim()), (10, 60));
        assert_eq!(c.num_layers, 36);
    }
    #[test]
    fn reject_inconsistent_or_overflowing_dimensions() {
        let mut c = SeedVr2Config::standard_7b();
        c.embedding_dim -= 1;
        assert!(c.validate().is_err());
        c = SeedVr2Config::standard_7b();
        c.num_heads = usize::MAX;
        assert!(c.validate().is_err());
        c = SeedVr2Config::standard_7b();
        c.norm_eps = f32::NAN;
        assert!(c.validate().is_err());
    }
    #[test]
    fn standard_vae_has_causal_temporal_compression() {
        let c = SeedVr2VaeConfig::standard();
        c.validate().unwrap();
        assert_eq!(c.block_channels, [128, 256, 512, 512]);
        assert_eq!((c.latent_channels, c.temporal_downsample_factor), (16, 4));
        assert_eq!(c.scaling_factor, 0.9152);
        assert_eq!(c.shifting_factor, 0.0);
    }
}
