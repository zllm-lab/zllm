//! FLUX.2 Klein 平台与执行无关的图像 DiT 架构配置。

/// FLUX.2 Klein 4B 的 Transformer 配置。
#[derive(Clone, Debug)]
pub struct Flux2KleinConfig {
    pub input_channels: usize,
    pub text_dim: usize,
    pub hidden_size: usize,
    pub double_layers: usize,
    pub single_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_hidden_size: usize,
    pub axes_dim: [usize; 4],
    pub rope_theta: f32,
    pub timestep_dim: usize,
    pub norm_eps: f32,
}

impl Flux2KleinConfig {
    pub fn klein_4b() -> Self {
        Self {
            input_channels: 128,
            text_dim: 7_680,
            hidden_size: 3_072,
            double_layers: 5,
            single_layers: 20,
            num_heads: 24,
            head_dim: 128,
            mlp_hidden_size: 9_216,
            axes_dim: [32, 32, 32, 32],
            rope_theta: 2_000.0,
            timestep_dim: 256,
            norm_eps: 1.0e-6,
        }
    }

    pub fn attention_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.input_channels == 0 || self.text_dim == 0 || self.hidden_size == 0 || self.double_layers == 0 || self.single_layers == 0 || self.num_heads == 0 || self.head_dim == 0 || self.mlp_hidden_size == 0 || self.timestep_dim == 0 {
            return Err("FLUX.2 Klein 架构维度必须非零".to_owned());
        }
        if self.attention_dim() != self.hidden_size {
            return Err(format!("FLUX.2 Klein attention={}，期望 hidden_size={}", self.attention_dim(), self.hidden_size));
        }
        if self.axes_dim.iter().any(|dim| *dim == 0 || !dim.is_multiple_of(2)) || self.axes_dim.iter().sum::<usize>() != self.head_dim {
            return Err(format!("FLUX.2 Klein axes_dim={:?} 必须是正偶数且总和等于 head_dim={}", self.axes_dim, self.head_dim));
        }
        if self.mlp_hidden_size != self.hidden_size * 3 {
            return Err(format!("FLUX.2 Klein MLP hidden={}，期望 hidden_size*3={}", self.mlp_hidden_size, self.hidden_size * 3));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 || !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err("FLUX.2 Klein RoPE theta 与 norm eps 必须为正有限数".to_owned());
        }
        Ok(())
    }
}

/// FLUX.2 Klein 图像 VAE 解码器配置。
#[derive(Clone, Debug)]
pub struct Flux2KleinVaeConfig {
    pub packed_channels: usize,
    pub latent_channels: usize,
    pub patch: [usize; 2],
    pub block_channels: [usize; 4],
    pub layers_per_block: usize,
    pub norm_groups: usize,
    pub norm_eps: f32,
    pub batch_norm_eps: f32,
}

impl Flux2KleinVaeConfig {
    pub fn standard() -> Self {
        Self { packed_channels: 128, latent_channels: 32, patch: [2, 2], block_channels: [128, 256, 512, 512], layers_per_block: 2, norm_groups: 32, norm_eps: 1.0e-6, batch_norm_eps: 1.0e-4 }
    }

    pub fn validate(&self) -> Result<(), String> {
        let patch_area = self.patch.into_iter().product::<usize>();
        if self.latent_channels == 0 || self.packed_channels != self.latent_channels * patch_area || self.patch.contains(&0) || self.block_channels.contains(&0) || self.layers_per_block == 0 || self.norm_groups == 0 {
            return Err(format!("FLUX.2 Klein VAE 配置不一致: {self:?}"));
        }
        if self.block_channels.iter().any(|channels| !channels.is_multiple_of(self.norm_groups)) || !self.norm_eps.is_finite() || self.norm_eps <= 0.0 || !self.batch_norm_eps.is_finite() || self.batch_norm_eps <= 0.0 {
            return Err(format!("FLUX.2 Klein VAE norm 配置非法: {self:?}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_klein_4b_config_is_consistent() {
        let config = Flux2KleinConfig::klein_4b();
        config.validate().unwrap();
        assert_eq!(config.attention_dim(), 3_072);
        assert_eq!(config.axes_dim.iter().sum::<usize>(), 128);
    }

    #[test]
    fn official_vae_config_is_consistent() {
        Flux2KleinVaeConfig::standard().validate().unwrap();
    }
}
