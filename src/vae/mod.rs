//! 视频/音频编解码器领域层。
//!
//! 与 attention/、moe/、norm.rs 平行的独立领域，保存编解码器规格、
//! 平台无关算法与 reference 实现。具体存储、kernel、同步和资源生命周期留在 backend/。

/// Conv3D 的完整逻辑 shape。Tensor 本身只保存 `[channels, spatial]`，
/// 时空维度由这份规格贴近调用点传给 backend。
/// Conv1D 的完整逻辑 shape。音频 VAE 的 `conv1d` / `conv1d_strided` /
/// `conv_transpose1d` 共用：普通 conv1d 恒为 stride=1，转置卷积恒为 dilation=1，
/// 未用到的字段由调用方按卷积种类填固定值。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dSpec {
    pub batch: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv3dSpec {
    pub input_channels: usize,
    pub output_channels: usize,
    pub input_shape: [usize; 3],
    pub kernel: [usize; 3],
    pub stride: [usize; 3],
    /// 非 causal 轴为两侧对称 padding；causal 时间轴只在前侧 padding。
    pub padding: [usize; 3],
    pub causal: bool,
}

impl Conv3dSpec {
    pub fn output_shape(&self) -> Result<[usize; 3], String> {
        if self.input_channels == 0 || self.output_channels == 0 || self.input_shape.contains(&0) || self.kernel.contains(&0) || self.stride.contains(&0) {
            return Err(format!("Conv3D 规格含零维度: {self:?}"));
        }
        let mut output = [0; 3];
        for axis in 0..3 {
            let padding = if self.causal && axis == 0 { self.padding[axis] } else { self.padding[axis].checked_mul(2).ok_or("Conv3D padding 溢出")? };
            let padded = self.input_shape[axis].checked_add(padding).ok_or("Conv3D 输入 shape 溢出")?;
            if padded < self.kernel[axis] {
                return Err(format!("Conv3D axis {axis} padded={padded} 小于 kernel={}", self.kernel[axis]));
            }
            output[axis] = (padded - self.kernel[axis]) / self.stride[axis] + 1;
        }
        Ok(output)
    }

    pub fn input_spatial(&self) -> Result<usize, String> {
        self.input_shape.into_iter().try_fold(1usize, |size, dim| size.checked_mul(dim).ok_or_else(|| "Conv3D input spatial 溢出".to_owned()))
    }

    pub fn output_spatial(&self) -> Result<usize, String> {
        self.output_shape()?.into_iter().try_fold(1usize, |size, dim| size.checked_mul(dim).ok_or_else(|| "Conv3D output spatial 溢出".to_owned()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelShuffleSpec {
    /// 输出通道数；输入通道数为 `channels * upscale^2`。
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub upscale: usize,
}

impl PixelShuffleSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.channels == 0 || self.height == 0 || self.width == 0 || self.upscale == 0 {
            return Err(format!("pixel shuffle 规格含零维度: {self:?}"));
        }
        Ok(())
    }
}

/// VAE 编解码器规格。
#[derive(Clone, Debug)]
pub enum VaeSpec {
    /// H3 视频 VAE：时空因果卷积，空间 16× / 时间 4× 压缩，24 latent channels。
    H3Video(H3VideoVaeSpec),
    /// H3 音频 VAE：32kHz → 40Hz，32 latent channels。
    H3Audio(H3AudioVaeSpec),
}

/// H3 视频 VAE 规格。
#[derive(Clone, Debug)]
pub struct H3VideoVaeSpec {
    /// latent 通道数(H3 = 24)。
    pub latent_channels: usize,
    /// 空间压缩倍率(H3 = 16)。
    pub spatial_compression: usize,
    /// 时间压缩倍率(H3 = 4)。
    pub temporal_compression: usize,
    /// 官方因果 VAE 的时间 chunk 帧数。
    pub temporal_clip_length: usize,
    /// 相邻时间 chunk 复用的尾部 token 语义。
    pub temporal_token_drop: usize,
    /// VisualVAE latent 写入 DiT reference condition 时的 patch。
    pub condition_patch: [usize; 3],
    pub encoder_channels: [usize; 6],
    pub encoder_space_strides: [usize; 6],
    pub encoder_time_strides: [usize; 6],
    pub encoder_res_blocks: usize,
    pub encoder_norm_groups: usize,
    pub encoder_norm_eps: f32,
    pub decoder_hidden_size: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_head_dim: usize,
    pub decoder_ffn_size: usize,
    pub decoder_register_tokens: usize,
    pub decoder_rope_dim: usize,
    pub decoder_rope_theta: f32,
    pub decoder_norm_eps: f32,
}

impl H3VideoVaeSpec {
    /// H3 官方默认配置。
    pub fn standard() -> Self {
        Self {
            latent_channels: 24,
            spatial_compression: 16,
            temporal_compression: 4,
            temporal_clip_length: 17,
            temporal_token_drop: 3,
            condition_patch: [1, 2, 2],
            encoder_channels: [128, 256, 256, 512, 512, 1024],
            encoder_space_strides: [2, 2, 2, 2, 1, 1],
            encoder_time_strides: [1, 2, 2, 1, 1, 1],
            encoder_res_blocks: 2,
            encoder_norm_groups: 32,
            encoder_norm_eps: 1.0e-6,
            decoder_hidden_size: 2048,
            decoder_layers: 36,
            decoder_heads: 32,
            decoder_head_dim: 64,
            decoder_ffn_size: 8192,
            decoder_register_tokens: 4,
            decoder_rope_dim: 48,
            decoder_rope_theta: 100.0,
            decoder_norm_eps: 1.0e-5,
        }
    }
}

/// H3 音频 VAE 规格。
#[derive(Clone, Debug)]
pub struct H3AudioVaeSpec {
    /// latent 通道数(H3 = 32)。
    pub latent_channels: usize,
    /// 采样率(H3 = 32000 Hz)。
    pub sample_rate: usize,
    /// latent 帧率(H3 = 40 Hz)。
    pub latent_rate: usize,
    /// 编码器隐藏维度。
    pub encoder_dim: usize,
    /// 解码器隐藏维度。
    pub decoder_dim: usize,
    /// VAE 内部连续表示维度。
    pub latent_dim: usize,
    /// 左右声道分别作为 batch 解码。
    pub output_channels: usize,
    pub encoder_rates: [usize; 5],
    pub decoder_rates: [usize; 7],
    pub decoder_kernels: [usize; 7],
    pub resblock_kernels: [usize; 3],
    pub resblock_dilations: [usize; 3],
    pub alias_filter_kernel: usize,
}

impl H3AudioVaeSpec {
    /// H3 官方默认配置。
    pub fn standard() -> Self {
        Self {
            latent_channels: 32,
            sample_rate: 32000,
            latent_rate: 40,
            encoder_dim: 64,
            decoder_dim: 1024,
            latent_dim: 2048,
            output_channels: 2,
            encoder_rates: [2, 4, 4, 5, 5],
            decoder_rates: [5, 5, 2, 2, 2, 2, 2],
            decoder_kernels: [9, 9, 4, 4, 4, 4, 4],
            resblock_kernels: [3, 7, 11],
            resblock_dilations: [1, 3, 5],
            alias_filter_kernel: 12,
        }
    }
}
