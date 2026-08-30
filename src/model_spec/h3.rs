//! H3 平台与执行无关的 DiT 架构配置。

// H3-Omni-Transformer 是 33B 参数稠密单流 Transformer，
// 使用 CFG-distilled 联合视频/音频扩散生成。
// 与文本 LLM 的 Model trait 不同，H3 是 Diffusion Transformer(DiT)，
// 使用 AdaLN 调制 + 时间步嵌入 + MM-RoPE。
//
// 架构参数来源：HuggingFace MiniMaxAI/MiniMax-H3 config.json。

/// H3-Omni-Transformer 配置（DiT 主干）。
#[derive(Debug, Clone)]
pub struct H3Config {
    /// Transformer 隐藏维度（H3 = 5376）。
    pub hidden_size: usize,
    /// DiT 层数（H3 = 50）。
    pub num_layers: usize,
    /// Token refiner 层数（H3 = 2，输入侧细化）。
    pub token_refiner_num_layers: usize,
    /// 注意力头数（H3 = 56）。
    pub num_attention_heads: usize,
    /// 每头维度（H3 = 128）。
    pub attention_head_dim: usize,
    /// FFN 中间维度（H3 = 14336）。
    pub ffn_hidden_size: usize,
    /// 视频 latent 输入通道数（H3 = 24，H3-VAE 输出）。
    pub video_latent_channels: usize,
    /// 音频 latent 输入通道数（H3 = 32）。
    pub audio_latent_channels: usize,
    /// 视频 patch 大小，按 [time, height, width] 排列。
    pub patch_size: [usize; 3],
    /// 文本编码器输出维度（H3 = 5120，Qwen3-VL text encoder）。
    pub text_dim: usize,
    /// 时间步嵌入维度（H3 = 2688）。
    pub time_embed_dim: usize,
    /// 时间步嵌入隐藏维度（= hidden_size）。
    pub time_embed_hidden_size: usize,
    /// 时间步频率嵌入维度（H3 = 256）。
    pub timestep_input_dim: usize,
    /// RoPE 频率维度（H3 = 16）。
    pub rope_inv_freq_len: usize,
    /// 视频 Flow Matching sigma shift（H3 = 12）。
    pub sigma_shift_video: f32,
    /// 音频 Flow Matching sigma shift（H3 = 3）。
    pub sigma_shift_audio: f32,
    /// Transformer LayerNorm eps。
    pub norm_eps: f32,
    /// 最终归一化 eps。
    pub final_norm_eps: f32,
    /// QK 归一化 eps。
    pub qk_norm_eps: f32,
    /// AdaLN 输出特征数（H3 = 96768）。
    pub adaln_out_features: usize,
    /// 最终 AdaLN 输出特征数（H3 = 10752）。
    pub final_adaln_out_features: usize,
    /// AdaLN 曲线网格点数;Some 时走 curve 模式(用 adaln_t_table 替代 time embedder)。
    /// ComfyUI 的 curve checkpoint 把 TimeEmbedder 烘焙成 [grid, time_embed_dim] 查表,
    /// forward 时对归一化时间 t∈[0,1] 线性插值。None 时走标准 sinusoidal+MLP。
    pub adaln_curve_grid: Option<usize>,
}

impl H3Config {
    /// H3 官方默认配置。
    pub fn standard() -> Self {
        Self {
            hidden_size: 5376,
            num_layers: 50,
            token_refiner_num_layers: 2,
            num_attention_heads: 56,
            attention_head_dim: 128,
            ffn_hidden_size: 14336,
            video_latent_channels: 24,
            audio_latent_channels: 32,
            patch_size: [1, 2, 2],
            text_dim: 5120,
            time_embed_dim: 2688,
            time_embed_hidden_size: 5376,
            timestep_input_dim: 256,
            rope_inv_freq_len: 16,
            sigma_shift_video: 12.0,
            sigma_shift_audio: 3.0,
            norm_eps: 1e-5,
            final_norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            adaln_out_features: 96768,
            final_adaln_out_features: 10752,
            adaln_curve_grid: None,
        }
    }

    /// ComfyUI curve + FP8 checkpoint 配置(rzgar/minimax_h3_fl2va_fp16attn_fp8)。
    ///
    /// 这类 checkpoint 把 TimeEmbedder 压缩成 adaln_t_table[1025, 8] 查表,
    /// time_embed_dim 从标准 2688 降到 8(1025 个时间步 × 8 维 basis)。
    /// MLP(fc1/fc2)用 F8_E4M3 per-tensor scale,attention 保持 F16。
    /// 其他维度与 standard() 一致。
    pub fn curve_fp8() -> Self {
        Self {
            hidden_size: 5376,
            num_layers: 50,
            token_refiner_num_layers: 2,
            num_attention_heads: 56,
            attention_head_dim: 128,
            ffn_hidden_size: 14336,
            video_latent_channels: 24,
            audio_latent_channels: 32,
            patch_size: [1, 2, 2],
            text_dim: 5120,
            time_embed_dim: 8,
            time_embed_hidden_size: 5376,
            timestep_input_dim: 256,
            rope_inv_freq_len: 16,
            sigma_shift_video: 12.0,
            sigma_shift_audio: 3.0,
            norm_eps: 1e-5,
            final_norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            adaln_out_features: 96768,
            final_adaln_out_features: 10752,
            adaln_curve_grid: Some(1025),
        }
    }

    /// 注意力总维度 = heads × head_dim。
    pub fn attention_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// 一个视频 patch 展平后的输入/输出维度。
    pub fn video_patch_dim(&self) -> usize {
        self.video_latent_channels * self.patch_size.iter().product::<usize>()
    }

    /// 官方 checkpoint 的结构不变量。
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 || self.num_layers == 0 || self.token_refiner_num_layers == 0 || self.patch_size.contains(&0) {
            return Err("H3 hidden/layer/refiner/patch 维度必须非零".to_owned());
        }
        if self.adaln_out_features != self.hidden_size * 18 {
            return Err(format!("H3 AdaLN 输出 {}，期望 hidden_size*18={}", self.adaln_out_features, self.hidden_size * 18,));
        }
        if self.final_adaln_out_features != self.hidden_size * 2 {
            return Err(format!("H3 final AdaLN 输出 {}，期望 hidden_size*2={}", self.final_adaln_out_features, self.hidden_size * 2,));
        }
        if !self.sigma_shift_video.is_finite() || !self.sigma_shift_audio.is_finite() || self.sigma_shift_video <= 0.0 || self.sigma_shift_audio <= 0.0 {
            return Err(format!("H3 sigma shift 必须是正有限数: video={} audio={}", self.sigma_shift_video, self.sigma_shift_audio,));
        }
        Ok(())
    }
}
