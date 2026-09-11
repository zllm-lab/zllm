//! SenseVoice-Small 架构常量（平台与执行无关）。
//!
//! 数值来源（2026-09-08 对照钉死，不凭记忆）：
//! - FunASR `funasr/models/sense_voice/model.py`（SenseVoiceSmall / SenseVoiceEncoderSmall）
//! - FunAudioLLM/SenseVoiceSmall `config.yaml`
//! - sherpa-onnx `offline-recognizer-sense-voice-impl.h` / `offline-sense-voice-model.cc`
//!
//! 结构：kaldi fbank(80) → LFR(窗口 7/步长 6 → 560 维) → CMVN →
//! [lang,event,emo,textnorm] 4 个可学习 query + 帧序列整体乘 √512 加正弦 PE →
//! 50 层 SAN-M（LayerNorm + 全局 attention + FSMN 深度卷积记忆 + ReLU FFN，
//! 首层 560→512 无残差）→ after_norm → 20 层 tp SAN-M → tp_norm →
//! CTC 线性头 → 逐帧 argmax → 贪心合并（blank=0，前 4 帧为控制 token 预测）。

/// 特殊 query 的 embedding 表索引（SenseVoiceSmall.lid_dict / textnorm_dict，
/// 指向 16 行可学习 query embedding，不是词表 id）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenseVoiceLanguage {
    Auto,
    Zh,
    En,
    Yue,
    Ja,
    Ko,
}

impl SenseVoiceLanguage {
    /// query embedding 行号：auto=0, zh=3, en=4, yue=7, ja=11, ko=12。
    pub fn query_index(self) -> usize {
        match self {
            Self::Auto => 0,
            Self::Zh => 3,
            Self::En => 4,
            Self::Yue => 7,
            Self::Ja => 11,
            Self::Ko => 12,
        }
    }

    /// 配置文本（yaml / CLI）→ 语言；SenseVoice 不支持的语言直接报错。
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "zh" => Ok(Self::Zh),
            "en" => Ok(Self::En),
            "yue" => Ok(Self::Yue),
            "ja" => Ok(Self::Ja),
            "ko" => Ok(Self::Ko),
            other => Err(format!("SenseVoice language={other} 不支持，可选 auto/zh/en/yue/ja/ko")),
        }
    }
}

/// 文本规整模式（同样是 query embedding 行号：withitn=14, woitn=15）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenseVoiceTextNorm {
    WithItn,
    WithoutItn,
}

impl SenseVoiceTextNorm {
    pub fn query_index(self) -> usize {
        match self {
            Self::WithItn => 14,
            Self::WithoutItn => 15,
        }
    }

    /// 配置文本（yaml / CLI）→ 文本规整模式。
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "withitn" => Ok(Self::WithItn),
            "woitn" => Ok(Self::WithoutItn),
            other => Err(format!("SenseVoice text_norm={other} 不支持，可选 withitn/woitn")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SenseVoiceConfig {
    // ---- 前端：kaldi fbank（torchaudio.compliance.kaldi 语义）----
    pub sample_rate: usize,
    pub frame_ms: usize,
    pub frame_shift_ms: usize,
    /// 512 = round_to_power_of_two(400)。
    pub fft_length: usize,
    pub mel_bins: usize,
    /// low_freq=20Hz；high_freq=0 表示 Nyquist。
    pub mel_low_hz: f32,
    pub mel_high_hz: f32,
    pub preemphasis: f32,
    pub remove_dc_offset: bool,
    /// 幅度谱平方（use_power=true）后取 log。
    pub use_power: bool,

    // ---- LFR 帧拼接 ----
    /// 拼接窗口（lfr_m=7）：LFR 帧维度 = mel_bins * lfr_window = 560。
    pub lfr_window: usize,
    /// 拼接步长（lfr_n=6）：60ms/帧。
    pub lfr_stride: usize,

    // ---- encoder（SenseVoiceEncoderSmall）----
    /// 4 头 × 128。
    pub attention_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    /// encoders0(1) + encoders(49)；首层 in=560 无残差。
    pub layer_count: usize,
    /// after_norm 之后的 tp_encoders（20）。
    pub tp_layer_count: usize,
    /// FSMN 记忆分支的 depthwise 卷积核；sanm_shfit=0 → 左右对称 padding。
    pub fsmn_kernel: usize,
    /// torch LayerNorm 默认 eps。
    pub norm_eps: f32,

    // ---- query embedding / CTC ----
    /// 可学习 query embedding 行数：7 + 7 语言 + 2 textnorm。
    pub query_embed_count: usize,
    /// encoder 输入前缀长度（lang/event/emo/textnorm），CTC 文本从第 4 帧起。
    pub query_prefix: usize,
    /// CTC blank（词表 id 0）。
    pub blank_id: u32,
}

impl SenseVoiceConfig {
    pub fn standard() -> Self {
        Self {
            sample_rate: 16_000,
            frame_ms: 25,
            frame_shift_ms: 10,
            fft_length: 512,
            mel_bins: 80,
            mel_low_hz: 20.0,
            mel_high_hz: 0.0,
            preemphasis: 0.97,
            remove_dc_offset: true,
            use_power: true,
            lfr_window: 7,
            lfr_stride: 6,
            attention_heads: 4,
            head_dim: 128,
            d_model: 512,
            ffn_dim: 2048,
            layer_count: 50,
            tp_layer_count: 20,
            fsmn_kernel: 11,
            norm_eps: 1e-5,
            query_embed_count: 16,
            query_prefix: 4,
            blank_id: 0,
        }
    }

    /// 帧长采样数（25ms × 16kHz）。
    pub fn frame_length(&self) -> usize {
        self.frame_ms * self.sample_rate / 1000
    }

    /// 帧移采样数（10ms × 16kHz）。
    pub fn frame_shift(&self) -> usize {
        self.frame_shift_ms * self.sample_rate / 1000
    }

    /// LFR 后每帧维度（encoder 输入维度）。
    pub fn encoder_input_dim(&self) -> usize {
        self.mel_bins * self.lfr_window
    }

    /// FSMN depthwise 卷积的对称 padding。
    pub fn fsmn_padding(&self) -> usize {
        (self.fsmn_kernel - 1) / 2
    }

    /// CTC 头之前的总层数（50 主 + 20 tp）。
    pub fn total_layer_count(&self) -> usize {
        self.layer_count + self.tp_layer_count
    }

    pub fn validate(&self) -> Result<(), String> {
        let problems = [
            (self.frame_length() != 400, "frame_length 必须是 400"),
            (self.frame_shift() != 160, "frame_shift 必须是 160"),
            (self.fft_length < self.frame_length(), "fft_length 小于帧长"),
            (self.d_model != self.attention_heads * self.head_dim, "d_model != heads*head_dim"),
            (self.encoder_input_dim() == 0, "encoder 输入维度为零"),
            (self.fsmn_kernel % 2 != 1, "FSMN kernel 必须为奇数（对称 padding）"),
        ];
        for (failed, message) in problems {
            if failed {
                return Err(format!("SenseVoice 配置非法：{message}"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_config_derived_values() {
        let config = SenseVoiceConfig::standard();
        assert_eq!(config.frame_length(), 400);
        assert_eq!(config.frame_shift(), 160);
        assert_eq!(config.encoder_input_dim(), 560);
        assert_eq!(config.fsmn_padding(), 5);
        assert_eq!(config.total_layer_count(), 70);
        assert_eq!(config.query_prefix, 4);
        config.validate().unwrap();
    }
}
