//! SenseVoice 音频前端：kaldi fbank → LFR → CMVN → encoder 输入组装。
//!
//! 数值语义与 kaldi / kaldi-native-fbank（sherpa-onnx 的 SenseVoice 路径）逐位对齐：
//! - fbank：去直流 → preemphasis（**kaldi 规则 `y[0] = (1-c)·x[0]`**，注意 torchaudio
//!   是 `y[0] = x[0]`，两者每条语音只差第一个样本；选 kaldi 规则以对齐 sherpa 验收资产）
//!   → 对称 hamming 窗 → 512 点 FFT → 功率谱 |X|²（无归一化）→ kaldi mel 三角滤波
//!   （20Hz..Nyquist，mel 域线性三角）→ log(max(v, f32::EPSILON))
//! - LFR：首帧前补 3 份（(7-1)/2），步长 6、窗口 7 拼接，尾部以重复末帧补齐
//! - CMVN：LFR 后逐维 `(x + shift) * scale`（am.mvn，560 维）
//! - 组装：[lang,event,emo,textnorm] 4 个 query 行 + LFR 帧，整体乘 √512，
//!   加正弦 PE（位置从 1 开始，前 280 列 sin、后 280 列 cos，
//!   `its[j] = exp(-j·ln(10000)/279)`）
//!
//! fbank 数值已与 kaldi-native-fbank（venv 实测）对拍一致；单测内嵌该对拍 fixture。

use crate::model_spec::sensevoice::{SenseVoiceConfig, SenseVoiceLanguage, SenseVoiceTextNorm};
use crate::weight::model::sensevoice::SenseVoiceCmvn;

/// 预计算的窗、mel 滤波器组与 FFT 旋转因子。
pub struct FbankState {
    window: Vec<f32>,
    banks: Vec<f32>,
    /// 旋转因子表（按 FFT 蝶形级数展开），`cos/sin` 交错存放。
    twiddles: Vec<f32>,
}

impl FbankState {
    pub fn new(config: &SenseVoiceConfig) -> Self {
        let frame_length = config.frame_length();
        // kaldi 对称 hamming：0.54 - 0.46·cos(2πn/(N-1))
        let window = (0..frame_length).map(|n| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / (frame_length - 1) as f32).cos()).collect();
        let banks = mel_filterbanks(config);
        let twiddles = fft_twiddles(config.fft_length);
        Self { window, banks, twiddles }
    }

    /// 提取 fbank 特征，输出 `[rows][mel_bins]` 行优先。
    /// `samples` 必须是 int16 尺度（FunASR `upscale_samples`：[-1,1] × 32768）。
    pub fn compute(&self, config: &SenseVoiceConfig, samples: &[f32]) -> Result<Vec<f32>, String> {
        let frame_length = config.frame_length();
        let frame_shift = config.frame_shift();
        if samples.len() < frame_length {
            return Err(format!("SenseVoice 输入样本 {} 少于单帧 {}", samples.len(), frame_length));
        }
        let rows = (samples.len() - frame_length) / frame_shift + 1;
        let mut frames = vec![0.0f32; rows * config.mel_bins];
        let mut windowed = vec![0.0f32; config.fft_length];
        for row in 0..rows {
            let start = row * frame_shift;
            let frame = &samples[start..start + frame_length];
            windowed[..frame_length].copy_from_slice(frame);
            process_window(config, &self.window, &mut windowed[..frame_length]);
            let power = real_fft_power(config.fft_length, &windowed, &self.twiddles);
            let target = &mut frames[row * config.mel_bins..(row + 1) * config.mel_bins];
            for (bin, output) in target.iter_mut().enumerate() {
                let bank = &self.banks[bin * (config.fft_length / 2 + 1)..(bin + 1) * (config.fft_length / 2 + 1)];
                let energy = bank.iter().zip(&power).map(|(&weight, &value)| weight * value).sum::<f32>();
                *output = energy.max(f32::EPSILON).ln();
            }
        }
        Ok(frames)
    }
}

/// 帧内预处理：去直流 → kaldi preemphasis（原地）。FFT 长度以外的样本保持 0。
fn process_window(config: &SenseVoiceConfig, window: &[f32], frame: &mut [f32]) {
    let mean = frame.iter().sum::<f32>() / frame.len() as f32;
    for value in frame.iter_mut() {
        *value -= mean;
    }
    let coefficient = config.preemphasis;
    for index in (1..frame.len()).rev() {
        frame[index] -= coefficient * frame[index - 1];
    }
    frame[0] -= coefficient * frame[0];
    for (value, &weight) in frame.iter_mut().zip(window) {
        *value *= weight;
    }
}

/// kaldi mel 三角滤波器组：`[mel_bins][fft_length/2+1]` 行优先。
///
/// bin b 的三角在 mel 域由 `(b, b+1, b+2) × mel_freq_delta` 构成（mel_low 起算）；
/// 对每个 FFT bin 求其 mel 频率落在三角内的线性权重。
fn mel_filterbanks(config: &SenseVoiceConfig) -> Vec<f32> {
    let fft_bins = config.fft_length / 2 + 1;
    let bin_width = config.sample_rate as f32 / config.fft_length as f32;
    let high = if config.mel_high_hz <= 0.0 { 0.5 * config.sample_rate as f32 } else { config.mel_high_hz };
    let mel = |frequency: f32| 1127.0 * (1.0 + frequency / 700.0).ln();
    let mel_low = mel(config.mel_low_hz);
    let mel_high = mel(high);
    let delta = (mel_high - mel_low) / (config.mel_bins as f32 + 1.0);
    let mut banks = vec![0.0f32; config.mel_bins * fft_bins];
    for bin in 0..config.mel_bins {
        let left = mel_low + bin as f32 * delta;
        let center = left + delta;
        let right = center + delta;
        let row = &mut banks[bin * fft_bins..(bin + 1) * fft_bins];
        for (index, weight) in row.iter_mut().enumerate() {
            let value = mel(bin_width * index as f32);
            if value > left && value < right {
                *weight = if value <= center { (value - left) / (center - left) } else { (right - value) / (right - center) };
            }
        }
    }
    banks
}

/// 展开的旋转因子表：蝶形级数 s（长度 2^s 的蝶形）的 cos/sin 依次存放。
fn fft_twiddles(length: usize) -> Vec<f32> {
    let mut twiddles = Vec::new();
    let mut span = 2;
    while span <= length {
        for index in 0..span / 2 {
            let angle = -2.0 * std::f32::consts::PI * index as f32 / span as f32;
            twiddles.push(angle.cos());
            twiddles.push(angle.sin());
        }
        span *= 2;
    }
    twiddles
}

/// 实输入基-2 FFT，返回功率谱 `[length/2+1]`。`input` 长度必须等于 `length`。
fn real_fft_power(length: usize, input: &[f32], twiddles: &[f32]) -> Vec<f32> {
    debug_assert!(input.len() == length);
    let mut real = input.to_vec();
    let mut imag = vec![0.0f32; length];
    // 位反转重排
    let bits = length.trailing_zeros();
    for index in 0..length {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);
        if reversed > index {
            real.swap(index, reversed);
            imag.swap(index, reversed);
        }
    }
    // 蝶形迭代，逐级消耗 twiddles
    let mut twiddle_base = 0;
    let mut span = 2;
    while span <= length {
        let half = span / 2;
        let mut offset = 0;
        while offset < length {
            let mut pair = 0;
            while pair < half {
                let twiddle = twiddle_base + pair * 2;
                let (cosine, sine) = (twiddles[twiddle], twiddles[twiddle + 1]);
                let left = offset + pair;
                let right = left + half;
                let product_real = real[right] * cosine - imag[right] * sine;
                let product_imag = real[right] * sine + imag[right] * cosine;
                real[right] = real[left] - product_real;
                imag[right] = imag[left] - product_imag;
                real[left] += product_real;
                imag[left] += product_imag;
                pair += 1;
            }
            offset += span;
        }
        twiddle_base += half * 2;
        span *= 2;
    }
    (0..=length / 2).map(|bin| real[bin] * real[bin] + imag[bin] * imag[bin]).collect()
}

/// LFR 帧拼接：首帧前补 `(window-1)/2` 份，步长 `stride`、窗口 `window`；
/// 末尾以重复末帧补齐，保证输出 `ceil(rows/stride)` 行。
pub fn apply_lfr(config: &SenseVoiceConfig, frames: &[f32], frame_rows: usize) -> Result<(Vec<f32>, usize), String> {
    let window = config.lfr_window;
    let stride = config.lfr_stride;
    let width = config.mel_bins;
    if frames.len() != frame_rows * width {
        return Err(format!("SenseVoice LFR 输入元素 {} 与 rows({frame_rows})×{width} 不符", frames.len()));
    }
    let rows = frame_rows.div_ceil(stride);
    let left_padding = (window - 1) / 2;
    let padded_rows = left_padding + frame_rows;
    let required = (rows - 1) * stride + window;
    let total = required.max(padded_rows);
    let mut padded = Vec::with_capacity(total * width);
    padded.extend_from_slice(&frames[..width]);
    for _ in 1..left_padding {
        padded.extend_from_slice(&frames[..width]);
    }
    padded.extend_from_slice(frames);
    while padded.len() / width < required {
        let last = padded[padded.len() - width..].to_vec();
        padded.extend_from_slice(&last);
    }
    let dim = width * window;
    let mut output = Vec::with_capacity(rows * dim);
    for row in 0..rows {
        let start = row * stride;
        output.extend_from_slice(&padded[start * width..(start + window) * width]);
    }
    Ok((output, rows))
}

/// LFR 帧逐维 CMVN（原地）。
pub fn apply_cmvn(cmvn: &SenseVoiceCmvn, frames: &mut [f32]) -> Result<(), String> {
    if cmvn.shift.len() != cmvn.scale.len() {
        return Err(format!("SenseVoice CMVN shift {} 与 scale {} 维度不一致", cmvn.shift.len(), cmvn.scale.len()));
    }
    for row in frames.chunks_exact_mut(cmvn.shift.len()) {
        for (value, (&shift, &scale)) in row.iter_mut().zip(cmvn.shift.iter().zip(&cmvn.scale)) {
            *value = (*value + shift) * scale;
        }
    }
    Ok(())
}

/// 组装 encoder 输入：`[lang, event(1), emo(2), textnorm]` 4 个 query 行 + LFR 帧，
/// 整体乘 √d_model，加位置从 1 开始的正弦 PE。返回 `[rows+4][input_dim]` 行优先。
pub fn assemble_encoder_input(config: &SenseVoiceConfig, query_table: &[f32], language: SenseVoiceLanguage, textnorm: SenseVoiceTextNorm, frames: &[f32], frame_rows: usize) -> Result<Vec<f32>, String> {
    let dim = config.encoder_input_dim();
    if query_table.len() != config.query_embed_count * dim {
        return Err(format!("SenseVoice query 表 {} 元素，期望 {}×{dim}", query_table.len(), config.query_embed_count));
    }
    if frames.len() != frame_rows * dim {
        return Err(format!("SenseVoice LFR 帧 {} 元素与 rows({frame_rows})×{dim} 不符", frames.len()));
    }
    let rows = frame_rows + config.query_prefix;
    let mut output = Vec::with_capacity(rows * dim);
    // 输入顺序（FunASR encode()）：[lang, event, emo, textnorm, frames...]
    let query_rows = [language.query_index(), 1, 2, textnorm.query_index()];
    for &query in &query_rows {
        output.extend_from_slice(&query_table[query * dim..(query + 1) * dim]);
    }
    output.extend_from_slice(frames);

    let scale = (config.d_model as f32).sqrt();
    let half = dim / 2;
    let log_timescale_step = 10_000f32.ln() / (half as f32 - 1.0);
    for row in 0..rows {
        let target = &mut output[row * dim..(row + 1) * dim];
        let position = (row + 1) as f32;
        for (index, value) in target.iter_mut().enumerate() {
            *value *= scale;
            let column = if index < half { index } else { index - half };
            let angle = position * (-(column as f32) * log_timescale_step).exp();
            *value += if index < half { angle.sin() } else { angle.cos() };
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_spec::sensevoice::SenseVoiceConfig;

    /// 与 kaldi-native-fbank（sherpa-onnx 的 SenseVoice 前端）对拍的 fixture：
    /// 确定性三音信号（f64 计算后 cast f32，×32768），
    /// 期望值由 knf（dither=0）实测生成 /tmp/sv/gen_fixture.py。
    #[test]
    fn fbank_matches_kaldi_native_fbank() {
        let config = SenseVoiceConfig::standard();
        let state = FbankState::new(&config);
        let samples: Vec<f32> = (0..2720)
            .map(|t| {
                let t = t as f64;
                let value = 0.5 * (2.0 * std::f64::consts::PI * 440.0 * t / 16000.0).sin() + 0.25 * (2.0 * std::f64::consts::PI * 1234.5 * t / 16000.0).sin() + 0.1 * (2.0 * std::f64::consts::PI * 3000.0 * t / 16000.0).sin();
                (value * 32768.0) as f32
            })
            .collect();
        let frames = state.compute(&config, &samples).unwrap();
        assert_eq!(frames.len() / config.mel_bins, 15);
        let expected: [(usize, [f32; 12]); 4] = [
            (0, [13.798299, 14.525374, 14.701557, 14.304954, 13.905722, 13.637063, 14.485916, 15.354865, 15.413881, 14.164124, 13.434804, 15.017626]),
            (1, [14.682981, 15.593143, 15.387144, 14.335776, 13.548647, 14.320696, 15.536963, 15.994514, 15.464849, 12.272886, 14.489904, 14.134635]),
            (7, [13.876390, 14.787746, 14.552106, 13.435379, 12.740306, 13.700094, 14.862075, 15.263745, 14.604184, 10.243785, 13.845453, 14.195386]),
            (14, [14.200655, 15.135700, 14.963826, 13.954972, 13.124979, 13.824815, 15.106249, 15.632879, 15.192304, 12.187090, 14.106235, 15.257939]),
        ];
        for (row, values) in expected {
            let frame = &frames[row * config.mel_bins..];
            for (actual, expected) in frame.iter().zip(values) {
                assert!((actual - expected).abs() < 0.05, "frame {row} actual={actual} expected={expected}");
            }
        }
    }

    /// LFR：7 帧输入 → 首帧前补 3、末尾重复末帧 3，共 ceil(7/6)=2 行。
    #[test]
    fn lfr_pads_left_with_first_and_repeats_tail() {
        let config = SenseVoiceConfig::standard();
        let width = config.mel_bins;
        let frames: Vec<f32> = (0..7 * width).map(|index| index as f32).collect();
        let (output, rows) = apply_lfr(&config, &frames, 7).unwrap();
        assert_eq!(rows, 2);
        let dim = config.encoder_input_dim();
        assert_eq!(output.len(), rows * dim);
        // 第 0 行 = [f0×3(左补), f0, f1, f2, f3]
        assert_eq!(&output[..width], &frames[..width]);
        assert_eq!(&output[width..2 * width], &frames[..width]);
        assert_eq!(&output[3 * width..5 * width], &frames[..2 * width]);
        // 第 1 行 = [f3, f4, f5, f6, f6×3(右补)]
        assert_eq!(&output[dim..dim + 4 * width], &frames[3 * width..7 * width]);
        for repeat in 0..3 {
            assert_eq!(&output[(dim + (4 + repeat) as usize * width)..][..width], &frames[6 * width..7 * width]);
        }
    }

    /// 正弦 PE：位置从 1 起，前半 sin、后半 cos，its[j]=exp(-j·ln(1e4)/279)。
    #[test]
    fn sinusoidal_position_encoding_layout() {
        let config = SenseVoiceConfig::standard();
        let dim = config.encoder_input_dim();
        let query_table = vec![0.0f32; config.query_embed_count * dim];
        let frames = vec![1.0f32; 6 * dim];
        let output = assemble_encoder_input(&config, &query_table, SenseVoiceLanguage::Zh, SenseVoiceTextNorm::WithoutItn, &frames, 6).unwrap();
        assert_eq!(output.len(), (6 + 4) * dim);
        let scale = (config.d_model as f32).sqrt();
        let half = dim / 2;
        let step = 10_000f32.ln() / (half as f32 - 1.0);
        // 行 0 位置 1：sin(its[0]) = sin(1.0)
        assert!((output[0] - (0.0 * scale + 1.0f32.sin())).abs() < 1e-5);
        // 行 4（首个 LFR 帧，value=scale 后加 PE）
        let position = 5.0f32;
        let j = 13;
        let angle = position * (-(j as f32) * step).exp();
        assert!((output[4 * dim + j] - (scale + angle.sin())).abs() < 1e-4);
        assert!((output[4 * dim + half + j] - (scale + angle.cos())).abs() < 1e-4);
    }
}
