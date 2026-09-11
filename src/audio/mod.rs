//! 音频输入解析：WAV 解码与 16kHz 重采样。
//!
//! 从 Gemma4 多模态路径提取为共享实现（SenseVoice ASR 前端在 Android 目标上
//! 同样需要 WAV 输入，不再局限于 macOS）。只支持无压缩 PCM16 / F32。

use std::path::Path;

/// 读取 WAV 文件并重采样为 mono 16kHz f32（幅度 [-1, 1]）。
pub fn read_wav_16khz(path: &Path) -> Result<Vec<f32>, String> {
    let data = std::fs::read(path).map_err(|error| format!("读取 WAV {} 失败: {error}", path.display()))?;
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(format!("{} 不是 RIFF/WAVE 文件", path.display()));
    }
    let mut cursor = 12;
    let mut format = None;
    let mut pcm = None;
    while cursor + 8 <= data.len() {
        let id = &data[cursor..cursor + 4];
        let size = u32::from_le_bytes(data[cursor + 4..cursor + 8].try_into().expect("WAV chunk size")) as usize;
        let start = cursor + 8;
        let end = start.checked_add(size).ok_or("WAV chunk 大小溢出")?;
        if end > data.len() {
            return Err("WAV chunk 超出文件".to_owned());
        }
        if id == b"fmt " && size >= 16 {
            format = Some((
                u16::from_le_bytes(data[start..start + 2].try_into().unwrap()),
                u16::from_le_bytes(data[start + 2..start + 4].try_into().unwrap()) as usize,
                u32::from_le_bytes(data[start + 4..start + 8].try_into().unwrap()) as usize,
                u16::from_le_bytes(data[start + 14..start + 16].try_into().unwrap()),
            ));
        } else if id == b"data" {
            pcm = Some(&data[start..end]);
        }
        cursor = end + (size & 1);
    }
    let (encoding, channels, sample_rate, bits) = format.ok_or("WAV 缺少 fmt chunk")?;
    if channels == 0 || sample_rate == 0 {
        return Err(format!("WAV sample_rate={sample_rate} channels={channels} 无效"));
    }
    let pcm = pcm.ok_or("WAV 缺少 data chunk")?;
    let samples = match (encoding, bits) {
        (1, 16) => pcm.chunks_exact(2).map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0).collect::<Vec<_>>(),
        (3, 32) => pcm.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>(),
        _ => return Err(format!("WAV encoding={encoding} bits={bits} 暂不支持，只支持 PCM16/F32")),
    };
    if !samples.len().is_multiple_of(channels) {
        return Err("WAV sample 数不能整除 channels".to_owned());
    }
    let mono = samples.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32).collect::<Vec<f32>>();
    Ok(resample_to_16khz(&mono, sample_rate))
}

/// 线性插值上采样 / 重叠平均下采样到 16kHz。
pub fn resample_to_16khz(samples: &[f32], sample_rate: usize) -> Vec<f32> {
    const TARGET_RATE: usize = 16_000;
    if samples.is_empty() || sample_rate == TARGET_RATE {
        return samples.to_vec();
    }
    let output_len = samples.len().saturating_mul(TARGET_RATE).div_ceil(sample_rate);
    if sample_rate < TARGET_RATE {
        return (0..output_len)
            .map(|index| {
                let position = index as f64 * sample_rate as f64 / TARGET_RATE as f64;
                let left = position.floor() as usize;
                let right = (left + 1).min(samples.len() - 1);
                let fraction = (position - left as f64) as f32;
                samples[left] + (samples[right] - samples[left]) * fraction
            })
            .collect();
    }

    let scale = sample_rate as f64 / TARGET_RATE as f64;
    (0..output_len)
        .map(|index| {
            let start = index as f64 * scale;
            let end = ((index + 1) as f64 * scale).min(samples.len() as f64);
            let mut weighted_sum = 0.0f64;
            for source in start.floor() as usize..end.ceil() as usize {
                let overlap = (end.min((source + 1) as f64) - start.max(source as f64)).max(0.0);
                if overlap > 0.0 {
                    weighted_sum += samples[source] as f64 * overlap;
                }
            }
            (weighted_sum / (end - start)) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::resample_to_16khz;

    #[test]
    fn downsample_48khz_averages_each_target_interval() {
        let output = resample_to_16khz(&[0.0, 3.0, 6.0, 9.0, 12.0, 15.0], 48_000);
        assert_eq!(output, vec![3.0, 12.0]);
    }
}
