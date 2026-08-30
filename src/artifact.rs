//! 通用推理产物文件编解码：裸 F32、Y4M 视频与 WAV 音频。

use std::{
    fs,
    fs::File,
    io::{BufWriter, Read, Write},
    path::Path,
};

// 多个 H3 入口共享本文件，node 只使用编码函数。
#[allow(dead_code)]
pub fn finite_check(label: &str, values: &[f32]) -> Result<(), String> {
    let bad = values.iter().filter(|value| !value.is_finite()).count();
    let max_abs = values.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    println!("[h3] {label} finite={}/{} max_abs={max_abs:.4}", values.len() - bad, values.len());
    if bad == 0 { Ok(()) } else { Err(format!("{label} 含 {bad} 个非有限值(NaN/Inf)")) }
}

pub fn create_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|error| format!("创建 {}: {error}", parent.display()))?;
    }
    Ok(())
}

/// 读取 F32 little-endian 文件，校验元素数。
#[allow(dead_code)]
pub fn read_f32(path: &Path, elements: usize) -> Result<Vec<f32>, String> {
    let mut bytes = Vec::new();
    File::open(path).map_err(|error| format!("打开 {}: {error}", path.display()))?.read_to_end(&mut bytes).map_err(|error| format!("读取 {}: {error}", path.display()))?;
    if bytes.len() != elements.checked_mul(4).ok_or("F32 文件大小溢出")? {
        return Err(format!("{} bytes={}，期望 {}", path.display(), bytes.len(), elements * 4));
    }
    Ok(bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 chunk"))).collect())
}

/// 写入 F32 little-endian 文件。
#[allow(dead_code)]
pub fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    create_parent(path)?;
    let mut output = BufWriter::new(File::create(path).map_err(|error| format!("创建 {}: {error}", path.display()))?);
    for value in values {
        output.write_all(&value.to_le_bytes()).map_err(|error| format!("写入 {}: {error}", path.display()))?;
    }
    output.flush().map_err(|error| format!("刷新 {}: {error}", path.display()))
}

fn decoded_rgb(video: &[f32], decoded_frames: usize, height: usize, width: usize, frame: usize, row: usize, column: usize) -> [f32; 3] {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    std::array::from_fn(|channel| {
        let index = ((channel * decoded_frames + frame) * height + row) * width + column;
        (video[index] * STD[channel] + MEAN[channel]).clamp(0.0, 1.0)
    })
}

fn yuv(rgb: [f32; 3]) -> [f32; 3] {
    let [red, green, blue] = rgb;
    [16.0 + 65.481 * red + 128.553 * green + 24.966 * blue, 128.0 - 37.797 * red - 74.203 * green + 112.0 * blue, 128.0 + 112.0 * red - 93.786 * green - 18.214 * blue]
}

pub fn write_y4m(path: &Path, video: &[f32], frames: usize, decoded_frames: usize, height: usize, width: usize) -> Result<(), String> {
    if frames == 0 || frames > decoded_frames || height == 0 || width == 0 || !height.is_multiple_of(2) || !width.is_multiple_of(2) {
        return Err(format!("Y4M shape frames={frames}/{decoded_frames} height={height} width={width} 非法"));
    }
    let expected = 3usize.checked_mul(decoded_frames).and_then(|value| value.checked_mul(height)).and_then(|value| value.checked_mul(width)).ok_or("Y4M 元素数溢出")?;
    if video.len() != expected {
        return Err(format!("Y4M video elements={}，期望 {expected}", video.len()));
    }
    create_parent(path)?;
    let mut output = BufWriter::new(File::create(path).map_err(|error| format!("创建 {}: {error}", path.display()))?);
    writeln!(output, "YUV4MPEG2 W{width} H{height} F24:1 Ip A1:1 C420jpeg").map_err(|error| format!("写入 {}: {error}", path.display()))?;
    let mut y_plane = vec![0u8; height * width];
    let mut u_plane = vec![0u8; height * width / 4];
    let mut v_plane = vec![0u8; height * width / 4];
    for frame in 0..frames {
        for row in 0..height {
            for column in 0..width {
                y_plane[row * width + column] = yuv(decoded_rgb(video, decoded_frames, height, width, frame, row, column))[0].clamp(0.0, 255.0) as u8;
            }
        }
        for row in 0..height / 2 {
            for column in 0..width / 2 {
                let mut u = 0.0;
                let mut v = 0.0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let pixel = yuv(decoded_rgb(video, decoded_frames, height, width, frame, row * 2 + dy, column * 2 + dx));
                        u += pixel[1];
                        v += pixel[2];
                    }
                }
                let index = row * (width / 2) + column;
                u_plane[index] = (u * 0.25).clamp(0.0, 255.0) as u8;
                v_plane[index] = (v * 0.25).clamp(0.0, 255.0) as u8;
            }
        }
        output.write_all(b"FRAME\n").and_then(|_| output.write_all(&y_plane)).and_then(|_| output.write_all(&u_plane)).and_then(|_| output.write_all(&v_plane)).map_err(|error| format!("写入 {} frame {frame}: {error}", path.display()))?;
    }
    output.flush().map_err(|error| format!("刷新 {}: {error}", path.display()))
}

pub fn write_wav(path: &Path, audio: &[f32], channels: usize, available_samples: usize, samples: usize, sample_rate: usize) -> Result<(), String> {
    if channels == 0 || samples == 0 || samples > available_samples || audio.len() != channels.checked_mul(available_samples).ok_or("WAV 元素数溢出")? {
        return Err(format!("WAV shape channels={channels} samples={samples}/{available_samples} elements={} 非法", audio.len()));
    }
    let data_bytes = channels.checked_mul(samples).and_then(|value| value.checked_mul(2)).and_then(|value| u32::try_from(value).ok()).ok_or("WAV data 超过 RIFF u32")?;
    let channels_u16 = u16::try_from(channels).map_err(|_| "WAV channels 超过 u16".to_owned())?;
    let sample_rate_u32 = u32::try_from(sample_rate).map_err(|_| "WAV sample rate 超过 u32".to_owned())?;
    let block_align = channels_u16.checked_mul(2).ok_or("WAV block align 溢出")?;
    let byte_rate = sample_rate_u32.checked_mul(u32::from(block_align)).ok_or("WAV byte rate 溢出")?;
    create_parent(path)?;
    let mut output = BufWriter::new(File::create(path).map_err(|error| format!("创建 {}: {error}", path.display()))?);
    output
        .write_all(b"RIFF")
        .and_then(|_| output.write_all(&(36u32 + data_bytes).to_le_bytes()))
        .and_then(|_| output.write_all(b"WAVEfmt "))
        .and_then(|_| output.write_all(&16u32.to_le_bytes()))
        .and_then(|_| output.write_all(&1u16.to_le_bytes()))
        .and_then(|_| output.write_all(&channels_u16.to_le_bytes()))
        .and_then(|_| output.write_all(&sample_rate_u32.to_le_bytes()))
        .and_then(|_| output.write_all(&byte_rate.to_le_bytes()))
        .and_then(|_| output.write_all(&block_align.to_le_bytes()))
        .and_then(|_| output.write_all(&16u16.to_le_bytes()))
        .and_then(|_| output.write_all(b"data"))
        .and_then(|_| output.write_all(&data_bytes.to_le_bytes()))
        .map_err(|error| format!("写入 {} header: {error}", path.display()))?;
    for sample in 0..samples {
        for channel in 0..channels {
            let value = (audio[channel * available_samples + sample].clamp(-1.0, 1.0) * 32767.0) as i16;
            output.write_all(&value.to_le_bytes()).map_err(|error| format!("写入 {} sample {sample}: {error}", path.display()))?;
        }
    }
    output.flush().map_err(|error| format!("刷新 {}: {error}", path.display()))
}
