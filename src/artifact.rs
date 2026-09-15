//! 通用推理产物文件编解码：裸 F32、Y4M 视频与 WAV 音频。

use std::{
    fs,
    fs::File,
    io::{BufWriter, Read, Write},
    path::Path,
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
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

fn yuv(rgb: [f32; 3]) -> [f32; 3] {
    let [red, green, blue] = rgb;
    [16.0 + 65.481 * red + 128.553 * green + 24.966 * blue, 128.0 - 37.797 * red - 74.203 * green + 112.0 * blue, 128.0 + 112.0 * red - 93.786 * green - 18.214 * blue]
}

fn validate_y4m_video(video: &[f32], frames: usize, decoded_frames: usize, height: usize, width: usize) -> Result<(), String> {
    if frames == 0 || frames > decoded_frames || height == 0 || width == 0 || !height.is_multiple_of(2) || !width.is_multiple_of(2) {
        return Err(format!("Y4M shape frames={frames}/{decoded_frames} height={height} width={width} 非法"));
    }
    let expected = 3usize.checked_mul(decoded_frames).and_then(|value| value.checked_mul(height)).and_then(|value| value.checked_mul(width)).ok_or("Y4M 元素数溢出")?;
    if video.len() != expected {
        return Err(format!("Y4M video elements={}，期望 {expected}", video.len()));
    }
    Ok(())
}

/// 将 channel-major F32 分批写成 24fps、limited-range BT.601、C420jpeg 视频。
/// 每批可含多帧；调用方负责按时间顺序提交完成融合的帧，decoded_frames 是本批的通道步长。
/// 只保留一帧 YUV，Write 的同步写入直接提供背压，不缓存尚未消费的视频。
pub struct Y4mWriter<W: Write> {
    output: W,
    height: usize,
    width: usize,
    frame: usize,
    planes: Vec<u8>,
    rgb_scale: [f32; 3],
    rgb_bias: [f32; 3],
}

impl<W: Write> Y4mWriter<W> {
    pub fn new(output: W, height: usize, width: usize) -> Result<Self, String> {
        Self::with_rgb_transform(output, height, width, [0.229, 0.224, 0.225], [0.485, 0.456, 0.406])
    }

    /// RGB 反归一化由调用方指定，使不同模型共享同一流式编码路径。
    pub fn with_rgb_transform(mut output: W, height: usize, width: usize, rgb_scale: [f32; 3], rgb_bias: [f32; 3]) -> Result<Self, String> {
        if rgb_scale.iter().chain(&rgb_bias).any(|v| !v.is_finite()) {
            return Err("Y4M RGB 变换必须为有限数".to_owned());
        }
        if height == 0 || width == 0 || !height.is_multiple_of(2) || !width.is_multiple_of(2) {
            return Err(format!("Y4M shape height={height} width={width} 非法"));
        }
        let pixels = height.checked_mul(width).ok_or("Y4M 像素数溢出")?;
        let bytes = pixels.checked_add(pixels / 2).ok_or("Y4M 帧大小溢出")?;
        writeln!(output, "YUV4MPEG2 W{width} H{height} F24:1 Ip A1:1 C420jpeg").map_err(|error| format!("写入 Y4M header: {error}"))?;
        Ok(Self { output, height, width, frame: 0, planes: vec![0; bytes], rgb_scale, rgb_bias })
    }

    pub fn write_frames(&mut self, video: &[f32], frames: usize, decoded_frames: usize) -> Result<(), String> {
        let Self { output, height, width, frame: written, planes, rgb_scale, rgb_bias } = self;
        let (rgb_scale, rgb_bias) = (*rgb_scale, *rgb_bias);
        let (height, width) = (*height, *width);
        validate_y4m_video(video, frames, decoded_frames, height, width)?;
        let pixels = height * width;
        let channel_stride = decoded_frames * pixels;
        let (y_plane, uv) = planes.split_at_mut(pixels);
        let (u_plane, v_plane) = uv.split_at_mut(pixels / 4);
        let workers = thread::available_parallelism().map_or(1, |count| count.get()).min(4).min(height / 2).min((pixels / 65536).max(1));
        for frame in 0..frames {
            let offset = frame * pixels;
            let channels = std::array::from_fn::<_, 3, _>(|channel| &video[channel * channel_stride + offset..channel * channel_stride + offset + pixels]);
            // 限制转换并发，为编码器留出 CPU；小帧直接执行，避免建线程的固定开销。
            if workers == 1 {
                encode_y4m_rows(channels, width, y_plane, u_plane, v_plane, rgb_scale, rgb_bias);
            } else {
                let rows = (height / 2).div_ceil(workers) * 2;
                thread::scope(|scope| {
                    for (part, ((y, u), v)) in y_plane.chunks_mut(rows * width).zip(u_plane.chunks_mut(rows * width / 4)).zip(v_plane.chunks_mut(rows * width / 4)).enumerate() {
                        let channels = channels.map(|channel| &channel[part * rows * width..part * rows * width + y.len()]);
                        scope.spawn(move || encode_y4m_rows(channels, width, y, u, v, rgb_scale, rgb_bias));
                    }
                });
            }
            output.write_all(b"FRAME\n").and_then(|_| output.write_all(y_plane)).and_then(|_| output.write_all(u_plane)).and_then(|_| output.write_all(v_plane)).map_err(|error| format!("写入 Y4M frame {written}: {error}"))?;
            *written += 1;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        self.output.flush().map_err(|error| format!("刷新 Y4M: {error}"))
    }
}

fn encode_y4m_rows(channels: [&[f32]; 3], width: usize, y_plane: &mut [u8], u_plane: &mut [u8], v_plane: &mut [u8], rgb_scale: [f32; 3], rgb_bias: [f32; 3]) {
    // 同一个 2×2 块一次完成反归一化、Y 和 UV；保留原先 UV 累加次序与截断规则。
    for row in (0..y_plane.len() / width).step_by(2) {
        for column in (0..width).step_by(2) {
            let mut u = 0.0;
            let mut v = 0.0;
            for index in [row * width + column, row * width + column + 1, (row + 1) * width + column, (row + 1) * width + column + 1] {
                let rgb = std::array::from_fn(|channel| (channels[channel][index] * rgb_scale[channel] + rgb_bias[channel]).clamp(0.0, 1.0));
                let pixel = yuv(rgb);
                y_plane[index] = pixel[0].clamp(0.0, 255.0) as u8;
                u += pixel[1];
                v += pixel[2];
            }
            let index = row / 2 * (width / 2) + column / 2;
            u_plane[index] = (u * 0.25).clamp(0.0, 255.0) as u8;
            v_plane[index] = (v * 0.25).clamp(0.0, 255.0) as u8;
        }
    }
}

pub fn write_y4m(path: &Path, video: &[f32], frames: usize, decoded_frames: usize, height: usize, width: usize) -> Result<(), String> {
    validate_y4m_video(video, frames, decoded_frames, height, width)?;
    create_parent(path)?;
    let output = BufWriter::new(File::create(path).map_err(|error| format!("创建 {}: {error}", path.display()))?);
    let write = || {
        let mut writer = Y4mWriter::new(output, height, width)?;
        writer.write_frames(video, frames, decoded_frames)?;
        writer.flush()
    };
    write().map_err(|error| format!("写入 {}: {error}", path.display()))
}

/// 向编码进程的 stdin 输送 Y4M；调用方配置输出路径、音频输入和编码质量。
/// 写入线程受管道背压约束，当前线程在写入或编码阻塞时仍可响应取消并回收子进程。
pub fn pipe_y4m(command: &mut Command, video: &[f32], frames: usize, decoded_frames: usize, height: usize, width: usize, cancellation: &AtomicBool) -> Result<(), String> {
    validate_y4m_video(video, frames, decoded_frames, height, width)?;
    if cancellation.load(Ordering::Acquire) {
        return Err("Y4M 编码已取消".to_owned());
    }
    let mut child = command.stdin(Stdio::piped()).spawn().map_err(|error| format!("启动 Y4M 编码进程: {error}"))?;
    let input = child.stdin.take().expect("编码进程 stdin 已配置为 piped");
    thread::scope(|scope| {
        let writer = thread::Builder::new().spawn_scoped(scope, move || {
            let mut writer = Y4mWriter::new(input, height, width)?;
            writer.write_frames(video, frames, decoded_frames)?;
            writer.flush()
        });
        let mut writer = match writer {
            Ok(writer) => Some(writer),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("启动 Y4M 写入线程: {error}"));
            }
        };
        let join = |writer: thread::ScopedJoinHandle<'_, Result<(), String>>| writer.join().map_err(|_| "Y4M 写入线程异常退出".to_owned()).and_then(|result| result);
        let mut written = None;
        let result = loop {
            if cancellation.load(Ordering::Acquire) {
                break Err("Y4M 编码已取消".to_owned());
            }
            if writer.as_ref().is_some_and(|writer| writer.is_finished()) {
                written = Some(join(writer.take().expect("写入线程存在")));
                if let Some(Err(error)) = &written {
                    break Err(error.clone());
                }
            }
            match child.try_wait() {
                Ok(Some(status)) => break if status.success() { Ok(()) } else { Err(format!("Y4M 编码进程退出状态 {status}")) },
                Err(error) => break Err(format!("查询 Y4M 编码进程: {error}")),
                Ok(None) => thread::sleep(Duration::from_millis(10)),
            }
        };
        // 先终止消费者，解除可能阻塞的 pipe write，再 join；所有正常/错误出口都 wait，避免僵尸进程。
        if result.is_err() {
            let _ = child.kill();
        }
        let waited = child.wait().map_err(|error| format!("回收 Y4M 编码进程: {error}"));
        let written = written.unwrap_or_else(|| join(writer.take().expect("写入线程存在")));
        result.and(waited.map(|_| ())).and(written)
    })
}

/// 从 channel-major 视频张量抽取一张小尺寸 JPEG，供任务执行中回传真实缩略图。
/// `available_frames` 是已经完成的帧数，`frame_stride` 是张量每个通道的总帧跨度。
pub fn video_preview_jpeg(video: &[f32], frame: usize, available_frames: usize, frame_stride: usize, height: usize, width: usize) -> Result<Vec<u8>, String> {
    if frame >= available_frames || available_frames > frame_stride {
        return Err(format!("预览帧={frame} 超出已解码帧={available_frames}/跨度={frame_stride}"));
    }
    validate_y4m_video(video, frame.saturating_add(1), frame_stride, height, width)?;
    let preview_width = width.min(320);
    let preview_height = (height * preview_width / width).max(1);
    let pixels = height.checked_mul(width).ok_or("预览图像素数溢出")?;
    let channel_stride = frame_stride.checked_mul(pixels).ok_or("预览图通道跨度溢出")?;
    let offset = frame.checked_mul(pixels).ok_or("预览图帧偏移溢出")?;
    let mut rgb = vec![0u8; preview_width * preview_height * 3];
    for row in 0..preview_height {
        let source_row = row * height / preview_height;
        for column in 0..preview_width {
            let source_column = column * width / preview_width;
            let source = offset + source_row * width + source_column;
            let target = (row * preview_width + column) * 3;
            for channel in 0..3 {
                let value = video[channel * channel_stride + source] * [0.229, 0.224, 0.225][channel] + [0.485, 0.456, 0.406][channel];
                rgb[target + channel] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 72).encode(&rgb, preview_width as u32, preview_height as u32, image::ExtendedColorType::Rgb8).map_err(|error| format!("编码视频预览 JPEG: {error}"))?;
    Ok(jpeg)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    // 保留原来两遍像素转换作为 oracle，独立覆盖融合后仍须一致的舍入与 UV 顺序。
    fn reference(video: &[f32], frames: usize, decoded: usize, height: usize, width: usize) -> Vec<u8> {
        let mut output = format!("YUV4MPEG2 W{width} H{height} F24:1 Ip A1:1 C420jpeg\n").into_bytes();
        for frame in 0..frames {
            let pixel = |row: usize, column: usize| {
                let rgb = std::array::from_fn(|channel| {
                    let value = video[((channel * decoded + frame) * height + row) * width + column];
                    (value * [0.229, 0.224, 0.225][channel] + [0.485, 0.456, 0.406][channel]).clamp(0.0, 1.0)
                });
                let [r, g, b] = rgb;
                [16.0 + 65.481 * r + 128.553 * g + 24.966 * b, 128.0 - 37.797 * r - 74.203 * g + 112.0 * b, 128.0 + 112.0 * r - 93.786 * g - 18.214 * b]
            };
            output.extend_from_slice(b"FRAME\n");
            for row in 0..height {
                for column in 0..width {
                    output.push(pixel(row, column)[0].clamp(0.0, 255.0) as u8);
                }
            }
            for channel in 1..3 {
                for row in 0..height / 2 {
                    for column in 0..width / 2 {
                        let mut sum = 0.0;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                sum += pixel(row * 2 + dy, column * 2 + dx)[channel];
                            }
                        }
                        output.push((sum * 0.25).clamp(0.0, 255.0) as u8);
                    }
                }
            }
        }
        output
    }

    #[test]
    fn y4m_matches_reference_and_chunk_order() {
        for (width, height, decoded) in [(2, 2, 3), (10, 6, 7), (514, 258, 5)] {
            let pixels = width * height;
            let mut seed = 42u32;
            let mut video: Vec<f32> = (0..3 * decoded * pixels)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed as f32 / u32::MAX as f32 - 0.5) * 8.0
                })
                .collect();
            for (index, value) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0].into_iter().enumerate() {
                video[index] = value;
            }
            let expected = reference(&video, decoded - 1, decoded, height, width);
            let mut output = Vec::new();
            let mut writer = Y4mWriter::new(&mut output, height, width).unwrap();
            writer.write_frames(&video, decoded - 1, decoded).unwrap();
            writer.flush().unwrap();
            assert_eq!(output, expected);

            let mut chunks = Vec::new();
            let mut writer = Y4mWriter::new(&mut chunks, height, width).unwrap();
            for (start, count) in [(0, 1), (1, decoded - 2)] {
                let batch: Vec<f32> = (0..3).flat_map(|c| video[(c * decoded + start) * pixels..(c * decoded + start + count) * pixels].iter().copied()).collect();
                writer.write_frames(&batch, count, count).unwrap();
            }
            writer.flush().unwrap();
            assert_eq!(chunks, expected);
        }
    }

    #[test]
    fn y4m_color_range_and_header() {
        let mut video = Vec::new();
        for (mean, std) in [(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)] {
            for rgb in [0.0, 1.0] {
                video.extend_from_slice(&[(rgb - mean) / std; 4]);
            }
        }
        let mut output = Vec::new();
        let mut writer = Y4mWriter::new(&mut output, 2, 2).unwrap();
        writer.write_frames(&video, 2, 2).unwrap();
        let header = b"YUV4MPEG2 W2 H2 F24:1 Ip A1:1 C420jpeg\n";
        assert!(output.starts_with(header));
        // 白色 V 的旧公式截断为 127；保留原路径的浮点舍入，不能偷偷改成四舍五入。
        assert_eq!(&output[header.len()..], b"FRAME\n\x10\x10\x10\x10\x80\x80FRAME\n\xeb\xeb\xeb\xeb\x80\x7f");
    }

    #[test]
    fn y4m_rejects_invalid_shapes_and_reports_io_errors() {
        assert!(Y4mWriter::new(io::sink(), 0, 2).is_err());
        assert!(Y4mWriter::new(io::sink(), 3, 2).is_err());
        assert!(Y4mWriter::new(io::sink(), usize::MAX - 1, 2).is_err());
        let mut writer = Y4mWriter::new(io::sink(), 2, 2).unwrap();
        assert!(writer.write_frames(&[0.0; 12], 0, 1).is_err());
        assert!(writer.write_frames(&[0.0; 12], 2, 1).is_err());
        assert!(writer.write_frames(&[0.0; 11], 1, 1).is_err());
        assert!(writer.write_frames(&[], 1, usize::MAX).is_err());

        struct ShortWriter {
            bytes: usize,
            fail_at: usize,
        }
        impl Write for ShortWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.bytes >= self.fail_at {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "consumer exited"));
                }
                let count = bytes.len().min(2).min(self.fail_at - self.bytes);
                self.bytes += count;
                Ok(count)
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::other("flush failed"))
            }
        }
        let mut writer = Y4mWriter::new(ShortWriter { bytes: 0, fail_at: 45 }, 2, 2).unwrap();
        assert!(writer.write_frames(&[0.0; 12], 1, 1).unwrap_err().contains("frame 0"));
        let mut writer = Y4mWriter::new(ShortWriter { bytes: 0, fail_at: 100 }, 2, 2).unwrap();
        writer.write_frames(&[0.0; 12], 1, 1).unwrap();
        assert!(writer.flush().unwrap_err().contains("flush failed"));
    }

    #[test]
    fn y4m_symmetric_rgb_range_keeps_black_and_white() {
        let mut input = Vec::new();
        for _ in 0..3 {
            input.extend_from_slice(&[-1.0; 4]);
            input.extend_from_slice(&[1.0; 4]);
        }
        let mut output = Vec::new();
        Y4mWriter::with_rgb_transform(&mut output, 2, 2, [0.5; 3], [0.5; 3]).unwrap().write_frames(&input, 2, 2).unwrap();
        let header = b"YUV4MPEG2 W2 H2 F24:1 Ip A1:1 C420jpeg\n";
        assert_eq!(&output[header.len()..], b"FRAME\n\x10\x10\x10\x10\x80\x80FRAME\n\xeb\xeb\xeb\xeb\x80\x7f");
    }

    #[test]
    #[cfg(unix)]
    fn y4m_pipe_exit_and_cancellation() {
        let cancellation = AtomicBool::new(false);
        let run = |script: &str, cancellation: &AtomicBool| {
            let mut command = Command::new("sh");
            command.args(["-c", script]).stdout(Stdio::null()).stderr(Stdio::null());
            pipe_y4m(&mut command, &[0.0; 12], 1, 1, 2, 2, cancellation)
        };
        assert!(run("cat >/dev/null", &cancellation).is_ok());
        assert!(run("exit 7", &cancellation).is_err());
        assert!(pipe_y4m(&mut Command::new("/nonexistent-y4m-encoder"), &[0.0; 12], 1, 1, 2, 2, &cancellation).is_err());
        // exec 确保消费者就是被 kill/wait 的直接子进程，阻塞时 stdin 的 reader 不留在后代中。
        let start = std::time::Instant::now();
        thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(50));
                cancellation.store(true, Ordering::Release);
            });
            let mut command = Command::new("sh");
            command.args(["-c", "exec sleep 30"]).stdout(Stdio::null()).stderr(Stdio::null());
            let video = vec![0.0; 3 * 512 * 512];
            assert!(pipe_y4m(&mut command, &video, 1, 1, 512, 512, &cancellation).unwrap_err().contains("取消"));
        });
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(run("cat >/dev/null", &cancellation).unwrap_err().contains("取消"));
        cancellation.store(false, Ordering::Release);
        assert!(run("cat >/dev/null", &cancellation).is_ok());
    }
}
