//! 视频文件解码与 Qwen 系 2 FPS 帧采样。
//!
//! 只负责文件到原始 RGB 帧的转换(ffmpeg/ffprobe 子进程)，不含模型算法；
//! H3 node 与 Qwen3.6 Metal 入口共用。

use std::{
    io::{self, Read},
    path::Path,
    process::{Command, Stdio},
};

use super::RgbImage;

const MAX_DECODED_VIDEO_BYTES: usize = 512 << 20;
const MAX_FFMPEG_ERROR_BYTES: usize = 64 << 10;

/// 用 ffprobe 读取视频首流的宽高。
pub fn video_dimensions(path: &Path) -> Result<(usize, usize), String> {
    let probe = Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0"]).arg(path).output().map_err(|error| format!("启动 ffprobe 失败: {error}"))?;
    if !probe.status.success() {
        return Err(format!("ffprobe {} 失败: {}", path.display(), String::from_utf8_lossy(&probe.stderr)));
    }
    let dimensions = String::from_utf8_lossy(&probe.stdout);
    let (width, height) = dimensions.trim().split_once('x').ok_or_else(|| format!("ffprobe 返回的视频尺寸 {:?} 无效", dimensions.trim()))?;
    let width = width.parse::<usize>().map_err(|error| format!("视频 width={width:?} 无效: {error}"))?;
    let height = height.parse::<usize>().map_err(|error| format!("视频 height={height:?} 无效: {error}"))?;
    if width == 0 || height == 0 {
        return Err("视频尺寸为 0".to_owned());
    }
    Ok((width, height))
}

/// 用 ffmpeg 解码为 24 FPS 的 RGB24 原始帧序列。
pub fn decode_video(path: &Path) -> Result<Vec<RgbImage>, String> {
    let (width, height) = video_dimensions(path)?;
    let frame_bytes = width.checked_mul(height).and_then(|value| value.checked_mul(3)).ok_or("视频 frame bytes 溢出")?;
    if frame_bytes == 0 {
        return Err("视频尺寸为 0".to_owned());
    }

    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-an", "-vf", "fps=24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("启动 ffmpeg 视频解码失败: {error}"))?;
    let mut stdout = child.stdout.take().ok_or("ffmpeg stdout pipe 缺失")?;
    let stderr = child.stderr.take().ok_or("ffmpeg stderr pipe 缺失")?;
    // stderr 必须并行排空，否则 ffmpeg 错误输出填满 pipe 后会与 stdout 读取互相等待。
    let stderr_reader = std::thread::spawn(move || read_bounded(stderr, MAX_FFMPEG_ERROR_BYTES));

    let frames = read_rgb_frames(&mut stdout, width, height, MAX_DECODED_VIDEO_BYTES);
    drop(stdout);
    if frames.is_err() {
        let _ = child.kill();
    }
    let status = child.wait().map_err(|error| format!("等待 ffmpeg 视频解码失败: {error}"))?;
    let stderr = stderr_reader.join().map_err(|_| "ffmpeg stderr reader panic".to_owned())?.map_err(|error| format!("读取 ffmpeg stderr 失败: {error}"))?;
    let frames = frames?;
    if !status.success() {
        return Err(format!("ffmpeg 视频解码 {} 失败: {}", path.display(), String::from_utf8_lossy(&stderr)));
    }
    if frames.is_empty() {
        return Err(format!("ffmpeg 视频解码 {} 没有输出帧", path.display()));
    }
    Ok(frames)
}

fn read_rgb_frames(reader: &mut impl Read, width: usize, height: usize, max_bytes: usize) -> Result<Vec<RgbImage>, String> {
    let frame_bytes = width.checked_mul(height).and_then(|value| value.checked_mul(3)).ok_or("视频 frame bytes 溢出")?;
    if frame_bytes == 0 || frame_bytes > max_bytes {
        return Err(format!("视频单帧 bytes={frame_bytes} 超过解码上限 {max_bytes}"));
    }
    let mut frames = Vec::new();
    let mut pixels = vec![0; frame_bytes];
    loop {
        let mut filled = 0;
        while filled < frame_bytes {
            match reader.read(&mut pixels[filled..]) {
                Ok(0) if filled == 0 => return Ok(frames),
                Ok(0) => return Err(format!("ffmpeg 视频末帧只有 {filled}/{frame_bytes} bytes")),
                Ok(read) => filled += read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(format!("读取 ffmpeg 视频帧失败: {error}")),
            }
        }
        let decoded_bytes = frames.len().checked_add(1).and_then(|count| count.checked_mul(frame_bytes)).ok_or("视频解码大小溢出")?;
        if decoded_bytes > max_bytes {
            return Err(format!("视频解码数据超过 {max_bytes} bytes 上限: frame_bytes={frame_bytes} frames>{}", max_bytes / frame_bytes));
        }
        frames.push(RgbImage { width, height, pixels });
        pixels = vec![0; frame_bytes];
    }
}

fn read_bounded(mut reader: impl Read, limit: usize) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(output);
        }
        let keep = read.min(limit.saturating_sub(output.len()));
        output.extend_from_slice(&buffer[..keep]);
    }
}

/// Qwen 视频约定:2 FPS 采样(24 FPS 输入 `step_by(12)`),帧数补成偶数以凑满
/// temporal_patch_size=2。返回 (采样帧, 每对帧的平均时间戳秒)。
pub fn qwen_video_frames(frames: &[RgbImage]) -> Result<(Vec<RgbImage>, Vec<f32>), String> {
    if frames.is_empty() {
        return Err("Qwen video 没有帧".to_owned());
    }
    let mut sampled = frames.iter().enumerate().step_by(12).map(|(index, image)| (index, image.clone())).collect::<Vec<_>>();
    if sampled.len() % 2 != 0 {
        let (index, image) = sampled.last().expect("sampled 至少一帧");
        sampled.push((*index, image.clone()));
    }
    let timestamps = sampled.chunks_exact(2).map(|pair| (pair[0].0 as f32 / 24.0 + pair[1].0 as f32 / 24.0) * 0.5).collect();
    Ok((sampled.into_iter().map(|(_, image)| image).collect(), timestamps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn rgb_frame_reader_streams_complete_frames() {
        let bytes = (0..18).collect::<Vec<_>>();
        let frames = read_rgb_frames(&mut Cursor::new(bytes), 2, 1, 18).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].pixels, [0, 1, 2, 3, 4, 5]);
        assert_eq!(frames[2].pixels, [12, 13, 14, 15, 16, 17]);
    }

    #[test]
    fn rgb_frame_reader_rejects_truncated_or_oversized_output() {
        assert!(read_rgb_frames(&mut Cursor::new(vec![0; 5]), 2, 1, 12).unwrap_err().contains("末帧"));
        assert!(read_rgb_frames(&mut Cursor::new(vec![0; 18]), 2, 1, 12).unwrap_err().contains("超过"));
    }

    #[test]
    fn bounded_reader_drains_but_only_keeps_limit() {
        assert_eq!(read_bounded(Cursor::new(vec![7; 32]), 5).unwrap(), vec![7; 5]);
    }
}
