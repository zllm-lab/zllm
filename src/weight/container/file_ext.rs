//! 跨平台预读抽象, 统一 GGUF / Safetensors / NVFP4 加载的 `read_exact_at` 调用。
//!
//! Linux / macOS 走 POSIX pread; Windows 走 OVERLAPPED I/O
//! (`std::os::windows::fs::FileExt::seek_read`)。两个实现都不修改 file 的共享
//! seek offset, 因此在多线程并行加载同一 shard 时安全。

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::sync::Arc;

/// 在指定偏移读取并填充 `buf`, 行为对齐 `std::io::Read::read_exact`:
/// 循环到填满 `buf` 或在 EOF 提前结束 (后者返回 `UnexpectedEof`)。
pub trait FileExt {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
}

#[cfg(unix)]
impl FileExt for File {
    #[inline]
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, offset)
    }
}

#[cfg(windows)]
impl FileExt for File {
    #[inline]
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        // Windows std 没有 read_exact_at; 这里循环 seek_read 直到填满或 EOF。
        // seek_read 内部用 ReadFile + OVERLAPPED, 不持有 file 共享的 seek offset。
        let mut filled = 0usize;
        while filled < buf.len() {
            let n = std::os::windows::fs::FileExt::seek_read(self, &mut buf[filled..], offset + filled as u64)?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "seek_read 提前返回 0 字节"));
            }
            filled += n;
        }
        Ok(())
    }
}

/// `Arc<File>` 转发到内部 `File` 的实现。zllm 加载路径把 file 包成 `Arc`
/// 共享给 cache + 读取方, 不能要求调用方再解引用。
impl FileExt for Arc<File> {
    #[inline]
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        self.deref().read_exact_at(buf, offset)
    }
}
