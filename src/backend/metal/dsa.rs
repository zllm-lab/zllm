//! Metal DSA 的 key archive、Top-K scratch 与跨层选择状态。

use std::{mem, path::Path};

use crate::backend::metal::api::Buffer;
use half::f16;

use crate::kernel::metal as ops;

use super::{MetalContext, MetalTensor};

const DSA_LAYOUT: &str = "prefix-rope-f16-v1\n";

fn migrate_tail_rope_archive(bytes: &mut [u8], rows: usize, head_dim: usize, rope_dim: usize, theta: f32) -> Result<(), String> {
    if head_dim != rope_dim * 2 || !rope_dim.is_multiple_of(2) {
        return Err(format!("旧 DSA archive 迁移只支持 head_dim=2*rope_dim，实际 {head_dim}/{rope_dim}"));
    }
    let half = rope_dim / 2;
    let read = |data: &[u8], index: usize| f16::from_bits(u16::from_le_bytes([data[index * 2], data[index * 2 + 1]])).to_f32();
    for row in 0..rows {
        let row_begin = row * head_dim * 2;
        let row_bytes = &mut bytes[row_begin..row_begin + head_dim * 2];
        let mut raw = vec![0.0f32; head_dim];
        for (index, value) in raw.iter_mut().take(rope_dim).enumerate() {
            *value = read(row_bytes, index);
        }
        let tail = head_dim - rope_dim;
        for pair in 0..half {
            let angle = row as f32 * theta.powf(-(pair as f32) / half as f32);
            let (cosine, sine) = (angle.cos(), angle.sin());
            let real = read(row_bytes, tail + pair);
            let imag = read(row_bytes, tail + half + pair);
            raw[tail + pair * 2] = real * cosine + imag * sine;
            raw[tail + pair * 2 + 1] = -real * sine + imag * cosine;
        }
        let mut output = vec![0.0f32; head_dim];
        for pair in 0..half {
            let angle = row as f32 * theta.powf(-(pair as f32) / half as f32);
            let (cosine, sine) = (angle.cos(), angle.sin());
            let real = raw[pair * 2];
            let imag = raw[pair * 2 + 1];
            output[pair] = real * cosine - imag * sine;
            output[half + pair] = imag * cosine + real * sine;
        }
        output[rope_dim..].copy_from_slice(&raw[rope_dim..]);
        for (index, value) in output.into_iter().enumerate() {
            let bits = f16::from_f32(value).to_bits().to_le_bytes();
            row_bytes[index * 2..index * 2 + 2].copy_from_slice(&bits);
        }
    }
    Ok(())
}

/// 旧 checkpoint 不含 indexer key 时保持无效，Backend 会明确回退 dense attention。
pub struct MetalDsaState {
    key_caches: Vec<Option<Buffer>>,
    lengths: Vec<usize>,
    ordered_scores: Buffer,
    selection: Buffer,
    radix_state: Buffer,
    capacity: usize,
    head_dim: usize,
    top_k: usize,
    selection_valid: bool,
    selection_rows: usize,
}

impl MetalDsaState {
    pub fn new(ctx: &MetalContext, layer_count: usize, capacity: usize, head_dim: usize, top_k: usize) -> Result<Self, String> {
        if layer_count == 0 || capacity == 0 || head_dim == 0 || top_k == 0 || top_k > capacity {
            return Err(format!("DSA state 参数非法: layers={layer_count} capacity={capacity} head_dim={head_dim} top_k={top_k}"));
        }
        Ok(Self {
            key_caches: (0..layer_count).map(|_| None).collect(),
            lengths: vec![0; layer_count],
            ordered_scores: ctx.shared_buffer_zeros(capacity * mem::size_of::<u32>()),
            selection: ctx.shared_buffer_zeros(top_k * mem::size_of::<u32>()),
            radix_state: ctx.shared_buffer_zeros(2 * mem::size_of::<u32>()),
            capacity,
            head_dim,
            top_k,
            selection_valid: false,
            selection_rows: 0,
        })
    }

    /// 从连续 raw F16 archive 恢复 indexer key；读取使用 `read`，不使用 mmap。
    pub fn load_checkpoint(&mut self, ctx: &MetalContext, dir: &Path, prompt_len: usize, layers: &[usize], rope_dim: usize, theta: f32) -> Result<usize, String> {
        if prompt_len > self.capacity {
            return Err(format!("DSA checkpoint prompt {prompt_len} 超过 capacity {}", self.capacity));
        }
        let expected = prompt_len.checked_mul(self.head_dim).and_then(|n| n.checked_mul(mem::size_of::<f16>())).ok_or("DSA checkpoint 大小溢出")?;
        let prefix_layout = std::fs::read_to_string(dir.join("indexer-layout.txt")).is_ok_and(|layout| layout == DSA_LAYOUT);
        let mut loaded = 0;
        for &layer in layers {
            if layer >= self.key_caches.len() {
                return Err(format!("DSA checkpoint layer {layer} 越界"));
            }
            let path = dir.join(format!("indexer-layer{layer:03}.f16le"));
            let mut bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(format!("读取 {}: {error}", path.display())),
            };
            if bytes.len() != expected {
                return Err(format!("{} 长度 {}，期望 {expected}", path.display(), bytes.len()));
            }
            if !prefix_layout {
                migrate_tail_rope_archive(&mut bytes, prompt_len, self.head_dim, rope_dim, theta)?;
            }
            let cache = ctx.shared_buffer_zeros(self.capacity * self.head_dim * mem::size_of::<f16>());
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), cache.contents().cast::<u8>(), bytes.len());
            }
            self.key_caches[layer] = Some(cache);
            self.lengths[layer] = prompt_len;
            loaded += 1;
        }
        Ok(loaded)
    }

    pub(crate) fn can_update(&mut self, layer: usize, position: usize, head_dim: usize, top_k: usize) -> bool {
        self.selection_valid = false;
        self.selection_rows = 0;
        head_dim == self.head_dim && top_k == self.top_k && position < self.capacity && self.lengths.get(layer).copied() == Some(position) && self.key_caches.get(layer).and_then(Option::as_ref).is_some()
    }

    pub(crate) fn append_key(&mut self, ctx: &MetalContext, layer: usize, position: usize, key: &MetalTensor) -> Result<(), String> {
        let cache = self.key_caches.get(layer).and_then(Option::as_ref).ok_or_else(|| format!("L{layer} DSA key cache 未加载"))?;
        ops::mla::dsa_store_keys(ctx, key, cache, position, self.capacity)?;
        self.lengths[layer] = position + 1;
        Ok(())
    }

    pub(crate) fn append_prefill_keys(&mut self, ctx: &MetalContext, layer: usize, keys: &MetalTensor) -> Result<(), String> {
        if layer >= self.key_caches.len() || keys.cols != self.head_dim || keys.rows > self.capacity {
            return Err(format!("L{layer} DSA prefill key 形状 [{},{}] 不符，capacity={} head_dim={}", keys.rows, keys.cols, self.capacity, self.head_dim));
        }
        if self.lengths[layer] != 0 {
            return Err(format!("L{layer} DSA prefill key 已存在 {} token", self.lengths[layer]));
        }
        let cache = ctx.shared_buffer_zeros(self.capacity * self.head_dim * mem::size_of::<f16>());
        ops::mla::dsa_store_keys(ctx, keys, &cache, 0, self.capacity)?;
        self.key_caches[layer] = Some(cache);
        self.lengths[layer] = keys.rows;
        Ok(())
    }

    /// 逐层 checkpoint 只提交当前层；manifest 由 session 层在本文件提交后更新。
    pub fn dump_layer_checkpoint(&self, dir: &Path, layer: usize) -> Result<usize, String> {
        let Some(cache) = self.key_caches.get(layer).ok_or_else(|| format!("DSA checkpoint layer {layer} 越界"))? else {
            return Ok(0);
        };
        std::fs::create_dir_all(dir).map_err(|error| format!("创建 {}: {error}", dir.display()))?;
        let bytes_len = self.lengths[layer] * self.head_dim * mem::size_of::<f16>();
        let bytes = unsafe { std::slice::from_raw_parts(cache.contents().cast::<u8>(), bytes_len) };
        let path = dir.join(format!("indexer-layer{layer:03}.f16le"));
        let tmp = dir.join(format!("indexer-layer{layer:03}.f16le.tmp"));
        std::fs::write(&tmp, bytes).map_err(|error| format!("写 {}: {error}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|error| format!("提交 {}: {error}", path.display()))?;
        std::fs::write(dir.join("indexer-layout.txt"), DSA_LAYOUT).map_err(|error| format!("写 indexer layout: {error}"))?;
        Ok(bytes_len)
    }

    pub(crate) fn select(&mut self, ctx: &MetalContext, layer: usize, query: &MetalTensor, head_weights: &MetalTensor, head_count: usize) -> Result<(), String> {
        let rows = self.lengths[layer];
        if rows <= self.top_k {
            return Ok(());
        }
        let cache = self.key_caches[layer].as_ref().ok_or_else(|| format!("L{layer} DSA key cache 未加载"))?;
        ops::mla::dsa_select_topk(ctx, query, cache, head_weights, &self.ordered_scores, &self.selection, &self.radix_state, rows, head_count, self.head_dim, self.top_k)?;
        self.selection_valid = true;
        self.selection_rows = 1;
        Ok(())
    }

    pub(crate) fn select_prefill(&mut self, ctx: &MetalContext, layer: usize, query: &MetalTensor, head_weights: &MetalTensor, head_count: usize) -> Result<(), String> {
        let rows = self.lengths[layer];
        if rows <= self.top_k {
            self.selection_valid = false;
            self.selection_rows = 0;
            return Ok(());
        }
        let cache = self.key_caches[layer].as_ref().ok_or_else(|| format!("L{layer} DSA key cache 未加载"))?;
        let bytes = rows.checked_mul(self.top_k).and_then(|n| n.checked_mul(mem::size_of::<u32>())).ok_or("DSA prefill selection 大小溢出")?;
        let selection = ctx.shared_buffer_zeros(bytes);
        ops::mla::dsa_select_prefill(ctx, query, cache, head_weights, &selection, rows, head_count, self.head_dim, self.top_k)?;
        self.selection = selection;
        self.selection_valid = true;
        self.selection_rows = rows;
        Ok(())
    }

    pub fn selection_valid(&self) -> bool {
        self.selection_valid
    }

    pub(crate) fn selection(&self) -> &Buffer {
        &self.selection
    }

    pub(crate) fn prefill_selection(&self, rows: usize) -> Option<&Buffer> {
        (self.selection_valid && self.selection_rows == rows).then_some(&self.selection)
    }

    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }
}
