use rayon::prelude::*;
use serde::Deserialize;
use std::{
    borrow::Cow,
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap},
    error::Error,
    fs, io,
    path::Path,
};
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Copy)]
pub struct Merge {
    pub rank: u32,
    pub id: u32,
}

const MERGE_ID_BITS: u32 = 18;
const MERGE_RANK_BITS: u32 = 19;
const MERGE_ID_LIMIT: u32 = 1 << MERGE_ID_BITS;
const MERGE_RANK_LIMIT: usize = 1 << MERGE_RANK_BITS;
const MERGE_RANK_MASK: u64 = (1 << MERGE_RANK_BITS) - 1;

pub struct Merges {
    entries: Box<[u64]>,
    ids: Box<[u32]>,
}

impl Merges {
    /// Entries must follow tokenizer.json merge order. Each item is
    /// `(left_id, right_id, merged_id)`; its position becomes the rank.
    pub fn new<I>(entries: I) -> io::Result<Self>
    where
        I: IntoIterator<Item = (u32, u32, u32)>,
    {
        let mut ranked = Vec::new();
        for (rank, (left, right, id)) in entries.into_iter().enumerate() {
            let rank = u32::try_from(rank).map_err(|_| invalid_data("merge rank 超出 u32".to_owned()))?;
            ranked.push((left, right, id, rank));
        }
        Self::from_ranked(ranked)
    }

    /// tiktoken 只保存合并后 token 的 rank；同一 token 的每个有效二分都使用该 rank。
    fn from_ranked<I>(entries: I) -> io::Result<Self>
    where
        I: IntoIterator<Item = (u32, u32, u32, u32)>,
    {
        let entries: Vec<(u32, u32, u32, u32)> = entries.into_iter().collect();

        let mut packed = Vec::with_capacity(entries.len());

        for &(left, right, id, rank) in &entries {
            if left >= MERGE_ID_LIMIT || right >= MERGE_ID_LIMIT || id >= MERGE_ID_LIMIT {
                return Err(invalid_data(format!("merge token id 超过 18 bits: left={left} right={right} merged={id}")));
            }
            if rank as usize >= MERGE_RANK_LIMIT {
                return Err(invalid_data(format!("merge rank 超过 19 bits: {rank}")));
            }
            packed.push(((merge_pair(left, right) << MERGE_RANK_BITS) | rank as u64, id));
        }

        packed.sort_unstable_by_key(|entry| entry.0 >> MERGE_RANK_BITS);
        if let Some(pair) = packed.windows(2).find(|pair| pair[0].0 >> MERGE_RANK_BITS == pair[1].0 >> MERGE_RANK_BITS) {
            let encoded = pair[0].0 >> MERGE_RANK_BITS;
            return Err(invalid_data(format!("merge pair 重复: left={} right={}", encoded >> MERGE_ID_BITS, encoded & (MERGE_ID_LIMIT - 1) as u64)));
        }
        let (entries, ids): (Vec<_>, Vec<_>) = packed.into_iter().unzip();
        Ok(Self { entries: entries.into_boxed_slice(), ids: ids.into_boxed_slice() })
    }

    #[inline]
    pub fn get(&self, left: u32, right: u32) -> Option<Merge> {
        if left >= MERGE_ID_LIMIT || right >= MERGE_ID_LIMIT {
            return None;
        }
        let pair = merge_pair(left, right);
        let index = self.entries.binary_search_by_key(&pair, |entry| entry >> MERGE_RANK_BITS).ok()?;
        let rank = (self.entries[index] & MERGE_RANK_MASK) as u32;
        Some(Merge { rank, id: self.ids[index] })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn memory_bytes(&self) -> usize {
        self.entries.len() * size_of::<u64>() + self.ids.len() * size_of::<u32>()
    }
}

#[inline(always)]
const fn merge_pair(left: u32, right: u32) -> u64 {
    ((left as u64) << MERGE_ID_BITS) | right as u64
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct VocabGroup {
    byte_offset: u32,
    item_offset: u32,
}

pub struct Vocab {
    groups: Box<[VocabGroup]>,
    keys: Box<[u8]>,
    ids: Box<[u32]>,
}

impl Vocab {
    /// vocab 来自外部 tokenizer.json;过大输入返回 Err 而非 panic。
    pub fn new<I, K>(entries: I) -> Result<Self, &'static str>
    where
        I: IntoIterator<Item = (K, u32)>,
        K: Into<Vec<u8>>,
    {
        let mut entries: Vec<(Vec<u8>, u32)> = entries.into_iter().map(|(key, id)| (key.into(), id)).collect();
        entries.sort_unstable_by(|a, b| a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0)));

        const VOCAB_LIMIT: usize = 4 << 20;
        let max_len = entries.last().map_or(0, |entry| entry.0.len());
        if max_len >= u32::MAX as usize {
            return Err("vocab key is too long");
        }
        if entries.len() >= VOCAB_LIMIT {
            return Err("vocab has too many entries");
        }
        let total_bytes: usize = entries.iter().map(|entry| entry.0.len()).sum();
        if total_bytes >= u32::MAX as usize {
            return Err("vocab byte pool is too large");
        }

        let mut groups = vec![VocabGroup::default(); max_len + 2];
        let mut keys = Vec::with_capacity(total_bytes);
        let mut ids = Vec::with_capacity(entries.len());
        let mut entry = 0;

        for len in 0..=max_len {
            groups[len] = VocabGroup { byte_offset: keys.len() as u32, item_offset: ids.len() as u32 };
            while entry < entries.len() && entries[entry].0.len() == len {
                if entry > 0 && entries[entry - 1].0 == entries[entry].0 {
                    return Err("duplicate vocab key");
                }
                keys.extend_from_slice(&entries[entry].0);
                ids.push(entries[entry].1);
                entry += 1;
            }
        }
        groups[max_len + 1] = VocabGroup { byte_offset: keys.len() as u32, item_offset: ids.len() as u32 };

        Ok(Self { groups: groups.into_boxed_slice(), keys: keys.into_boxed_slice(), ids: ids.into_boxed_slice() })
    }

    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<u32> {
        let len = key.len();
        let group = *self.groups.get(len)?;
        let next = *self.groups.get(len + 1)?;
        let first = group.item_offset as usize;
        let mut low = first;
        let mut high = next.item_offset as usize;

        while low < high {
            let middle = low + (high - low) / 2;
            let start = group.byte_offset as usize + (middle - first) * len;
            match self.keys[start..start + len].cmp(key) {
                Ordering::Less => low = middle + 1,
                Ordering::Greater => high = middle,
                Ordering::Equal => return Some(self.ids[middle]),
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn memory_bytes(&self) -> usize {
        self.groups.len() * size_of::<VocabGroup>() + self.keys.len() + self.ids.len() * size_of::<u32>()
    }
}

pub struct Bpe {
    vocab: Vocab,
    merges: Merges,
    byte_ids: [u32; 256],
    utf8_byte_fallback: bool,
}

impl Bpe {
    pub fn new(vocab: Vocab, merges: Merges) -> io::Result<Self> {
        let mut byte_ids = [0; 256];
        for (byte, id) in byte_ids.iter_mut().enumerate() {
            *id = vocab.get(&[byte as u8]).ok_or_else(|| invalid_data(format!("ByteLevel vocab 缺少字节 {byte}")))?;
        }
        Ok(Self { vocab, merges, byte_ids, utf8_byte_fallback: false })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        Self::from_slice(&fs::read(path)?)
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
        let json: TokenizerJson = serde_json::from_slice(bytes)?;
        Self::from_json(&json)
    }

    fn from_json(json: &TokenizerJson) -> Result<Self, Box<dyn Error>> {
        if json.model.kind != "BPE" {
            return Err(invalid_data(format!("unsupported tokenizer model {:?}", json.model.kind)).into());
        }

        let mut merge_entries = Vec::with_capacity(json.model.merges.len());
        for merge in &json.model.merges {
            let (left, right) = merge.pair()?;
            let left_id = *json.model.vocab.get(left).ok_or_else(|| invalid_data(format!("merge token {left:?} is missing from vocab")))?;
            let right_id = *json.model.vocab.get(right).ok_or_else(|| invalid_data(format!("merge token {right:?} is missing from vocab")))?;
            let mut merged = String::with_capacity(left.len() + right.len());
            merged.push_str(left);
            merged.push_str(right);
            let merged_id = *json.model.vocab.get(&merged).ok_or_else(|| invalid_data(format!("merged token {merged:?} is missing from vocab")))?;
            merge_entries.push((left_id, right_id, merged_id));
        }

        let added: HashMap<u32, &str> = json.added_tokens.iter().map(|token| (token.id, token.content.as_str())).collect();
        let mut vocab_entries = Vec::with_capacity(json.model.vocab.len());
        for (token, &id) in &json.model.vocab {
            let bytes = if let Some(content) = added.get(&id) {
                content.as_bytes().to_vec()
            } else if json.model.byte_fallback {
                token.as_bytes().to_vec()
            } else {
                decode_byte_level(token)?
            };
            vocab_entries.push((bytes, id));
        }
        let vocab = Vocab::new(vocab_entries)?;
        let merges = Merges::new(merge_entries)?;
        if !json.model.byte_fallback {
            return Ok(Self::new(vocab, merges)?);
        }
        let mut byte_ids = [0u32; 256];
        for (byte, id) in byte_ids.iter_mut().enumerate() {
            let token = format!("<0x{byte:02X}>");
            *id = *json.model.vocab.get(&token).ok_or_else(|| invalid_data(format!("byte fallback vocab 缺少 {token}")))?;
        }
        Ok(Self { vocab, merges, byte_ids, utf8_byte_fallback: true })
    }

    #[inline(always)]
    fn piece_id(&self, piece: &[u8]) -> Option<u32> {
        self.vocab.get(piece)
    }

    #[inline(always)]
    fn byte_id(&self, byte: u8) -> u32 {
        self.byte_ids[byte as usize]
    }

    #[inline]
    fn merge(&self, ids: &mut Vec<u32>, start: usize) {
        let count = ids.len() - start;
        if count < 2 {
            return;
        }
        let mut values = ids[start..].to_vec();
        let mut previous = (0..count).map(|index| index.checked_sub(1)).collect::<Vec<_>>();
        let mut next = (0..count).map(|index| (index + 1 < count).then_some(index + 1)).collect::<Vec<_>>();
        let mut versions = vec![0u32; count];
        let mut active = vec![true; count];
        let mut heap = BinaryHeap::new();
        for left in 0..count - 1 {
            if let Some(candidate) = merge_candidate(&self.merges, &values, &versions, &next, left) {
                heap.push(candidate);
            }
        }

        while let Some(Reverse((rank, left, left_version, right, right_version, merged_id))) = heap.pop() {
            if !active[left] || !active[right] || next[left] != Some(right) || versions[left] != left_version || versions[right] != right_version {
                continue;
            }
            let Some(merge) = self.merges.get(values[left], values[right]) else { continue };
            if merge.rank != rank || merge.id != merged_id {
                continue;
            }

            values[left] = merged_id;
            versions[left] = versions[left].wrapping_add(1);
            active[right] = false;
            versions[right] = versions[right].wrapping_add(1);
            let after = next[right];
            next[left] = after;
            if let Some(after) = after {
                previous[after] = Some(left);
            }
            if let Some(before) = previous[left]
                && let Some(candidate) = merge_candidate(&self.merges, &values, &versions, &next, before)
            {
                heap.push(candidate);
            }
            if let Some(candidate) = merge_candidate(&self.merges, &values, &versions, &next, left) {
                heap.push(candidate);
            }
        }

        ids.truncate(start);
        let mut current = Some(0usize);
        while let Some(index) = current {
            ids.push(values[index]);
            current = next[index];
        }
    }

    fn append_piece(&self, piece: &[u8], ids: &mut Vec<u32>) {
        let start = ids.len();
        if self.utf8_byte_fallback {
            if let Ok(text) = std::str::from_utf8(piece) {
                for character in text.chars() {
                    let mut buffer = [0u8; 4];
                    let bytes = character.encode_utf8(&mut buffer).as_bytes();
                    if let Some(id) = self.piece_id(bytes) {
                        ids.push(id);
                    } else {
                        ids.extend(bytes.iter().copied().map(|byte| self.byte_id(byte)));
                    }
                }
            } else {
                ids.extend(piece.iter().copied().map(|byte| self.byte_id(byte)));
            }
        } else {
            ids.extend(piece.iter().copied().map(|byte| self.byte_id(byte)));
        }
        self.merge(ids, start);
    }

    pub fn memory_bytes(&self) -> usize {
        self.vocab.memory_bytes() + self.merges.memory_bytes() + size_of_val(&self.byte_ids)
    }
}

type MergeCandidate = Reverse<(u32, usize, u32, usize, u32, u32)>;

#[inline]
fn merge_candidate(merges: &Merges, values: &[u32], versions: &[u32], next: &[Option<usize>], left: usize) -> Option<MergeCandidate> {
    let right = next[left]?;
    let merge = merges.get(values[left], values[right])?;
    Some(Reverse((merge.rank, left, versions[left], right, versions[right], merge.id)))
}

pub struct Tokenizer {
    pub bpe: Bpe,
    added_tokens: AddedTokens,
    pretokenizer: Pretokenizer,
    normalize_nfc: bool,
    space_replacement: Option<Box<[u8]>>,
}

struct DecodedToken {
    bytes: Vec<u8>,
    special: bool,
}

/// Reverse tokenizer used by streaming generation. Token bytes are restored
/// directly from tokenizer.json's ByteLevel vocabulary without constructing
/// intermediate per-token Strings.
pub struct Detokenizer {
    tokens: Box<[Option<DecodedToken>]>,
}

impl Detokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        if looks_like_json(&bytes) {
            Self::from_slice(&bytes)
        } else {
            let special = load_tiktoken_special_tokens(path)?;
            Self::from_tiktoken_slice(&bytes, &special)
        }
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
        let json: TokenizerJson = serde_json::from_slice(bytes)?;
        let byte_fallback = json.model.byte_fallback;
        let max_id = json.model.vocab.values().copied().chain(json.added_tokens.iter().map(|token| token.id)).max().unwrap_or(0) as usize;
        // vocab id 来自外部文件;封顶防止损坏/恶意 tokenizer.json 触发无界分配 abort。
        const DECODER_ID_LIMIT: usize = 4 << 20;
        if max_id >= DECODER_ID_LIMIT {
            return Err(format!("tokenizer vocab id {max_id} 超过解码表上限 {DECODER_ID_LIMIT}").into());
        }
        let mut tokens = Vec::with_capacity(max_id + 1);
        tokens.resize_with(max_id + 1, || None);
        let added: HashMap<u32, &str> = json.added_tokens.iter().map(|token| (token.id, token.content.as_str())).collect();
        for (token, id) in json.model.vocab {
            let slot = &mut tokens[id as usize];
            if slot.is_some() {
                return Err(invalid_data(format!("duplicate decoder token ID {id}")).into());
            }
            let bytes = match added.get(&id) {
                Some(content) => content.as_bytes().to_vec(),
                None => decode_vocab_token(&token, byte_fallback)?,
            };
            *slot = Some(DecodedToken { bytes, special: false });
        }
        for token in json.added_tokens {
            let slot = &mut tokens[token.id as usize];
            let bytes = token.content.into_bytes();
            if let Some(existing) = slot {
                if existing.bytes != bytes {
                    return Err(invalid_data(format!("decoder token ID {} 内容冲突", token.id)).into());
                }
                existing.special |= token.special;
            } else {
                *slot = Some(DecodedToken { bytes, special: token.special });
            }
        }
        Ok(Self { tokens: tokens.into_boxed_slice() })
    }

    pub fn from_bpe_tokens(tokens: &[String], special: &[bool]) -> Result<Self, Box<dyn Error>> {
        if tokens.len() != special.len() {
            return Err(invalid_data(format!("BPE tokens={} 与 special={} 数量不一致", tokens.len(), special.len())).into());
        }
        let tokens = tokens
            .iter()
            .zip(special)
            .map(|(token, special)| {
                let bytes = if *special { token.as_bytes().to_vec() } else { decode_byte_level(token)? };
                Ok(Some(DecodedToken { bytes, special: *special }))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self { tokens: tokens.into_boxed_slice() })
    }

    /// gemma4 GGUF 的 tokens 是字面文本：普通 token 把 ▁(U+2581) 还原为空格，
    /// <0xXX> byte fallback 还原为对应单字节；special token 按原文输出。
    pub fn from_gguf_gemma4_tokens(tokens: &[String], special: &[bool]) -> Result<Self, Box<dyn Error>> {
        if tokens.len() != special.len() {
            return Err(invalid_data(format!("gemma4 tokens={} 与 special={} 数量不一致", tokens.len(), special.len())).into());
        }
        let tokens = tokens
            .iter()
            .zip(special)
            .map(|(token, special)| {
                let bytes = if *special { token.as_bytes().to_vec() } else { decode_vocab_token(token, true)? };
                Ok(Some(DecodedToken { bytes, special: *special }))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self { tokens: tokens.into_boxed_slice() })
    }

    fn from_tiktoken_slice(bytes: &[u8], special: &HashMap<u32, String>) -> Result<Self, Box<dyn Error>> {
        let entries = parse_tiktoken(bytes)?;
        let base_count = entries.len();
        let added = tiktoken_added_tokens(base_count, special)?;
        let mut tokens = Vec::with_capacity(base_count + added.len());
        tokens.resize_with(base_count + added.len(), || None);
        for (bytes, id) in entries {
            tokens[id as usize] = Some(DecodedToken { bytes, special: false });
        }
        for token in added {
            tokens[token.id as usize] = Some(DecodedToken { bytes: token.content.into_bytes(), special: true });
        }
        Ok(Self { tokens: tokens.into_boxed_slice() })
    }

    pub fn decode_bytes(&self, ids: &[u32], skip_special_tokens: bool) -> io::Result<Vec<u8>> {
        let capacity = ids.iter().try_fold(0usize, |total, &id| {
            let token = self.tokens.get(id as usize).and_then(Option::as_ref).ok_or_else(|| invalid_data(format!("unknown decoder token ID {id}")))?;
            total.checked_add(if skip_special_tokens && token.special { 0 } else { token.bytes.len() }).ok_or_else(|| invalid_data("decoded text size overflow".to_owned()))
        })?;
        let mut output = Vec::with_capacity(capacity);
        for &id in ids {
            let token = self.tokens.get(id as usize).and_then(Option::as_ref).ok_or_else(|| invalid_data(format!("unknown decoder token ID {id}")))?;
            if !skip_special_tokens || !token.special {
                output.extend_from_slice(&token.bytes);
            }
        }
        Ok(output)
    }
}

/// 从 Hugging Face 风格的 vocab.json、added_tokens.json 和 merges.txt
/// 构造 ByteLevel BPE tokenizer 与 detokenizer。
pub fn load_bpe_directory(root: impl AsRef<Path>) -> Result<(Tokenizer, Detokenizer), Box<dyn Error>> {
    let root = root.as_ref();
    let vocab: HashMap<String, u32> = serde_json::from_slice(&fs::read(root.join("vocab.json"))?)?;
    let added: HashMap<String, u32> = serde_json::from_slice(&fs::read(root.join("added_tokens.json"))?)?;
    let token_count = vocab.values().chain(added.values()).copied().max().ok_or_else(|| invalid_data("BPE vocab 为空".to_owned()))? as usize + 1;
    let mut tokens = vec![None; token_count];
    let mut special = vec![false; token_count];
    for (token, id) in vocab {
        let slot = tokens.get_mut(id as usize).ok_or_else(|| invalid_data(format!("BPE vocab token id {id} 越界")))?;
        if slot.replace(token).is_some() {
            return Err(invalid_data(format!("BPE vocab token id {id} 重复")).into());
        }
    }
    for (token, id) in added {
        let index = id as usize;
        let slot = tokens.get_mut(index).ok_or_else(|| invalid_data(format!("BPE added token id {id} 越界")))?;
        match slot {
            Some(existing) if existing != &token => return Err(invalid_data(format!("BPE token id {id} 内容冲突")).into()),
            Some(_) => {}
            None => *slot = Some(token),
        }
        special[index] = true;
    }
    let tokens = tokens.into_iter().enumerate().map(|(id, token)| token.ok_or_else(|| invalid_data(format!("BPE tokenizer 缺少 token id {id}")))).collect::<io::Result<Vec<_>>>()?;
    let merges = fs::read_to_string(root.join("merges.txt"))?.lines().filter(|line| !line.is_empty() && !line.starts_with('#')).map(str::to_owned).collect::<Vec<_>>();
    let tokenizer = Tokenizer::from_bpe_tokens(&tokens, &merges, &special)?;
    let detokenizer = Detokenizer::from_bpe_tokens(&tokens, &special)?;
    Ok((tokenizer, detokenizer))
}

/// 跨 token 保留不完整的 UTF-8 字节，只在字符完整后交给流式响应。
#[derive(Default)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    output.push_str(text);
                    self.pending.clear();
                    break;
                }
                Err(error) if error.valid_up_to() > 0 => {
                    let valid = error.valid_up_to();
                    output.push_str(std::str::from_utf8(&self.pending[..valid]).expect("valid_up_to 保证 UTF-8"));
                    self.pending.drain(..valid);
                }
                Err(error) if error.error_len().is_some() => {
                    let invalid = error.error_len().expect("刚确认 error_len");
                    output.push('\u{fffd}');
                    self.pending.drain(..invalid);
                }
                Err(_) => break,
            }
        }
        output
    }

    pub fn finish(&mut self) -> String {
        // pending 只可能是不完整但仍可能合法的 UTF-8 前缀；生成在长度上限处
        // 停止时没有后续字节可补齐，不能把截断字节伪造成 U+FFFD。
        self.pending.clear();
        String::new()
    }
}

impl Tokenizer {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        if looks_like_json(&bytes) {
            Self::from_slice(&bytes)
        } else {
            let special = load_tiktoken_special_tokens(path)?;
            Self::from_tiktoken_slice(&bytes, &special)
        }
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
        let json: TokenizerJson = serde_json::from_slice(bytes)?;
        let bpe = Bpe::from_json(&json)?;
        let added_tokens = AddedTokens::new(&json.added_tokens)?;
        let pretokenizer = json.pretokenizer();
        let (normalize_nfc, space_replacement) = json.normalization()?;
        Ok(Self { bpe, added_tokens, pretokenizer, normalize_nfc, space_replacement })
    }

    pub fn from_bpe_tokens(tokens: &[String], merges: &[String], special: &[bool]) -> Result<Self, Box<dyn Error>> {
        if tokens.len() != special.len() {
            return Err(invalid_data(format!("BPE tokens={} 与 special={} 数量不一致", tokens.len(), special.len())).into());
        }
        let ids: HashMap<&str, u32> = tokens.iter().enumerate().map(|(id, token)| Ok((token.as_str(), u32::try_from(id).map_err(|_| invalid_data("BPE token id 超出 u32".to_owned()))?))).collect::<io::Result<_>>()?;
        let mut merge_entries = Vec::with_capacity(merges.len());
        for merge in merges {
            let (left, right) = merge.split_once(' ').ok_or_else(|| invalid_data(format!("invalid BPE merge {merge:?}")))?;
            let left_id = *ids.get(left).ok_or_else(|| invalid_data(format!("merge token {left:?} 不在 vocab")))?;
            let right_id = *ids.get(right).ok_or_else(|| invalid_data(format!("merge token {right:?} 不在 vocab")))?;
            let merged = format!("{left}{right}");
            let merged_id = *ids.get(merged.as_str()).ok_or_else(|| invalid_data(format!("merged token {merged:?} 不在 vocab")))?;
            merge_entries.push((left_id, right_id, merged_id));
        }
        let mut vocab =
            tokens.iter().zip(special).enumerate().map(|(id, (token, special))| Ok((if *special { token.as_bytes().to_vec() } else { decode_byte_level(token)? }, id as u32, *special))).collect::<io::Result<Vec<(Vec<u8>, u32, bool)>>>()?;
        // GGUF 里同一字符串可能既是 merge vocab token 又是 added/special token(Laguna 等);
        // string→id 匹配按 HF 语义让 special 条目胜出,同类取低 id。
        // special 优先、同类低 id 优先的去重;不能只靠相邻 dedup(重复项在长度排序下
        // 才相邻,这里顺序不同),用 seen 集合按保留序取首个。
        vocab.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
        let mut seen = std::collections::HashSet::with_capacity(vocab.len());
        vocab.retain(|(key, _, _)| seen.insert(key.clone()));
        let vocab = vocab.into_iter().map(|(key, id, _)| (key, id)).collect::<Vec<_>>();
        let added = tokens
            .iter()
            .zip(special)
            .enumerate()
            .filter(|(_, (_, special))| **special)
            .map(|(id, (content, _))| AddedTokenJson { id: id as u32, content: content.clone(), special: true, single_word: false, lstrip: false, rstrip: false })
            .collect::<Vec<_>>();
        Ok(Self { bpe: Bpe::new(Vocab::new(vocab)?, Merges::new(merge_entries)?)?, added_tokens: AddedTokens::new(&added)?, pretokenizer: Pretokenizer::Legacy, normalize_nfc: false, space_replacement: None })
    }

    /// 从 GGUF 的 BPE metadata 构造 tokenizer；`tokenizer.ggml.pre` 决定模型使用的
    /// 预切分规则，未知 preset 保持 llama.cpp 兼容的 Legacy 行为。
    pub fn from_gguf_bpe_tokens(tokens: &[String], merges: &[String], special: &[bool], preset: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = Self::from_bpe_tokens(tokens, merges, special)?;
        match preset {
            Some("glm4" | "chatglm-bpe") => tokenizer.pretokenizer = Pretokenizer::Cl100k,
            Some("joyai-llm") => tokenizer.pretokenizer = Pretokenizer::JoyAi,
            _ => {}
        }
        Ok(tokenizer)
    }

    /// gemma4 GGUF 的 tokens 是字面文本：空格用 ▁(U+2581) 表示，byte fallback 用 <0xXX> 表示，
    /// 与官方 tokenizer.json 的 byte_fallback BPE 同构。因此 vocab 不做 ByteLevel 解码，
    /// encode 前把空格替换为 ▁，整个输入作为一个 pretoken 做 BPE。
    pub fn from_gguf_gemma4_tokens(tokens: &[String], merges: &[String], special: &[bool]) -> Result<Self, Box<dyn Error>> {
        if tokens.len() != special.len() {
            return Err(invalid_data(format!("gemma4 tokens={} 与 special={} 数量不一致", tokens.len(), special.len())).into());
        }
        let ids: HashMap<&str, u32> = tokens.iter().enumerate().map(|(id, token)| Ok((token.as_str(), u32::try_from(id).map_err(|_| invalid_data("gemma4 token id 超出 u32".to_owned()))?))).collect::<io::Result<_>>()?;
        let mut merge_entries = Vec::with_capacity(merges.len());
        for merge in merges {
            let (left, right) = merge.split_once(' ').ok_or_else(|| invalid_data(format!("invalid gemma4 merge {merge:?}")))?;
            let left_id = *ids.get(left).ok_or_else(|| invalid_data(format!("merge token {left:?} 不在 vocab")))?;
            let right_id = *ids.get(right).ok_or_else(|| invalid_data(format!("merge token {right:?} 不在 vocab")))?;
            let merged = format!("{left}{right}");
            let merged_id = *ids.get(merged.as_str()).ok_or_else(|| invalid_data(format!("merged token {merged:?} 不在 vocab")))?;
            merge_entries.push((left_id, right_id, merged_id));
        }
        let vocab = tokens.iter().enumerate().map(|(id, token)| (token.clone().into_bytes(), id as u32)).collect::<Vec<_>>();
        // byte fallback：<0xXX> token 的 id 作为单字节回退 id，与 tokenizer.json 路径一致。
        let mut byte_ids = [0u32; 256];
        for (byte, id) in byte_ids.iter_mut().enumerate() {
            let token = format!("<0x{byte:02X}>");
            *id = *ids.get(token.as_str()).ok_or_else(|| invalid_data(format!("gemma4 vocab 缺少 byte fallback token {token}")))?;
        }
        let added = tokens
            .iter()
            .zip(special)
            .enumerate()
            .filter(|(_, (_, special))| **special)
            .map(|(id, (content, _))| AddedTokenJson { id: id as u32, content: content.clone(), special: true, single_word: false, lstrip: false, rstrip: false })
            .collect::<Vec<_>>();
        let bpe = Bpe { vocab: Vocab::new(vocab)?, merges: Merges::new(merge_entries)?, byte_ids, utf8_byte_fallback: true };
        Ok(Self { bpe, added_tokens: AddedTokens::new(&added)?, pretokenizer: Pretokenizer::Whole, normalize_nfc: false, space_replacement: Some("▁".as_bytes().into()) })
    }

    fn from_tiktoken_slice(bytes: &[u8], special: &HashMap<u32, String>) -> Result<Self, Box<dyn Error>> {
        let entries = parse_tiktoken(bytes)?;
        let base_count = entries.len();
        let ids: HashMap<Vec<u8>, u32> = entries.iter().cloned().collect();
        let mut merges = Vec::new();
        for (token, &id) in &ids {
            for split in 1..token.len() {
                let Some(&left) = ids.get(&token[..split]) else { continue };
                let Some(&right) = ids.get(&token[split..]) else { continue };
                merges.push((left, right, id, id));
            }
        }
        let added = tiktoken_added_tokens(base_count, special)?;
        Ok(Self { bpe: Bpe::new(Vocab::new(entries)?, Merges::from_ranked(merges)?)?, added_tokens: AddedTokens::new(&added)?, pretokenizer: Pretokenizer::KimiK3, normalize_nfc: false, space_replacement: None })
    }

    pub fn tokenize(&self, input: &[u8]) -> Vec<u32> {
        self.tokenize_with_special(input, true)
    }

    /// `allow_special_tokens=false` 用于用户和工具文本，防止文本伪造 XTML 控制 token。
    pub fn tokenize_with_special(&self, input: &[u8], allow_special_tokens: bool) -> Vec<u32> {
        let input = self.prepare_input(input);
        if input.len() < PARALLEL_BPE_MIN_BYTES {
            return self.tokenize_prepared_serial(&input, allow_special_tokens);
        }
        self.tokenize_prepared_parallel(&input, allow_special_tokens)
    }

    fn prepare_input<'a>(&self, input: &'a [u8]) -> Cow<'a, [u8]> {
        let input = if self.normalize_nfc {
            match std::str::from_utf8(input) {
                Ok(input) => Cow::Owned(input.nfc().collect::<String>().into_bytes()),
                Err(_) => Cow::Borrowed(input),
            }
        } else {
            Cow::Borrowed(input)
        };
        if let Some(replacement) = &self.space_replacement {
            let spaces = input.iter().filter(|&&byte| byte == b' ').count();
            let mut output = Vec::with_capacity(input.len() + spaces.saturating_mul(replacement.len().saturating_sub(1)));
            for &byte in input.as_ref() {
                if byte == b' ' {
                    output.extend_from_slice(replacement);
                } else {
                    output.push(byte);
                }
            }
            Cow::Owned(output)
        } else {
            input
        }
    }

    fn tokenize_prepared_serial(&self, input: &[u8], allow_special_tokens: bool) -> Vec<u32> {
        let mut cursor = Cursor::new(input, &self.added_tokens, self.pretokenizer, allow_special_tokens);
        let mut ids = Vec::new();
        while cursor.pos < cursor.input.len() {
            if let Some(id) = cursor.added() {
                ids.push(id);
                continue;
            }

            let start = cursor.pos;
            let piece = match cursor.piece() {
                Some(piece) => piece,
                None => {
                    cursor.pos += 1;
                    &cursor.input[start..cursor.pos]
                }
            };

            if let Some(id) = self.bpe.piece_id(piece) {
                ids.push(id);
                continue;
            }

            self.bpe.append_piece(piece, &mut ids);
        }
        ids
    }

    fn tokenize_prepared_parallel(&self, input: &[u8], allow_special_tokens: bool) -> Vec<u32> {
        let mut cursor = Cursor::new(input, &self.added_tokens, self.pretokenizer, allow_special_tokens);
        let mut pieces = Vec::new();
        while cursor.pos < cursor.input.len() {
            if let Some(id) = cursor.added() {
                pieces.push(TokenPiece::Added(id));
                continue;
            }
            let start = cursor.pos;
            let piece = match cursor.piece() {
                Some(piece) => piece,
                None => {
                    cursor.pos += 1;
                    &cursor.input[start..cursor.pos]
                }
            };
            pieces.push(TokenPiece::Bytes(piece));
        }

        // BPE merge 不能跨 pre-tokenizer piece；按 piece 分片并行不会改变 token 序列。
        // 每个 worker 保留多个 piece，避免为代码文本里的短词创建海量微任务。
        let workers = rayon::current_num_threads().max(1);
        let chunk_size = pieces.len().div_ceil(workers.saturating_mul(4)).max(1);
        let chunks = pieces
            .par_chunks(chunk_size)
            .map(|pieces| {
                let mut ids = Vec::new();
                for &piece in pieces {
                    match piece {
                        TokenPiece::Added(id) => ids.push(id),
                        TokenPiece::Bytes(piece) => {
                            if let Some(id) = self.bpe.piece_id(piece) {
                                ids.push(id);
                            } else {
                                self.bpe.append_piece(piece, &mut ids);
                            }
                        }
                    }
                }
                ids
            })
            .collect::<Vec<_>>();
        let mut ids = Vec::with_capacity(chunks.iter().map(Vec::len).sum());
        for chunk in chunks {
            ids.extend(chunk);
        }
        ids
    }

    pub fn memory_bytes(&self) -> usize {
        self.bpe.memory_bytes() + self.added_tokens.memory_bytes()
    }
}

const PARALLEL_BPE_MIN_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy)]
enum TokenPiece<'a> {
    Added(u32),
    Bytes(&'a [u8]),
}

struct AddedToken {
    id: u32,
    content: Box<[u8]>,
    special: bool,
}

struct AddedTokens {
    entries: Box<[AddedToken]>,
    starts: [u32; 257],
}

impl AddedTokens {
    fn new(tokens: &[AddedTokenJson]) -> io::Result<Self> {
        let mut ids = HashMap::with_capacity(tokens.len());
        let mut entries = Vec::with_capacity(tokens.len());
        for token in tokens {
            if token.content.is_empty() {
                return Err(invalid_data("added token 不能为空".to_owned()));
            }
            if token.single_word || token.lstrip || token.rstrip {
                return Err(invalid_data(format!("added token {:?} 使用了尚不支持的边界或空白规则", token.content)));
            }
            if ids.insert(token.id, ()).is_some() {
                return Err(invalid_data(format!("duplicate added token ID {}", token.id)));
            }
            entries.push(AddedToken { id: token.id, content: token.content.as_bytes().into(), special: token.special });
        }
        entries.sort_unstable_by(|a, b| a.content[0].cmp(&b.content[0]).then_with(|| b.content.len().cmp(&a.content.len())).then_with(|| a.content.cmp(&b.content)));
        if let Some(pair) = entries.windows(2).find(|pair| pair[0].content == pair[1].content) {
            return Err(invalid_data(format!("duplicate added token {:?}", String::from_utf8_lossy(&pair[0].content))));
        }

        let mut starts = [0u32; 257];
        let mut index = 0;
        for byte in 0..256 {
            starts[byte] = index as u32;
            while index < entries.len() && entries[index].content[0] == byte as u8 {
                index += 1;
            }
        }
        starts[256] = entries.len() as u32;
        Ok(Self { entries: entries.into_boxed_slice(), starts })
    }

    #[inline]
    fn get(&self, input: &[u8], allow_special: bool) -> Option<(u32, usize)> {
        let first = *input.first()? as usize;
        self.entries[self.starts[first] as usize..self.starts[first + 1] as usize].iter().find(|token| (allow_special || !token.special) && input.starts_with(&token.content)).map(|token| (token.id, token.content.len()))
    }

    #[inline]
    fn next(&self, input: &[u8], mut pos: usize, allow_special: bool) -> usize {
        while pos < input.len() {
            if self.get(&input[pos..], allow_special).is_some() {
                return pos;
            }
            pos += 1;
        }
        input.len()
    }

    fn memory_bytes(&self) -> usize {
        size_of_val(&self.starts) + self.entries.len() * size_of::<AddedToken>() + self.entries.iter().map(|token| token.content.len()).sum::<usize>()
    }
}

#[derive(Clone, Copy)]
enum Pretokenizer {
    Legacy,
    Cl100k,
    MiniMaxM3,
    KimiK3,
    JoyAi,
    Whole,
}

struct Cursor<'a> {
    input: &'a [u8],
    pos: usize,
    next_added: usize,
    added_tokens: &'a AddedTokens,
    pretokenizer: Pretokenizer,
    allow_special: bool,
}

impl<'a> Cursor<'a> {
    #[inline(always)]
    fn new(input: &'a [u8], added_tokens: &'a AddedTokens, pretokenizer: Pretokenizer, allow_special: bool) -> Self {
        let next_added = added_tokens.next(input, 0, allow_special);
        Self { input, pos: 0, next_added, added_tokens, pretokenizer, allow_special }
    }

    #[inline(always)]
    fn added(&mut self) -> Option<u32> {
        if self.pos != self.next_added {
            return None;
        }
        match self.added_tokens.get(self.input.get(self.pos..)?, self.allow_special) {
            Some((id, size)) => {
                self.pos += size;
                self.next_added = self.added_tokens.next(self.input, self.pos, self.allow_special);
                Some(id)
            }
            None => {
                // 该位置的 added token 被禁用(allow_special=false)或不匹配:
                // 必须推进扫描窗口,否则 piece() 的 limit 停在此处,后续输入
                // 会退化为逐字节 token 且再也检测不到真正的 added token。
                self.next_added = self.added_tokens.next(self.input, self.pos + 1, self.allow_special);
                None
            }
        }
    }

    #[inline]
    fn piece(&mut self) -> Option<&'a [u8]> {
        let start = self.pos;
        let limit = self.next_added;
        let input = &self.input[..limit];
        let (first, first_size) = char_at(input, start)?;
        let first_end = start + first_size;

        match self.pretokenizer {
            Pretokenizer::Whole => return self.take(limit),
            Pretokenizer::Legacy | Pretokenizer::Cl100k => {
                let cl100k = matches!(self.pretokenizer, Pretokenizer::Cl100k);
                // GPT-2 的缩写后缀大小写敏感；cl100k 的 `(?i:...)` 不敏感。
                if let Some(size) = contraction(&input[start..], !cl100k) {
                    return self.take(start + size);
                }
                if is_letter(first) {
                    return self.take(scan(input, first_end, is_letter));
                }
                if !is_newline(first)
                    && !is_number(first)
                    && let Some((next, size)) = char_at(input, first_end)
                    && is_letter(next)
                {
                    return self.take(scan(input, first_end + size, is_letter));
                }
                // 只有 GPT-2 ` ?\p{N}{1,3}` 把前导空格和数字放在同一块；
                // cl100k 的 `\p{N}{1,3}` 必须让空格走后面的 whitespace 分支。
                if !cl100k && first == ' ' && char_at(input, first_end).is_some_and(|(next, _)| is_number(next)) {
                    return self.take(scan_limited_number(input, first_end));
                }
            }
            Pretokenizer::MiniMaxM3 => {
                if let Some(mut end) = minimax_word_end(input, start) {
                    if let Some(size) = contraction(&input[end..], false) {
                        end += size;
                    }
                    return self.take(end);
                }
                if !is_newline(first)
                    && !is_letter(first)
                    && !is_number(first)
                    && let Some(mut end) = minimax_word_end(input, first_end)
                {
                    if let Some(size) = contraction(&input[end..], false) {
                        end += size;
                    }
                    return self.take(end);
                }
            }
            Pretokenizer::KimiK3 => {
                if is_han(first) {
                    return self.take(scan(input, first_end, is_han));
                }
                if let Some(mut end) = kimi_word_end(input, start) {
                    if let Some(size) = contraction(&input[end..], false) {
                        end += size;
                    }
                    return self.take(end);
                }
                if !is_newline(first)
                    && !is_letter(first)
                    && !is_number(first)
                    && let Some(mut end) = kimi_word_end(input, first_end)
                {
                    if let Some(size) = contraction(&input[end..], false) {
                        end += size;
                    }
                    return self.take(end);
                }
            }
            Pretokenizer::JoyAi => {
                // DeepSeek-V4 的 joyai tokenizer 先隔离中日韩文字块，再应用
                // ByteLevel 主正则；这一步不能退化为普通 Unicode letter 扫描。
                if is_joy_cjk(first) {
                    return self.take(scan(input, first_end, is_joy_cjk));
                }
                if first.is_ascii_punctuation()
                    && let Some((next, size)) = char_at(input, first_end)
                    && next.is_ascii_alphabetic()
                {
                    return self.take(scan(input, first_end + size, |character| character.is_ascii_alphabetic()));
                }
                if is_letter_or_mark(first) {
                    return self.take(scan(input, first_end, |character| is_letter_or_mark(character) && !is_joy_cjk(character)));
                }
                if !is_newline(first)
                    && !is_letter_or_mark(first)
                    && !is_symbol(first)
                    && let Some((next, size)) = char_at(input, first_end)
                    && is_letter_or_mark(next)
                {
                    return self.take(scan(input, first_end + size, |character| is_letter_or_mark(character) && !is_joy_cjk(character)));
                }
            }
        }

        if is_number(first) {
            let mut end = first_end;
            for _ in 1..3 {
                let Some((next, size)) = char_at(input, end) else { break };
                if !is_number(next) {
                    break;
                }
                end += size;
            }
            return self.take(end);
        }

        let body = if first == ' ' { first_end } else { start };
        if let Some((next, _)) = char_at(input, body)
            && is_symbol(next)
        {
            return self.take(scan(input, scan(input, body, is_symbol), is_newline));
        }

        if first.is_whitespace() {
            let mut end = first_end;
            let mut count = 1;
            let mut last_start = start;
            let mut last_newline_end = is_newline(first).then_some(first_end);
            while let Some((next, size)) = char_at(input, end) {
                if !next.is_whitespace() {
                    break;
                }
                last_start = end;
                end += size;
                count += 1;
                if is_newline(next) {
                    last_newline_end = Some(end)
                }
            }
            if let Some(end) = last_newline_end {
                return self.take(end);
            }
            return self.take(if end < input.len() && count > 1 { last_start } else { end });
        }

        None
    }

    #[inline(always)]
    fn take(&mut self, end: usize) -> Option<&'a [u8]> {
        let start = self.pos;
        self.pos = end;
        self.input.get(start..end)
    }
}

#[derive(Deserialize)]
struct TokenizerJson {
    model: ModelJson,
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    special: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
}

#[derive(Deserialize, Default)]
struct TokenizerConfigJson {
    #[serde(default)]
    added_tokens_decoder: HashMap<u32, TokenizerConfigToken>,
}

#[derive(Deserialize)]
struct TokenizerConfigToken {
    content: String,
}

#[derive(Deserialize)]
struct ModelJson {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    merges: Vec<MergeJson>,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MergeJson {
    Pair([String; 2]),
    Text(String),
}

impl MergeJson {
    fn pair(&self) -> io::Result<(&str, &str)> {
        match self {
            Self::Pair(pair) => Ok((&pair[0], &pair[1])),
            Self::Text(text) => text.split_once(' ').ok_or_else(|| invalid_data(format!("invalid merge {text:?}"))),
        }
    }
}

impl TokenizerJson {
    fn normalization(&self) -> io::Result<(bool, Option<Box<[u8]>>)> {
        let Some(normalizer) = &self.normalizer else { return Ok((false, None)) };
        match normalizer.get("type").and_then(serde_json::Value::as_str) {
            Some("NFC") => Ok((true, None)),
            Some("Sequence") if normalizer.get("normalizers").and_then(serde_json::Value::as_array).is_some_and(Vec::is_empty) => Ok((false, None)),
            Some("Replace") => {
                let pattern = normalizer.get("pattern").and_then(|pattern| pattern.get("String")).and_then(serde_json::Value::as_str);
                let content = normalizer.get("content").and_then(serde_json::Value::as_str);
                if pattern != Some(" ") {
                    return Err(invalid_data(format!("unsupported Replace normalizer pattern {pattern:?}")));
                }
                let content = content.ok_or_else(|| invalid_data("Replace normalizer 缺少 content".to_owned()))?;
                Ok((false, Some(content.as_bytes().into())))
            }
            Some(kind) => Err(invalid_data(format!("unsupported tokenizer normalizer {kind:?}"))),
            None => Err(invalid_data("tokenizer normalizer 缺少 type".to_owned())),
        }
    }

    fn pretokenizer(&self) -> Pretokenizer {
        let whole = self.model.byte_fallback
            && self.pre_tokenizer.as_ref().and_then(|value| value.get("type")).and_then(serde_json::Value::as_str) == Some("Split")
            && self.pre_tokenizer.as_ref().and_then(|value| value.get("pattern")).and_then(|pattern| pattern.get("String")).and_then(serde_json::Value::as_str) == Some(" ");
        if whole {
            return Pretokenizer::Whole;
        }
        let patterns = self
            .pre_tokenizer
            .as_ref()
            .and_then(|value| value.get("pretokenizers"))
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter(|item| item.get("type").and_then(serde_json::Value::as_str) == Some("Split"))
            .filter_map(|split| split.get("pattern").and_then(|pattern| pattern.get("Regex")).and_then(serde_json::Value::as_str));
        let mut minimax = false;
        let mut cl100k = false;
        for pattern in patterns {
            if pattern.contains("一-龥") && pattern.contains("぀-ゟ") && pattern.contains("゠-ヿ") {
                return Pretokenizer::JoyAi;
            }
            minimax |= pattern.contains("\\p{Lu}") && pattern.contains("\\p{Ll}");
            cl100k |= pattern.contains("(?i:") && pattern.contains("\\p{N}{1,3}") && pattern.contains("\\s+(?!\\S)");
        }
        if minimax {
            Pretokenizer::MiniMaxM3
        } else if cl100k {
            Pretokenizer::Cl100k
        } else {
            Pretokenizer::Legacy
        }
    }
}

fn decode_byte_level(token: &str) -> io::Result<Vec<u8>> {
    token.chars().map(decode_byte).collect()
}

fn decode_vocab_token(token: &str, byte_fallback: bool) -> io::Result<Vec<u8>> {
    if !byte_fallback {
        return decode_byte_level(token);
    }
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        let byte = u8::from_str_radix(&token[3..5], 16).map_err(|_| invalid_data(format!("invalid byte fallback token {token:?}")))?;
        return Ok(vec![byte]);
    }
    Ok(token.replace('▁', " ").into_bytes())
}

fn looks_like_json(bytes: &[u8]) -> bool {
    bytes.iter().copied().find(|byte| !byte.is_ascii_whitespace()) == Some(b'{')
}

fn load_tiktoken_special_tokens(path: &Path) -> Result<HashMap<u32, String>, Box<dyn Error>> {
    let Some(root) = path.parent() else { return Ok(HashMap::new()) };
    let config_path = root.join("tokenizer_config.json");
    if !config_path.is_file() {
        return Ok(HashMap::new());
    }
    let config: TokenizerConfigJson = serde_json::from_slice(&fs::read(config_path)?)?;
    Ok(config.added_tokens_decoder.into_iter().map(|(id, token)| (id, token.content)).collect())
}

fn parse_tiktoken(bytes: &[u8]) -> io::Result<Vec<(Vec<u8>, u32)>> {
    let mut entries = Vec::new();
    for (line_index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split(|byte| byte.is_ascii_whitespace()).filter(|field| !field.is_empty());
        let encoded = fields.next().ok_or_else(|| invalid_data(format!("tiktoken 第 {} 行缺少 token", line_index + 1)))?;
        let rank = fields.next().ok_or_else(|| invalid_data(format!("tiktoken 第 {} 行缺少 rank", line_index + 1)))?;
        if fields.next().is_some() {
            return Err(invalid_data(format!("tiktoken 第 {} 行字段过多", line_index + 1)));
        }
        let rank = std::str::from_utf8(rank).ok().and_then(|rank| rank.parse::<u32>().ok()).ok_or_else(|| invalid_data(format!("tiktoken 第 {} 行 rank 非法", line_index + 1)))?;
        entries.push((decode_base64(encoded)?, rank));
    }
    entries.sort_unstable_by_key(|entry| entry.1);
    for (expected, (_, rank)) in entries.iter().enumerate() {
        if *rank as usize != expected {
            return Err(invalid_data(format!("tiktoken rank 不连续: index={expected}, rank={rank}")));
        }
    }
    if entries.is_empty() || entries.len() + KIMI_RESERVED_SPECIAL_TOKENS >= MERGE_ID_LIMIT as usize {
        return Err(invalid_data(format!("tiktoken vocab 大小 {} 无效", entries.len())));
    }
    Ok(entries)
}

const KIMI_RESERVED_SPECIAL_TOKENS: usize = 256;

fn tiktoken_added_tokens(base_count: usize, overrides: &HashMap<u32, String>) -> io::Result<Vec<AddedTokenJson>> {
    let end = base_count.checked_add(KIMI_RESERVED_SPECIAL_TOKENS).ok_or_else(|| invalid_data("tiktoken special token 范围溢出".to_owned()))?;
    for &id in overrides.keys() {
        if (id as usize) < base_count || (id as usize) >= end {
            return Err(invalid_data(format!("tiktoken special token id {id} 不在 {base_count}..{end}")));
        }
    }
    Ok((base_count..end)
        .map(|id| AddedTokenJson { id: id as u32, content: overrides.get(&(id as u32)).cloned().unwrap_or_else(|| format!("<|reserved_token_{id}|>")), special: true, single_word: false, lstrip: false, rstrip: false })
        .collect())
}

fn decode_base64(input: &[u8]) -> io::Result<Vec<u8>> {
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    for &byte in input {
        if byte == b'=' {
            break;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(invalid_data("tiktoken token 不是合法 base64".to_owned())),
        };
        accumulator = (accumulator << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((accumulator >> bits) as u8);
            accumulator &= (1 << bits) - 1;
        }
    }
    Ok(output)
}

#[inline(always)]
fn decode_byte(character: char) -> io::Result<u8> {
    match character as u32 {
        code @ (33..=126 | 161..=172 | 174..=255) => Ok(code as u8),
        code @ 256..=288 => Ok((code - 256) as u8),
        code @ 289..=322 => Ok((code - 162) as u8),
        323 => Ok(173),
        _ => Err(invalid_data(format!("invalid ByteLevel character {character:?}"))),
    }
}

#[inline]
fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[inline(always)]
fn char_at(input: &[u8], pos: usize) -> Option<(char, usize)> {
    let size = match *input.get(pos)? {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return None,
    };
    let bytes = input.get(pos..pos.checked_add(size)?)?;
    Some((std::str::from_utf8(bytes).ok()?.chars().next()?, size))
}

#[inline(always)]
fn scan(input: &[u8], mut pos: usize, accept: impl Fn(char) -> bool) -> usize {
    while let Some((character, size)) = char_at(input, pos) {
        if !accept(character) {
            break;
        }
        pos += size;
    }
    pos
}

#[inline(always)]
/// `case_sensitive=true` 对应 GPT-2 的 `'s|'t|'re|...`(大小写敏感);
/// cl100k 系用 `(?i:...)`,传 false。
fn contraction(input: &[u8], case_sensitive: bool) -> Option<usize> {
    if input.first() != Some(&b'\'') {
        return None;
    }
    let second = *input.get(1)?;
    let matches = |actual: u8, expected: u8| if case_sensitive { actual == expected } else { actual.to_ascii_lowercase() == expected };
    if [b's', b't', b'm', b'd'].iter().any(|&expected| matches(second, expected)) {
        return Some(2);
    }
    let third = *input.get(2)?;
    if matches(second, b'r') && matches(third, b'e') {
        return Some(3);
    }
    if matches(second, b'v') && matches(third, b'e') {
        return Some(3);
    }
    if matches(second, b'l') && matches(third, b'l') {
        return Some(3);
    }
    None
}

/// GPT-2 `\p{N}{1,3}`:从 from 开始最多吸收 3 个数字。
fn scan_limited_number(input: &[u8], mut from: usize) -> usize {
    let mut count = 0;
    while count < 3 {
        let Some((character, size)) = char_at(input, from) else { break };
        if !is_number(character) {
            break;
        }
        from += size;
        count += 1;
    }
    from
}

#[inline(always)]
fn is_letter(character: char) -> bool {
    matches!(get_general_category(character), GeneralCategory::UppercaseLetter | GeneralCategory::LowercaseLetter | GeneralCategory::TitlecaseLetter | GeneralCategory::ModifierLetter | GeneralCategory::OtherLetter)
}

#[inline(always)]
fn is_number(character: char) -> bool {
    matches!(get_general_category(character), GeneralCategory::DecimalNumber | GeneralCategory::LetterNumber | GeneralCategory::OtherNumber)
}

#[inline(always)]
fn is_minimax_upper(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::UppercaseLetter | GeneralCategory::TitlecaseLetter | GeneralCategory::ModifierLetter | GeneralCategory::OtherLetter | GeneralCategory::NonspacingMark | GeneralCategory::SpacingMark | GeneralCategory::EnclosingMark
    )
}

#[inline(always)]
fn is_minimax_lower(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::LowercaseLetter | GeneralCategory::ModifierLetter | GeneralCategory::OtherLetter | GeneralCategory::NonspacingMark | GeneralCategory::SpacingMark | GeneralCategory::EnclosingMark
    )
}

#[inline]
fn minimax_word_end(input: &[u8], start: usize) -> Option<usize> {
    let (first, size) = char_at(input, start)?;
    let first_end = start + size;
    if is_minimax_lower(first) {
        return Some(scan(input, first_end, is_minimax_lower));
    }
    if !is_minimax_upper(first) {
        return None;
    }

    let mut end = scan(input, first_end, is_minimax_upper);
    if let Some((next, size)) = char_at(input, end)
        && is_minimax_lower(next)
    {
        end = scan(input, end + size, is_minimax_lower);
    }
    Some(end)
}

#[inline(always)]
fn is_han(character: char) -> bool {
    matches!(
        character as u32,
        0x2e80..=0x2ef3
            | 0x2f00..=0x2fd5
            | 0x3005
            | 0x3007
            | 0x3021..=0x3029
            | 0x3038..=0x303b
            | 0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0xf900..=0xfa6d
            | 0xfa70..=0xfad9
            | 0x16fe2..=0x16fe3
            | 0x20000..=0x2a6df
            | 0x2a700..=0x2b81d
            | 0x2b820..=0x2cead
            | 0x2ceb0..=0x2ebe0
            | 0x2f800..=0x2fa1d
            | 0x30000..=0x3134a
            | 0x31350..=0x323af
    )
}

#[inline(always)]
fn is_joy_cjk(character: char) -> bool {
    matches!(character as u32, 0x4e00..=0x9fa5 | 0x3040..=0x309f | 0x30a0..=0x30ff)
}

#[inline(always)]
fn is_letter_or_mark(character: char) -> bool {
    is_letter(character) || matches!(get_general_category(character), GeneralCategory::NonspacingMark | GeneralCategory::SpacingMark | GeneralCategory::EnclosingMark)
}

#[inline(always)]
fn is_kimi_upper(character: char) -> bool {
    !is_han(character) && is_minimax_upper(character)
}

#[inline(always)]
fn is_kimi_lower(character: char) -> bool {
    !is_han(character) && is_minimax_lower(character)
}

#[inline]
fn kimi_word_end(input: &[u8], start: usize) -> Option<usize> {
    let (first, size) = char_at(input, start)?;
    let first_end = start + size;
    if is_kimi_lower(first) {
        return Some(scan(input, first_end, is_kimi_lower));
    }
    if !is_kimi_upper(first) {
        return None;
    }
    let mut end = scan(input, first_end, is_kimi_upper);
    if let Some((next, size)) = char_at(input, end)
        && is_kimi_lower(next)
    {
        end = scan(input, end + size, is_kimi_lower);
    }
    Some(end)
}

#[inline(always)]
fn is_newline(character: char) -> bool {
    matches!(character, '\r' | '\n')
}

#[inline(always)]
fn is_symbol(character: char) -> bool {
    !character.is_whitespace() && !is_letter(character) && !is_number(character)
}

#[cfg(test)]
mod tests {

    #[test]
    fn duplicate_vocab_key_prefers_special_token() {
        // 完整字节字母表 + merge token(id 256,"aa")与 added/special token(id 257)同串:
        // special 胜出,不再报 duplicate vocab key。
        let mut tokens: Vec<String> = (0..=255u8).map(|byte| encode_byte(byte).to_string()).collect();
        let mut special = vec![false; 256];
        tokens.push("laguna".to_owned());
        special.push(false);
        tokens.push("laguna".to_owned());
        special.push(true);
        let merges: Vec<String> = Vec::new();
        let tokenizer = Tokenizer::from_bpe_tokens(&tokens, &merges, &special).expect("重复 vocab key 应被去重");
        assert_eq!(tokenizer.tokenize(b"laguna"), vec![257]);
    }

    use super::*;
    use serde_json::json;

    fn encode_byte(byte: u8) -> char {
        let code = match byte {
            code @ (33..=126 | 161..=172 | 174..=255) => code as u32,
            code @ 0..=32 => code as u32 + 256,
            code @ 127..=160 => code as u32 + 162,
            173 => 323,
        };
        char::from_u32(code).unwrap()
    }

    fn minimax_tokenizer_json() -> Vec<u8> {
        let mut vocab = serde_json::Map::new();
        for byte in 0..=255u8 {
            vocab.insert(encode_byte(byte).to_string(), json!(byte));
        }

        let it = "it";
        let it_quote = "it'";
        let its = "it's";
        let e_acute = format!("{}{}", encode_byte(0xc3), encode_byte(0xa9));
        vocab.insert(it.to_owned(), json!(256));
        vocab.insert(it_quote.to_owned(), json!(257));
        vocab.insert(its.to_owned(), json!(258));
        vocab.insert(e_acute.clone(), json!(259));
        vocab.insert("camel".to_owned(), json!(300));
        vocab.insert("Case".to_owned(), json!(301));
        vocab.insert("camelCase".to_owned(), json!(302));

        serde_json::to_vec(&json!({
            "version": "1.0",
            "added_tokens": [{
                "id": 200000,
                "content": "<minimax>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [{
                    "type": "Split",
                    "pattern": {"Regex": "\\p{Lu}+\\p{Ll}+"},
                    "behavior": "Removed",
                    "invert": true
                }, {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true,
                    "use_regex": false
                }]
            },
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": [
                    ["i", "t"],
                    [it, "'"],
                    [it_quote, "s"],
                    [encode_byte(0xc3).to_string(), encode_byte(0xa9).to_string()]
                ]
            }
        }))
        .unwrap()
    }

    #[test]
    fn parallel_long_bpe_matches_serial_tokens() {
        let tokenizer = Tokenizer::from_slice(&minimax_tokenizer_json()).unwrap();
        let input = "camelCase it's cafe\u{301} <minimax> 12345\n".repeat(2048);
        let prepared = tokenizer.prepare_input(input.as_bytes());
        assert!(prepared.len() >= PARALLEL_BPE_MIN_BYTES);
        let serial = tokenizer.tokenize_prepared_serial(&prepared, true);
        let parallel = tokenizer.tokenize_prepared_parallel(&prepared, true);
        assert_eq!(parallel, serial);
        assert_eq!(tokenizer.tokenize(input.as_bytes()), serial);
        let serial_without_special = tokenizer.tokenize_prepared_serial(&prepared, false);
        assert_eq!(tokenizer.tokenize_prepared_parallel(&prepared, false), serial_without_special);
        assert_eq!(tokenizer.tokenize_with_special(input.as_bytes(), false), serial_without_special);
    }

    fn cl100k_tokenizer_json() -> Vec<u8> {
        let mut vocab = serde_json::Map::new();
        for byte in 0..=255u8 {
            vocab.insert(encode_byte(byte).to_string(), json!(byte));
        }
        vocab.insert(format!("{}1", encode_byte(b' ')), json!(256));

        serde_json::to_vec(&json!({
            "version": "1.0",
            "added_tokens": [],
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [{
                    "type": "Split",
                    "pattern": {
                        "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    },
                    "behavior": "Isolated",
                    "invert": false
                }, {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true,
                    "use_regex": false
                }]
            },
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": [[encode_byte(b' ').to_string(), "1"]]
            }
        }))
        .unwrap()
    }

    #[test]
    fn minimax_uses_vocab_byte_ids_and_dynamic_added_tokens() {
        let tokenizer = Tokenizer::from_slice(&minimax_tokenizer_json()).unwrap();
        assert_eq!(tokenizer.tokenize("🙂".as_bytes()), vec![0xf0, 0x9f, 0x99, 0x82]);
        assert_eq!(tokenizer.tokenize(b"a<minimax>b"), vec![b'a' as u32, 200000, b'b' as u32]);
    }

    #[test]
    fn cl100k_does_not_merge_space_with_number() {
        let tokenizer = Tokenizer::from_slice(&cl100k_tokenizer_json()).unwrap();
        assert_eq!(tokenizer.tokenize(b" 1"), vec![b' ' as u32, b'1' as u32]);
    }

    #[test]
    fn gguf_glm4_presets_use_cl100k_pretokenizer() {
        let mut tokens = (0..=255u8).map(|byte| encode_byte(byte).to_string()).collect::<Vec<_>>();
        tokens.push(format!("{}1", encode_byte(b' ')));
        let merges = [format!("{} 1", encode_byte(b' '))];
        let special = vec![false; tokens.len()];

        for preset in ["glm4", "chatglm-bpe"] {
            let tokenizer = Tokenizer::from_gguf_bpe_tokens(&tokens, &merges, &special, Some(preset)).unwrap();
            assert_eq!(tokenizer.tokenize(b" 1"), vec![b' ' as u32, b'1' as u32]);
        }
        let legacy = Tokenizer::from_gguf_bpe_tokens(&tokens, &merges, &special, None).unwrap();
        assert_eq!(legacy.tokenize(b" 1"), vec![256]);
    }

    #[test]
    fn minimax_applies_nfc_and_keeps_contractions_in_one_piece() {
        let tokenizer = Tokenizer::from_slice(&minimax_tokenizer_json()).unwrap();
        assert_eq!(tokenizer.tokenize(b"it's"), vec![258]);
        assert_eq!(tokenizer.tokenize("e\u{301}".as_bytes()), vec![259]);
        assert_eq!(tokenizer.tokenize(b"camelCase"), vec![300, 301]);
    }

    #[test]
    fn minimax_detokenizer_obeys_special_flag() {
        let bytes = minimax_tokenizer_json();
        let detokenizer = Detokenizer::from_slice(&bytes).unwrap();
        assert_eq!(detokenizer.decode_bytes(&[b'a' as u32, 200000, b'b' as u32], true).unwrap(), b"ab");
        assert_eq!(detokenizer.decode_bytes(&[b'a' as u32, 200000, b'b' as u32], false).unwrap(), b"a<minimax>b");
    }

    #[test]
    fn damaged_tokenizer_tables_return_errors() {
        assert!(Vocab::new([(b"x".to_vec(), 0), (b"x".to_vec(), 1)]).is_err());
        assert!(Merges::new([(MERGE_ID_LIMIT, 1, 2)]).is_err());
        assert!(Merges::new([(1, 2, 3), (1, 2, 4)]).is_err());

        let vocab = Vocab::new([(b"x".to_vec(), 0)]).unwrap();
        let merges = Merges::new([]).unwrap();
        let error = Bpe::new(vocab, merges).err().expect("缺少 ByteLevel 字节必须报错");
        assert!(error.to_string().contains("缺少字节 0"));
    }

    #[test]
    fn utf8_stream_decoder_joins_token_byte_fragments() {
        let mut stream = Utf8StreamDecoder::default();
        assert_eq!(stream.push(&[0xe2]), "");
        assert_eq!(stream.push(&[0x82]), "");
        assert_eq!(stream.push(&[0xac, b'!']), "€!");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn utf8_stream_decoder_replaces_invalid_bytes_and_discards_incomplete_tail() {
        let mut stream = Utf8StreamDecoder::default();
        assert_eq!(stream.push(b"ok\xfftail"), "ok�tail");
        assert_eq!(stream.push(&[0xe2]), "");
        assert_eq!(stream.finish(), "");
    }

    fn base64(data: &[u8]) -> String {
        const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut output = String::new();
        for chunk in data.chunks(3) {
            let value = ((chunk[0] as u32) << 16) | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8) | chunk.get(2).copied().unwrap_or(0) as u32;
            output.push(TABLE[((value >> 18) & 63) as usize] as char);
            output.push(TABLE[((value >> 12) & 63) as usize] as char);
            output.push(if chunk.len() > 1 { TABLE[((value >> 6) & 63) as usize] as char } else { '=' });
            output.push(if chunk.len() > 2 { TABLE[(value & 63) as usize] as char } else { '=' });
        }
        output
    }

    fn kimi_tiktoken_model() -> Vec<u8> {
        let mut tokens = (0..=255u8).map(|byte| vec![byte]).collect::<Vec<_>>();
        tokens.extend([b"it".to_vec(), b"it'".to_vec(), b"it's".to_vec(), "你".as_bytes().to_vec()]);
        let mut model = String::new();
        for (rank, token) in tokens.iter().enumerate() {
            model.push_str(&format!("{} {rank}\n", base64(token)));
        }
        model.into_bytes()
    }

    #[test]
    fn kimi_tiktoken_handles_han_contractions_and_special_control() {
        let model = kimi_tiktoken_model();
        let special = HashMap::from([(260, "[BOS]".to_owned())]);
        let tokenizer = Tokenizer::from_tiktoken_slice(&model, &special).unwrap();
        assert_eq!(tokenizer.tokenize(b"it's"), vec![258]);
        assert_eq!(tokenizer.tokenize("你".as_bytes()), vec![259]);
        assert_eq!(tokenizer.tokenize(b"[BOS]"), vec![260]);
        assert_ne!(tokenizer.tokenize_with_special(b"[BOS]", false), vec![260]);

        let detokenizer = Detokenizer::from_tiktoken_slice(&model, &special).unwrap();
        assert_eq!(detokenizer.decode_bytes(&[258, 259], false).unwrap(), "it's你".as_bytes());
        assert_eq!(detokenizer.decode_bytes(&[260], true).unwrap(), b"");
    }

    // 合成最小 gemma4 GGUF 词表：0/1 是 special，2..=257 是 <0x00>..<0xFF> byte fallback，
    // 其余是字面文本 token(空格用 ▁ 表示)。
    fn gemma4_gguf_parts() -> (Vec<String>, Vec<String>, Vec<bool>) {
        let mut tokens = vec!["<bos>".to_owned(), "<eos>".to_owned()];
        for byte in 0..=255u8 {
            tokens.push(format!("<0x{byte:02X}>"));
        }
        tokens.extend(["▁", "a", "b", "▁a", "ab", "你", "好", "你好", "▁你", "▁你好"].iter().map(|&token| token.to_owned()));
        let merges = ["▁ a", "a b", "你 好", "▁ 你", "▁你 好", "▁ 你好"].iter().map(|&merge| merge.to_owned()).collect();
        let special = tokens.iter().map(|token| token.starts_with('<') && !token.starts_with("<0x")).collect();
        (tokens, merges, special)
    }

    const GEMMA4_A: u32 = 259;
    const GEMMA4_B: u32 = 260;
    const GEMMA4_AB: u32 = 262;
    const GEMMA4_SPACE_NIHAO: u32 = 267;

    #[test]
    fn gemma4_gguf_replaces_space_and_merges_whole_piece() {
        let (tokens, merges, special) = gemma4_gguf_parts();
        let tokenizer = Tokenizer::from_gguf_gemma4_tokens(&tokens, &merges, &special).unwrap();
        // 空格先替换为 ▁，整个输入作为一个 pretoken 做 BPE。
        assert_eq!(tokenizer.tokenize("ab 你好".as_bytes()), vec![GEMMA4_AB, GEMMA4_SPACE_NIHAO]);
        assert_eq!(tokenizer.tokenize(b"a<bos>b"), vec![GEMMA4_A, 0, GEMMA4_B]);

        let detokenizer = Detokenizer::from_gguf_gemma4_tokens(&tokens, &special).unwrap();
        // ▁ 还原为空格，special token 按原文输出或被跳过。
        assert_eq!(detokenizer.decode_bytes(&[GEMMA4_AB, GEMMA4_SPACE_NIHAO], false).unwrap(), "ab 你好".as_bytes());
        assert_eq!(detokenizer.decode_bytes(&[0, GEMMA4_A], true).unwrap(), b"a");
        assert_eq!(detokenizer.decode_bytes(&[0, GEMMA4_A], false).unwrap(), b"<bos>a");
    }

    #[test]
    fn gemma4_gguf_byte_fallback_roundtrips_unknown_bytes() {
        let (tokens, merges, special) = gemma4_gguf_parts();
        let tokenizer = Tokenizer::from_gguf_gemma4_tokens(&tokens, &merges, &special).unwrap();
        // 🙂 不在词表：逐字节回退到 <0xXX> token(id 从 2 开始)。
        let ids = tokenizer.tokenize("🙂".as_bytes());
        assert_eq!(ids, vec![2 + 0xf0, 2 + 0x9f, 2 + 0x99, 2 + 0x82]);

        let detokenizer = Detokenizer::from_gguf_gemma4_tokens(&tokens, &special).unwrap();
        assert_eq!(detokenizer.decode_bytes(&ids, false).unwrap(), "🙂".as_bytes());
    }
}
