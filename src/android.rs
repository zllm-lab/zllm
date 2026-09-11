//! Android App 的最小 C ABI：zLLM 持有对话模板、GGUF tokenizer 与 detokenizer。

use std::{
    ffi::{CStr, CString, c_char},
    panic::catch_unwind,
    path::PathBuf,
    ptr,
    sync::Mutex,
    time::SystemTime,
};

use crate::tokenizer::Detokenizer;
use crate::weight::container::gguf::GgufReader;

fn owned(value: Result<String, String>) -> *mut c_char {
    let text = match value {
        Ok(value) => value,
        Err(error) => format!("ERROR: {error}"),
    };
    CString::new(text.replace('\0', "")).map_or(ptr::null_mut(), CString::into_raw)
}

unsafe fn input<'a>(value: *const c_char, name: &str) -> Result<&'a str, String> {
    if value.is_null() {
        return Err(format!("{name} 为空"));
    }
    unsafe { CStr::from_ptr(value) }.to_str().map_err(|error| format!("{name} 不是 UTF-8: {error}"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zllm_minicpm5_prompt(user: *const c_char) -> *mut c_char {
    catch_unwind(|| {
        owned((|| {
            let user = unsafe { input(user, "user")? };
            Ok(crate::runtime::minicpm5::minicpm5_instruct_prompt(user, None, false))
        })())
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zllm_minicpm5_encode(gguf: *const c_char, prompt: *const c_char) -> *mut c_char {
    // 计数、模板拼接和正式输入共用词表，避免一次对话反复构建 BPE 合并表。
    static ENCODER: Mutex<Option<((PathBuf, u64, SystemTime), crate::tokenizer::Tokenizer)>> = Mutex::new(None);
    catch_unwind(|| {
        owned((|| {
            let path = std::path::Path::new(unsafe { input(gguf, "gguf")? });
            let metadata = path.metadata().map_err(|error| format!("读取 tokenizer 文件 {}: {error}", path.display()))?;
            let identity = (path.to_owned(), metadata.len(), metadata.modified().map_err(|error| error.to_string())?);
            let mut encoder = ENCODER.lock().map_err(|error| format!("tokenizer 缓存不可用: {error}"))?;
            if encoder.as_ref().is_none_or(|(saved, _)| *saved != identity) {
                let reader = GgufReader::open(path).map_err(|error| error.to_string())?;
                *encoder = Some((identity, reader.bpe_tokenizer()?));
            }
            let tokens = encoder.as_ref().unwrap().1.tokenize(unsafe { input(prompt, "prompt")? }.as_bytes());
            serde_json::to_string(&tokens).map_err(|error| error.to_string())
        })())
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zllm_minicpm5_decode(gguf: *const c_char, tokens_json: *const c_char) -> *mut c_char {
    // 流式显示会逐 token 调用；仅缓存词表，文件更新时重新加载，不重复解析整份 GGUF。
    static DECODER: Mutex<Option<((PathBuf, u64, SystemTime), Detokenizer)>> = Mutex::new(None);
    catch_unwind(|| {
        owned((|| {
            let path = std::path::Path::new(unsafe { input(gguf, "gguf")? });
            let metadata = path.metadata().map_err(|error| format!("读取 tokenizer 文件 {}: {error}", path.display()))?;
            let identity = (path.to_owned(), metadata.len(), metadata.modified().map_err(|error| error.to_string())?);
            let mut decoder = DECODER.lock().map_err(|error| format!("detokenizer 缓存不可用: {error}"))?;
            if decoder.as_ref().is_none_or(|(saved, _)| *saved != identity) {
                let reader = GgufReader::open(path).map_err(|error| error.to_string())?;
                *decoder = Some((identity, reader.bpe_detokenizer()?));
            }
            let tokens: Vec<u32> = serde_json::from_str(unsafe { input(tokens_json, "tokens")? }).map_err(|error| error.to_string())?;
            let bytes = decoder.as_ref().unwrap().1.decode_bytes(&tokens, true).map_err(|error| error.to_string())?;
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        })())
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zllm_string_free(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}

/// 录音仅在这里做音频前处理；完整 encoder / CTC head 由 QNN HTP 执行。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zllm_sensevoice_prepare(model: *const c_char, pcm: *const c_char, output: *const c_char, rows: u32) -> *mut c_char {
    catch_unwind(|| {
        owned((|| {
            use crate::model_spec::sensevoice::{SenseVoiceLanguage, SenseVoiceTextNorm};
            use crate::runtime::sensevoice::frontend::{FbankState, apply_cmvn, apply_lfr, assemble_encoder_input};
            let weights = crate::weight::model::sensevoice::SenseVoiceWeights::open(std::path::Path::new(unsafe { input(model, "ASR model")? }))?;
            let config = weights.config();
            let rows = rows as usize;
            if rows <= config.query_prefix || rows > 2048 {
                return Err(format!("ASR 图输入行数无效: {rows}"));
            }
            let pcm = std::fs::read(unsafe { input(pcm, "PCM path")? }).map_err(|e| format!("读取录音: {e}"))?;
            if pcm.len() % 2 != 0 || pcm.len() < config.sample_rate / 5 * 2 || pcm.len() > config.sample_rate * 30 * 2 {
                return Err("录音需要为 0.2–30 秒、16kHz 单声道 PCM16".into());
            }
            let samples: Vec<f32> = pcm.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32).collect();
            let output = std::path::Path::new(unsafe { input(output, "ASR output")? });
            std::fs::create_dir_all(output).map_err(|e| format!("创建 ASR 临时目录: {e}"))?;
            let chunk_samples = ((rows - config.query_prefix) * config.lfr_stride - 1) * config.frame_shift() + config.frame_length();
            let fbank = FbankState::new(config);
            let cmvn = weights.cmvn()?;
            let query = weights.query_table()?;
            let mut files = Vec::new();
            // 尾段向前取足窗口，保留短语上下文，避免把句尾独立识别成无语音。
            let mut starts = vec![0];
            let stride = chunk_samples.saturating_sub(config.sample_rate / 2).max(1);
            while starts.last().unwrap() + chunk_samples < samples.len() {
                starts.push((starts.last().unwrap() + stride).min(samples.len().saturating_sub(chunk_samples)));
            }
            for (i, &start) in starts.iter().enumerate() {
                let chunk = &samples[start..(start + chunk_samples).min(samples.len())];
                // 延续 LFR 的尾帧复制规则；数字静音会产生极端 log-mel 值，不能用来填满静态图。
                let mut padded = chunk.to_vec();
                padded.resize(chunk.len().max(config.frame_length()), *chunk.last().unwrap());
                let frames = fbank.compute(config, &padded)?;
                let (mut lfr, lfr_rows) = apply_lfr(config, &frames, frames.len() / config.mel_bins)?;
                apply_cmvn(&cmvn, &mut lfr)?;
                let tail = lfr[(lfr_rows - 1) * config.encoder_input_dim()..].to_vec();
                for _ in lfr_rows..rows - config.query_prefix {
                    lfr.extend_from_slice(&tail);
                }
                let encoded = assemble_encoder_input(config, &query, SenseVoiceLanguage::Auto, SenseVoiceTextNorm::WithoutItn, &lfr, rows - config.query_prefix)?;
                let path = output.join(format!("input_{i}.f32"));
                let bytes: Vec<u8> = encoded.into_iter().flat_map(f32::to_le_bytes).collect();
                std::fs::write(&path, bytes).map_err(|e| format!("写入 ASR 特征: {e}"))?;
                files.push(path);
            }
            serde_json::to_string(&files).map_err(|e| e.to_string())
        })())
    })
    .unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zllm_sensevoice_decode(model: *const c_char, frames: *const c_char) -> *mut c_char {
    catch_unwind(|| {
        owned((|| {
            let weights = crate::weight::model::sensevoice::SenseVoiceWeights::open(std::path::Path::new(unsafe { input(model, "ASR model")? }))?;
            let frames: Vec<Vec<u32>> = serde_json::from_str(unsafe { input(frames, "ASR frame ids")? }).map_err(|e| e.to_string())?;
            let vocab = weights.tokens()?;
            let mut text = String::new();
            for ids in frames {
                if ids.iter().any(|&id| id as usize >= weights.vocab_size()) {
                    return Err("ASR CTC id 超出词表".into());
                }
                let ids = crate::runtime::sensevoice::decode::ctc_greedy(&ids, 0, weights.config().query_prefix);
                let decoded = vocab.decode(&ids);
                let decoded = decoded.trim();
                // 相邻音频窗有重叠；只合并真实匹配的文本，不猜测或删除不匹配的词。
                let overlap = decoded.char_indices().map(|(i, _)| i).chain(std::iter::once(decoded.len())).filter(|&i| i > 0 && decoded[..i].chars().count() >= 2 && text.ends_with(&decoded[..i])).max().unwrap_or(0);
                if overlap == 0 && text.ends_with(|c: char| c.is_ascii_alphanumeric()) && decoded.starts_with(|c: char| c.is_ascii_alphanumeric()) {
                    text.push(' ');
                }
                text.push_str(&decoded[overlap..]);
            }
            Ok(text.trim().to_owned())
        })())
    })
    .unwrap_or(ptr::null_mut())
}
