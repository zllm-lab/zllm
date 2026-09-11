//! SenseVoice CPU oracle 组合入口：WAV → fbank/LFR/CMVN/PE → encoder → CTC 文本。
//!
//! 既是正确性基准（HTP 图的执行后审计），也是跨后端一致性的锚点。

use std::path::Path;

use super::{SenseVoiceLanguage, SenseVoiceTextNorm, decode, frontend, sensevoice_encode};
use crate::backend::VaeBackend;
use crate::backend::cpu::CpuContext;
use crate::model_spec::sensevoice::SenseVoiceConfig;
use crate::weight::model::sensevoice::SenseVoiceWeights;

pub struct SenseVoiceTranscription {
    pub text: String,
    /// CTC 贪心合并后的转写 token 序列（已剔除控制位与 blank）。
    pub token_ids: Vec<u32>,
    /// 逐帧 argmax 的原始 id 序列（含 4 个控制位，供 HTP 图逐帧审计）。
    pub frame_ids: Vec<u32>,
}

/// 转写单个 WAV 文件（mono 16kHz 或可重采样）。
pub fn transcribe(config: &SenseVoiceConfig, weights: &SenseVoiceWeights, wav: &Path, language: SenseVoiceLanguage, textnorm: SenseVoiceTextNorm) -> Result<SenseVoiceTranscription, String> {
    let samples = crate::audio::read_wav_16khz(wav)?;
    if samples.is_empty() {
        return Err(format!("SenseVoice 输入 {} 为空", wav.display()));
    }
    // FunASR WavFrontend upscale_samples：[-1,1] × 32768 后提 fbank
    let samples: Vec<f32> = samples.iter().map(|&value| value * 32768.0).collect();

    let fbank = frontend::FbankState::new(config);
    let frames = fbank.compute(config, &samples)?;
    let frame_rows = frames.len() / config.mel_bins;
    let (mut lfr, lfr_rows) = frontend::apply_lfr(config, &frames, frame_rows)?;
    frontend::apply_cmvn(&weights.cmvn()?, &mut lfr)?;
    let query = weights.query_table()?;
    let input = frontend::assemble_encoder_input(config, &query, language, textnorm, &lfr, lfr_rows)?;

    let backend = CpuContext;
    let rows = lfr_rows + config.query_prefix;
    let tensor = backend.vae_tensor_from_f32(input, rows, config.encoder_input_dim()).map_err(|error| format!("SenseVoice 上传输入失败: {error:?}"))?;
    let started = std::time::Instant::now();
    let logits = sensevoice_encode(&backend, config, weights, tensor).map_err(|error| format!("SenseVoice encoder 失败: {error:?}"))?;
    let elapsed = started.elapsed();
    let logits = backend.vae_tensor_to_f32(&logits).map_err(|error| format!("SenseVoice 下载 logits 失败: {error:?}"))?;
    let vocab = weights.vocab_size();
    let frame_ids = logits.chunks_exact(vocab).map(|row| row.iter().copied().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(id, _)| id as u32).unwrap_or(0)).collect::<Vec<_>>();
    let token_ids = decode::ctc_greedy(&frame_ids, config.blank_id, config.query_prefix);
    let text = weights.tokens()?.decode(&token_ids);
    eprintln!("[sensevoice-cpu] frames={frame_rows} lfr_rows={lfr_rows} encode={elapsed:?} tokens={}", token_ids.len());
    Ok(SenseVoiceTranscription { text, token_ids, frame_ids })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// 真权重 + sherpa 测试音频对拍（资产缺失时跳过，保持 cargo test --lib 无权重全绿）。
    /// 资产：/path/to/sensevoice-small-f32/{model.safetensors,tokens.txt,am.mvn}
    /// 与 test_wavs（sherpa-onnx-sense-voice 发布包）。期望文本在首次转写并与
    /// sherpa 官方输出人工核对后钉死（记录见 docs/sensevoice-qnn-android.md）。
    #[test]
    fn transcribes_sherpa_test_wavs() {
        let root = PathBuf::from("/path/to/sensevoice-small-f32");
        if !root.join("model.safetensors").is_file() {
            eprintln!("skip: {} 不存在", root.display());
            return;
        }
        // zh/en 与 sherpa 官方文档输出逐字一致（docs/onnx/sense-voice/code/2024-07-17.txt）；
        // ja/ko/yue 为本 oracle 首次转写（多语种流畅正确）后钉死。
        let expected: &[(&str, &str)] = &[
            ("zh.wav", "开饭时间早上九点至下午五点"),
            ("en.wav", "the tribal chieftain called for the boy and presented him with fifty pieces of gold"),
            ("ja.wav", "うちの中学は弁当制で持っていきない場合は50円の学校販売のパンを買う"),
            ("ko.wav", "조금만 생각을 하면서 살면 훨씬 편할 거야"),
            ("yue.wav", "呢几个字都表达唔到我想讲嘅意思"),
        ];
        let config = SenseVoiceConfig::standard();
        let weights = SenseVoiceWeights::open(&root).unwrap();
        for (wav, expected) in expected {
            let path = root.join("test_wavs").join(wav);
            if !path.is_file() {
                continue;
            }
            let result = transcribe(&config, &weights, &path, SenseVoiceLanguage::Auto, SenseVoiceTextNorm::WithoutItn).unwrap();
            eprintln!("[sensevoice-cpu] {wav}: {}", result.text);
            assert_eq!(result.text, *expected, "{wav} 转写与钉定期望不一致");
        }
    }
}
