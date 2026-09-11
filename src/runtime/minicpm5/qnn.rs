//! MiniCPM5 Android QNN HTP 正式进程组合。
//!
//! 模型计算由设备侧 QNN engine 完成；本模块只负责 GGUF tokenizer、运行配置、
//! 子进程生命周期与结果核验。任何 QNN 图失败都会直接返回错误，不存在 CPU 算子回退。

use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::model_spec::sensevoice::{SenseVoiceLanguage, SenseVoiceTextNorm};
use crate::weight::container::gguf::GgufReader;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QnnRuntimeConfig {
    version: u32,
    model: ModelConfig,
    backend: BackendConfig,
    /// 可选 ASR 验收段（SenseVoice）：设备图转写 + 进程内 CPU oracle 审计。
    #[serde(default)]
    asr: Option<AsrConfig>,
}

/// SenseVoice ASR 验收配置。`weights` 是设备侧 F32 safetensors 资产目录
/// （model.safetensors + tokens.txt + am.mvn），同一目录同时供 CPU oracle 审计。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AsrConfig {
    weights: PathBuf,
    wav: PathBuf,
    /// ASR 图执行器（带 sensevoice 算子分解的 replay 构建）；缺省复用 LLM runner。
    #[serde(default)]
    runner: Option<PathBuf>,
    runner_config: PathBuf,
    #[serde(default = "default_asr_language")]
    language: String,
    #[serde(default = "default_asr_text_norm")]
    text_norm: String,
    /// 可选钉定期望；缺省只要求与 CPU oracle 一致。
    #[serde(default)]
    expected_text: Option<String>,
}

fn default_asr_language() -> String {
    "auto".to_owned()
}

fn default_asr_text_norm() -> String {
    "woitn".to_owned()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    weights: PathBuf,
    prompt: String,
    prompt_tokens: Vec<u32>,
    expected_tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendConfig {
    runner: PathBuf,
    verify_config: PathBuf,
    decode_config: PathBuf,
    library_path: String,
    adsp_library_path: String,
}

impl QnnRuntimeConfig {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config: Self = serde_yaml::from_slice(&fs::read(path)?)?;
        if config.version != 1 {
            return Err(format!("MiniCPM5 QNN 配置 version={}，期望 1", config.version).into());
        }
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let mut relative = vec![&mut config.model.weights, &mut config.backend.runner, &mut config.backend.verify_config, &mut config.backend.decode_config];
        if let Some(asr) = &mut config.asr {
            relative.extend([&mut asr.weights, &mut asr.wav, &mut asr.runner_config]);
            if let Some(runner) = &mut asr.runner
                && runner.is_relative()
            {
                *runner = base.join(&*runner);
            }
        }
        for item in relative {
            if item.is_relative() {
                *item = base.join(&*item);
            }
        }
        if config.model.prompt_tokens.is_empty() || config.model.expected_tokens.len() < 3 {
            return Err("MiniCPM5 QNN 配置至少需要一个 prompt token 和三个 expected token".into());
        }
        Ok(config)
    }

    pub fn check(&self) -> Result<(), Box<dyn std::error::Error>> {
        for (name, path) in [("GGUF", &self.model.weights), ("QNN engine", &self.backend.runner), ("端到端配置", &self.backend.verify_config), ("常驻解码配置", &self.backend.decode_config)] {
            if !path.is_file() {
                return Err(format!("{name} 文件不存在: {}", path.display()).into());
            }
        }
        if let Some(asr) = &self.asr {
            for (name, path) in [
                ("ASR 权重", &asr.weights.join("model.safetensors")),
                ("ASR 词表", &asr.weights.join("tokens.txt")),
                ("ASR CMVN", &asr.weights.join("am.mvn")),
                ("ASR 音频", &asr.wav),
                ("ASR 图配置", &asr.runner_config),
                ("ASR 执行器", asr.runner.as_ref().unwrap_or(&self.backend.runner)),
            ] {
                if !path.is_file() {
                    return Err(format!("{name} 文件不存在: {}", path.display()).into());
                }
            }
            SenseVoiceLanguage::parse(&asr.language)?;
            SenseVoiceTextNorm::parse(&asr.text_norm)?;
        }
        let reader = GgufReader::open(&self.model.weights)?;
        let actual = reader.bpe_tokenizer()?.tokenize(self.model.prompt.as_bytes());
        if actual != self.model.prompt_tokens {
            return Err(format!("prompt 与 QNN calibration 不一致: actual={actual:?} expected={:?}", self.model.prompt_tokens).into());
        }
        Ok(())
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.check()?;
        eprintln!("[minicpm5-qnn] prompt_tokens={} device=HTP cpu_fallback=false", self.model.prompt_tokens.len());
        if let Some(asr) = &self.asr {
            self.run_asr(asr)?;
        }

        let verified = self.invoke(&self.backend.verify_config, true)?;
        let actual = parse_stage_tokens(&verified);
        let staged_expected = &self.model.expected_tokens[..3];
        if actual != staged_expected {
            return Err(format!("QNN HTP 端到端 token 不一致: actual={actual:?} expected={staged_expected:?}").into());
        }
        eprintln!("[minicpm5-qnn-e2e] tokens={actual:?} status=passed");

        let resident = self.invoke(&self.backend.decode_config, false)?;
        let decoded = parse_resident_tokens(&resident);
        if decoded != self.model.expected_tokens[1..] {
            return Err(format!("QNN HTP 连续解码 token 不一致: actual={decoded:?} expected={:?}", &self.model.expected_tokens[1..]).into());
        }
        let milliseconds = parse_float_field(&resident, "mean_ms=").ok_or("常驻图缺少 HOT mean_ms")?;
        let tokens_per_second = 1000.0 / milliseconds;
        let reader = GgufReader::open(&self.model.weights)?;
        let text = String::from_utf8_lossy(&reader.bpe_detokenizer()?.decode_bytes(&self.model.expected_tokens, true)?).into_owned();
        println!("{text}");
        eprintln!("[minicpm5-qnn-summary] generated_tokens={} decode_ms={milliseconds:.3} tok_per_s={tokens_per_second:.3} cpu_fallback=false", self.model.expected_tokens.len());
        if tokens_per_second < 10.0 {
            return Err(format!("QNN HTP 吞吐 {tokens_per_second:.3} tok/s 低于验收线 10 tok/s").into());
        }
        Ok(())
    }

    /// SenseVoice ASR 验收：设备侧 HTP 图转写，随后本进程 CPU oracle 执行后审计。
    fn run_asr(&self, asr: &AsrConfig) -> Result<(), Box<dyn std::error::Error>> {
        let runner = asr.runner.as_ref().unwrap_or(&self.backend.runner);
        let output = self.invoke_with(runner, &asr.runner_config, false)?;
        let device_text = output.lines().find_map(|line| line.strip_prefix("SENSEVOICE ")?.split("text=").nth(1).map(str::to_owned)).ok_or("ASR 图输出缺少 SENSEVOICE text=")?;
        let weights = crate::weight::model::sensevoice::SenseVoiceWeights::open(&asr.weights)?;
        let oracle = crate::runtime::sensevoice::cpu::transcribe(weights.config(), &weights, &asr.wav, SenseVoiceLanguage::parse(&asr.language)?, SenseVoiceTextNorm::parse(&asr.text_norm)?)?;
        eprintln!("[sensevoice-qnn] device=\"{device_text}\" oracle=\"{}\"", oracle.text);
        if device_text != oracle.text {
            return Err(format!("SenseVoice HTP 转写与 CPU oracle 不一致:\ndevice=\"{device_text}\"\noracle=\"{}\"", oracle.text).into());
        }
        if let Some(expected) = &asr.expected_text
            && device_text != *expected
        {
            return Err(format!("SenseVoice HTP 转写与钉定期望不一致:\ndevice=\"{device_text}\"\nexpected=\"{expected}\"").into());
        }
        eprintln!("[sensevoice-qnn-e2e] status=passed tokens={}", oracle.token_ids.len());
        Ok(())
    }

    fn invoke(&self, config: &Path, accept_completed_audit: bool) -> Result<String, Box<dyn std::error::Error>> {
        self.invoke_with(&self.backend.runner, config, accept_completed_audit)
    }

    fn invoke_with(&self, runner: &Path, config: &Path, accept_completed_audit: bool) -> Result<String, Box<dyn std::error::Error>> {
        let output =
            Command::new(runner).args(["--config", config.as_os_str().to_str().ok_or("QNN 配置路径不是 UTF-8")?]).env("LD_LIBRARY_PATH", &self.backend.library_path).env("ADSP_LIBRARY_PATH", &self.backend.adsp_library_path).output()?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        // W4 端到端审计还会报告逐元素量化偏差；只要所有阶段执行完，最终
        // token 判定由调用方严格核对。常驻性能图必须原样返回 0。
        if !output.status.success() && !(accept_completed_audit && stdout.contains("stages=") && stdout.contains("TOKEN actual=")) {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("QNN engine 失败 status={}:\n{}\n{}", output.status, tail(&stdout), tail(&stderr)).into());
        }
        Ok(stdout)
    }
}

fn parse_stage_tokens(output: &str) -> Vec<u32> {
    output.lines().filter_map(|line| line.strip_prefix("TOKEN actual=")).filter_map(|value| value.split_whitespace().next()?.parse().ok()).collect()
}

fn parse_resident_tokens(output: &str) -> Vec<u32> {
    output.lines().filter(|line| line.starts_with("RESIDENT_DECODE ") || line.starts_with("RESIDENT_CONTINUOUS ")).filter_map(|line| line.split(" token=").nth(1)?.split_whitespace().next()?.parse().ok()).collect()
}

fn parse_float_field(output: &str, prefix: &str) -> Option<f64> {
    output.lines().find_map(|line| line.find(prefix).and_then(|at| line[at + prefix.len()..].split_whitespace().next()?.parse().ok()))
}

fn tail(text: &str) -> &str {
    let start = text.char_indices().rev().nth(3999).map_or(0, |(index, _)| index);
    &text[start..]
}
