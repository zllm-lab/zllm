//! SenseVoice-Small safetensors loader。
//!
//! 资产由 sherpa-onnx 官方 fp32 ONNX 一次性转换（脚本在 zllm-bench），目录布局：
//! - `model.safetensors`：F32 权重，键名保留 FunASR state_dict 路径
//!   （`embed.weight`、`encoder.encoders0.0.*`、`encoder.encoders.{i}.*`、
//!   `encoder.tp_encoders.{i}.*`、`encoder.after_norm.*`、`encoder.tp_norm.*`、`ctc.ctc_lo.*`）
//! - `tokens.txt`：sherpa 词表，每行 `<piece> <id>`
//! - `am.mvn`：kaldi 全局 CMVN（`<AddShift>`/`<Rescale>`，560 维，LFR 之后施加）
//!
//! 已验证目标：sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17（25035 词表）。
//! 结构常量见 `model_spec::sensevoice`；本文件只做命名映射、shape 校验与惰性加载。

use std::path::{Path, PathBuf};

use crate::model_spec::sensevoice::SenseVoiceConfig;
use crate::weight::container::safetensor::SafetensorStore;

/// 全局 CMVN：LFR 后 560 维逐维 `(x + shift) * scale`。
pub struct SenseVoiceCmvn {
    pub shift: Vec<f32>,
    pub scale: Vec<f32>,
}

/// 单层 SAN-M 权重（host 侧，按层惰性加载）。
///
/// 布局约定（与 torch state_dict 一致，除 fsmn）：
/// - 线性权重 `[out, in]` 行优先（torch nn.Linear 原生布局，无需转置）
/// - `fsmn_weight` 已从 torch 的 `[channels, 1, kernel]`（通道主序）重排为
///   `[kernel][channels]`（tap 主序），与 `Backend::depthwise_conv1d` 约定一致
pub struct SenseVoiceLayerWeights {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub qkv_weight: Vec<f32>,
    pub qkv_bias: Vec<f32>,
    pub attn_out_weight: Vec<f32>,
    pub attn_out_bias: Vec<f32>,
    pub fsmn_weight: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub ffn_up_weight: Vec<f32>,
    pub ffn_up_bias: Vec<f32>,
    pub ffn_down_weight: Vec<f32>,
    pub ffn_down_bias: Vec<f32>,
}

/// 词表：id → piece（sherpa tokens.txt）。
pub struct SenseVoiceTokens {
    pub pieces: Vec<String>,
}

impl SenseVoiceTokens {
    /// piece 顺序拼接，`▁` 映射为空格（sentencepiece 语义）。
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(piece) = self.pieces.get(id as usize) {
                text.push_str(&piece.replace('▁', " "));
            }
        }
        text
    }
}

/// 整个 SenseVoice-Small 权重封装（safetensors store + 目录内伴生资产）。
pub struct SenseVoiceWeights {
    root: PathBuf,
    store: SafetensorStore,
    config: SenseVoiceConfig,
    vocab_size: usize,
}

impl SenseVoiceWeights {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let store = SafetensorStore::open(dir).map_err(|error| format!("打开 SenseVoice 目录 {} 失败: {error}", dir.display()))?;
        let config = SenseVoiceConfig::standard();
        config.validate()?;
        let model = Self { root: dir.to_path_buf(), store, config, vocab_size: 0 };
        let vocab_size = model.expect_ctc_vocab()?;
        let model = Self { vocab_size, ..model };
        model.validate_tensors()?;
        model.validate_sidecar_files()?;
        Ok(model)
    }

    pub fn config(&self) -> &SenseVoiceConfig {
        &self.config
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn tensor_shape(&self, name: &str) -> Result<Vec<usize>, String> {
        self.store.tensor_info(name).map(|info| info.shape).map_err(|error| format!("SenseVoice 缺少 tensor {name}: {error}"))
    }

    fn expect_shape(&self, name: &str, expected: &[usize]) -> Result<(), String> {
        let shape = self.tensor_shape(name)?;
        if shape.as_slice() != expected {
            return Err(format!("SenseVoice {name} shape {shape:?}，期望 {expected:?}"));
        }
        Ok(())
    }

    fn expect_ctc_vocab(&self) -> Result<usize, String> {
        let shape = self.tensor_shape("ctc.ctc_lo.weight")?;
        if shape.len() != 2 || shape[1] != self.config.d_model {
            return Err(format!("SenseVoice ctc.ctc_lo.weight shape {shape:?}，期望 [vocab, {}]", self.config.d_model));
        }
        Ok(shape[0])
    }

    /// 全部 tensor 的存在性与 shape 校验（只读元数据，不触碰数据区）。
    fn validate_tensors(&self) -> Result<(), String> {
        let cfg = &self.config;
        let input_dim = cfg.encoder_input_dim();
        let attention_dim = cfg.attention_heads * cfg.head_dim;
        let qkv_dim = 3 * attention_dim;
        self.expect_shape("embed.weight", &[cfg.query_embed_count, input_dim])?;
        self.expect_shape("ctc.ctc_lo.weight", &[self.vocab_size, cfg.d_model])?;
        self.expect_shape("ctc.ctc_lo.bias", &[self.vocab_size])?;
        self.expect_shape("encoder.after_norm.weight", &[cfg.d_model])?;
        self.expect_shape("encoder.after_norm.bias", &[cfg.d_model])?;
        self.expect_shape("encoder.tp_norm.weight", &[cfg.d_model])?;
        self.expect_shape("encoder.tp_norm.bias", &[cfg.d_model])?;
        for index in 0..cfg.total_layer_count() {
            let prefix = self.layer_prefix(index);
            // 首层 norm1/qkv 输入是 560 维 LFR 帧，其余层是 512。
            let input_width = if index == 0 { input_dim } else { cfg.d_model };
            self.expect_shape(&format!("{prefix}.norm1.weight"), &[input_width])?;
            self.expect_shape(&format!("{prefix}.norm1.bias"), &[input_width])?;
            self.expect_shape(&format!("{prefix}.self_attn.linear_q_k_v.weight"), &[qkv_dim, input_width])?;
            self.expect_shape(&format!("{prefix}.self_attn.linear_q_k_v.bias"), &[qkv_dim])?;
            self.expect_shape(&format!("{prefix}.self_attn.linear_out.weight"), &[attention_dim, attention_dim])?;
            self.expect_shape(&format!("{prefix}.self_attn.linear_out.bias"), &[attention_dim])?;
            self.expect_shape(&format!("{prefix}.self_attn.fsmn_block.weight"), &[attention_dim, 1, cfg.fsmn_kernel])?;
            self.expect_shape(&format!("{prefix}.norm2.weight"), &[cfg.d_model])?;
            self.expect_shape(&format!("{prefix}.norm2.bias"), &[cfg.d_model])?;
            self.expect_shape(&format!("{prefix}.feed_forward.w_1.weight"), &[cfg.ffn_dim, cfg.d_model])?;
            self.expect_shape(&format!("{prefix}.feed_forward.w_1.bias"), &[cfg.ffn_dim])?;
            self.expect_shape(&format!("{prefix}.feed_forward.w_2.weight"), &[cfg.d_model, cfg.ffn_dim])?;
            self.expect_shape(&format!("{prefix}.feed_forward.w_2.bias"), &[cfg.d_model])?;
        }
        Ok(())
    }

    fn validate_sidecar_files(&self) -> Result<(), String> {
        if !self.root.join("tokens.txt").is_file() {
            return Err(format!("SenseVoice 目录缺少 tokens.txt: {}", self.root.display()));
        }
        if !self.root.join("am.mvn").is_file() {
            return Err(format!("SenseVoice 目录缺少 am.mvn: {}", self.root.display()));
        }
        Ok(())
    }

    /// 0 = encoders0.0（首层，560 入）；1..layer_count = encoders.{i-1}；
    /// layer_count..layer_count+tp_layer_count = tp_encoders.{i-layer_count}。
    fn layer_prefix(&self, index: usize) -> String {
        layer_prefix(index, self.config.layer_count)
    }

    fn load_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        self.store.load(name).map_err(|error| format!("读取 SenseVoice {name} 失败: {error}"))?.to_f32().map_err(|error| format!("解码 SenseVoice {name} 失败: {error}"))
    }

    /// 16 行特殊 query embedding（lang/event/emo/textnorm），`[query_embed_count][input_dim]`。
    pub fn query_table(&self) -> Result<Vec<f32>, String> {
        self.load_f32("embed.weight")
    }

    pub fn load_layer(&self, index: usize) -> Result<SenseVoiceLayerWeights, String> {
        if index >= self.config.total_layer_count() {
            return Err(format!("SenseVoice layer {index} 越界，共 {} 层", self.config.total_layer_count()));
        }
        let prefix = self.layer_prefix(index);
        let channels = self.config.attention_heads * self.config.head_dim;
        let fsmn = self.load_f32(&format!("{prefix}.self_attn.fsmn_block.weight"))?;
        if fsmn.len() != channels * self.config.fsmn_kernel {
            return Err(format!("SenseVoice {prefix} fsmn 元素数 {}，期望 {}*{}", fsmn.len(), channels, self.config.fsmn_kernel));
        }
        // torch [channels, 1, kernel] → tap 主序 [kernel][channels]
        let mut fsmn_weight = vec![0.0f32; fsmn.len()];
        for (channel, values) in fsmn.chunks_exact(self.config.fsmn_kernel).enumerate() {
            for (tap, &value) in values.iter().enumerate() {
                fsmn_weight[tap * channels + channel] = value;
            }
        }
        Ok(SenseVoiceLayerWeights {
            norm1_weight: self.load_f32(&format!("{prefix}.norm1.weight"))?,
            norm1_bias: self.load_f32(&format!("{prefix}.norm1.bias"))?,
            qkv_weight: self.load_f32(&format!("{prefix}.self_attn.linear_q_k_v.weight"))?,
            qkv_bias: self.load_f32(&format!("{prefix}.self_attn.linear_q_k_v.bias"))?,
            attn_out_weight: self.load_f32(&format!("{prefix}.self_attn.linear_out.weight"))?,
            attn_out_bias: self.load_f32(&format!("{prefix}.self_attn.linear_out.bias"))?,
            fsmn_weight,
            norm2_weight: self.load_f32(&format!("{prefix}.norm2.weight"))?,
            norm2_bias: self.load_f32(&format!("{prefix}.norm2.bias"))?,
            ffn_up_weight: self.load_f32(&format!("{prefix}.feed_forward.w_1.weight"))?,
            ffn_up_bias: self.load_f32(&format!("{prefix}.feed_forward.w_1.bias"))?,
            ffn_down_weight: self.load_f32(&format!("{prefix}.feed_forward.w_2.weight"))?,
            ffn_down_bias: self.load_f32(&format!("{prefix}.feed_forward.w_2.bias"))?,
        })
    }

    /// after_norm（主 encoder 栈末）与 tp_norm（tp 栈末）的 (weight, bias)。
    pub fn final_norm(&self, tp: bool) -> Result<(Vec<f32>, Vec<f32>), String> {
        let name = if tp { "encoder.tp_norm" } else { "encoder.after_norm" };
        Ok((self.load_f32(&format!("{name}.weight"))?, self.load_f32(&format!("{name}.bias"))?))
    }

    /// CTC 头：`(weight [vocab, d_model], bias [vocab])`。
    pub fn ctc_head(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((self.load_f32("ctc.ctc_lo.weight")?, self.load_f32("ctc.ctc_lo.bias")?))
    }

    /// 解析 kaldi 全局 CMVN 文本（<AddShift> 行跟随 560 个 shift，<Rescale> 行跟随 560 个 scale）。
    pub fn cmvn(&self) -> Result<SenseVoiceCmvn, String> {
        let text = std::fs::read_to_string(self.root.join("am.mvn")).map_err(|error| format!("读取 am.mvn 失败: {error}"))?;
        let expected = self.config.encoder_input_dim();
        let mut shift = None;
        let mut scale = None;
        let lines: Vec<&str> = text.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.first() == Some(&"<AddShift>") || fields.first() == Some(&"<Rescale>") {
                let values = lines.get(index + 1).map(|next| next.split_whitespace().collect::<Vec<_>>()).unwrap_or_default();
                // kaldi mvn 第二行形如 "<LearnRateCoef> [ count v1 v2 ... ]"
                if values.first() != Some(&"<LearnRateCoef>") || values.len() < expected + 3 {
                    return Err(format!("am.mvn {} 段格式非法", fields[0]));
                }
                let parsed = values[3..3 + expected].iter().map(|value| value.parse::<f32>()).collect::<Result<Vec<_>, _>>().map_err(|error| format!("am.mvn {} 数值解析失败: {error}", fields[0]))?;
                if fields.first() == Some(&"<AddShift>") {
                    shift = Some(parsed);
                } else {
                    scale = Some(parsed);
                }
            }
        }
        Ok(SenseVoiceCmvn { shift: shift.ok_or("am.mvn 缺少 <AddShift> 段")?, scale: scale.ok_or("am.mvn 缺少 <Rescale> 段")? })
    }

    /// 解析 sherpa tokens.txt（每行 `<piece> <id>`）。
    pub fn tokens(&self) -> Result<SenseVoiceTokens, String> {
        let text = std::fs::read_to_string(self.root.join("tokens.txt")).map_err(|error| format!("读取 tokens.txt 失败: {error}"))?;
        let mut pieces = vec![String::new(); self.vocab_size];
        let mut seen = 0usize;
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let (piece, id) = line.rsplit_once(' ').ok_or_else(|| format!("tokens.txt 行缺少 id: {line:?}"))?;
            let id: usize = id.parse().map_err(|error| format!("tokens.txt id 解析失败: {error}"))?;
            if id >= pieces.len() {
                return Err(format!("tokens.txt id={id} 超出词表 {}", pieces.len()));
            }
            pieces[id] = piece.to_owned();
            seen += 1;
        }
        if seen != self.vocab_size {
            return Err(format!("tokens.txt 覆盖 {seen} 个 id，期望 {}", self.vocab_size));
        }
        Ok(SenseVoiceTokens { pieces })
    }
}

/// 层索引 → state_dict 前缀。0 = encoders0.0（首层，560 入）；
/// 1..layer_count = encoders.{i-1}；其后 = tp_encoders.{i-layer_count}。
fn layer_prefix(index: usize, layer_count: usize) -> String {
    if index == 0 {
        "encoder.encoders0.0".to_owned()
    } else if index < layer_count {
        format!("encoder.encoders.{}", index - 1)
    } else {
        format!("encoder.tp_encoders.{}", index - layer_count)
    }
}

#[cfg(test)]
mod tests {
    use super::layer_prefix;

    #[test]
    fn layer_prefix_mapping() {
        assert_eq!(layer_prefix(0, 50), "encoder.encoders0.0");
        assert_eq!(layer_prefix(1, 50), "encoder.encoders.0");
        assert_eq!(layer_prefix(49, 50), "encoder.encoders.48");
        assert_eq!(layer_prefix(50, 50), "encoder.tp_encoders.0");
        assert_eq!(layer_prefix(69, 50), "encoder.tp_encoders.19");
    }
}
