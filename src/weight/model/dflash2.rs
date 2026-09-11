//! DFlash2 Safetensors 命名与形状契约；embedding/LM head 由 target 所有者提供。

use crate::{
    model_spec::dflash2::Dflash2Config,
    weight::container::safetensor::{SafetensorStore, TensorData},
};
use serde::Deserialize;
use std::path::Path;

/// 持久化身份包含实际权重内容；同一路径替换 checkpoint 不能复用旧 aux/KV。
pub fn checkpoint_fingerprint(root: &Path) -> Result<String, String> {
    let mut hash = blake3::Hasher::new();
    let config = std::fs::read(root.join("config.json")).map_err(|e| format!("读取 DFlash2 config fingerprint: {e}"))?;
    parse_config(&config)?;
    hash.update(&config);
    let index_path = root.join("model.safetensors.index.json");
    let files = if index_path.exists() {
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let map = index.get("weight_map").and_then(serde_json::Value::as_object).ok_or("DFlash2 index 缺少 weight_map")?;
        map.values().map(|v| v.as_str().map(str::to_owned).ok_or_else(|| "DFlash2 shard 名称非字符串".to_owned())).collect::<Result<std::collections::BTreeSet<_>, _>>()?
    } else {
        std::collections::BTreeSet::from(["model.safetensors".to_owned()])
    };
    for file in files {
        hash.update(&(file.len() as u64).to_le_bytes());
        hash.update(file.as_bytes());
        let mut input = std::fs::File::open(root.join(&file)).map_err(|e| format!("DFlash2 fingerprint {file}: {e}"))?;
        hash.update_reader(&mut input).map_err(|e| format!("DFlash2 fingerprint {file}: {e}"))?;
    }
    Ok(hash.finalize().to_hex().to_string())
}

#[derive(Deserialize)]
struct CheckpointConfig {
    architectures: Vec<String>,
    model_type: String,
    dtype: String,
    attention_bias: bool,
    attention_dropout: f32,
    hidden_act: String,
    is_causal: bool,
    layer_types: Vec<String>,
    use_sliding_window: bool,
    tie_word_embeddings: bool,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    num_target_layers: usize,
    sliding_window: usize,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    rope_parameters: RopeConfig,
    dflash_config: DraftConfig,
}

#[derive(Deserialize)]
struct RopeConfig {
    rope_type: String,
    rope_theta: f32,
}

#[derive(Deserialize)]
struct DraftConfig {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: u32,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
}

pub fn parse_config(json: &[u8]) -> Result<Dflash2Config, String> {
    let raw: CheckpointConfig = serde_json::from_slice(json).map_err(|e| format!("解析 DFlash2 config: {e}"))?;
    if raw.architectures != ["DFlash2DraftModel"]
        || raw.model_type != "qwen3"
        || raw.dtype != "bfloat16"
        || raw.attention_bias
        || raw.attention_dropout != 0.0
        || raw.hidden_act != "silu"
        || raw.is_causal
        || !raw.use_sliding_window
        || raw.tie_word_embeddings
        || raw.rope_parameters.rope_type != "default"
        || raw.layer_types.len() != raw.num_hidden_layers
        || raw.layer_types.iter().any(|kind| kind != "sliding_attention")
    {
        return Err("DFlash2 当前仅支持 BF16 Qwen3/SILU、无 bias/dropout、非因果滑窗、default RoPE 的 DFlash2DraftModel".into());
    }
    let draft = raw.dflash_config;
    let config = Dflash2Config {
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        layer_count: raw.num_hidden_layers,
        head_count: raw.num_attention_heads,
        kv_head_count: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        vocab_size: raw.vocab_size,
        target_layer_count: raw.num_target_layers,
        target_layer_ids: draft.target_layer_ids,
        block_size: draft.block_size,
        sliding_window: raw.sliding_window,
        mask_token_id: draft.mask_token_id,
        conv_group_size: draft.conv_group_size,
        conv_kernel_size: draft.conv_kernel_size,
        selector_rank: draft.selector_rank,
        selector_top_k: draft.selector_top_k,
        rms_eps: raw.rms_norm_eps,
        rope_theta: raw.rope_parameters.rope_theta,
        max_position_embeddings: raw.max_position_embeddings,
    };
    config.validate()?;
    Ok(config)
}

#[derive(Clone)]
pub struct Dflash2Checkpoint {
    store: SafetensorStore,
    config: Dflash2Config,
}

impl Dflash2Checkpoint {
    pub fn open(root: &Path) -> Result<Self, String> {
        let path = root.join("config.json");
        let config = parse_config(&std::fs::read(&path).map_err(|e| format!("读取 {}: {e}", path.display()))?)?;
        let checkpoint = Self { store: SafetensorStore::open(root)?, config };
        let expected = tensor_shapes(&checkpoint.config);
        for (name, shape) in &expected {
            let info = checkpoint.store.tensor_info(name)?;
            if info.dtype != "BF16" || info.shape != *shape {
                return Err(format!("DFlash2 {name}: {} {:?}，期望 BF16 {shape:?}", info.dtype, info.shape));
            }
        }
        // 不静默忽略不同网络变体增加的权重，尤其不能把 DSpark 当成 DFlash2。
        for name in checkpoint.store.tensor_names() {
            if !expected.iter().any(|(key, _)| key == &name) {
                return Err(format!("DFlash2 未知张量 {name}，checkpoint 与执行规格不一致"));
            }
        }
        Ok(checkpoint)
    }

    pub fn config(&self) -> &Dflash2Config {
        &self.config
    }

    pub fn load(&self, name: &str) -> Result<TensorData, String> {
        self.store.load(name)
    }

    /// 每个 target stage 只装载自己的 FC 列切片，跨 stage 累加投影后的 hidden。
    pub fn capture_projection(&self, index: usize) -> Result<TensorData, String> {
        if index >= self.config.target_layer_ids.len() {
            return Err(format!("DFlash2 capture index={index} 超出 {}", self.config.target_layer_ids.len()));
        }
        let h = self.config.hidden_size;
        self.store.load_columns("fc.weight", index * h..(index + 1) * h)
    }
}

fn tensor_shapes(c: &Dflash2Config) -> Vec<(String, Vec<usize>)> {
    let h = c.hidden_size;
    let mut shapes = vec![
        ("fc.weight".into(), vec![h, h * c.target_layer_ids.len()]),
        ("hidden_norm.weight".into(), vec![h]),
        ("norm.weight".into(), vec![h]),
        ("candidate_selector.hidden_projection.weight".into(), vec![c.selector_rank, h]),
        ("candidate_selector.predecessor_codebook".into(), vec![c.vocab_size, c.selector_rank]),
        ("candidate_selector.successor_codebook".into(), vec![c.vocab_size, c.selector_rank]),
    ];
    for layer in 0..c.layer_count {
        let q = c.head_count * c.head_dim;
        let kv = c.kv_head_count * c.head_dim;
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.q_norm.weight", vec![c.head_dim]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.k_norm.weight", vec![c.head_dim]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("mlp.gate_proj.weight", vec![c.intermediate_size, h]),
            ("mlp.up_proj.weight", vec![c.intermediate_size, h]),
            ("mlp.down_proj.weight", vec![h, c.intermediate_size]),
            ("attention_conv.base_kernel", vec![2, c.conv_kernel_size, h]),
            ("attention_conv.kernel_projection.weight", vec![2 * c.conv_kernel_size * (h / c.conv_group_size), h]),
            ("mlp_conv.base_kernel", vec![2, c.conv_kernel_size, h]),
            ("mlp_conv.kernel_projection.weight", vec![2 * c.conv_kernel_size * (h / c.conv_group_size), h]),
        ] {
            shapes.push((format!("layers.{layer}.{suffix}"), shape));
        }
    }
    shapes
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    pub(crate) struct Fixture(pub PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn config_json(c: &Dflash2Config) -> serde_json::Value {
        json!({"architectures":["DFlash2DraftModel"],"model_type":"qwen3","dtype":"bfloat16", "attention_bias":false,"attention_dropout":0.0,
            "hidden_act":"silu","is_causal":false,"use_sliding_window":true,"tie_word_embeddings":false,
            "layer_types":vec!["sliding_attention"; c.layer_count],"hidden_size":c.hidden_size,"intermediate_size":c.intermediate_size,
            "num_hidden_layers":c.layer_count,"num_attention_heads":c.head_count,"num_key_value_heads":c.kv_head_count,"head_dim":c.head_dim,
            "vocab_size":c.vocab_size,"num_target_layers":c.target_layer_count,"sliding_window":c.sliding_window,"rms_norm_eps":c.rms_eps,
            "max_position_embeddings":c.max_position_embeddings,"rope_parameters":{"rope_type":"default","rope_theta":c.rope_theta},
            "dflash_config":{"block_size":c.block_size,"conv_group_size":c.conv_group_size,"conv_kernel_size":c.conv_kernel_size,
                "mask_token_id":c.mask_token_id,"selector_rank":c.selector_rank,"selector_top_k":c.selector_top_k,"target_layer_ids":c.target_layer_ids}})
    }

    #[test]
    fn dflash2_fingerprint_detects_replaced_weights() {
        let files = fixture(&tiny_config());
        let first = checkpoint_fingerprint(&files.0).unwrap();
        assert_eq!(first, checkpoint_fingerprint(&files.0).unwrap());
        let path = files.0.join("model.safetensors");
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(path, bytes).unwrap();
        assert_ne!(first, checkpoint_fingerprint(&files.0).unwrap());
    }

    pub(crate) fn tiny_config() -> Dflash2Config {
        Dflash2Config {
            hidden_size: 4,
            intermediate_size: 6,
            layer_count: 2,
            head_count: 2,
            kv_head_count: 1,
            head_dim: 2,
            vocab_size: 5,
            target_layer_count: 4,
            target_layer_ids: vec![0, 2],
            block_size: 3,
            sliding_window: 3,
            mask_token_id: 4,
            conv_group_size: 2,
            conv_kernel_size: 2,
            selector_rank: 2,
            selector_top_k: 3,
            rms_eps: 1e-5,
            rope_theta: 10000.,
            max_position_embeddings: 16,
        }
    }

    pub(crate) fn fixture(c: &Dflash2Config) -> Fixture {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!("zllm-dflash2-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("config.json"), serde_json::to_vec(&config_json(c)).unwrap()).unwrap();
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape) in tensor_shapes(c) {
            let seed: usize = name.bytes().map(usize::from).sum();
            let begin = data.len();
            for i in 0..shape.iter().product() {
                let mut value = (((i * 7 + seed) % 17) as f32 - 8.) / 32.;
                if name.ends_with("norm.weight") {
                    value = 1. + ((i + seed) % 3) as f32 / 16.;
                }
                if name.ends_with("base_kernel") && (i / c.hidden_size) % c.conv_kernel_size == 0 {
                    value += 1.;
                }
                data.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
            }
            header.insert(name, json!({"dtype":"BF16","shape":shape,"data_offsets":[begin,data.len()]}));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(data);
        std::fs::write(root.join("model.safetensors"), file).unwrap();
        Fixture(root)
    }

    #[test]
    fn official_config_and_invalid_variants() {
        let c = Dflash2Config::glm53();
        assert_eq!(tensor_shapes(&c).iter().map(|(_, shape)| shape.iter().product::<usize>()).sum::<usize>(), 2_459_424_256);
        assert_eq!(parse_config(&serde_json::to_vec(&config_json(&c)).unwrap()).unwrap(), c);
        for (field, value) in [("is_causal", json!(true)), ("attention_bias", json!(true)), ("num_hidden_layers", json!(7)), ("hidden_act", json!("gelu"))] {
            let mut raw = config_json(&c);
            raw[field] = value;
            assert!(parse_config(&serde_json::to_vec(&raw).unwrap()).is_err(), "{field}");
        }
        for (field, value) in [("target_layer_ids", json!([5, 5])), ("target_layer_ids", json!([78])), ("conv_group_size", json!(0)), ("selector_top_k", json!(154881))] {
            let mut raw = config_json(&c);
            raw["dflash_config"][field] = value;
            assert!(parse_config(&serde_json::to_vec(&raw).unwrap()).is_err(), "{field}");
        }
        let mut overflow = c;
        overflow.hidden_size = usize::MAX - 15;
        assert!(overflow.validate().is_err());
    }

    #[test]
    fn checkpoint_has_no_target_head_and_slices_fc_columns() {
        let c = tiny_config();
        let files = fixture(&c);
        let checkpoint = Dflash2Checkpoint::open(&files.0).unwrap();
        assert!(checkpoint.load("embed_tokens.weight").is_err());
        assert!(checkpoint.load("lm_head.weight").is_err());
        let full = checkpoint.load("fc.weight").unwrap().to_f32().unwrap();
        for capture in 0..2 {
            let slice = checkpoint.capture_projection(capture).unwrap();
            assert_eq!(slice.shape, [4, 4]);
            let expected: Vec<_> = full.chunks_exact(8).flat_map(|row| row[capture * 4..capture * 4 + 4].iter().copied()).collect();
            assert_eq!(slice.to_f32().unwrap(), expected);
        }
        assert!(checkpoint.capture_projection(2).is_err());
        // shape 相同数量不同的层也不能静默缺失。
        let mut raw = config_json(&c);
        raw["num_hidden_layers"] = json!(3);
        raw["layer_types"] = json!(["sliding_attention", "sliding_attention", "sliding_attention"]);
        std::fs::write(files.0.join("config.json"), serde_json::to_vec(&raw).unwrap()).unwrap();
        assert!(Dflash2Checkpoint::open(&files.0).is_err());
    }

    #[test]
    fn rejects_wrong_dtype_shape_and_extra_weights() {
        for case in 0..3 {
            let files = fixture(&tiny_config());
            let path = files.0.join("model.safetensors");
            let file = std::fs::read(&path).unwrap();
            let length = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
            let mut header: serde_json::Value = serde_json::from_slice(&file[8..8 + length]).unwrap();
            match case {
                0 => header["norm.weight"]["dtype"] = json!("F16"),
                1 => header["fc.weight"]["shape"] = json!([8, 4]),
                _ => header["unexpected.weight"] = header["norm.weight"].clone(),
            }
            let header = serde_json::to_vec(&header).unwrap();
            let mut changed = (header.len() as u64).to_le_bytes().to_vec();
            changed.extend(header);
            changed.extend(&file[8 + length..]);
            std::fs::write(path, changed).unwrap();
            let error = Dflash2Checkpoint::open(&files.0).err().expect("非法 checkpoint 必须拒绝");
            assert!(
                error.contains(match case {
                    0 => "norm.weight",
                    1 => "fc.weight",
                    _ => "unexpected.weight",
                }),
                "{error}"
            );
        }
    }
}
