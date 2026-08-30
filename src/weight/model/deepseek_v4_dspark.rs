//! DeepSeek-V4 DSpark 权重规格；执行复用 DeepSeek layer 与通用 speculative 协议。

use std::path::Path;

use serde::Deserialize;

use crate::{
    model_spec::deepseek_v4::DeepSeekV4Config,
    speculative::{BlockDraftSpec, HiddenStateCapturePlan},
    weight::{
        container::safetensor::TensorData,
        format::block_fp8::BlockFp8Matrix,
        model::deepseek_v4::{DeepSeekV4ExpertSource, DeepSeekV4HeadWeights, DeepSeekV4HyperConnectionWeights, DeepSeekV4LayerWeights, DeepSeekV4Weights},
    },
};

#[derive(Clone, Debug, Deserialize)]
pub struct DeepSeekV4DsparkConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: u32,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
}

impl DeepSeekV4DsparkConfig {
    pub fn read(root: &Path, target: &DeepSeekV4Config) -> Result<Self, String> {
        let path = root.join("config.json");
        let config: Self = serde_json::from_slice(&std::fs::read(&path).map_err(|error| format!("读取 {} 失败: {error}", path.display()))?).map_err(|error| format!("解析 {} 失败: {error}", path.display()))?;
        config.validate(target)?;
        Ok(config)
    }

    pub fn block_spec(&self) -> Result<BlockDraftSpec, String> {
        BlockDraftSpec::new(self.dspark_block_size, self.dspark_block_size, 1).map_err(|error| error.to_string())
    }

    pub fn capture_plan(&self, verifier_layer_count: usize) -> Result<HiddenStateCapturePlan, String> {
        let boundaries = self.dspark_target_layer_ids.iter().map(|layer| layer.checked_add(1).ok_or("DeepSeek-V4 DSpark target layer boundary 溢出")).collect::<Result<Vec<_>, _>>()?;
        HiddenStateCapturePlan::new(boundaries, verifier_layer_count).map_err(|error| error.to_string())
    }

    fn validate(&self, target: &DeepSeekV4Config) -> Result<(), String> {
        if self.vocab_size != target.vocab_size || self.hidden_size != target.hidden_size || self.num_hidden_layers != target.layer_count {
            return Err(format!("DeepSeek-V4 DSpark target 规格不匹配: vocab={}/{} hidden={}/{} layers={}/{}", self.vocab_size, target.vocab_size, self.hidden_size, target.hidden_size, self.num_hidden_layers, target.layer_count));
        }
        if self.dspark_target_layer_ids.len() != target.mtp_layer_count || self.dspark_target_layer_ids.windows(2).any(|pair| pair[0] >= pair[1]) || self.dspark_target_layer_ids.iter().any(|&layer| layer >= target.layer_count) {
            return Err(format!("DeepSeek-V4 DSpark target layers 非法: {:?}, mtp={}", self.dspark_target_layer_ids, target.mtp_layer_count));
        }
        if self.dspark_markov_rank == 0 || usize::try_from(self.dspark_noise_token_id).unwrap_or(usize::MAX) >= self.vocab_size {
            return Err(format!("DeepSeek-V4 DSpark Markov/noise 非法: rank={} noise={}", self.dspark_markov_rank, self.dspark_noise_token_id));
        }
        self.block_spec()?;
        self.capture_plan(target.layer_count)?;
        Ok(())
    }
}

pub struct DeepSeekV4DsparkInputWeights {
    pub projection: BlockFp8Matrix,
    pub norm: TensorData,
}

pub struct DeepSeekV4DsparkHeadWeights {
    pub hyper_connection: DeepSeekV4HyperConnectionWeights,
    pub norm: TensorData,
    pub markov_embedding: TensorData,
    pub markov_projection: TensorData,
    pub confidence_projection: TensorData,
}

#[derive(Clone)]
pub struct DeepSeekV4DsparkCheckpoint {
    weights: DeepSeekV4Weights,
    pub config: DeepSeekV4DsparkConfig,
}

impl DeepSeekV4DsparkCheckpoint {
    pub fn open(root: &Path, target: DeepSeekV4Config) -> Result<Self, String> {
        let config = DeepSeekV4DsparkConfig::read(root, &target)?;
        let checkpoint = Self { weights: DeepSeekV4Weights::open(root, target)?, config };
        checkpoint.validate_head_tensors()?;
        Ok(checkpoint)
    }

    pub fn target_weights(&self) -> &DeepSeekV4Weights {
        &self.weights
    }

    pub fn load_input(&self) -> Result<DeepSeekV4DsparkInputWeights, String> {
        let hidden = self.config.hidden_size;
        let captures = self.config.dspark_target_layer_ids.len();
        Ok(DeepSeekV4DsparkInputWeights {
            projection: self.weights.load_block_fp8("mtp.0.main_proj", hidden, hidden.checked_mul(captures).ok_or("DeepSeek-V4 DSpark main projection 维度溢出")?)?,
            norm: self.weights.load_dense("mtp.0.main_norm.weight", &[hidden])?,
        })
    }

    pub fn load_layer(&self, layer: usize) -> Result<DeepSeekV4LayerWeights, String> {
        self.weights.load_mtp_layer(layer)
    }

    pub fn expert_source(&self) -> DeepSeekV4ExpertSource {
        self.weights.mtp_expert_source()
    }

    pub fn load_head(&self) -> Result<DeepSeekV4DsparkHeadWeights, String> {
        let hidden = self.config.hidden_size;
        let rank = self.config.dspark_markov_rank;
        let copies = self.weights.config().hyper_connection_copies;
        let last = self.config.dspark_target_layer_ids.len() - 1;
        let prefix = format!("mtp.{last}");
        Ok(DeepSeekV4DsparkHeadWeights {
            hyper_connection: DeepSeekV4HyperConnectionWeights {
                function: self.weights.load_f32(&format!("{prefix}.hc_head_fn"), &[copies, copies.checked_mul(hidden).ok_or("DeepSeek-V4 DSpark head hidden 维度溢出")?])?,
                base: self.weights.load_f32(&format!("{prefix}.hc_head_base"), &[copies])?,
                scale: self.weights.load_f32(&format!("{prefix}.hc_head_scale"), &[1])?,
            },
            norm: self.weights.load_dense(&format!("{prefix}.norm.weight"), &[hidden])?,
            markov_embedding: self.weights.load_dense(&format!("{prefix}.markov_head.markov_w1.weight"), &[self.config.vocab_size, rank])?,
            markov_projection: self.weights.load_dense(&format!("{prefix}.markov_head.markov_w2.weight"), &[self.config.vocab_size, rank])?,
            confidence_projection: self.weights.load_dense(&format!("{prefix}.confidence_head.proj.weight"), &[1, hidden + rank])?,
        })
    }

    pub fn target_head(&self) -> Result<DeepSeekV4HeadWeights, String> {
        self.weights.output_head()
    }

    fn validate_head_tensors(&self) -> Result<(), String> {
        self.load_input()?;
        self.load_head()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_official_dspark_config() {
        let config: DeepSeekV4DsparkConfig = serde_json::from_str(
            r#"{"vocab_size":129280,"hidden_size":4096,"num_hidden_layers":43,
            "dspark_block_size":5,"dspark_noise_token_id":128799,
            "dspark_target_layer_ids":[40,41,42],"dspark_markov_rank":256}"#,
        )
        .unwrap();
        let target = DeepSeekV4Config::flash();
        config.validate(&target).unwrap();
        assert_eq!(config.capture_plan(43).unwrap().boundaries(), [41, 42, 43]);
        assert_eq!(config.block_spec().unwrap(), BlockDraftSpec { block_size: 5, speculative_tokens: 5, verifier_accept_k: 1 });
    }
}
