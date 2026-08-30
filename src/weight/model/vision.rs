//! 通用 Vision Transformer 权重分组。
//!
//! 保留 safetensors 原始 dtype，CPU/Metal backend 在真正消费时决定解码或上传。

use crate::weight::container::safetensor::TensorData;

#[derive(Debug)]
pub struct LinearWeights {
    pub weight: TensorData,
    pub bias: Option<TensorData>,
}

#[derive(Debug)]
pub struct LayerNormWeights {
    pub weight: TensorData,
    pub bias: TensorData,
}

#[derive(Debug)]
pub struct VisionAttentionWeights {
    pub query: LinearWeights,
    pub key: LinearWeights,
    pub value: LinearWeights,
    pub output: LinearWeights,
}

#[derive(Debug)]
pub struct VisionMlpWeights {
    pub input: LinearWeights,
    pub output: LinearWeights,
}

#[derive(Debug)]
pub struct VisionEncoderLayerWeights {
    pub input_norm: LayerNormWeights,
    pub attention: VisionAttentionWeights,
    pub post_attention_norm: LayerNormWeights,
    pub mlp: VisionMlpWeights,
}
